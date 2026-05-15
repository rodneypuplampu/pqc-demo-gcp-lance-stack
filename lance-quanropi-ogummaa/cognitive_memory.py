"""
core/cognitive_memory.py
========================
OgummaaAI Cognitive Memory — 5-Pillar Agentic Cognition over LanceDB.

Pillars:
  1. Extract  – Gemini decomposes raw text into atomic facts
  2. Encode   – Arrow schema + Blob V2 storage in LanceDB
  3. Consolidate – Vector-similarity dedup / contradiction resolution
  4. Recall   – Composite scoring: Similarity × W_sim + Recency × W_rec + Importance × W_imp
  5. Forget   – TTL-based pruning to prevent unbounded memory growth
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import lancedb
import pyarrow as pa
from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768  # text-embedding-004 dimensionality

MEMORY_SCHEMA = pa.schema([
    pa.field("memory_id",   pa.string()),
    pa.field("session_id",  pa.string()),
    pa.field("scope",       pa.string()),          # e.g. "historical", "geospatial", "session"
    pa.field("content",     pa.string()),           # the atomic fact
    pa.field("importance",  pa.float32()),          # 0.0 – 1.0
    pa.field("timestamp",   pa.float64()),          # unix epoch
    pa.field("media_blob",  pa.binary()),           # Blob V2 — raw image / tile bytes (nullable)
    pa.field("vector",      pa.list_(pa.float32(), EMBEDDING_DIM)),
])

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

W_SIM = 0.60
W_REC = 0.20
W_IMP = 0.20
CONSOLIDATION_THRESHOLD = 0.92   # cosine similarity above which we overwrite
MAX_RECALL_CANDIDATES   = 20
TTL_SECONDS             = 86_400 * 30   # 30-day default TTL

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    memory_id:  str
    session_id: str
    scope:      str
    content:    str
    importance: float
    timestamp:  float
    media_blob: bytes | None
    vector:     list[float]


# ---------------------------------------------------------------------------
# CognitiveMemory
# ---------------------------------------------------------------------------

class CognitiveMemory:
    """
    Active cognitive substrate backed by LanceDB + Google Gemini.

    Usage::

        mem = CognitiveMemory(lancedb_path="gs://lance-data-adk-02/gummaa_workspace.lance")
        await mem.remember("session-abc", "Ashanti kingdom traded gold along the Niger...")
        facts = await mem.recall("session-abc", "Gold trade routes in West Africa", k=5)
    """

    def __init__(
        self,
        lancedb_path: str,
        table_name: str = "cognitive_memory",
        gemini_client: genai.Client | None = None,
        project_id: str | None = None,
        location: str = "us-central1",
    ) -> None:
        self.lancedb_path = lancedb_path
        self.table_name   = table_name
        self.project_id   = project_id or os.getenv("PROJECT_ID", "multi-agent-lab01")
        self.location     = location

        # Gemini client (Vertex AI backend)
        self._client = gemini_client or genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

        # LanceDB connection
        self._db    = lancedb.connect(lancedb_path)
        self._table = self._get_or_create_table()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_table(self) -> lancedb.table.Table:
        """Return existing table or create it with the cognitive schema."""
        existing = self._db.table_names()
        if self.table_name in existing:
            return self._db.open_table(self.table_name)
        return self._db.create_table(self.table_name, schema=MEMORY_SCHEMA)

    def _embed(self, text: str) -> list[float]:
        """Generate a 768-dim embedding via Vertex AI text-embedding-004."""
        response = self._client.models.embed_content(
            model="text-embedding-004",
            contents=text,
        )
        return response.embeddings[0].values

    def _extract_atomic_facts(self, raw_text: str, session_id: str) -> list[dict[str, Any]]:
        """
        Pillar 1 — Extract.
        Ask Gemini to decompose raw text into self-contained atomic facts with
        importance scores and scopes. Returns a list of dicts.
        """
        prompt = f"""You are a knowledge extraction engine for a historical geospatial AI system.
Decompose the following text into self-contained atomic facts.

For each fact, output a JSON object with:
  - "content": the atomic fact as a concise declarative sentence
  - "importance": float 0.0–1.0 (higher = more historically / geographically significant)
  - "scope": one of ["historical", "geospatial", "political", "cultural", "session"]

Return ONLY a JSON array of these objects, no markdown fences.

TEXT:
{raw_text}
"""
        response = self._client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = response.text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            facts = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Fact extraction returned non-JSON; treating as single fact.")
            facts = [{"content": raw_text, "importance": 0.5, "scope": "session"}]
        return facts

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x ** 2 for x in a) ** 0.5
        mag_b = sum(x ** 2 for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ------------------------------------------------------------------
    # Pillar 2+3: Encode + Consolidate
    # ------------------------------------------------------------------

    def remember(
        self,
        session_id: str,
        raw_text: str,
        media_blob: bytes | None = None,
    ) -> list[str]:
        """
        Extract atomic facts from *raw_text*, embed them, consolidate against
        existing memories, and persist to LanceDB.

        Returns a list of memory_ids that were written (new or updated).
        """
        facts       = self._extract_atomic_facts(raw_text, session_id)
        written_ids = []

        for fact in facts:
            content    = fact.get("content", "")
            importance = float(fact.get("importance", 0.5))
            scope      = fact.get("scope", "session")

            if not content:
                continue

            vector = self._embed(content)

            # Pillar 3 — Consolidate: check for near-duplicate existing facts
            consolidated = self._consolidate(session_id, content, vector, importance, scope, media_blob)
            written_ids.append(consolidated)

        return written_ids

    def _consolidate(
        self,
        session_id: str,
        content: str,
        vector: list[float],
        importance: float,
        scope: str,
        media_blob: bytes | None,
    ) -> str:
        """
        If a near-identical memory already exists (cosine ≥ threshold),
        overwrite it (update importance + timestamp). Otherwise insert new.
        """
        try:
            results = (
                self._table
                .search(vector)
                .where(f"session_id = '{session_id}'")
                .limit(1)
                .to_list()
            )
        except Exception:
            results = []

        now = time.time()

        if results:
            top      = results[0]
            existing = top.get("vector", [])
            sim      = self._cosine_similarity(vector, existing)

            if sim >= CONSOLIDATION_THRESHOLD:
                mid = top["memory_id"]
                # Merge: keep higher importance, refresh timestamp
                merged_importance = max(importance, top.get("importance", 0.0))
                self._table.update(
                    where=f"memory_id = '{mid}'",
                    values={
                        "importance": merged_importance,
                        "timestamp":  now,
                        "content":    content,   # newer phrasing wins
                        "vector":     vector,
                    },
                )
                logger.debug("Consolidated memory %s (sim=%.3f)", mid, sim)
                return mid

        # Insert new memory
        mid = str(uuid.uuid4())
        row = {
            "memory_id":  mid,
            "session_id": session_id,
            "scope":      scope,
            "content":    content,
            "importance": importance,
            "timestamp":  now,
            "media_blob": media_blob or b"",
            "vector":     vector,
        }
        self._table.add([row])
        return mid

    # ------------------------------------------------------------------
    # Pillar 4: Adaptive Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        session_id: str,
        query: str,
        k: int = 5,
        scope_filter: str | None = None,
    ) -> list[MemoryRecord]:
        """
        Retrieve the top-*k* memories using composite scoring:

            Score = (Similarity × 0.60) + (Recency × 0.20) + (Importance × 0.20)

        Recency is normalised so the most-recent memory scores 1.0.
        """
        query_vec = self._embed(query)

        where_clause = f"session_id = '{session_id}'"
        if scope_filter:
            where_clause += f" AND scope = '{scope_filter}'"

        try:
            raw = (
                self._table
                .search(query_vec)
                .where(where_clause)
                .limit(MAX_RECALL_CANDIDATES)
                .to_list()
            )
        except Exception as exc:
            logger.warning("Recall search failed: %s", exc)
            return []

        if not raw:
            return []

        now        = time.time()
        timestamps = [r["timestamp"] for r in raw]
        min_ts     = min(timestamps)
        max_ts     = max(timestamps)
        ts_range   = max_ts - min_ts or 1.0

        scored = []
        for r in raw:
            # LanceDB ANN search returns _distance (L2); convert to cosine-ish similarity
            distance   = r.get("_distance", 0.0)
            similarity = 1.0 / (1.0 + distance)   # monotone proxy

            recency    = (r["timestamp"] - min_ts) / ts_range
            importance = float(r.get("importance", 0.5))
            score      = W_SIM * similarity + W_REC * recency + W_IMP * importance
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            MemoryRecord(
                memory_id  = r["memory_id"],
                session_id = r["session_id"],
                scope      = r["scope"],
                content    = r["content"],
                importance = r["importance"],
                timestamp  = r["timestamp"],
                media_blob = r.get("media_blob") or None,
                vector     = r["vector"],
            )
            for _, r in scored[:k]
        ]

    # ------------------------------------------------------------------
    # Pillar 5: Forget (TTL pruning)
    # ------------------------------------------------------------------

    def forget(self, ttl_seconds: float = TTL_SECONDS) -> int:
        """
        Delete memories older than *ttl_seconds*. Returns the count removed.
        """
        cutoff = time.time() - ttl_seconds
        try:
            before = self._table.count_rows()
            self._table.delete(f"timestamp < {cutoff}")
            after  = self._table.count_rows()
            removed = before - after
            logger.info("Forget pruning removed %d memories.", removed)
            return removed
        except Exception as exc:
            logger.warning("Forget operation failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return basic table statistics."""
        try:
            count = self._table.count_rows()
        except Exception:
            count = -1
        return {
            "table":    self.table_name,
            "path":     self.lancedb_path,
            "rows":     count,
            "emb_dim":  EMBEDDING_DIM,
        }
