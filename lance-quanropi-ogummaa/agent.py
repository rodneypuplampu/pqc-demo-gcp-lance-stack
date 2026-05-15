"""
core/agent.py
=============
lance-agno Compute-Bound Orchestrator — Gummaa-Atlas.

Multi-agent workflow:
  Coordinator   → decomposes the user query
  GraphAgent    → cross-correlates historical knowledge graph relationships
  DiscovAgent   → surface-discovers geospatial entities and routes geocoding
  Synthesizer   → collapses all agent outputs into a coherent response

Key mechanics:
  - Cognitive memory recall injected before each LLM call
  - safe_execute() uses Lance .checkout() Time-Travel for zero-penalty rollback
  - All agents share the same CognitiveMemory instance
"""

from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from google import genai
from google.genai import types as genai_types

from core.cognitive_memory import CognitiveMemory
from core.tools import (
    build_map_overlay,
    classify_region,
    geocode,
    haversine_distance,
    reverse_geocode,
)

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Agent result container
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    agent_name:  str
    output:      str
    geojson:     dict | None = None
    map_points:  list[dict] = field(default_factory=list)
    error:       str | None = None


# ---------------------------------------------------------------------------
# GummaaAgent — Compute-Bound Orchestrator
# ---------------------------------------------------------------------------

class GummaaAgent:
    """
    Orchestrates the Coordinator / GraphAgent / DiscovAgent / Synthesizer
    workflow with cognitive memory recall and lance Time-Travel safety.
    """

    def __init__(
        self,
        lancedb_path: str,
        project_id: str | None = None,
        location: str = "us-central1",
    ) -> None:
        self.project_id = project_id or os.getenv("PROJECT_ID", "multi-agent-lab01")
        self.location   = location

        self._client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

        self.memory = CognitiveMemory(
            lancedb_path=lancedb_path,
            project_id=self.project_id,
            location=self.location,
            gemini_client=self._client,
        )

        logger.info(
            "GummaaAgent initialised. Memory stats: %s",
            self.memory.stats(),
        )

    # ------------------------------------------------------------------
    # Time-Travel safe execution
    # ------------------------------------------------------------------

    def safe_execute(
        self,
        fn: Callable[[], Any],
        rollback_version: int | None = None,
    ) -> tuple[Any, bool]:
        """
        Execute *fn* inside a LanceDB time-travel checkpoint.

        On success  → return (result, True)
        On failure  → roll back to *rollback_version* (or latest) and
                      return (error_string, False)
        """
        table = self.memory._table

        # Snapshot current version
        try:
            current_version = table.version()
        except Exception:
            current_version = None

        try:
            result = fn()
            return result, True
        except Exception as exc:
            logger.warning("safe_execute caught exception: %s", exc)
            # Roll back
            target = rollback_version if rollback_version is not None else current_version
            if target is not None:
                try:
                    table.checkout(target)
                    logger.info("Rolled back to LanceDB version %d", target)
                except Exception as rb_exc:
                    logger.error("Rollback failed: %s", rb_exc)
            return str(exc), False

    # ------------------------------------------------------------------
    # Sub-agents
    # ------------------------------------------------------------------

    def _coordinator(
        self,
        session_id: str,
        user_query: str,
        coordinates: dict | None,
        memory_context: str,
    ) -> AgentResult:
        """
        Coordinator: parse query intent, extract named historical entities,
        and produce a structured task description for downstream agents.
        """
        coord_str = ""
        if coordinates:
            lat = coordinates.get("lat")
            lng = coordinates.get("lng")
            if lat and lng:
                region   = classify_region(float(lat), float(lng))
                location = reverse_geocode(float(lat), float(lng))
                coord_str = (
                    f"\nThe user has clicked on the map at "
                    f"lat={lat:.4f}, lng={lng:.4f} ({location}). "
                    f"This falls within the historical region: {region}."
                )

        prompt = f"""You are the Coordinator in a historical geospatial AI system called Gummaa-Atlas.

## Cognitive Memory Context (cross-correlated):
{memory_context or "No prior context."}

## User Query:
{user_query}{coord_str}

Your task:
1. Identify the core historical intent and any named places, kingdoms, empires, or trade routes.
2. Determine whether the query requires geospatial resolution (place → coordinates) or primarily historical narrative.
3. Output a structured task brief in plain English for the GraphAgent and DiscovAgent.
4. Be concise — this output feeds other agents, not the user directly.
"""
        response = self._client.models.generate_content(
            model=MODEL_ID, contents=prompt
        )
        return AgentResult(agent_name="Coordinator", output=response.text)

    def _graph_agent(
        self,
        session_id: str,
        coordinator_brief: str,
        memory_context: str,
    ) -> AgentResult:
        """
        GraphAgent: produce a narrative knowledge graph of historical relationships,
        causality chains, and temporal context.
        """
        prompt = f"""You are the GraphAgent in Gummaa-Atlas, an expert in African and world history.

## Coordinator Brief:
{coordinator_brief}

## Cognitive Memory Context:
{memory_context or "No prior context."}

Your task:
1. Identify key historical entities (kingdoms, empires, trade routes, leaders, events).
2. Describe their relationships and influence chains.
3. Provide temporal context (approximate dates / centuries).
4. Surface any contradictions with the memory context and resolve them.
5. Produce a rich, scholarly narrative paragraph (3–6 sentences) summarising the historical significance.

Focus on Pan-African history, pre-colonial African civilisations, the Trans-Saharan trade,
the Atlantic world, and global imperial systems where relevant.
"""
        response = self._client.models.generate_content(
            model=MODEL_ID, contents=prompt
        )
        return AgentResult(agent_name="GraphAgent", output=response.text)

    def _discovery_agent(
        self,
        session_id: str,
        coordinator_brief: str,
        user_query: str,
        coordinates: dict | None,
    ) -> AgentResult:
        """
        DiscovAgent: resolve named places to coordinates, build GeoJSON overlay.
        """
        prompt = f"""You are the DiscovAgent in Gummaa-Atlas, specialised in geospatial discovery.

## Coordinator Brief:
{coordinator_brief}

## Original Query:
{user_query}

Extract up to 5 specific historical places, cities, kingdoms, or geographic features from the brief.
For each, output ONLY a JSON array (no markdown) of objects:
  {{"name": "...", "description": "...", "importance": 0.0-1.0, "scope": "historical"}}

Return ONLY the JSON array. No explanatory text.
"""
        response = self._client.models.generate_content(
            model=MODEL_ID, contents=prompt
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        import json
        try:
            places_raw = json.loads(raw)
        except Exception:
            places_raw = []

        map_points: list[dict] = []
        for p in places_raw:
            name = p.get("name", "")
            if not name:
                continue
            place = geocode(name)
            if place:
                map_points.append({
                    "title":       name,
                    "description": p.get("description", ""),
                    "importance":  p.get("importance", 0.5),
                    "scope":       p.get("scope", "historical"),
                    "lat":         place.lat,
                    "lng":         place.lng,
                })

        # Also include user-clicked coordinate if provided
        if coordinates and coordinates.get("lat") and coordinates.get("lng"):
            lat = float(coordinates["lat"])
            lng = float(coordinates["lng"])
            map_points.append({
                "title":       "Selected Location",
                "description": reverse_geocode(lat, lng),
                "importance":  0.8,
                "scope":       "geospatial",
                "lat":         lat,
                "lng":         lng,
            })

        geojson = build_map_overlay(map_points, "Gummaa-Atlas Discovery") if map_points else None
        return AgentResult(
            agent_name="DiscovAgent",
            output=f"Resolved {len(map_points)} geospatial entities.",
            geojson=geojson,
            map_points=map_points,
        )

    def _synthesizer(
        self,
        user_query: str,
        coordinator_output: str,
        graph_output: str,
        discov_output: str,
        map_points: list[dict],
        memory_context: str,
    ) -> str:
        """
        Synthesizer: merge all agent outputs into the final user-facing response.
        """
        points_summary = ""
        if map_points:
            points_summary = "\n".join(
                f"  • {p['title']} ({p['lat']:.3f}, {p['lng']:.3f})"
                for p in map_points
            )

        prompt = f"""You are the Synthesizer for Gummaa-Atlas, a historical AI mapping assistant.

## Original User Query:
{user_query}

## Coordinator Analysis:
{coordinator_output}

## Historical Knowledge Graph (GraphAgent):
{graph_output}

## Geospatial Discoveries ({len(map_points)} locations mapped):
{points_summary or "None identified."}

## Cognitive Memory Context:
{memory_context or "No prior context."}

Synthesize all of the above into a single, fluent, scholarly yet accessible response for the user.
Requirements:
- Open with the most important historical insight.
- Weave in the geographic context naturally.
- Reference the mapped locations by name where relevant.
- End with an invitation for the user to explore further (one sentence).
- Maintain the voice of a knowledgeable Pan-African historian.
- Do NOT use bullet points — write in coherent paragraphs.
- Length: 150–250 words.
"""
        response = self._client.models.generate_content(
            model=MODEL_ID, contents=prompt
        )
        return response.text

    # ------------------------------------------------------------------
    # Main orchestration entry point
    # ------------------------------------------------------------------

    def run(
        self,
        session_id: str,
        user_query: str,
        coordinates: dict | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full multi-agent pipeline and return a structured result dict.

        Returns::

            {
                "response":   str,          # synthesized final answer
                "geojson":    dict | None,  # GeoJSON FeatureCollection for the map
                "map_points": list[dict],   # raw point list
                "session_id": str,
            }
        """
        # ── Recall cognitive context ──────────────────────────────────
        memory_records = self.memory.recall(session_id, user_query, k=6)
        memory_context = "\n".join(r.content for r in memory_records)

        # ── Coordinator ───────────────────────────────────────────────
        coord_result, _ = self.safe_execute(
            lambda: self._coordinator(session_id, user_query, coordinates, memory_context)
        )
        if not isinstance(coord_result, AgentResult):
            coord_result = AgentResult(agent_name="Coordinator", output=str(coord_result))

        # ── GraphAgent ────────────────────────────────────────────────
        graph_result, _ = self.safe_execute(
            lambda: self._graph_agent(session_id, coord_result.output, memory_context)
        )
        if not isinstance(graph_result, AgentResult):
            graph_result = AgentResult(agent_name="GraphAgent", output=str(graph_result))

        # ── DiscovAgent ───────────────────────────────────────────────
        discov_result, _ = self.safe_execute(
            lambda: self._discovery_agent(session_id, coord_result.output, user_query, coordinates)
        )
        if not isinstance(discov_result, AgentResult):
            discov_result = AgentResult(agent_name="DiscovAgent", output=str(discov_result), map_points=[])

        # ── Synthesizer ───────────────────────────────────────────────
        final_response, _ = self.safe_execute(
            lambda: self._synthesizer(
                user_query,
                coord_result.output,
                graph_result.output,
                discov_result.output,
                discov_result.map_points or [],
                memory_context,
            )
        )
        if not isinstance(final_response, str):
            final_response = str(final_response)

        # ── Remember this interaction ─────────────────────────────────
        self.safe_execute(
            lambda: self.memory.remember(
                session_id,
                f"Q: {user_query}\nA: {final_response}",
            )
        )

        return {
            "response":   final_response,
            "geojson":    discov_result.geojson,
            "map_points": discov_result.map_points or [],
            "session_id": session_id,
        }
