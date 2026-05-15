"""
server.py
=========
Gummaa-Atlas FastAPI Backend.

Endpoints:
  GET  /              → serves index.html
  GET  /health        → health check
  POST /api/chat      → multi-agent orchestration + cognitive memory
  GET  /api/memory    → memory stats for a given session
  DELETE /api/memory  → prune expired memories (TTL)
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.agent import GummaaAgent

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gummaa.server")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ID   = os.getenv("PROJECT_ID",   "multi-agent-lab01")
LOCATION     = os.getenv("LOCATION",     "us-central1")
LANCEDB_PATH = os.getenv("LANCEDB_PATH", "/app/gummaa_workspace.lance")
PORT         = int(os.getenv("PORT", 8080))

STATIC_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Agent (singleton — shared across requests)
# ---------------------------------------------------------------------------
logger.info("Initialising GummaaAgent at path: %s", LANCEDB_PATH)
agent = GummaaAgent(
    lancedb_path=LANCEDB_PATH,
    project_id=PROJECT_ID,
    location=LOCATION,
)
logger.info("GummaaAgent ready.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Gummaa-Atlas API",
    description="Cognitive Multi-Agent Historical Mapping System",
    version="1.0.0",
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class Coordinates(BaseModel):
    lat: float | None = None
    lng: float | None = None


class ChatRequest(BaseModel):
    session_id:  str        = Field(default_factory=lambda: str(uuid.uuid4()))
    message:     str        = Field(..., min_length=1, max_length=4096)
    coordinates: Coordinates | None = None


class ChatResponse(BaseModel):
    session_id:  str
    response:    str
    geojson:     dict | None = None
    map_points:  list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_index():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index))


@app.get("/health")
async def health():
    stats = agent.memory.stats()
    return {"status": "ok", "memory": stats}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main endpoint — runs the full Coordinator → GraphAgent → DiscovAgent → Synthesizer
    pipeline with cognitive memory recall / remember.
    """
    logger.info(
        "Chat request | session=%s | query='%s' | coords=%s",
        req.session_id,
        req.message[:80],
        req.coordinates,
    )

    coords_dict = None
    if req.coordinates:
        coords_dict = {
            "lat": req.coordinates.lat,
            "lng": req.coordinates.lng,
        }

    try:
        result = agent.run(
            session_id=req.session_id,
            user_query=req.message,
            coordinates=coords_dict,
        )
    except Exception as exc:
        logger.error("Agent run failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return ChatResponse(
        session_id=req.session_id,
        response=result["response"],
        geojson=result.get("geojson"),
        map_points=result.get("map_points", []),
    )


@app.get("/api/memory")
async def memory_stats(session_id: str | None = None):
    """Return cognitive memory table stats."""
    stats = agent.memory.stats()
    if session_id:
        # Surface session-specific recall snapshot
        records = agent.memory.recall(session_id, "summary", k=10)
        stats["session_records"] = [
            {
                "memory_id": r.memory_id,
                "scope":     r.scope,
                "content":   r.content[:120],
                "importance": r.importance,
            }
            for r in records
        ]
    return stats


@app.delete("/api/memory")
async def prune_memory(ttl_days: float = 30.0):
    """Prune memories older than ttl_days. Returns count removed."""
    removed = agent.memory.forget(ttl_seconds=ttl_days * 86_400)
    return {"pruned": removed}


# ---------------------------------------------------------------------------
# Entry point (local dev)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)
