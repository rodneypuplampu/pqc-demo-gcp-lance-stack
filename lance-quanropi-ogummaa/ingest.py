"""
ingest.py
=========
Gummaa-Atlas — Historical Cartographic Data Ingestion.

Pre-populates the LanceDB cognitive memory lakehouse with a curated seed
dataset of Pan-African historical facts, geospatial anchors, and trade
route data so the system has rich context from day one.

Usage::

    python ingest.py

Environment:
    PROJECT_ID      — GCP project (default: multi-agent-lab01)
    LANCEDB_PATH    — LanceDB path, GCS or local (default: ./gummaa_workspace.lance)
    LOCATION        — Vertex AI region (default: us-central1)
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gummaa.ingest")

# ---------------------------------------------------------------------------
# Seed data — Pan-African historical knowledge
# ---------------------------------------------------------------------------

SEED_FACTS: list[dict] = [
    # ── Empires & Kingdoms ─────────────────────────────────────────────────
    {
        "scope":      "historical",
        "content":    "The Ghana Empire (Wagadou) flourished in West Africa from approximately 300 CE to 1200 CE, controlling Trans-Saharan gold-salt trade routes between the Sahara and the forest zones.",
        "importance": 0.95,
        "session_id": "seed",
    },
    {
        "scope":      "historical",
        "content":    "The Mali Empire reached its zenith under Mansa Musa (reigned 1312–1337 CE) and encompassed the gold fields of Bambuk and Bure, making it the wealthiest state in the medieval world.",
        "importance": 0.97,
        "session_id": "seed",
    },
    {
        "scope":      "historical",
        "content":    "The Songhai Empire under Askia Muhammad I (1493–1528 CE) extended from the Atlantic coast to the Hausa states, with Timbuktu as a global centre of Islamic scholarship.",
        "importance": 0.95,
        "session_id": "seed",
    },
    {
        "scope":      "historical",
        "content":    "The Kingdom of Kush (Nubia) dominated the upper Nile from approximately 1070 BCE to 350 CE, at times ruling Egypt as the 25th Dynasty (c. 744–656 BCE).",
        "importance": 0.93,
        "session_id": "seed",
    },
    {
        "scope":      "historical",
        "content":    "Great Zimbabwe served as the capital of the Kingdom of Zimbabwe (c. 1220–1450 CE), a powerful Shona polity that controlled gold trade between the interior and Indian Ocean ports.",
        "importance": 0.92,
        "session_id": "seed",
    },
    {
        "scope":      "historical",
        "content":    "The Ashanti Confederacy (Asante Empire), founded c. 1701 by Osei Tutu, became the dominant power in the Gold Coast region, renowned for its kente weaving, gold work, and the sacred Golden Stool.",
        "importance": 0.91,
        "session_id": "seed",
    },
    {
        "scope":      "historical",
        "content":    "The Kingdom of Benin (present-day Nigeria), founded c. 1180 CE, produced world-renowned bronze plaques and maintained sophisticated diplomatic relations with Portugal from the 15th century.",
        "importance": 0.90,
        "session_id": "seed",
    },
    {
        "scope":      "historical",
        "content":    "Axum (Aksum), a powerful trading empire in the northern Horn of Africa (c. 100–940 CE), minted its own currency, adopted Christianity c. 330 CE, and controlled trade through the Red Sea.",
        "importance": 0.93,
        "session_id": "seed",
    },
    # ── Trade Routes ──────────────────────────────────────────────────────
    {
        "scope":      "geospatial",
        "content":    "The Trans-Saharan trade network linked sub-Saharan West Africa to North Africa and the Mediterranean, with major arteries passing through Timbuktu, Gao, Sijilmasa, and Tripoli.",
        "importance": 0.90,
        "session_id": "seed",
    },
    {
        "scope":      "geospatial",
        "content":    "The Indian Ocean trade network connected East African city-states (Kilwa, Mombasa, Zanzibar) to Arabia, Persia, India, and China from at least the 9th century CE.",
        "importance": 0.90,
        "session_id": "seed",
    },
    {
        "scope":      "geospatial",
        "content":    "The Niger River served as the primary inland waterway for West African commerce, connecting the savanna economies of Mali and Songhai with forest-zone goods including kola nuts and ivory.",
        "importance": 0.87,
        "session_id": "seed",
    },
    # ── Pan-Africanism & Political Philosophy ────────────────────────────
    {
        "scope":      "political",
        "content":    "Kwame Nkrumah, first President of Ghana (1960–1966), articulated the doctrine of Pan-Africanism and Consciencism, arguing that African unity was a prerequisite for genuine political and economic liberation from neo-colonialism.",
        "importance": 0.98,
        "session_id": "seed",
    },
    {
        "scope":      "political",
        "content":    "Nkrumah defined neo-colonialism as the final stage of imperialism, wherein the outward forms of political independence mask continued economic and strategic control by former colonial powers.",
        "importance": 0.97,
        "session_id": "seed",
    },
    {
        "scope":      "political",
        "content":    "The Organisation of African Unity (OAU), founded in Addis Ababa in 1963, was the institutional expression of Pan-Africanist aspirations; it was reconstituted as the African Union (AU) in 2002.",
        "importance": 0.88,
        "session_id": "seed",
    },
    {
        "scope":      "political",
        "content":    "Marcus Garvey's Universal Negro Improvement Association (UNIA), founded 1914, promoted African repatriation, economic self-reliance, and Black pride, directly inspiring 20th-century African nationalist movements.",
        "importance": 0.89,
        "session_id": "seed",
    },
    # ── Colonial Geography ────────────────────────────────────────────────
    {
        "scope":      "historical",
        "content":    "The Berlin Conference of 1884–85 formalised the 'Scramble for Africa', partitioning the continent among European powers largely without African participation and drawing borders that ignored pre-existing ethnic, linguistic, and political boundaries.",
        "importance": 0.96,
        "session_id": "seed",
    },
    {
        "scope":      "historical",
        "content":    "The Kingdom of Ethiopia under Emperor Menelik II defeated the Italian army at the Battle of Adwa (1 March 1896), becoming the most celebrated symbol of African resistance to colonialism.",
        "importance": 0.96,
        "session_id": "seed",
    },
    # ── Geospatial Anchors ────────────────────────────────────────────────
    {
        "scope":      "geospatial",
        "content":    "Timbuktu (16.7735° N, 3.0074° W), Mali — served as the intellectual and commercial capital of the Sahel, housing the Sankore University and over 700,000 manuscripts in its libraries.",
        "importance": 0.94,
        "session_id": "seed",
    },
    {
        "scope":      "geospatial",
        "content":    "Kilwa Kisiwani (8.9583° S, 39.5125° E), Tanzania — the wealthiest medieval East African trading port, described by Ibn Battuta (1331 CE) as one of the most beautiful cities in the world.",
        "importance": 0.91,
        "session_id": "seed",
    },
    {
        "scope":      "geospatial",
        "content":    "Great Zimbabwe ruins (20.2673° S, 30.9337° E) represent the largest stone structure in sub-Saharan Africa south of the Sahara, covering 722 hectares and built without mortar.",
        "importance": 0.90,
        "session_id": "seed",
    },
]

SESSION_ID = "seed"


def ingest(lancedb_path: str, project_id: str, location: str) -> None:
    from core.cognitive_memory import CognitiveMemory

    logger.info("Connecting to LanceDB at: %s", lancedb_path)
    mem = CognitiveMemory(
        lancedb_path=lancedb_path,
        project_id=project_id,
        location=location,
    )

    logger.info("Beginning ingestion of %d seed facts …", len(SEED_FACTS))

    for i, fact in enumerate(SEED_FACTS, 1):
        content    = fact["content"]
        session_id = fact.get("session_id", SESSION_ID)

        logger.info("[%d/%d] Encoding: %s …", i, len(SEED_FACTS), content[:60])

        # Bypass the Gemini extraction step — write atomic facts directly
        from core.cognitive_memory import CognitiveMemory
        vector = mem._embed(content)
        mem._consolidate(
            session_id=session_id,
            content=content,
            vector=vector,
            importance=fact.get("importance", 0.5),
            scope=fact.get("scope", "historical"),
            media_blob=None,
        )

    logger.info("Ingestion complete. Final stats: %s", mem.stats())


if __name__ == "__main__":
    ingest(
        lancedb_path=os.getenv("LANCEDB_PATH", "./gummaa_workspace.lance"),
        project_id=os.getenv("PROJECT_ID",     "multi-agent-lab01"),
        location=os.getenv("LOCATION",         "us-central1"),
    )
