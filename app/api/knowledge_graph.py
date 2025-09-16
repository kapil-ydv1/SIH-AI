from fastapi import APIRouter
from app.models.schemas import KnowledgeGraphIngestPayload, KnowledgeGraphQueryParams
import os
from dotenv import load_dotenv
from app.services.knowledge_service import KnowledgeService


router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


def get_kg_service() -> KnowledgeService | None:
    load_dotenv()
    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        return None
    return KnowledgeService(uri, user, password)


@router.post("/ingest")
def ingest_kg(payload: KnowledgeGraphIngestPayload):
    service = get_kg_service()
    if service is None:
        return {"error": "Knowledge Graph disabled: missing NEO4J_PASSWORD"}
    entities = service.extract_entities(payload.text)
    service.ingest_entities_and_relationships(entities, payload.document_id)
    service.close()
    return {"document_id": payload.document_id, "entities_ingested": len(entities), "status": "ingested"}


@router.get("/query")
def query_kg(q: str, limit: int = 10):
    # Placeholder: implement Neo4j queries as needed
    return {"status": "not_implemented", "query": q, "limit": limit}


