from fastapi import APIRouter
from app.services.embedding_service import EmbeddingService


router = APIRouter(tags=["index"])
embedding_service = EmbeddingService()


@router.post("/reindex")
def reindex_root(body: dict | None = None):
    # Spec: POST /api/reindex returns { "status": "reindexing_started" }
    if embedding_service.chunks:
        embedding_service.build_index(embedding_service.chunks)
    return {"status": "reindexing_started"}


