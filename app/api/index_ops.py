from fastapi import APIRouter
from app.services.embedding_service import EmbeddingService


router = APIRouter(prefix="/index", tags=["index"])
embedding_service = EmbeddingService()


@router.get("/stats")
def index_stats():
    return embedding_service.get_index_stats()


@router.post("/reindex")
def reindex(body: dict | None = None):
    # Align to spec response
    if not embedding_service.chunks:
        return {"status": "reindexing_started"}
    embedding_service.build_index(embedding_service.chunks)
    return {"status": "reindexing_started"}


@router.delete("/{document_id}")
def delete_document(document_id: str):
    before = len(embedding_service.chunks)
    embedding_service.chunks = [c for c in embedding_service.chunks if c.get("filename") != document_id]
    if embedding_service.chunks:
        embedding_service.build_index(embedding_service.chunks)
    else:
        # Reset in-memory structures
        embedding_service.index = None
        embedding_service.embeddings_matrix = None
    after = len(embedding_service.chunks)
    return {"document_id": document_id, "removed": (before - after) > 0}


