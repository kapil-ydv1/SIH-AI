from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.language import router as language_router
from app.api.knowledge_graph import router as kg_router
from app.api.index_ops import router as index_router
from app.api.processing import router as processing_router
from app.api.querying import router as query_router
from app.api.summaries import router as summaries_router
from app.api.compliance import router as compliance_router
from app.api.reindex_root import router as reindex_root_router


app = FastAPI(title="SIH-AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(language_router, prefix="/api")
app.include_router(kg_router, prefix="/api")
app.include_router(index_router, prefix="/api")
app.include_router(processing_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(summaries_router, prefix="/api")
app.include_router(compliance_router, prefix="/api")
app.include_router(reindex_root_router, prefix="/api")


@app.get("/")
def health_root():
    # Enriched basic health according to reference (booleans best-effort)
    try:
        from app.services.ocr_service import AdvancedDocumentProcessor  # noqa
        ocr_ok = True
    except Exception:
        ocr_ok = False
    try:
        from app.services.embedding_service import HAS_FAISS  # noqa
        faiss_ok = True
    except Exception:
        faiss_ok = False
    try:
        import openai  # noqa
        openai_ok = True
    except Exception:
        openai_ok = False
    try:
        import spacy  # noqa
        spacy_ok = True
    except Exception:
        spacy_ok = False
    try:
        import neo4j  # noqa
        neo4j_ok = True
    except Exception:
        neo4j_ok = False
    return {
        "status": "ok",
        "services": {
            "ocr": ocr_ok,
            "embeddings": True,
            "faiss": faiss_ok,
            "openai": openai_ok,
            "spacy_model": spacy_ok,
            "neo4j": neo4j_ok
        }
    }


@app.get("/api/health/full")
def health_full():
    # Deep health detail flags, best-effort
    checks = {}
    try:
        import pytesseract  # noqa
        checks["tesseract"] = True
    except Exception:
        checks["tesseract"] = False
    try:
        import faiss  # type: ignore # noqa
        checks["faiss"] = True
    except Exception:
        checks["faiss"] = False
    try:
        import torch  # noqa
        checks["torch"] = True
    except Exception:
        checks["torch"] = False
    try:
        import openai  # noqa
        checks["openai"] = True
    except Exception:
        checks["openai"] = False
    try:
        import spacy  # noqa
        checks["spacy_model"] = True
    except Exception:
        checks["spacy_model"] = False
    try:
        import neo4j  # noqa
        checks["neo4j"] = True
    except Exception:
        checks["neo4j"] = False
    try:
        import redis  # noqa
        checks["redis"] = True
    except Exception:
        checks["redis"] = False
    return checks


