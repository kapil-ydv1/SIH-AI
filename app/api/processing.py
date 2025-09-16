from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status
from app.models.schemas import ProcessDocumentPayload, GenerateEmbeddingsPayload, BatchProcessPayload
from app.services.ocr_service import AdvancedDocumentProcessor
from app.services.embedding_service import EmbeddingService
import os
import shutil
import uuid


router = APIRouter(tags=["processing"])
ocr_processor = AdvancedDocumentProcessor()
embedding_service = EmbeddingService()

# In-memory registry for demo purposes
DOCUMENT_PATHS: dict[str, str] = {}
DOCUMENT_LAST_RESULT: dict[str, dict] = {}


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    uploads_dir = os.path.join("documents")
    os.makedirs(uploads_dir, exist_ok=True)
    file_id = uuid.uuid4().hex
    # Preserve original filename on disk
    file_path = os.path.join(uploads_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    size_bytes = os.path.getsize(file_path)
    DOCUMENT_PATHS[file_id] = file_path
    return {
        "document_id": file_id,
        "filename": file.filename,
        "size_bytes": size_bytes,
        "status": "uploaded"
    }


@router.post("/process/{document_id}")
def process_document(document_id: str, body: dict | None = None):
    file_path = DOCUMENT_PATHS.get(document_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="document_not_found")
    # Optional flag (currently not altering internal behavior)
    _ = (body or {}).get("auto_translate_to_en", True)
    result = ocr_processor.process_document(file_path)
    # Cache last result for embeddings/summaries
    DOCUMENT_LAST_RESULT[document_id] = result
    # Shape per spec
    response = {
        "document_id": document_id,
        "pages": result.get("pages", len(result.get("page_details", []))),
        "text": result.get("text", ""),
        "metadata": result.get("metadata", {}),
        "images_found": result.get("images_found", 0),
        "tables_detected": result.get("tables_detected", 0),
        "quality_analysis": result.get("quality_analysis", {}),
        "status": "processed"
    }
    return response


@router.post("/embeddings/{document_id}")
def generate_embeddings(document_id: str, body: dict | None = None):
    last = DOCUMENT_LAST_RESULT.get(document_id)
    if not last:
        raise HTTPException(status_code=404, detail="document_not_processed")
    page_details = last.get("page_details", [])
    filename = os.path.basename(DOCUMENT_PATHS.get(document_id, document_id))
    chunk_size = (body or {}).get("chunk_size", 1000)
    overlap = (body or {}).get("overlap", 200)
    chunks = embedding_service.process_and_chunk_pages(filename, page_details, int(chunk_size), int(overlap))
    result = embedding_service.build_index(chunks)
    return {
        "document_id": document_id,
        **{k: v for k, v in result.items() if k in {"total_chunks", "embedding_dimension", "index_type", "status"}},
        "status": result.get("status", "indexed")
    }


@router.get("/chunks/{document_id}")
def get_chunks(document_id: str):
    filename = os.path.basename(DOCUMENT_PATHS.get(document_id, document_id))
    chunks = [c for c in embedding_service.chunks if c.get("filename") in {filename, document_id}]
    return {"document_id": document_id, "chunks": chunks}


@router.post("/process/batch")
def batch_process(body: dict, background_tasks: BackgroundTasks):
    # Align with spec: { "folder": "./documents", "auto_translate_to_en": true }
    folder = body.get("folder")
    if not folder or not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail="invalid_folder")
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for fname in files:
        path = os.path.join(folder, fname)
        background_tasks.add_task(ocr_processor.process_document, path)
    return {"submitted": True, "files": len(files)}


