## API Reference (Requests and Responses)

The following documents the expected request and response formats for the current endpoints and additional APIs supported by the existing microservices.

### 1) POST `/api/upload` — Upload Document
- Content-Type: `multipart/form-data`
- Body:
  - `file`: binary file (pdf, png, jpg, etc.)
  - `title`: string (document title)
  - `category`: string (e.g., "policy", "safety", "procurement")
  - `description`: string
  - `department`: string (owning department)
  - `language`: string ISO code (e.g., `en`, `ml`)
  - `priority`: string (e.g., `low`, `medium`, `high`, `urgent`)
  - `access_scope`: enum(`private`, `department`, `shared`)
  - `shared_departments[]`: string[] (required if `access_scope` == `shared`)
- 201 Created:
```json
{
  "document_id": "abc123",
  "filename": "Sample.pdf",
  "size_bytes": 123456,
  "metadata": {
    "title": "Q1 Safety Update",
    "category": "safety",
    "description": "Monthly summary of incidents and mitigations",
    "department": "Operations",
    "language": "en",
    "priority": "high",
    "access": {
      "scope": "shared",
      "shared_departments": ["Operations", "Compliance"]
    }
  },
  "status": "uploaded"
}
```

### 2) POST `/api/process/{document_id}` — Run OCR / Extraction
- Path params: `document_id`
- Optional JSON body:
```json
{ "auto_translate_to_en": true }
```
- 200 OK:
```json
{
  "document_id": "abc123",
  "pages": 10,
  "text": "...extracted text...",
  "metadata": {
    "title": "",
    "author": "",
    "producer": "",
    "creation_date": ""
  },
  "images_found": 3,
  "tables_detected": 1,
  "quality_analysis": {
    "overall_quality_score": 82,
    "extraction_success_rate": 90.0
  },
  "status": "processed"
}
```

### 3) POST `/api/embeddings/{document_id}` — Generate Embeddings
- Path params: `document_id`
- Optional JSON body:
```json
{ "chunk_size": 1000, "overlap": 200 }
```
- 200 OK:
```json
{
  "document_id": "abc123",
  "total_chunks": 19,
  "embedding_dimension": 1024,
  "index_type": "IndexFlatIP",
  "status": "indexed"
}
```

### 4) POST `/api/query` — Semantic Search + Answer (RAG)
- Content-Type: `application/json`
- Body:
```json
{ "query": "What are the key takeaways?", "k": 5, "threshold": 0.4 }
```
- 200 OK:
```json
{
  "query": "What are the key takeaways?",
  "results": [
    {
      "filename": "Sample.pdf",
      "chunk_id": 7,
      "text": "...",
      "similarity_score": 0.83,
      "confidence": "HIGH"
    }
  ],
  "answer": "...LLM generated answer...",
  "citations": [{ "filename": "Sample.pdf", "chunk_id": 7 }]
}
```

### 5) GET `/api/summaries` — Role-Based Summaries
- Query params (example):
  - `type`: `short` | `bullet` (default `short`)
  - `document_id` (optional)
- 200 OK:
```json
{ "summary": "...generated summary...", "word_count": 124, "status": "success" }
```

### 6) GET `/api/compliance-alerts` — Compliance Outputs (predefined)
- Query params (example): `document_id`, `ruleset` (optional)
- 200 OK:
```json
{ "analysis": "- Rule 1: COMPLIANT\n- Rule 2: ISSUE: ...", "status": "success" }
```

### 7) GET `/api/knowledge-graph/query` — Query Knowledge Graph
- Query params (examples): `q`, `document_id` (optional)
- 200 OK (shape may vary):
```json
{
  "nodes": [{ "id": "Document:Sample.pdf", "labels": ["Document"] }],
  "relationships": [{ "type": "MENTIONED_IN", "from": "ORG:KMRL", "to": "Document:Sample.pdf" }]
}
```

### 8) GET `/` — Health Check (basic)
- 200 OK:
```json
{
  "status": "ok",
  "services": {
    "ocr": true,
    "embeddings": true,
    "faiss": true,
    "openai": true,
    "spacy_model": true,
    "neo4j": false
  }
}
```

---

## Additional APIs supported/required by current services

### 9) POST `/api/language/detect` — Language Detection
- Body:
```json
{ "text": "..." }
```
- 200 OK:
```json
{ "language": "en" }
```

### 10) POST `/api/translate` — Translation (Malayalam → English supported)
- Body:
```json
{ "text": "...", "source": "auto", "target": "en" }
```
- 200 OK:
```json
{ "translated_text": "..." }
```

### 11) POST `/api/knowledge-graph/ingest` — Extract & Ingest Entities
- Body:
```json
{ "document_id": "abc123", "text": "...document text..." }
```
- 202 Accepted:
```json
{ "document_id": "abc123", "entities_ingested": 42, "status": "ingested" }
```

### 12) GET `/api/index/stats` — Index/Embedding Stats
- 200 OK:
```json
{
  "status": "ready",
  "total_chunks": 19,
  "total_files": 1,
  "index_size": 19,
  "model_name": "BAAI/bge-large-en-v1.5",
  "files": { "Sample.pdf": { "chunks": 19, "total_words": 8123 } }
}
```

### 13) POST `/api/reindex` — Rebuild Index (optional)
- Body (optional):
```json
{ "force": true }
```
- 202 Accepted:
```json
{ "status": "reindexing_started" }
```

### 14) DELETE `/api/index/{document_id}` — Remove Document From Index (optional)
- Path params: `document_id`
- 200 OK:
```json
{ "document_id": "abc123", "removed": true }
```

### 15) GET `/api/chunks/{document_id}` — Return Chunks + Metadata
- Path params: `document_id`
- 200 OK:
```json
{
  "document_id": "abc123",
  "chunks": [
    { "chunk_id": 0, "word_count": 250, "text": "...", "filename": "Sample.pdf" }
  ]
}
```

### 16) POST `/api/process/batch` — Batch OCR + Embeddings for a Folder
- Body:
```json
{ "folder": "./documents", "auto_translate_to_en": true }
```
- 202 Accepted:
```json
{ "submitted": true, "files": 12 }
```

### 17) POST `/api/compliance/check` — Ad-hoc Compliance Check
- Body:
```json
{ "text": "...", "rules": ["Rule A", "Rule B"] }
```
- 200 OK:
```json
{ "analysis": "- Rule A: COMPLIANT\n- Rule B: ISSUE: ...", "status": "success" }
```

### 18) GET `/api/health/full` — Deep Health
- 200 OK:
```json
{
  "tesseract": true,
  "faiss": true,
  "torch": true,
  "openai": true,
  "spacy_model": true,
  "neo4j": false,
  "redis": false
}
```

---

Notes
- Error responses use standard HTTP codes with JSON: `{ "error": "message" }`.
- Some responses depend on external services (Tesseract, FAISS, PyTorch/Transformers, OpenAI, spaCy model, Neo4j, Redis). Ensure environment variables and installations are in place.
- Certain endpoints marked optional are recommended for observability and maintenance and can be enabled as the corresponding service code is completed.
