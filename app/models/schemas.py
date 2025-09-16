from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any


class TextPayload(BaseModel):
    text: str = Field(..., min_length=1)


class TranslatePayload(BaseModel):
    text: str = Field(..., min_length=1)
    source: Optional[str] = Field(default="auto")
    target: str = Field(default="en", min_length=2, max_length=5)


class KnowledgeGraphIngestPayload(BaseModel):
    document_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class BatchProcessItem(BaseModel):
    document_id: str
    file_path: Optional[str] = None
    text: Optional[str] = None


class BatchProcessPayload(BaseModel):
    items: List[BatchProcessItem]


class ProcessDocumentPayload(BaseModel):
    file_path: str


class GenerateEmbeddingsPayload(BaseModel):
    filename: str
    page_details: List[Dict[str, Any]]
    chunk_size: int = 1000
    overlap: int = 200


class QueryPayload(BaseModel):
    query: str
    k: int = 5
    threshold: float = 0.5


class SummaryRole(str):
    pass


class ComplianceAlertsParams(BaseModel):
    old: str
    new: str


class KnowledgeGraphQueryParams(BaseModel):
    q: str
    limit: int = 10
