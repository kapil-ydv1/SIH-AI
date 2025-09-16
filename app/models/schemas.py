
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class QueryRequest(BaseModel):
    query_text: str
    top_k: int = 5

class QueryResult(BaseModel):
    document_id: str
    snippet: str
    score: float

class QueryResponse(BaseModel):
    query_id: str
    results: List[QueryResult]

class SummaryResponse(BaseModel):
    document_id: str
    summary_text: str
    source_filename: str

class ComplianceAlert(BaseModel):
    alert_id: str
    document_id: str
    alert_type: str = Field(..., example="Deadline Missed")
    description: str
    severity: str = Field(..., example="High")

class KnowledgeGraphResponse(BaseModel):
    query: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]