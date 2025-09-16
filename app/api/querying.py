from fastapi import APIRouter
from app.models.schemas import QueryPayload
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
import os
from dotenv import load_dotenv


router = APIRouter(tags=["query"])
embedding_service = EmbeddingService()


def get_llm_service() -> LLMService | None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return LLMService(api_key=api_key)


@router.post("/query")
def query_docs(payload: QueryPayload):
    results = embedding_service.search(payload.query, k=payload.k, threshold=payload.threshold)
    citations = []
    for r in results:
        citations.append({"filename": r.get("filename"), "chunk_id": r.get("chunk_id")})

    answer = None
    if results:
        context_parts = []
        for chunk in results:
            page_num = chunk.get("page_number", "N/A")
            context_parts.append(f"[Source: Page {page_num}]: {chunk['text']}")
        context = "\n\n".join(context_parts)

        llm = get_llm_service()
        if llm:
            answer = llm.generate_answer(payload.query, context)

    return {
        "query": payload.query,
        "results": results,
        "answer": answer,
        "citations": citations
    }


