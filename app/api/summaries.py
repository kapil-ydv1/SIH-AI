from fastapi import APIRouter
from app.services.llm_service import LLMService
from app.services.summarization_service import SummarizationService
from dotenv import load_dotenv
import os


router = APIRouter(tags=["summaries"])


def get_summarizer() -> SummarizationService | None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    llm = LLMService(api_key=api_key)
    return SummarizationService(llm_service=llm)


@router.get("/summaries")
def get_summary(type: str = "short", document_id: str | None = None, text: str | None = None):
    summarizer = get_summarizer()
    if summarizer is None:
        return {"error": "Missing OPENAI_API_KEY"}
    # For now, use "text" directly; document_id support can look up cached text
    source_text = text or ""
    summary = summarizer.summarize_document(source_text, summary_type=type)
    return {"summary": summary, "word_count": len(summary.split()), "status": "success"}


