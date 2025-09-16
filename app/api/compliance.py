from fastapi import APIRouter
from app.services.llm_service import LLMService
from app.services.compliance_service import ComplianceService
from dotenv import load_dotenv
import os


router = APIRouter(tags=["compliance"])


def get_compliance_service() -> ComplianceService | None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    llm = LLMService(api_key=api_key)
    return ComplianceService(llm_service=llm)


@router.get("/compliance-alerts")
def get_compliance_alerts(document_id: str | None = None, ruleset: str | None = None, old: str | None = None, new: str | None = None):
    service = get_compliance_service()
    if service is None:
        return {"error": "Missing OPENAI_API_KEY"}
    # For now, support ad-hoc compare via old/new. Document-based flow can be added later.
    old_text = old or ""
    new_text = new or ""
    report = service.compare_documents(old_text, new_text)
    return {"analysis": report, "status": "success"}


@router.post("/compliance/check")
def compliance_check(body: dict):
    # Spec: POST /api/compliance/check with { text, rules }
    service = get_compliance_service()
    if service is None:
        return {"error": "Missing OPENAI_API_KEY"}
    text = body.get("text", "")
    rules = body.get("rules", [])
    # Reuse compare_documents by comparing text with itself + rules appended
    rules_text = "\n".join([f"Rule: {r}" for r in rules])
    report = service.compare_documents(text, text + "\n" + rules_text)
    return {"analysis": report, "status": "success"}


