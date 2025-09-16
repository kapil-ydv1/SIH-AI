from fastapi import APIRouter
from app.models.schemas import TextPayload, TranslatePayload
from app.services.language_service import LanguageService


router = APIRouter(tags=["language"])
language_service = LanguageService()


@router.post("/language/detect")
def detect_language(payload: TextPayload):
    lang = language_service.detect_language(payload.text)
    return {"language": lang}


@router.post("/translate")
def translate_text(payload: TranslatePayload):
    translated = language_service.translate_text(payload.text, source_lang=payload.source or "auto", target_lang=payload.target)
    return {"translated_text": translated}


