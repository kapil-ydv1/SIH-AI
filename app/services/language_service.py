from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
import time

class LanguageService:
    def __init__(self):
        """Initializes the translation service."""
        # No translator object needed here for deep-translator in this setup
        pass

    def detect_language(self, text: str) -> str:
        """
        Detects the language of a given text snippet.
        """
        sample_text = text[:500]
        try:
            lang_code = detect(sample_text)
            return lang_code
        except LangDetectException:
            return "unknown"
        except Exception as e:
            print(f"An error occurred during language detection: {e}")
            return "unknown"

    def translate_text(self, text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
        """
        Translates a large block of text to the target language using deep-translator.
        """
        try:
            # The library handles chunking internally, but we can set a timeout
            print(f"Translating document from '{source_lang}' to '{target_lang}'...")
            translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
            return translated_text
        except Exception as e:
            print(f"An error occurred during translation: {e}")
            return text # Return original text if translation fails

# --- Test Block for the Language Service ---
if __name__ == "__main__":
    service = LanguageService()

    # Test Case 1: English Text
    english_text = "This is a test of the language detection system. It should correctly identify English."
    detected_lang_en = service.detect_language(english_text)
    print(f"Detected language for English text: '{detected_lang_en}'")
    
    # Test Case 2: Malayalam Text
    malayalam_text = "ഇതൊരു പരീക്ഷണമാണ്. സിസ്റ്റം മലയാളം ശരിയായി തിരിച്ചറിയണം."
    detected_lang_ml = service.detect_language(malayalam_text)
    print(f"Detected language for Malayalam text: '{detected_lang_ml}'")
    
    # Test Case 3: Translation
    print("\n--- Testing Translation ---")
    translated_text = service.translate_text(malayalam_text, source_lang=detected_lang_ml)
    print(f"Original (Malayalam): {malayalam_text}")
    print(f"Translated (English): {translated_text}")