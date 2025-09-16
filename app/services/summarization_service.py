import os
import sys

# Add the current directory to the path to allow imports from other services
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- THIS IS THE CORRECTED IMPORT ---
from llm_service import LLMService 
from typing import Dict

class SummarizationService:
    def __init__(self, llm_service: LLMService):
        """
        Initializes the summarization service with an existing LLM service.
        """
        self.llm_service = llm_service

    def summarize_document(self, full_text: str, summary_type: str = 'general') -> str:
        """
        Generates a summary of a given text based on the specified type.

        Args:
            full_text (str): The entire text of the document to be summarized.
            summary_type (str): The role or type of summary required (e.g., 'general', 'executive', 'technical').

        Returns:
            str: The generated summary.
        """
        prompt_templates = {
            'general': """
                Provide a concise, well-structured summary of the following document.
                Capture the main ideas, key arguments, and overall conclusion.
            """,
            'executive': """
                You are summarizing a document for a busy executive.
                Provide a high-level summary in 3-4 bullet points.
                Focus ONLY on actionable items, key findings, strategic implications, and any mentioned deadlines or risks.
                Ignore background information and minor details.
            """,
            'technical': """
                Provide a detailed summary for a technical audience.
                Focus on the methodology, specific data points, experimental results, and technical conclusions mentioned in the document.
            """
        }

        system_prompt = prompt_templates.get(summary_type, prompt_templates['general'])
        user_prompt = f"Document Text:\n{full_text}"
        
        # We need to construct the messages array for the OpenAI call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            print(f"  - Generating a '{summary_type}' summary via OpenAI API...")
            # We call the OpenAI client directly through the passed llm_service object
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model_name,
                messages=messages,
                max_tokens=400,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating summary: {e}"


# --- Test Block for the Summarization Service ---
if __name__ == "__main__":
    from ocr_service import AdvancedDocumentProcessor
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found.")
        sys.exit(1)
        
    # 1. Initialize our services
    ocr_processor = AdvancedDocumentProcessor()
    llm_service = LLMService(api_key=api_key)
    summarization_service = SummarizationService(llm_service=llm_service)

    # 2. Extract the full text from the sample document
    pdf_file_path = os.path.join(current_dir, '..', '..', 'documents', 'sample.pdf')
    print(f"📄 Processing document: {pdf_file_path}")
    doc_result = ocr_processor.process_document(pdf_file_path)
    full_document_text = doc_result.get('text')

    if not full_document_text:
        print("❌ Could not extract text from the document. Aborting test.")
        sys.exit(1)

    # 3. Generate and print different types of summaries
    print("\n" + "="*50)
    print("🚀 Generating a GENERAL summary...")
    general_summary = summarization_service.summarize_document(full_document_text, summary_type='general')
    print("✅ General Summary:\n", general_summary)
    print("="*50)
    
    print("\n" + "="*50)
    print("🚀 Generating an EXECUTIVE summary...")
    executive_summary = summarization_service.summarize_document(full_document_text, summary_type='executive')
    print("✅ Executive Summary:\n", executive_summary)
    print("="*50)