import os
import sys

# Add the current directory to the path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from llm_service import LLMService
from typing import Dict

class ComplianceService:
    def __init__(self, llm_service: LLMService):
        """
        Initializes the compliance service with an existing LLM service.
        """
        self.llm_service = llm_service

    def compare_documents(self, old_text: str, new_text: str) -> str:
        """
        Compares two versions of a document and generates a compliance report.
        """
        # This is the core of the service: a highly detailed prompt.
        system_prompt = """
        You are an expert compliance analyst. Your task is to compare two versions of a policy document and generate a clear, structured report on the changes.

        Analyze the "Old Version" and "New Version" provided. Identify all additions, removals, and modifications.
        For each change, provide a brief summary and assess its potential risk level (Low, Medium, High).

        Your final output MUST be in the following format:

        **Compliance Change Report**

        **1. Added Policies:**
        - [Summary of Added Policy 1] (Risk: [Low/Medium/High])
        - [Summary of Added Policy 2] (Risk: [Low/Medium/High])
        - *No changes if none*

        **2. Removed Policies:**
        - [Summary of Removed Policy 1] (Risk: [Low/Medium/High])
        - *No changes if none*

        **3. Modified Policies:**
        - **Original:** [Quote the original text]
          **Updated:** [Quote the updated text]
          **Change Summary:** [Explain the modification] (Risk: [Low/Medium/High])
        - *No changes if none*
        """

        user_prompt = f"Please generate a compliance report based on the following documents:\n\n--- OLD VERSION ---\n{old_text}\n\n--- NEW VERSION ---\n{new_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            print("  - Generating compliance report via OpenAI API...")
            response = self.llm_service.client.chat.completions.create(
                model="gpt-4o", # Use the most powerful model for this analytical task
                messages=messages,
                max_tokens=1024,
                temperature=0.1, # Low temperature for factual analysis
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating compliance report: {e}"

# --- Test Block for the Compliance Service ---
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found.")
        sys.exit(1)
        
    # 1. Initialize our services
    llm_service = LLMService(api_key=api_key)
    compliance_service = ComplianceService(llm_service=llm_service)

    # 2. Create sample "old" and "new" policy documents for testing
    old_policy_text = """
    Safety Policy Document - v1.0
    1. All employees must wear hard hats in construction zones.
    2. Weekly safety meetings are mandatory for all site staff.
    3. All incidents must be reported to the site manager within 24 hours.
    """

    new_policy_text = """
    Safety Policy Document - v1.1
    1. All personnel, including contractors, must wear hard hats in designated construction zones.
    2. All incidents must be reported to the Health & Safety Officer immediately, not exceeding 8 hours.
    3. A daily safety briefing is now required for all teams before starting work.
    """

    # 3. Generate and print the compliance report
    print("🚀 Comparing document versions to generate a compliance report...")
    compliance_report = compliance_service.compare_documents(old_policy_text, new_policy_text)
    
    print("\n" + "="*50)
    print("✅ Compliance Report:\n")
    print(compliance_report)
    print("="*50)