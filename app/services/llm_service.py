import openai
import os

class LLMService:
    def __init__(self, api_key: str):
        """Initializes the LLM service with an OpenAI client."""
        # This is the key change: we create a client instance.
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = "gpt-4o"

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates a response using the OpenAI API, instructing it to cite sources.
        """
        system_prompt = """
        You are an expert Q&A assistant. Your task is to answer the user's question based ONLY on the provided context.
        The context is a series of text snippets, each marked with its source page number (e.g., [Source: Page X]).

        When you formulate your answer, you MUST follow these rules:
        1. Directly answer the user's question.
        2. For each piece of information you use, you MUST cite the source page number at the end of the sentence, like this: (Page X).
        3. If the context does not contain the answer, you MUST state that the information is not available in the provided document.
        4. Do not make up information.
        """
        
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            print("  - Calling OpenAI API with citation instruction...")
            # We now use the client instance to make the call.
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}"