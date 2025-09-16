from sentence_transformers import SentenceTransformer
from llm_service import LLMService
import numpy as np
import faiss
import pickle
import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # This line loads the variables from .env into your script

logger = logging.getLogger(__name__)

class EmbeddingService:
    def process_and_chunk_pages(self, filename: str, page_details: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
        """
        Processes text from a list of pages, chunks it, and adds page number metadata.
        """
        all_chunks_with_metadata = []
        chunk_id_counter = 0
        
        # Loop through each page's data from the OCR service
        for page in page_details:
            page_number = page['page_number']
            page_text = page['text']
            
            if not page_text.strip():
                continue # Skip empty pages

            # Use our existing smart chunking logic on the text of a single page
            chunks_from_page = self.smart_chunk_text(page_text, chunk_size, overlap)

            # For each chunk created from this page, add the crucial page_number metadata
            for chunk in chunks_from_page:
                chunk_data = {
                    'text': chunk['text'],
                    'filename': filename,
                    'page_number': page_number,  # <-- The essential new piece of metadata
                    'chunk_id': chunk_id_counter,
                    'word_count': len(chunk.get('text', '').split())
                }
                all_chunks_with_metadata.append(chunk_data)
                chunk_id_counter += 1
                
        return all_chunks_with_metadata
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5'):
        """
        Initialize embedding service with specified model
        all-MiniLM-L6-v2: Lightweight, fast, good performance
        all-mpnet-base-v2: Better quality but slower
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.index_path = "embeddings/document_index.faiss"
        self.chunks_path = "embeddings/chunks.pkl"
        self.metadata_path = "embeddings/metadata.json"
        
        # Ensure embeddings directory exists
        os.makedirs("embeddings", exist_ok=True)
        
        # Load existing index if available
        self.load_index()
    
    def smart_chunk_text(self, text: str, chunk_size: int = 1000, 
                        overlap: int = 200, min_chunk_size: int = 100) -> List[Dict]:
        """
        Advanced text chunking that preserves sentence boundaries
        """
        # Split by sentences first
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_size + sentence_size > chunk_size and current_chunk:
                if len(current_chunk.split()) >= min_chunk_size:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'word_count': len(current_chunk.split()),
                        'start_sentence': len(chunks)
                    })
                
                # Start new chunk with overlap
                if overlap > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence
                    current_size = len(current_chunk.split())
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk and len(current_chunk.split()) >= min_chunk_size:
            chunks.append({
                'text': current_chunk.strip(),
                'word_count': len(current_chunk.split()),
                'start_sentence': len(chunks)
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple delimiters"""
        import re
        
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _get_overlap_text(self, text: str, overlap_words: int) -> str:
        """Get last N words for overlap"""
        words = text.split()
        if len(words) <= overlap_words:
            return text
        return " ".join(words[-overlap_words:])
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Create embeddings in batches for memory efficiency"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch, 
                convert_to_tensor=True,
                show_progress_bar=True if len(texts) > 100 else False
            )
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
        def process_document(self, file_path: str) -> Dict[str, Any]:
            """Enhanced main document processing function with language handling."""
        # ... (the initial part of the method is the same, checking if file exists, etc.) ...
        # --- The code below should replace the existing try...except block ---
            if not os.path.exists(file_path):
                return {"error": "File not found", "text": ""}

            file_ext = os.path.splitext(file_path)[1].lower()
            # ... (rest of initial setup) ...

            # --- NEW INTEGRATION LOGIC STARTS HERE ---
            try:
                # First, get the page details and full text using the existing logic
                raw_result = {}
                if file_ext == '.pdf':
                    raw_result = self.extract_text_from_pdf(file_path)
                elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    # Note: This part needs to be updated to return page_details if it doesn't already
                    raw_result = self.extract_text_from_image(file_path) 
                elif file_ext == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    raw_result = {
                        "text": text,
                        "page_details": [{"page_number": 1, "text": text, "word_count": len(text.split())}]
                    }

                full_text = raw_result.get("text", "")
                if not full_text.strip():
                    return raw_result # Return early if no text was extracted

                # Step 2: Detect language from the extracted text
                detected_language = self.language_service.detect_language(full_text)
                raw_result["detected_language"] = detected_language
                print(f"Detected document language: {detected_language}")

                # Step 3: Translate if not English
                if detected_language != 'en':
                    print("Translation required. Translating text to English...")
                    translated_text = self.language_service.translate_text(full_text)

                    # We need to update the page_details with translated text.
                    # For simplicity, we'll put all translated text on page 1 for now.
                    # A more advanced implementation could translate page by page.
                    raw_result['text'] = translated_text
                    raw_result['page_details'] = [
                        {"page_number": 1, "text": translated_text, "word_count": len(translated_text.split())}
                    ]
                    raw_result['original_language'] = detected_language
                    print("Translation complete.")

                return raw_result

            except Exception as e:
                logger.error(f"Enhanced document processing failed for {os.path.basename(file_path)}: {e}")
                return {"error": str(e), "text": ""}
    
    def build_index(self, document_chunks: List[Dict]) -> Dict:
        """Build FAISS index from document chunks"""
        if not document_chunks:
            return {'error': 'No chunks provided'}
        
        try:
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in document_chunks]
            
            print(f"Creating embeddings for {len(texts)} chunks...")
            embeddings = self.create_embeddings_batch(texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            
            # Use IndexHNSWFlat for better performance on larger datasets
            if len(embeddings) > 1000:
                self.index = faiss.IndexHNSWFlat(dimension, 32)
                self.index.hnsw.efConstruction = 40
            else:
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings.astype('float32'))
            
            # Store chunks
            self.chunks = document_chunks
            
            # Save everything
            self.save_index()
            
            return {
                'status': 'success',
                'total_chunks': len(document_chunks),
                'embedding_dimension': dimension,
                'index_type': type(self.index).__name__,
                'model_used': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            return {'error': str(e)}
    
    def search(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """Search for similar chunks with confidence scoring"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            query_embedding = query_embedding.cpu().numpy()
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.chunks)))
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks) and score >= threshold:
                    result = self.chunks[idx].copy()
                    result['similarity_score'] = float(score)
                    result['confidence'] = self._calculate_confidence(score)
                    results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _calculate_confidence(self, similarity_score: float) -> str:
        """Convert similarity score to confidence level"""
        if similarity_score >= 0.8:
            return "HIGH"
        elif similarity_score >= 0.6:
            return "MEDIUM"
        elif similarity_score >= 0.4:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def save_index(self):
        """Save index, chunks, and metadata"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
            
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            metadata = {
                'model_name': self.model_name,
                'total_chunks': len(self.chunks),
                'last_updated': datetime.now().isoformat(),
                'index_type': type(self.index).__name__ if self.index else None
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Index saved with {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self) -> bool:
        """Load existing index and chunks"""
        try:
            if (os.path.exists(self.index_path) and 
                os.path.exists(self.chunks_path) and 
                os.path.exists(self.metadata_path)):
                
                self.index = faiss.read_index(self.index_path)
                
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                logger.info(f"Loaded index with {len(self.chunks)} chunks")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
        
        return False
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index"""
        if not self.index or not self.chunks:
            return {'status': 'empty'}
        
        # Analyze chunks by filename
        file_stats = {}
        for chunk in self.chunks:
            filename = chunk['filename']
            if filename not in file_stats:
                file_stats[filename] = {'chunks': 0, 'total_words': 0}
            file_stats[filename]['chunks'] += 1
            file_stats[filename]['total_words'] += chunk.get('word_count', 0)
        
        return {
            'status': 'ready',
            'total_chunks': len(self.chunks),
            'total_files': len(file_stats),
            'index_size': self.index.ntotal,
            'model_name': self.model_name,
            'files': file_stats
        }

# --- This is the FINAL, CORRECT test block ---
# --- This is the FINAL, DYNAMIC test block ---
if __name__ == "__main__":
    import sys
    import os
    
    # Add the current directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    try:
        from ocr_service import AdvancedDocumentProcessor
        from llm_service import LLMService
    except ImportError as e:
        print(f"Error importing a service: {e}")
        sys.exit(1)

    # Load API Key
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found.")
        sys.exit(1)

    print("🎬 Starting the full document processing and embedding pipeline with citations...")
    
    # Initialize services
    ocr_processor = AdvancedDocumentProcessor()
    embedding_service = EmbeddingService(model_name='BAAI/bge-large-en-v1.5')
    llm_service = LLMService(api_key=api_key)

    # Define path to the PDF
    pdf_file_path = os.path.join(current_dir, '..', '..', 'documents', 'sample.pdf')
    
    # Extract page-structured text
    print("\n📄 Step 1: Extracting text with page numbers...")
    page_details = ocr_processor.process_document(pdf_file_path).get('page_details', [])
    print(f"✅ Text extracted from {len(page_details)} pages.")

    # Chunk the text while preserving page numbers
    print("\n🧠 Step 2: Chunking document...")
    word_count = sum(p['word_count'] for p in page_details)
    chunk_size, overlap = (250, 50) if word_count > 500 else (10000, 0)
    document_chunks = embedding_service.process_and_chunk_pages("sample.pdf", page_details, chunk_size, overlap)
    build_result = embedding_service.build_index(document_chunks)
    print("✅ Index built successfully.")
    
    # --- Step 3: Test the full RAG pipeline with citations ---
    print("\n🔍 Step 3: Testing the full RAG pipeline with citations...")
    test_queries = [
        "What are the main objectives of research?",
        "What are the qualities of good research?",
        "Who is the author of this document?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        # Retrieval step: find relevant chunks
        results = embedding_service.search(query, k=5, threshold=0.4)
        
        if results:
            # --- KEY CHANGE HAPPENS HERE ---
            # We now format the context to include page numbers for the LLM
            context_parts = []
            for chunk in results:
                page_num = chunk.get('page_number', 'N/A')
                context_parts.append(f"[Source: Page {page_num}]: {chunk['text']}")
            
            context = "\n\n".join(context_parts)
            
            # Generation step: use the LLM to generate an answer
            answer = llm_service.generate_answer(query, context)
            print(f"✅ Generated Answer:\n{answer}")
        else:
            print("❌ No relevant results found for this query.")