import os
import sys

# Add the parent directory to the Python path to allow importing ocr_service.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ocr_service import AdvancedDocumentProcessor

def run_pdf_test(file_path):
    """
    Initializes the processor and runs the test on the specified PDF file.
    """
    processor = AdvancedDocumentProcessor()
    
    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return
    
    print(f"🚀 Processing PDF document: {file_path}")
    
    # Call the main processing function
    pdf_result = processor.process_document(file_path)
    
    print("\n================ PDF Processing Report ================")
    
    if pdf_result.get("error"):
        print(f"❌ Processing failed: {pdf_result['error']}")
    else:
        print("✅ Processing Success: True")
        print(f"📄 Filename: {pdf_result.get('filename')}")
        print(f"🔢 Total Pages: {pdf_result.get('pages', 0)}")
        print(f"📝 Total Word Count: {pdf_result.get('word_count', 0)}")
        print(f"🔤 Total Character Count: {pdf_result.get('char_count', 0)}")
        print(f"🖼️ Images Found: {pdf_result.get('images_found', 0)}")
        print(f"📊 Tables Detected: {pdf_result.get('tables_detected', 0)}")
        print(f"📌 Annotations: {len(pdf_result.get('annotations', []))}")
        print(f"🔗 Embedded Files: {pdf_result.get('embedded_files', 'None')}")
        
        # Print Extracted Metadata
        print("\n--- Extracted Metadata ---")
        for key, value in pdf_result.get("metadata", {}).items():
            print(f"| {key.capitalize().replace('_', ' '):<20}: {value}")
        
        # Print Quality Analysis
        print("\n--- Document Quality Analysis ---")
        quality_analysis = pdf_result.get("quality_analysis", {})
        for key, value in quality_analysis.items():
            if isinstance(value, list):
                value = ", ".join(value)
            print(f"| {key.capitalize().replace('_', ' '):<25}: {value}")
        
        # Print a detailed preview of each page
        print("\n--- Detailed Page Content ---")
        for page in pdf_result.get("page_details", []):
            print(f"➡️ Page {page['page_number']}:")
            print(f"   - Words: {page['word_count']}, Chars: {page['char_count']}")
            print(f"   - Images: {page['images']}, Tables: {page['tables']}")
            print(f"   - Structured Content: {page.get('structured_content', {}).get('text_blocks_count', 0)} blocks")
            print("   - Text Preview: " + page.get('text', '')[:200].replace('\n', ' ') + "...")
            print("-" * 20)
            
        print("\n--- Full Extracted Text ---")
        print(pdf_result.get('text', 'No text extracted.'))
        print("\n====================================================")

if __name__ == "__main__":
    # Define the path to the sample.pdf file
    pdf_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'documents','Sample2.pdf')
    
    run_pdf_test(pdf_file_path)