import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from language_service import LanguageService
import base64
import io
import re
from dotenv import load_dotenv
        
        

logger = logging.getLogger(__name__)

class AdvancedDocumentProcessor:
    def __init__(self):
        # Configure tesseract path if needed (Windows)
        self.language_service = LanguageService()
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        self.supported_formats = {
            '.pdf', '.txt', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp',
            '.docx', '.doc', '.rtf', '.odt', '.xps', '.epub', '.mobi'
        }
        
        # Advanced OCR configurations for different content types
        self.ocr_configs = {
            'general': r'--oem 3 --psm 6',
            'single_column': r'--oem 3 --psm 4',
            'single_text_line': r'--oem 3 --psm 7',
            'single_word': r'--oem 3 --psm 8',
            'table': r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'handwritten': r'--oem 3 --psm 13'
        }
    
    def advanced_image_preprocessing(self, image_path: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """Advanced image preprocessing with multiple enhancement techniques"""
        try:
            # Load image with OpenCV for advanced processing
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                # Fallback to PIL
                pil_image = Image.open(image_path)
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            original_shape = cv_image.shape
            preprocessing_log = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            preprocessing_log.append("converted_to_grayscale")
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            preprocessing_log.append("noise_reduction_applied")
            
            # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(denoised)
            preprocessing_log.append("contrast_enhanced_clahe")
            
            # Detect and correct skew
            skew_angle = self._detect_skew(contrast_enhanced)
            if abs(skew_angle) > 0.5:  # Only correct if significant skew
                contrast_enhanced = self._correct_skew(contrast_enhanced, skew_angle)
                preprocessing_log.append(f"skew_corrected_{skew_angle:.2f}_degrees")
            
            # Morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            morphed = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_CLOSE, kernel)
            preprocessing_log.append("morphological_cleanup")
            
            # Edge enhancement for text clarity
            sharpened = cv2.filter2D(morphed, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
            preprocessing_log.append("edge_enhancement")
            
            # Convert back to PIL Image
            final_image = Image.fromarray(sharpened)
            
            # Additional PIL enhancements
            final_image = ImageOps.autocontrast(final_image, cutoff=2)
            preprocessing_log.append("auto_contrast")
            
            preprocessing_info = {
                "original_size": original_shape[:2],
                "final_size": final_image.size,
                "skew_angle": skew_angle,
                "preprocessing_steps": preprocessing_log,
                "enhancement_score": self._calculate_enhancement_score(cv_image, np.array(final_image))
            }
            
            return final_image, preprocessing_info
            
        except Exception as e:
            logger.error(f"Advanced image preprocessing failed: {e}")
            # Fallback to basic preprocessing
            return self._basic_image_preprocessing(image_path), {"error": str(e)}
    
    def _detect_skew(self, image: np.ndarray) -> float:
        """Detect skew angle in the image"""
        try:
            # Use Hough Line Transform to detect skew
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:20]:  # Use first 20 lines
                    angle = np.degrees(theta) - 90
                    if abs(angle) < 45:  # Filter reasonable angles
                        angles.append(angle)
                
                if angles:
                    return np.median(angles)
            
            return 0.0
        except:
            return 0.0
    
    def _correct_skew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Correct skew in the image"""
        try:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return corrected
        except:
            return image
    
    def _calculate_enhancement_score(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate a score representing enhancement quality"""
        try:
            # Convert to grayscale if needed
            if len(original.shape) == 3:
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            
            # Calculate contrast improvement
            orig_std = np.std(original)
            proc_std = np.std(processed)
            
            contrast_improvement = proc_std / orig_std if orig_std > 0 else 1.0
            return min(contrast_improvement, 2.0)  # Cap at 2.0
        except:
            return 1.0
    
    def _basic_image_preprocessing(self, image_path: str) -> Image.Image:
        """Fallback basic preprocessing"""
        try:
            image = Image.open(image_path)
            if image.mode != 'L':
                image = image.convert('L')
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
        except Exception as e:
            logger.error(f"Basic image preprocessing failed: {e}")
            return Image.open(image_path)
    
    def extract_text_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """Advanced PDF text extraction with PyMuPDF, including table extraction."""
        try:
            doc = fitz.open(file_path)
            
            extracted_content = {
                "text": "",
                "pages": len(doc),
                "metadata": doc.metadata,
                "page_details": [],
                "images_found": 0,
                "tables_detected": 0,
            }
            
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # --- FIX IS HERE ---
                # 1. Find all tables and immediately convert the finder object to a list
                tables = page.find_tables()
                tables_list = list(tables) # Convert to a list
                
                # Now we can safely get the length
                extracted_content["tables_detected"] += len(tables_list)
                
                table_markdown = ""
                # And check the list's length
                if tables_list:
                    print(f"Found {len(tables_list)} table(s) on page {page_num + 1}.")
                    # Loop over the list to extract and format each table
                    for table in tables_list:
                        table_data = table.extract()
                        table_markdown += self._format_table_as_markdown(table_data)
                # --- END OF FIX ---

                # Extract regular text from the page
                page_text = page.get_text()
                
                # Combine the regular text with the formatted table text
                combined_page_text = page_text + table_markdown
                
                page_info = {
                    "page_number": page_num + 1,
                    "text": combined_page_text,
                    "word_count": len(combined_page_text.split()),
                    "char_count": len(combined_page_text)
                }
                extracted_content["page_details"].append(page_info)
                
                full_text += f"\n--- Page {page_num + 1} ---\n{combined_page_text}\n"
                extracted_content["images_found"] += len(page.get_images())

            extracted_content["text"] = full_text
            extracted_content["word_count"] = len(full_text.split())
            
            doc.close()
            return extracted_content
            
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return {"text": "", "error": str(e), "pages": 0, "page_details": []}
    def _extract_structured_text(self, text_dict: dict) -> Dict[str, Any]:
        """Extract structured information from PyMuPDF text dict"""
        try:
            fonts_used = set()
            text_sizes = []
            text_blocks = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            fonts_used.add(span.get("font", ""))
                            text_sizes.append(span.get("size", 0))
                            text_blocks.append({
                                "text": span.get("text", ""),
                                "font": span.get("font", ""),
                                "size": span.get("size", 0),
                                "flags": span.get("flags", 0),
                                "bbox": span.get("bbox", [])
                            })
            
            return {
                "fonts_used": list(fonts_used),
                "font_sizes": list(set(text_sizes)),
                "avg_font_size": np.mean(text_sizes) if text_sizes else 0,
                "text_blocks_count": len(text_blocks),
                "structured_blocks": text_blocks[:50]  # Limit for performance
            }
        except Exception as e:
            logger.error(f"Structured text extraction failed: {e}")
            return {"error": str(e)}
    
    def _detect_tables_in_page(self, page) -> List[Dict]:
        """Detect tables in PDF page"""
        try:
            # Simple table detection using text positioning
            text_dict = page.get_text("dict")
            potential_tables = []
            
            # Look for aligned text patterns that might indicate tables
            lines_by_y = {}
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        y_pos = round(line["bbox"][1])
                        if y_pos not in lines_by_y:
                            lines_by_y[y_pos] = []
                        lines_by_y[y_pos].append(line)
            
            # Find rows with multiple aligned elements
            for y_pos, lines in lines_by_y.items():
                if len(lines) >= 3:  # Potential table row
                    potential_tables.append({
                        "y_position": y_pos,
                        "columns": len(lines),
                        "bbox": [min(l["bbox"][0] for l in lines),
                                y_pos,
                                max(l["bbox"][2] for l in lines),
                                y_pos + max(l["bbox"][3] - l["bbox"][1] for l in lines)]
                    })
            
            return potential_tables
        except:
            return []
    
    def _extract_annotations(self, page) -> List[Dict]:
        """Extract annotations from PDF page"""
        try:
            annotations = []
            for annot in page.annots():
                annot_info = {
                    "type": annot.type[1],
                    "content": annot.info.get("content", ""),
                    "author": annot.info.get("title", ""),
                    "bbox": list(annot.rect),
                    "page": page.number + 1
                }
                annotations.append(annot_info)
            return annotations
        except:
            return []
    
    def _ocr_pdf_page(self, page) -> str:
        """Apply OCR to a PDF page when text extraction is insufficient"""
        try:
            # Render page to image
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Apply advanced preprocessing
            enhanced_image, _ = self.advanced_image_preprocessing_pil(image)
            
            # Perform OCR
            ocr_text = pytesseract.image_to_string(
                enhanced_image, 
                config=self.ocr_configs['general']
            )
            
            return ocr_text
        except Exception as e:
            logger.error(f"PDF page OCR failed: {e}")
            return ""
    def _format_table_as_markdown(self, table_data: list) -> str:
        """Converts a list of lists into a Markdown table string."""
        if not table_data:
            return ""
        
        markdown_table = ""
        # Create the header row
        header = table_data[0]
        markdown_table += "| " + " | ".join(str(h) if h is not None else '' for h in header) + " |\n"
        
        # Create the separator line
        markdown_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
        
        # Create the data rows
        for row in table_data[1:]:
            markdown_table += "| " + " | ".join(str(c) if c is not None else '' for c in row) + " |\n"
            
        return "\n" + markdown_table + "\n"
    
    def advanced_image_preprocessing_pil(self, pil_image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Advanced preprocessing for PIL images"""
        try:
            # Convert to numpy for OpenCV processing
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Apply advanced preprocessing
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Convert back to PIL
            final_image = Image.fromarray(enhanced)
            
            return final_image, {"enhancement": "applied"}
        except:
            return pil_image, {"enhancement": "failed"}
    
    def _analyze_document_quality(self, extracted_content: Dict) -> Dict[str, Any]:
        """Analyze overall document quality and extraction success"""
        try:
            total_pages = extracted_content.get("pages", 0)
            total_words = extracted_content.get("word_count", 0)
            total_chars = extracted_content.get("char_count", 0)
            
            # Calculate quality metrics
            avg_words_per_page = total_words / total_pages if total_pages > 0 else 0
            avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
            
            # Determine quality score
            quality_score = 0
            quality_notes = []
            
            if avg_words_per_page > 100:
                quality_score += 40
                quality_notes.append("good_text_density")
            elif avg_words_per_page > 50:
                quality_score += 20
                quality_notes.append("moderate_text_density")
            else:
                quality_notes.append("low_text_density")
            
            if extracted_content.get("images_found", 0) > 0:
                quality_score += 10
                quality_notes.append("contains_images")
            
            if extracted_content.get("tables_detected", 0) > 0:
                quality_score += 15
                quality_notes.append("contains_tables")
            
            if extracted_content.get("annotations", []):
                quality_score += 10
                quality_notes.append("has_annotations")
            
            if extracted_content["metadata"].get("title"):
                quality_score += 5
                quality_notes.append("has_metadata")
            
            # Text extraction success rate
            pages_with_text = sum(1 for page in extracted_content.get("page_details", []) 
                                if page.get("word_count", 0) > 10)
            extraction_success_rate = (pages_with_text / total_pages * 100) if total_pages > 0 else 0
            
            quality_score += min(extraction_success_rate / 4, 20)  # Up to 20 points for extraction success
            
            return {
                "overall_quality_score": min(quality_score, 100),
                "extraction_success_rate": extraction_success_rate,
                "avg_words_per_page": avg_words_per_page,
                "text_density": "high" if avg_words_per_page > 100 else "medium" if avg_words_per_page > 50 else "low",
                "content_richness": len(quality_notes),
                "quality_indicators": quality_notes,
                "recommendation": self._get_quality_recommendation(quality_score)
            }
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return {"error": str(e)}
    
    def _get_quality_recommendation(self, score: float) -> str:
        """Get recommendation based on quality score"""
        if score >= 80:
            return "excellent_extraction_quality"
        elif score >= 60:
            return "good_extraction_quality"
        elif score >= 40:
            return "moderate_quality_consider_ocr_enhancement"
        else:
            return "low_quality_manual_review_recommended"
    
    def extract_text_from_image(self, file_path: str) -> Dict[str, Any]:
        """Advanced OCR processing for images"""
        try:
            # Apply advanced preprocessing
            enhanced_image, preprocessing_info = self.advanced_image_preprocessing(file_path)
            
            # Multiple OCR attempts with different configurations
            ocr_results = {}
            
            for config_name, config in self.ocr_configs.items():
                try:
                    text = pytesseract.image_to_string(enhanced_image, config=config)
                    
                    # Get detailed OCR data
                    data = pytesseract.image_to_data(enhanced_image, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Word-level analysis
                    words = [word for word in data['text'] if word.strip()]
                    
                    ocr_results[config_name] = {
                        "text": text,
                        "confidence": avg_confidence,
                        "word_count": len(words),
                        "character_count": len(text),
                        "words_detected": len(words)
                    }
                    
                except Exception as e:
                    ocr_results[config_name] = {"error": str(e)}
            
            # Select best OCR result
            best_result = self._select_best_ocr_result(ocr_results)
            
            # Language detection
            detected_language = self._detect_language(best_result["text"])
            
            # Text quality analysis
            text_quality = self._analyze_text_quality(best_result["text"])
            
            return {
                "text": best_result["text"],
                "confidence": best_result["confidence"],
                "word_count": best_result["word_count"],
                "character_count": best_result["character_count"],
                "image_size": enhanced_image.size,
                "detected_language": detected_language,
                "preprocessing_info": preprocessing_info,
                "ocr_method_used": best_result["method"],
                "alternative_results": {k: v for k, v in ocr_results.items() if k != best_result["method"]},
                "text_quality_analysis": text_quality,
                "enhancement_applied": True
            }
            
        except Exception as e:
            logger.error(f"Advanced OCR processing error: {e}")
            return {"text": "", "error": str(e), "confidence": 0}
    
    def _select_best_ocr_result(self, ocr_results: Dict) -> Dict:
        """Select the best OCR result based on multiple criteria"""
        best_score = -1
        best_result = None
        best_method = None
        
        for method, result in ocr_results.items():
            if "error" in result:
                continue
                
            # Scoring criteria
            score = 0
            
            # Confidence weight (40%)
            score += result.get("confidence", 0) * 0.4
            
            # Word count weight (30%) - more words generally better
            word_count = result.get("word_count", 0)
            score += min(word_count / 100, 1) * 30
            
            # Text length weight (20%)
            char_count = result.get("character_count", 0)
            score += min(char_count / 1000, 1) * 20
            
            # Method-specific bonus (10%)
            method_bonus = {
                "general": 5,
                "single_column": 7,
                "table": 3,
                "handwritten": 2
            }
            score += method_bonus.get(method, 0)
            
            if score > best_score:
                best_score = score
                best_result = result
                best_method = method
        
        if best_result:
            best_result["method"] = best_method
            best_result["selection_score"] = best_score
            return best_result
        
        # Fallback
        return {"text": "", "confidence": 0, "word_count": 0, "character_count": 0, "method": "none"}
    
    def _detect_language(self, text: str) -> str:
        """Detect language of extracted text"""
        try:
            from langdetect import detect
            if len(text.strip()) > 20:
                return detect(text)
            return "unknown"
        except:
            return "unknown"
    
    def _analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """Analyze quality of extracted text"""
        try:
            # Basic metrics
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.split('\n'))
            
            # Character distribution analysis
            alpha_count = sum(1 for c in text if c.isalpha())
            digit_count = sum(1 for c in text if c.isdigit())
            space_count = sum(1 for c in text if c.isspace())
            punct_count = sum(1 for c in text if c in '.,!?;:"()[]{}')
            special_count = char_count - alpha_count - digit_count - space_count - punct_count
            
            # Quality indicators
            alpha_ratio = alpha_count / char_count if char_count > 0 else 0
            word_length_avg = char_count / word_count if word_count > 0 else 0
            
            # Detect potential OCR errors
            repeated_chars = len(re.findall(r'(.)\1{3,}', text))  # 4+ repeated chars
            suspicious_patterns = len(re.findall(r'[^\w\s]{3,}', text))  # 3+ special chars together
            
            quality_score = 100
            issues = []
            
            # Deduct points for issues
            if alpha_ratio < 0.5:
                quality_score -= 30
                issues.append("low_alphabetic_content")
            
            if repeated_chars > 0:
                quality_score -= repeated_chars * 5
                issues.append("repeated_characters_detected")
            
            if suspicious_patterns > text.count('\n') / 2:
                quality_score -= 20
                issues.append("suspicious_character_patterns")
            
            if word_length_avg < 3:
                quality_score -= 15
                issues.append("unusually_short_words")
            
            return {
                "quality_score": max(quality_score, 0),
                "character_distribution": {
                    "alphabetic": alpha_count,
                    "numeric": digit_count,
                    "whitespace": space_count,
                    "punctuation": punct_count,
                    "special": special_count
                },
                "text_metrics": {
                    "total_chars": char_count,
                    "total_words": word_count,
                    "total_lines": line_count,
                    "avg_word_length": word_length_avg,
                    "alphabetic_ratio": alpha_ratio
                },
                "potential_issues": issues,
                "ocr_confidence_indicators": {
                    "repeated_chars": repeated_chars,
                    "suspicious_patterns": suspicious_patterns
                }
            }
            
        except Exception as e:
            logger.error(f"Text quality analysis failed: {e}")
            return {"error": str(e)}
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Enhanced main document processing function with language handling."""
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        file_ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        
        # --- NEW INTEGRATION LOGIC STARTS HERE ---
        try:
            raw_result = {}
            # Step 1: Extract text using the appropriate method for the file type
            if file_ext == '.pdf':
                raw_result = self.extract_text_from_pdf(file_path)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                image_text = self.extract_text_from_image(file_path).get("text", "")
                raw_result = {
                    "text": image_text,
                    "page_details": [{"page_number": 1, "text": image_text, "word_count": len(image_text.split())}]
                }
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                raw_result = {
                    "text": text,
                    "page_details": [{"page_number": 1, "text": text, "word_count": len(text.split())}]
                }
            else:
                return {"error": f"Unsupported format: {file_ext}"}

            full_text = raw_result.get("text", "")
            if not full_text or not full_text.strip():
                print("No text content found in document.")
                return raw_result

            # Step 2: Detect the language from the extracted text
            detected_language = self.language_service.detect_language(full_text)
            raw_result["detected_language"] = detected_language
            print(f"Detected document language: {detected_language}")

            # Step 3: Translate the text if it's not English
            if detected_language != 'en' and detected_language != 'unknown':
                print(f"Translation required from '{detected_language}' to 'en'. Translating text...")
                
                # We need to translate the text from each page to preserve structure
                translated_page_details = []
                full_translated_text = ""

                for page in raw_result.get("page_details", []):
                    translated_text = self.language_service.translate_text(page['text'], source_lang=detected_language)
                    translated_page_details.append({
                        "page_number": page['page_number'],
                        "text": translated_text,
                        "word_count": len(translated_text.split())
                    })
                    full_translated_text += translated_text + "\n"
                # --- FINAL INTEGRATION STEP ---
                # If the knowledge service is active, process text for entities
                if self.knowledge_service and full_translated_text:
                    print("Updating knowledge graph...")
                    entities = self.knowledge_service.extract_entities(full_translated_text)
                    self.knowledge_service.ingest_entities_and_relationships(entities, filename)
                # --- END OF INTEGRATION ---
                raw_result['text'] = full_translated_text.strip()
                raw_result['page_details'] = translated_page_details
                raw_result['original_language'] = detected_language
                print("✅ Translation complete.")
            
            # Attach final metadata before returning
            raw_result.update({
                "filename": filename,
                "file_size": os.path.getsize(file_path),
                "format": file_ext
            })
            return raw_result

        except Exception as e:
            logger.error(f"Enhanced document processing failed for {filename}: {e}")
            return {"error": str(e)}

        load_dotenv()
        uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        # Initialize the Knowledge Service to be used later
        if password:
            from knowledge_service import KnowledgeService
            self.knowledge_service = KnowledgeService(uri, user, password)
        else:
            self.knowledge_service = None
            print("Warning: NEO4J_PASSWORD not found in .env file. Knowledge Graph will be disabled.")
# Test the enhanced implementation
import os

if __name__ == "__main__":
    processor = AdvancedDocumentProcessor()
    
    # Define the path to the sample.pdf file
    # The ".." moves up one directory from 'app/services' to 'app'
    # Then 'documents' and 'sample.pdf' point to the correct file
    pdf_file_path = os.path.join("..", "..", "documents", "sample.pdf")
    
    if not os.path.exists(pdf_file_path):
        print(f"Error: PDF file not found at {pdf_file_path}")
    else:
        print(f"🚀 Processing PDF document: {pdf_file_path}")
        
        # Call the main processing function
        pdf_result = processor.process_document(pdf_file_path)
        
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










