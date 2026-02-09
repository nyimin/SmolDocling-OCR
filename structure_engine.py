import os
import traceback
import time
from PIL import Image
import fitz # PyMuPDF
import cleaner # New cleaner module
import metadata_extractor
import openrouter_validator
import rapidocr_validator
import layout_analyzer # Metadata extraction and YAML frontmatter

# Enhanced Pipeline Integration (v2.0)
try:
    from enhanced_pipeline import EnhancedPipeline, process_document_enhanced
    ENHANCED_PIPELINE_AVAILABLE = True
    print("Enhanced pipeline loaded successfully.")
except ImportError as e:
    print(f"Enhanced pipeline not available: {e}")
    ENHANCED_PIPELINE_AVAILABLE = False

# GMFT Imports
try:
    from gmft.pdf_bindings import PyPDFium2Document
    from gmft.auto import AutoTableDetector, AutoTableFormatter, TATRFormatConfig
    print("GMFT imported successfully.")
except ImportError:
    print("GMFT not found. Table extraction will be disabled.")
    PyPDFium2Document = None

# RapidOCR Imports
try:
    from rapidocr_onnxruntime import RapidOCR
    print("RapidOCR imported successfully.")
except ImportError:
    print("RapidOCR not found. Scan mode disabled.")
    RapidOCR = None

# OpenRouter / OpenAI Imports
try:
    from openai import OpenAI
    import base64
    print("OpenAI library imported successfully (for OpenRouter).")
except ImportError:
    print("OpenAI library not found. OpenRouter OCR disabled.")
    OpenAI = None

# ... (skip lines) ...

# Initialize GMFT (Lightweight)
detector = None
formatter = None
if PyPDFium2Document:
    detector = AutoTableDetector()
    formatter = AutoTableFormatter()

# Initialize RapidOCR
ocr_engine = None
if RapidOCR:
    # default_model=True downloads models to ~/.rapidocr/ by default if not present
    try:
        ocr_engine = RapidOCR() 
    except Exception as e:
        print(f"Error initializing RapidOCR: {e}")

# ... (skip lines) ...

def extract_with_pymupdf4llm(pdf_path):
    """
    Use pymupdf4llm to extract digital PDFs to LLM-optimized Markdown.
    Returns: Markdown string, metadata dict
    """
    import pymupdf4llm
    
    try:
        # Extract with pymupdf4llm (optimized for LLMs)
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Get page count for metadata
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        
        # Extract document metadata
        doc_metadata = metadata_extractor.extract_metadata(pdf_path)
        
        # Add YAML frontmatter
        extraction_method = "pymupdf4llm (Digital PDF - LLM Optimized)"
        markdown_with_frontmatter = metadata_extractor.add_yaml_frontmatter(
            md_text,
            doc_metadata,
            extraction_method=extraction_method,
            confidence_score=None,
            language="en",
            ocr_engine="pymupdf4llm"
        )
        
        metadata = {
            "extraction_method": "Digital (pymupdf4llm)",
            "pages": page_count,
            "optimized_for": "LLM/RAG",
            "features": "Superior table extraction, document structure preservation"
        }
        
        return markdown_with_frontmatter, metadata
        
    except Exception as e:
        print(f"pymupdf4llm extraction failed: {e}")
        traceback.print_exc()
        return None, {"error": str(e)}



# ... (previous imports)
from gmft.pdf_bindings.base import BasePage
from gmft.base import Rect
import numpy as np

# ... (detector/formatter setup)

class RapidOCRPage(BasePage):
    """
    Adapter to make an Image + RapidOCR result look like a PDF page to GMFT.
    """
    def __init__(self, image, ocr_result, page_num=0):
        self._image = image
        self.width = image.width
        self.height = image.height
        self._ocr_result = ocr_result
        self.page_number = page_num
        
    def get_positions_and_text(self):
        # Yields (x0, y0, x1, y1, "string")
        if not self._ocr_result:
            return
            
        for line in self._ocr_result:
            box, text, score = line
            # RapidOCR box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, y0, x1, y1 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
            yield (x0, y0, x1, y1, text)
            
    def get_image(self, dpi=None, rect=None):
        # We process in native pixels, so we ignore DPI scaling requests mostly
        # or assume 72 DPI if forced? GMFT uses 72 DPI as base.
        # But here our "PDF coordinates" ARE pixels.
        if rect:
            # rect.bbox is (x0, y0, x1, y1)
            return self._image.crop(rect.bbox)
        return self._image
        
    def get_filename(self):
        return f"scan_page_{self.page_number}.png"
        
    def close(self):
        pass

def extract_with_rapidocr(input_path, dpi=300, lang="en", enhanced_mode=True):
    """
    Use RapidOCR + GMFT to extract text and tables from images/scans.
    
    Args:
        input_path: Path to PDF or image file
        dpi: DPI for PDF rendering
        lang: Language code for OCR
        enhanced_mode: Use v2.0 quality enhancement pipeline (default: True)
    
    Returns:
        tuple: (markdown_text, metadata_dict)
    """
    if not ocr_engine:
        return "Error: RapidOCR not initialized.", {}
        
    output_md = ""
    
    # Metadata dictionary to populate
    metadata = {
        "extraction_method": f"RapidOCR (lang={lang})",
        "dpi": dpi,
        "language": lang,
        "pages_processed": 0,
        "quality_score": 0.0,
        "validation_issues": 0
    }
    
    # Handle PDF input by converting to images first
    images = []
    if input_path.lower().endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(input_path)
            for page in doc:
                pix = page.get_pixmap(dpi=dpi) # Configurable DPI for OCR accuracy
                img_data = pix.tobytes("png")
                # Convert bytes to PIL Image
                import io
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                images.append(img)
            doc.close()
        except ImportError:
             return "Error: PyMuPDF (fitz) not found. Cannot process PDF for OCR."
    else:
        # It's an image file path (str)
        try:
            images = [Image.open(input_path).convert("RGB")]
        except Exception as e:
            return f"Error reading image file: {e}", {}

    # Collect all pages' elements first for global analysis (header/footer detection)
    all_pages_elements = []  # List of lists of elements

    # Process each image
    for i, image in enumerate(images):
        page_elements = []
        try:
            # 1. Run RapidOCR
            # RapidOCR expects an array or path. We pass the numpy array of the image.
            img_np = np.array(image)
            result, _ = ocr_engine(img_np)
            
            if not result:
                # Store empty page to keep index aligned
                all_pages_elements.append([])
                continue
                
            # 2. Create Adapter Page
            page_wrapper = RapidOCRPage(image, result, page_num=i)
            
            # 3. Try to detect tables with GMFT
            tables = []
            if detector:
                try:
                    tables = detector.extract(page_wrapper)
                except Exception as e:
                    print(f"GMFT Scan Detection failed: {e}")
            
            # 4. Format Output - Interleaved
            # We will collect "elements" (either text line or table) with their Y-position
            
            table_bboxes = [t.rect.bbox for t in tables] # (x0, y0, x1, y1)
            
            # Helper: Check if box is inside any table
            def is_in_table(box):
                bx0, by0, bx1, by1 = box
                b_center_x, b_center_y = (bx0+bx1)/2, (by0+by1)/2
                for (tx0, ty0, tx1, ty1) in table_bboxes:
                    if tx0 <= b_center_x <= tx1 and ty0 <= b_center_y <= ty1:
                        return True
                return False

            # Collect non-table text with full metadata for layout analysis
            text_elements = []
            for line in result:
                # line format: [bbox_points, (text, confidence)]
                box = line[0]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                l_box = (min(xs), min(ys), max(xs), max(ys))
                
                if not is_in_table(l_box):
                    text_elements.append({
                        "bbox": l_box,
                        "text": line[1],  # Text content
                        "confidence": line[2] if len(line) > 2 else 1.0  # OCR confidence
                    })
            
            # Apply layout analysis for intelligent reading order
            analyzer = layout_analyzer.LayoutAnalyzer(column_gap_threshold=50)
            layout_result = analyzer.analyze_page_layout(
                text_elements,
                confidence_threshold=0.7
            )
            
            # Convert analyzed elements to page_elements format
            for elem in layout_result['elements']:
                page_elements.append({
                    "y": elem['bbox'][1],
                    "type": "text",
                    "content": elem['text'],
                    "reading_order": elem.get('reading_order', 0),
                    "semantic_role": elem.get('semantic_role', 'paragraph'),
                    "confidence": elem.get('confidence', 1.0),
                    "uncertain": elem.get('uncertain', False)
                })
            
            # Collect tables
            # Custom config for scans
            scan_config = TATRFormatConfig(formatter_base_threshold=0.7)
            
            if tables and formatter:
                for table in tables:
                    try:
                        ft = formatter.extract(table, config_overrides=scan_config)
                        df = ft.df()
                        md_table = df.to_markdown()
                        
                        # Store as (y0, type="table", content)
                        # table.rect.bbox is (x0, y0, x1, y1)
                        if md_table:
                            page_elements.append({
                                "y": table.rect.bbox[1],
                                "type": "table",
                                "content": md_table
                            })
                    except Exception as e:
                         page_elements.append({
                            "y": table.rect.bbox[1],
                            "type": "text",
                            "content": f"[Error formatting detected table: {e}]"
                        })

            # Sort all elements by Y position
            page_elements.sort(key=lambda x: x["y"])
            all_pages_elements.append(page_elements)

        except Exception as e:
            print(f"Error during OCR/Processing page {i}: {e}")
            traceback.print_exc()
            all_pages_elements.append([]) # Append empty on error to keep alignment

    # --- GLOBAL CLEANING STEP ---
    # remove repetitive headers/footers
    # DISABLED: Delegated to EnhancedPipeline (noise_filter.py) for "Mark-Don't-Remove" strategy.
    # cleaned_pages = cleaner.detect_and_remove_headers_footers(all_pages_elements)
    cleaned_pages = all_pages_elements

    # --- ENHANCED PIPELINE (v2.0) ---
    if enhanced_mode and ENHANCED_PIPELINE_AVAILABLE:
        try:
            # Extract metadata first
            if input_path.lower().endswith('.pdf'):
                doc_metadata = metadata_extractor.extract_pdf_metadata(input_path)
            else:
                doc_metadata = metadata_extractor.extract_image_metadata(input_path)
            
            # Add extraction method
            doc_metadata['extraction_method'] = f"RapidOCR (lang={lang})"
            
            # Process through enhanced pipeline
            markdown, report = process_document_enhanced(
                cleaned_pages, 
                doc_metadata,
                quality_threshold=0.6
            )
            
            # Log quality info
            quality_score = report.get('quality', {}).get('score', '0.0')
            gate_passed = report.get('passed', True)
            print(f"Enhanced Pipeline: quality={quality_score}, gate_passed={gate_passed}")
            
            # Update metadata
            metadata['quality_score'] = quality_score
            metadata['pages_processed'] = len(cleaned_pages)
            
            return markdown, metadata
            
        except Exception as e:
            print(f"Enhanced pipeline failed, falling back: {e}")
            traceback.print_exc()
            # Fall through to legacy rendering

    # --- LEGACY RENDER STEP (fallback) ---
    # Render markdown with semantic annotations
    for i, page_elems in enumerate(cleaned_pages):
        output_md += f"\n\n<!-- page:{i + 1} -->\n\n"
        
        page_text_accumulator = ""
        
        for elem in page_elems:
            elem_type = elem.get("type", "text")
            content = elem.get("content", "")
            semantic_role = elem.get("semantic_role")
            reading_order = elem.get("reading_order", 0)
            uncertain = elem.get("uncertain", False)
            confidence = elem.get("confidence", 1.0)
            try:
                confidence = float(confidence) if isinstance(confidence, str) else confidence
            except (ValueError, TypeError):
                confidence = 1.0
            
            if elem_type == "text":
                # Add semantic role annotation
                if semantic_role:
                    if semantic_role == "heading":
                        page_text_accumulator += f"<!-- role:heading -->\n"
                    elif semantic_role != "paragraph":  # Don't annotate default paragraphs
                        page_text_accumulator += f"<!-- role:{semantic_role} -->\n"
                
                # Add confidence marker for low-confidence text
                if confidence < 0.7:
                    page_text_accumulator += f"<!-- confidence:{confidence:.2f} -->\n"
                
                # Handle uncertain text
                if uncertain:
                    page_text_accumulator += f"[uncertain: {content}]\n"
                else:
                    page_text_accumulator += content + "\n"
                    
            elif elem_type == "table":
                # Flush text accumulator first (with defrag)
                if page_text_accumulator:
                    # Fix hyphenation and defragment
                    fixed_text = cleaner.merge_hyphenated_words(page_text_accumulator)
                    output_md += fixed_text + "\n"
                    page_text_accumulator = ""
                
                # Add table with role annotation
                output_md += "<!-- role:table -->\n"
                output_md += "\n" + content + "\n\n"
        
        # Flush remaining text
        if page_text_accumulator:
             fixed_text = cleaner.merge_hyphenated_words(page_text_accumulator)
             output_md += fixed_text + "\n"
    
    # Extract document metadata and add YAML frontmatter
    if input_path.lower().endswith('.pdf'):
        doc_metadata = metadata_extractor.extract_pdf_metadata(input_path)
    else:
        doc_metadata = metadata_extractor.extract_image_metadata(input_path)
    
    # Add YAML frontmatter
    extraction_method = f"RapidOCR (lang={lang})"
    output_md_with_frontmatter = metadata_extractor.add_yaml_frontmatter(
        output_md,
        doc_metadata,
        extraction_method=extraction_method,
        confidence_score=None,  # TODO: Calculate average OCR confidence
        language=lang
    )

    # Update metadata for legacy path
    metadata['pages_processed'] = len(cleaned_pages)

    return output_md_with_frontmatter, metadata


# ============================================================================
# OpenRouter OCR Integration
# ============================================================================

# OpenRouter Model Configurations
OPENROUTER_MODELS = {
    "free": {
        "id": "nvidia/nemotron-nano-12b-v2-vl:free",
        "name": "Nemotron Nano 12B VL",
        "cost_per_1m": 0.0,
        "description": "FREE OCR specialist with video support"
    },
    "cheap": {
        "id": "google/gemini-2.0-flash-lite-001",
        "name": "Gemini 2.0 Flash Lite",
        "cost_per_1m": 0.000075,
        "description": "Ultra-fast, 1M context, multimodal"
    },
    "balanced": {
        "id": "qwen/qwen2.5-vl-32b-instruct",
        "name": "Qwen 2.5-VL 32B",
        "cost_per_1m": 0.00005,
        "description": "Best value, 75% accuracy (GPT-4o level)"
    },
    "quality": {
        "id": "qwen/qwen2.5-vl-72b-instruct",
        "name": "Qwen 2.5-VL 72B",
        "cost_per_1m": 0.00015,
        "description": "Highest accuracy, document specialist"
    },
    "premium": {
        "id": "mistralai/pixtral-large-2411",
        "name": "Mistral Pixtral Large",
        "cost_per_1m": 0.002,
        "description": "124B params, SOTA performance"
    }
}

def extract_with_openrouter(input_path, model="free", api_key=None):
    """
    Extract text from PDF or image using OpenRouter vision models.
    
    Args:
        input_path: Path to PDF or image file
        model: Model tier ("free", "cheap", "balanced", "quality", "premium")
        api_key: OpenRouter API key
    
    Returns:
        tuple: (markdown_text, metadata_dict)
        metadata includes: model_used, pages_processed, estimated_cost
    """
    if not OpenAI:
        return "Error: OpenAI library not installed. Run: pip install openai", {}
    
    if not api_key:
        return "Error: OpenRouter API key required. Get one at https://openrouter.ai", {}
    
    # Get model configuration
    if model not in OPENROUTER_MODELS:
        model = "free"  # Default to free model
    
    model_config = OPENROUTER_MODELS[model]
    model_id = model_config["id"]
    
    try:
        # Initialize OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # Convert PDF/image to base64
        images_base64 = []
        page_count = 0
        
        if input_path.lower().endswith('.pdf'):
            # Process PDF pages
            doc = fitz.open(input_path)
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc[page_num]
                # Render page to image (300 DPI for high fidelity)
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                images_base64.append(img_base64)
            
            doc.close()
        else:
            # Process single image
            with open(input_path, 'rb') as f:
                img_bytes = f.read()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                images_base64.append(img_base64)
            page_count = 1
        
        
        # Enhanced extraction prompt for RAG-optimized Markdown (High-Fidelity)
        extraction_prompt = """You are a precision document OCR specialist. Extract ALL visible text from this document with PERFECT accuracy.

🚨 CRITICAL RULES - NEVER VIOLATE:

1. FACTUALITY: Extract ONLY what you see. NEVER invent, infer, or hallucinate content.
2. UNCERTAINTY: If text is unclear/blurry, mark with [uncertain: best_guess]
3. COMPLETENESS: Extract ALL visible text. DO NOT summarize or skip sections.
4. ACCURACY: Preserve exact spelling, punctuation, and formatting.
5. NOISE HANDLING: DO NOT REMOVE headers, footers, or page numbers. Instead, wrap them in semantic tags (see below).

📋 OUTPUT FORMAT:

Use HTML comments for semantic annotations (invisible in rendering):

<!-- page:N --> - Mark page boundaries
<!-- role:TYPE --> - Before each element (heading, paragraph, table, list, figure, caption)
<!-- confidence:0.XX --> - For uncertain content (0.0-1.0)

🎨 STRICT STYLING RULES:

1. Lists: ALWAYS use hyphens (-) for unordered lists, never asterisks (*).
2. Spacing: EXACTLY one blank line between paragraphs/headers.
3. Links: Format visible URLs as [Link Text](url).
4. Tables: Align columns visually. Do not break rows across lines.

🎯 SEMANTIC ROLES:

- heading (with level:1-6 attribute)
- paragraph
- table (with caption attribute if present)
- list (with type:ordered|unordered attribute)
- figure (with caption attribute)
- caption
- footnote (with id attribute)
- header (for running headers)
- footer (for running footers)
- page-number

📐 MARKDOWN FORMATTING:

Headings: Use # ## ### based on visual hierarchy (size, bold, position)
Tables: Use GitHub-flavored Markdown with alignment (:---, :---:, ---:)
Lists: Preserve numbering and nesting. Use indentation for nested lists.
Formatting: **bold** *italic* `code` and [links](url)

🔄 READING ORDER:

Single-column: Top-to-bottom
Multi-column: Left-to-right, then top-to-bottom within each column.
Sidebars: Extract sidebars as separate blocks after the main section they appear next to, or at the end of the page if unrelated.

📊 TABLES & FIGURES:

- Extract captions if present (usually above/below)
- Preserve table header rows and column alignment
- For figures: ![Brief description](image)

❌ FORBIDDEN PHRASES (indicates hallucination):

- "Based on the image..."
- "As shown in the document..."
- "It appears that..."
- "I can see that..."
- "[Image of...]"

If you catch yourself using these, STOP and extract only visible text.

✅ QUALITY CHECKLIST (self-validate):

1. Did I extract ALL visible text?
2. Did I preserve the correct reading order?
3. Did I tag headers/footers with <!-- role:header/footer --> instead of removing them?
4. Did I avoid commentary?

🎯 BEGIN EXTRACTION:

Extract the document content following all rules above. Start with <!-- page:1 --> and proceed systematically."""
        
        # Prepare messages for OpenRouter
        content = [
            {
                "type": "text",
                "text": extraction_prompt
            }
        ]
        
        # Add all images
        for img_base64 in images_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
        
        # Call OpenRouter API with retry logic and timeout
        max_retries = 5  # More retries for intermittent connection issues
        retry_delay = 3  # Initial delay in seconds
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=16000,  # Generous limit for long documents
                    timeout=120.0  # 120-second timeout for slow/unstable connections
                )
                break  # Success - exit retry loop
                
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                
                # Check if we should retry
                if attempt < max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"OpenRouter API attempt {attempt + 1} failed ({error_type}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed - raise the exception
                    print(f"OpenRouter API failed after {max_retries} attempts. Last error: {error_type}")
                    raise Exception(f"OpenRouter API failed after {max_retries} attempts: {str(e)}") from e
        
        # Extract markdown text
        markdown_text = response.choices[0].message.content
        
        # Calculate cost estimate
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
        estimated_cost = (tokens_used / 1_000_000) * model_config["cost_per_1m"]
        
        # Extract document metadata
        if input_path.lower().endswith('.pdf'):
            doc_metadata = metadata_extractor.extract_pdf_metadata(input_path)
        else:
            doc_metadata = metadata_extractor.extract_image_metadata(input_path)
        
        # Normalize Markdown output (Phase 3: Post-Processing)
        print("Normalizing markdown output...")
        markdown_text = cleaner.normalize_markdown(markdown_text)
        
        # Validate OpenRouter output for quality and hallucinations
        expected_word_count = doc_metadata.get("text_layer_word_count", 0) if doc_metadata else 0
        
        validation_report = openrouter_validator.validate_openrouter_output(
            markdown_text,
            page_count=page_count,
            original_method=f"OpenRouter/{model_config['name']}",
            expected_word_count=expected_word_count
        )
        
        # Add YAML frontmatter with metadata and quality score
        extraction_method = f"OpenRouter/{model_config['name']}"
        markdown_with_frontmatter = metadata_extractor.add_yaml_frontmatter(
            markdown_text,
            doc_metadata,
            extraction_method=extraction_method,
            confidence_score=validation_report['quality_score'],  # Use validation quality score
            language="en"  # TODO: Auto-detect language
        )
        
        # Metadata for return (including validation results)
        metadata = {
            "model_used": model_config["name"],
            "model_id": model_id,
            "pages_processed": page_count,
            "tokens_used": tokens_used,
            "estimated_cost": estimated_cost,
            "cost_per_1m_tokens": model_config["cost_per_1m"],
            "quality_score": validation_report['quality_score'],
            "hallucination_count": validation_report['hallucination_count'],
            "semantic_annotations": {
                "page_markers": validation_report['semantic_annotations']['page_count'],
                "role_annotations": validation_report['semantic_annotations']['role_count']
            },
            "validation_issues": len(validation_report['issues'])
        }
        
        return markdown_with_frontmatter, metadata
        
    except Exception as e:
        error_msg = f"OpenRouter API Error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, {"error": str(e)}

def extract_smart_local(input_path, dpi=300, lang="en", enhanced_mode=True, progress_callback=None):
    """
    Smart local extraction with automatic digital/scan detection.
    - Digital PDFs: Use pymupdf4llm (fast, LLM-optimized)
    - Scanned PDFs: Use RapidOCR (layout-aware OCR)
    
    Returns:
        tuple: (markdown_text, metadata_dict)
    """
    metadata = {
        "extraction_method": "Smart Local (Auto)",
        "dpi": dpi,
        "language": lang,
        "quality_score": 0.0,
        "pages_processed": 0
    }
    
    if progress_callback:
        progress_callback(0.1, "🧠 Smart Mode: Analyzing document...")
    
    # 1. Try Digital Extraction (if PDF)
    if str(input_path).lower().endswith(".pdf"):
        try:
            if progress_callback:
                progress_callback(0.3, "🔍 Detecting document type...")
                
            print(f"Smart Local: Attempting digital extraction for {input_path}...")
            # Call pymupdf4llm extraction
            md_text, digital_meta = extract_with_pymupdf4llm(input_path)
            
            # Heuristic: Check if we got meaningful text length
            if md_text and len(md_text.strip()) > 100:
                print("Smart Local: Digital extraction successful.")
                if progress_callback:
                    progress_callback(1.0, "✅ Digital extraction complete")
                
                # Merge metadata
                metadata.update(digital_meta)
                return md_text, metadata
            else:
                print("Smart Local: Digital extraction returned insufficient text. Falling back to OCR.")
                if progress_callback:
                    progress_callback(0.4, "📄 Scanned document detected, running OCR...")
        except Exception as e:
            print(f"Smart Local: Digital extraction failed: {e}")
            if progress_callback:
                progress_callback(0.4, "⚠️ Digital extraction failed, falling back to OCR...")
            
    # 2. Fallback to RapidOCR
    print(f"Smart Local: Running RapidOCR...")
    if progress_callback:
        progress_callback(0.5, "📷 Starting RapidOCR scan...")
        
    return extract_with_rapidocr(input_path, dpi, lang, enhanced_mode)

