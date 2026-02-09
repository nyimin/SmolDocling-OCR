
import gradio as gr
import tempfile
import os
import traceback
import time
import zipfile
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pymupdf4llm
import fitz  # PyMuPDF for metadata and preview
import structure_engine

# Import service modules
from services.cache_manager import (
    get_file_hash, get_cached_result, save_to_cache, 
    clear_cache, get_cache_size
)
from services.history_manager import (
    load_history, save_history, get_history_display, clear_history
)
from utils.formatters import (
    count_stats, estimate_quality_score, estimate_cost,
    markdown_to_html, markdown_to_txt, markdown_to_docx,
    load_stats, save_stats, update_stats, get_stats_display
)

# Constants
MIN_TEXT_THRESHOLD = 50
IMAGES_DIR = "extracted_images"

# Create directories
os.makedirs(IMAGES_DIR, exist_ok=True)

# All cache, history, and stats functions now imported from modules

def get_pdf_metadata(file_path):
    try:
        doc = fitz.open(file_path)
        metadata = doc.metadata
        page_count = len(doc)
        doc.close()
        
        info = []
        if metadata.get("title"):
            info.append(f"**Title:** {metadata['title']}")
        if metadata.get("author"):
            info.append(f"**Author:** {metadata['author']}")
        if metadata.get("subject"):
            info.append(f"**Subject:** {metadata['subject']}")
        info.append(f"**Pages:** {page_count}")
        if metadata.get("creationDate"):
            date_str = metadata["creationDate"]
            if date_str.startswith("D:"):
                date_str = date_str[2:10]
                try:
                    formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    info.append(f"**Created:** {formatted}")
                except (IndexError, ValueError) as e:
                    # Date string format is invalid - skip this field
                    print(f"Warning: Could not parse creation date '{date_str}': {e}")
        
        return "\n".join(info) if info else "No metadata available"
    except Exception as e:
        return f"Could not read metadata: {e}"

def get_pdf_preview(file_path, page_num=0):
    """Generate preview image of PDF page."""
    try:
        doc = fitz.open(file_path)
        if page_num >= len(doc):
            page_num = 0
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        doc.close()
        
        # Save to temp file
        temp_path = os.path.join(tempfile.gettempdir(), f"preview_{page_num}.png")
        with open(temp_path, "wb") as f:
            f.write(img_data)
        return temp_path
    except Exception as e:
        print(f"Preview error: {e}")
        return None

def extract_images_from_pdf(file_path):
    """Extract all images from PDF."""
    images = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image
                img_filename = f"page{page_num+1}_img{img_index+1}.{base_image['ext']}"
                img_path = os.path.join(IMAGES_DIR, img_filename)
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                images.append(img_path)
        
        doc.close()
    except Exception as e:
        print(f"Image extraction error: {e}")
    
    return images

def markdown_to_html(markdown_text):
    html = markdown2.markdown(markdown_text, extras=["tables", "fenced-code-blocks"])
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        :root {{
            --bg-color: #ffffff;
            --text-color: #111827;
            --border-color: #e5e7eb;
            --code-bg: #f3f4f6;
            --header-bg: #f9fafb;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #0d1117;
                --text-color: #c9d1d9;
                --border-color: #30363d;
                --code-bg: #161b22;
                --header-bg: #161b22;
            }}
        }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif; 
            max-width: 800px; 
            margin: 40px auto; 
            padding: 20px; 
            line-height: 1.6;
            background-color: var(--bg-color);
            color: var(--text-color);
        }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid var(--border-color); padding: 12px; text-align: left; }}
        th {{ background-color: var(--header-bg); font-weight: 600; }}
        code {{ background: var(--code-bg); padding: 2px 6px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; }}
        pre {{ background: var(--code-bg); padding: 15px; border-radius: 6px; overflow-x: auto; border: 1px solid var(--border-color); }}
        img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        blockquote {{ border-left: 4px solid var(--border-color); margin: 0; padding-left: 16px; color: var(--text-color); opacity: 0.8; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""

def markdown_to_docx(markdown_text, output_path):
    doc = Document()
    for line in markdown_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('# '):
            doc.add_heading(line[2:], level=1)
        elif line.startswith('## '):
            doc.add_heading(line[3:], level=2)
        elif line.startswith('### '):
            doc.add_heading(line[4:], level=3)
        elif line.startswith('- ') or line.startswith('* '):
            doc.add_paragraph(line[2:], style='List Bullet')
        else:
            doc.add_paragraph(line)
    doc.save(output_path)

def markdown_to_txt(markdown_text):
    import re
    text = re.sub(r'#{1,6}\s', '', markdown_text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    return text

def get_openrouter_cost(model_name):
    """Get cost per 1M tokens for OpenRouter model."""
    model_costs = {
        "Nemotron Nano 12B VL (FREE) ⭐": 0.0,
        "Gemini 2.0 Flash Lite ($0.08/1K pages)": 0.000075,
        "Qwen 2.5-VL 32B ($0.05/1K pages)": 0.00005,
        "Qwen 2.5-VL 72B ($0.15/1K pages)": 0.00015,
        "Mistral Pixtral Large ($2/1K pages)": 0.002
    }
    return model_costs.get(model_name, 0.0)

def estimate_cost(model_name, pages=1):
    """Estimate cost for processing given number of pages."""
    cost_per_1m = get_openrouter_cost(model_name)
    if cost_per_1m == 0.0:
        return "**Estimated Cost:** FREE"
    
    # Estimate ~1000 tokens per page (conservative)
    tokens_estimate = pages * 1000
    cost_estimate = (tokens_estimate / 1_000_000) * cost_per_1m
    
    if cost_estimate < 0.01:
        return f"**Estimated Cost:** ~${cost_estimate:.4f} ({pages} page{'s' if pages > 1 else ''})"
    else:
        return f"**Estimated Cost:** ~${cost_estimate:.2f} ({pages} page{'s' if pages > 1 else ''})"

def toggle_settings(engine_choice):
    """Toggle visibility of OpenRouter vs Local OCR settings."""
    is_cloud = "Cloud OCR" in engine_choice
    return gr.update(visible=is_cloud), gr.update(visible=not is_cloud)

def clear_all():
    """Clear all inputs and reset UI to initial state."""
    return (
        None,  # file_input
        "",    # output_md_view
        "",    # output_raw_text
        None,  # download_btn
        "**Status:** Upload a file to get started",  # status_box
        "**Stats:** N/A",  # stats_display
        "**Quality:** N/A",  # quality_display
        "**Metadata:** N/A",  # metadata_display
        []     # image_gallery
    )


def process_single_file(file_path, dpi, ocr_lang, page_start, page_end, use_cache, ocr_engine="RapidOCR", openrouter_model="free", openrouter_api_key=None, progress_callback=None):
    # Check cache
    if use_cache and file_path.lower().endswith(".pdf"):
        file_hash = get_file_hash(file_path)
        cache_key_raw = f"{ocr_engine}_{openrouter_model if 'OpenRouter' in ocr_engine else dpi}"
        import re
        cache_key = re.sub(r'[\\/*?:"<>|]', '_', cache_key_raw)
        cached = get_cached_result(file_hash, cache_key, "md")
        if cached:
            return cached["markdown"], cached["method"] + " (cached)"
    
    markdown_text = None
    method_used = None
    
    # Route based on OCR engine selection
    if "OpenRouter" in ocr_engine:
        # Use OpenRouter for OCR
        
        # Map UI model names to tier keys
        model_map = {
            "Nemotron Nano 12B VL (FREE) ⭐": "free",
            "Gemini 2.0 Flash Lite ($0.08/1K pages)": "cheap",
            "Qwen 2.5-VL 32B ($0.05/1K pages)": "balanced",
            "Qwen 2.5-VL 72B ($0.15/1K pages)": "quality",
            "Mistral Pixtral Large ($2/1K pages)": "premium"
        }
        model_tier = model_map.get(openrouter_model, "free")
        
        # DEBUG: Check API Key
        if openrouter_api_key:
            print(f"DEBUG: calling OpenRouter with key: {openrouter_api_key[:8]}...")
        else:
            print("DEBUG: calling OpenRouter with NO API KEY (should fallback to env var)")

        try:
            markdown_text, metadata = structure_engine.extract_with_openrouter(
                file_path, model=model_tier, api_key=openrouter_api_key
            )
            method_used = f"OpenRouter ({metadata.get('model_used', 'Unknown')})"
            
            # Check for errors in returned text
            if markdown_text and "Error" in markdown_text:
                raise Exception("OpenRouter returned error text")
                
        except Exception as e:
            # Fallback to RapidOCR if OpenRouter fails (timeout, connection error, etc.)
            print(f"OpenRouter extraction failed: {type(e).__name__} - {e}")
            markdown_text, _ = structure_engine.extract_with_rapidocr(file_path, dpi=dpi, lang=ocr_lang)
            method_used = "RapidOCR (Fallback)"
    else:
        # SMART LOCAL MODE (Digital -> RapidOCR)
        # Use new smart local function
        markdown_text, metadata = structure_engine.extract_smart_local(
            file_path, dpi=dpi, lang=ocr_lang, progress_callback=progress_callback
        )
        method_used = metadata.get("extraction_method", "Smart Local")
        
        # If extraction failed completely
        if not markdown_text:
             method_used = "Failed"
    
    # Cache result
    if use_cache and markdown_text and file_path.lower().endswith(".pdf"):
        file_hash = get_file_hash(file_path)
        cache_key = f"{ocr_engine}_{openrouter_model if 'OpenRouter' in ocr_engine else dpi}"
        save_to_cache(file_hash, cache_key, "md", markdown_text, method_used)
    
    return markdown_text, method_used


def process_upload(files, export_format, dpi, ocr_lang, page_start, page_end, use_cache, ocr_engine, openrouter_model, openrouter_api_key, progress=gr.Progress()):
    if not files:
        return None, None, None, "Upload a PDF or image to get started.", "", gr.update(visible=False), "", [], ""
    
    if not isinstance(files, list):
        files = [files]
    
    start_time = time.time()
    results = []
    metadata_info = ""
    preview_image = None
    extracted_images = []
    quality_info = ""
    
    try:
        # Get metadata and preview for first PDF
        first_file = files[0].name if hasattr(files[0], 'name') else files[0]
        if first_file.lower().endswith(".pdf"):
            metadata_info = get_pdf_metadata(first_file)
            preview_image = get_pdf_preview(first_file, 0)
            extracted_images = extract_images_from_pdf(first_file)
        
        for idx, file in enumerate(files):
            input_path = file.name if hasattr(file, 'name') else file
            
            markdown_text, method_used = process_single_file(
                input_path, dpi, ocr_lang, page_start, page_end, use_cache,
                ocr_engine, openrouter_model, openrouter_api_key,
                progress_callback=progress  # Pass progress directly, no nested lambda
            )
            
            if markdown_text:
                results.append({
                    'filename': Path(input_path).stem,
                    'markdown': markdown_text,
                    'method': method_used
                })
        
        if not results:
            return None, None, None, "❌ Failed to extract content.", "", gr.update(visible=False), metadata_info, extracted_images, quality_info
        
        elapsed_time = time.time() - start_time
        total_words = sum(count_stats(r['markdown'])[0] for r in results)
        total_chars = sum(count_stats(r['markdown'])[1] for r in results)
        
        # Update stats
        update_stats(total_words, elapsed_time)
        
        # Quality score
        quality_score = estimate_quality_score(results[0]['markdown'], results[0]['method'])
        quality_info = f"🎯 **Quality Score:** {quality_score}/100"
        
        if len(results) == 1:
            markdown_text = results[0]['markdown']
            method_used = results[0]['method']
            filename = results[0]['filename']
            
            temp_dir = tempfile.mkdtemp()
            
            if export_format == "Markdown (.md)":
                output_path = os.path.join(temp_dir, f"{filename}.md")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(markdown_text)
            elif export_format == "HTML (.html)":
                output_path = os.path.join(temp_dir, f"{filename}.html")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(markdown_to_html(markdown_text))
            elif export_format == "Plain Text (.txt)":
                output_path = os.path.join(temp_dir, f"{filename}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(markdown_to_txt(markdown_text))
            elif export_format == "Word Document (.docx)":
                output_path = os.path.join(temp_dir, f"{filename}.docx")
                markdown_to_docx(markdown_text, output_path)
            
            stats_text = f"📊 **{total_words:,}** words • **{total_chars:,}** characters"
            status = f"✅ Converted using {method_used} in **{elapsed_time:.1f}s**"
            
            save_history({
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'words': total_words,
                'method': method_used
            })
            
            return markdown_text, markdown_text, output_path, status, stats_text, gr.update(visible=True), metadata_info, extracted_images, quality_info
        
        else:
            # Batch processing
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "converted_documents.zip")
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for result in results:
                    filename = result['filename']
                    md = result['markdown']
                    
                    if export_format == "Markdown (.md)":
                        file_path = os.path.join(temp_dir, f"{filename}.md")
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(md)
                    elif export_format == "HTML (.html)":
                        file_path = os.path.join(temp_dir, f"{filename}.html")
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(markdown_to_html(md))
                    elif export_format == "Plain Text (.txt)":
                        file_path = os.path.join(temp_dir, f"{filename}.txt")
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(markdown_to_txt(md))
                    elif export_format == "Word Document (.docx)":
                        file_path = os.path.join(temp_dir, f"{filename}.docx")
                        markdown_to_docx(md, file_path)
                    
                    zipf.write(file_path, os.path.basename(file_path))
            
            first_md = results[0]['markdown']
            stats_text = f"📊 **{len(results)} files** • **{total_words:,}** words"
            status = f"✅ Batch converted {len(results)} files in **{elapsed_time:.1f}s**"
            
            return first_md, first_md, zip_path, status, stats_text, gr.update(visible=True), metadata_info, extracted_images, quality_info
        
    except Exception as e:
        traceback.print_exc()
        return None, None, None, f"Error: {str(e)}", "", gr.update(visible=False), metadata_info, extracted_images, quality_info

def get_history_display():
    history = load_history()
    if not history:
        return "No recent conversions"
    
    lines = ["### Recent Conversions\n"]
    for entry in history[:5]:
        timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%m/%d %H:%M")
        lines.append(f"- **{entry['filename']}** ({entry['words']:,} words) - {timestamp}")
    
    return "\n".join(lines)

def get_stats_display():
    stats = load_stats()
    return f"""### 📈 Usage Stats
- **Total Conversions:** {stats['total_conversions']}
- **Total Words:** {stats['total_words']:,}
- **Avg Time:** {stats['avg_time']:.1f}s"""

def clear_cache():
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
    return "✅ Cache cleared!"

# CSS - Minimalist Design System
custom_css = """
/* ========================================
   DESIGN SYSTEM: PRIMITIVES
   ======================================== */

:root {
  /* Neutral Scale (Slate-ish Gray for modern feel) */
  --neutral-0:    #ffffff;
  --neutral-50:   #f8fafc;
  --neutral-100:  #f1f5f9;
  --neutral-200:  #e2e8f0;
  --neutral-300:  #cbd5e1;
  --neutral-400:  #94a3b8;
  --neutral-500:  #64748b;
  --neutral-600:  #475569;
  --neutral-700:  #334155;
  --neutral-800:  #1e293b;
  --neutral-900:  #0f172a;
  --neutral-950:  #020617;

  /* Brand Scale (Indigo - deeply professional) */
  --brand-50:     #eef2ff;
  --brand-100:    #e0e7ff;
  --brand-200:    #c7d2fe;
  --brand-300:    #a5b4fc;
  --brand-400:    #818cf8;
  --brand-500:    #6366f1;
  --brand-600:    #4f46e5;
  --brand-700:    #4338ca;
  --brand-800:    #3730a3;
  --brand-900:    #312e81;
  --brand-950:    #1e1b4b;

  /* Status Colors (Semantic) */
  --error-light:  #ef4444;
  --error-dark:   #f87171;
  --success-light:#10b981;
  --success-dark: #34d399;

  /* ========================================
     SEMANTIC TOKENS (Light Mode Default)
     ======================================== */
  
  /* Backgrounds */
  --bg-app:          var(--neutral-50);
  --bg-panel:        var(--neutral-0);
  --bg-element:      var(--neutral-0);
  --bg-element-alt:  var(--neutral-100);
  
  /* Text */
  --text-primary:    var(--neutral-900);
  --text-secondary:  var(--neutral-500);
  --text-tertiary:   var(--neutral-400);
  --text-on-accent:  var(--neutral-0);

  /* Borders */
  --border-subtle:   var(--neutral-200);
  --border-strong:   var(--neutral-300);
  --border-focus:    var(--brand-500);

  /* Interaction (Monochrome Primary) */
  --action-primary:        var(--neutral-950); /* Ultra-dark/Black */
  --action-primary-hover:  var(--neutral-800);
  --action-secondary:      var(--neutral-0);
  --action-secondary-hover:var(--neutral-50);

  /* Shadows (Flat Mode = None, but variables kept for logic) */
  --shadow-sm: none;
  --shadow-md: none;
}

/* ========================================
   DARK MODE OVERRIDES
   ======================================== */
.dark {
  /* Backgrounds */
  --bg-app:          var(--neutral-950);
  --bg-panel:        var(--neutral-900);
  --bg-element:      var(--neutral-900);
  --bg-element-alt:  var(--neutral-800);

  /* Text */
  --text-primary:    var(--neutral-50);
  --text-secondary:  var(--neutral-400);
  --text-tertiary:   var(--neutral-500);

  /* Borders */
  --border-subtle:   var(--neutral-800);
  --border-strong:   var(--neutral-700);
  
  /* Interaction (Monochrome Inverse) */
  --action-primary:        var(--neutral-50); /* White */
  --action-primary-hover:  var(--neutral-200);
  --action-secondary:      var(--neutral-800);
  --action-secondary-hover:var(--neutral-700);
  
  /* Status Adjustment */
  --error-light: var(--error-dark);
  --success-light: var(--success-dark);
}

/* ========================================
   GLOBAL RESET
   ======================================== */
* {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  transition: background-color 0.2s ease, color 0.1s ease, border-color 0.2s ease;
}

body, .gradio-container {
  background-color: var(--bg-app) !important;
  color: var(--text-primary) !important;
}

/* ========================================
   COMPONENT OVERRIDES
   ======================================== */

/* 1. Header & Typography */
h1, h2, h3, h4, h5, h6 {
  color: var(--text-primary) !important;
}

#header {
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 2rem;
  padding-bottom: 1rem;
}

#header h1 {
  font-weight: 700;
  letter-spacing: -0.025em;
}

/* 2. Panels & Cards (Inputs, Groups) */
.gradio-group, .gradio-box, .status-card, .upload-section, .quick-settings, .input-card, .output-markdown, .gradio-textbox textarea {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}

/* 3. Inputs (Text, Number, Dropdown) */
input[type="text"], input[type="password"], input[type="number"], textarea, select {
  background-color: var(--bg-element) !important;
  border: 1px solid var(--border-subtle) !important;
  color: var(--text-primary) !important;
  border-radius: 6px !important;
}

input:focus, textarea:focus, select:focus {
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 1px var(--border-focus) !important;
}

/* 4. Buttons */
button {
  box-shadow: none !important;
  font-weight: 500 !important;
  border-radius: 6px !important;
  text-transform: none !important;
}

/* Primary Button */
button.primary, .primary-btn {
  background: var(--action-primary) !important;
  color: var(--bg-app) !important; /* Invert text relative to button bg */
  border: 1px solid transparent !important;
}

button.primary:hover, .primary-btn:hover {
  background: var(--action-primary-hover) !important;
}

/* Secondary Button */
button.secondary, .secondary-btn {
  background: var(--bg-element) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-subtle) !important;
}

button.secondary:hover {
  background: var(--bg-element-alt) !important;
  border-color: var(--border-strong) !important;
}

/* 5. Tabs */
.gradio-tabs {
  border-bottom: 1px solid var(--border-subtle) !important;
}

.gradio-tab-nav {
  border: none !important;
  background: transparent !important;
}

.gradio-tab-nav button {
  color: var(--text-secondary) !important;
  border: none !important;
  background: transparent !important;
}

.gradio-tab-nav button.selected {
  color: var(--action-primary) !important;
  border-bottom: 2px solid var(--action-primary) !important;
  font-weight: 600 !important;
}

/* 6. Status & Stats Areas */
#status-card {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: none !important;
  padding: 1.5rem !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  text-align: center !important;
  gap: 1rem !important;
}

/* 7. Markdown/HTML Content */
.output-markdown code {
  background-color: var(--bg-element-alt) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
}

.output-markdown pre {
  background-color: var(--bg-element-alt) !important;
  border: 1px solid var(--border-subtle);
}

.output-markdown table {
  border-color: var(--border-subtle) !important;
}

.output-markdown th {
  background-color: var(--bg-element-alt) !important;
  color: var(--text-primary) !important;
  border-color: var(--border-subtle) !important;
}

.output-markdown td {
  border-color: var(--border-subtle) !important;
  color: var(--text-secondary) !important;
}

/* 8. Labels */
span.svelte-1gfkn6j, label, .block-title {
  color: var(--text-secondary) !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  margin-bottom: 0.5rem;
}

/* 9. Icons/SVGs */
svg {
  fill: currentColor;
}
"""


copy_js = """
function copyToClipboard() {
    const textArea = document.querySelector('textarea');
    if (textArea && textArea.value) {
        navigator.clipboard.writeText(textArea.value).then(() => {
            const btn = document.querySelector('.copy-btn');
            if (btn) {
                const originalText = btn.textContent;
                btn.textContent = '✓ Copied!';
                setTimeout(() => { btn.textContent = originalText; }, 2000);
            }
        });
    }
    return [];
}
"""

# Gradio Interface
with gr.Blocks(title="DocFlow") as demo:

    with gr.Row(elem_classes="container"):
        
        with gr.Column(scale=5):
            
            # Header
            with gr.Row():
                with gr.Column(elem_id="header"):
                    gr.Markdown("# DocFlow")
                    gr.Markdown("RAG-Optimized PDF & Image to Markdown Conversion")
            
            # Upload Section (TOP PRIORITY)
            gr.Markdown("## Upload Documents")
            with gr.Row(elem_classes="upload-section"):
                file_input = gr.File(
                    label="Upload PDF or Image Files",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    type="filepath", 
                    file_count="multiple", 
                    height=150
                )
            
            with gr.Row():
                process_btn = gr.Button("Convert to Markdown", variant="primary", size="lg", elem_classes="primary-btn")
                clear_btn = gr.Button("Clear", variant="secondary", size="lg")
            
            # Settings Section
            gr.Markdown("## Settings")
            with gr.Group(elem_classes="quick-settings"):
                with gr.Row():
                    ocr_engine = gr.Dropdown(
                        choices=["Cloud OCR (OpenRouter)", "Local OCR (Auto-Detect Digital/Scan)"],
                        value="Cloud OCR (OpenRouter)",
                        label="OCR Engine",
                        info="Cloud: 100+ languages. Local: Auto-detects digital PDFs vs scanned docs.",
                        scale=2
                    )
                    export_format = gr.Dropdown(
                        choices=["Markdown (.md)", "HTML (.html)", "Plain Text (.txt)", "Word Document (.docx)"],
                        value="Markdown (.md)", 
                        label="Export Format",
                        scale=1
                    )
                
                # Advanced Settings
                with gr.Accordion("Advanced Options", open=False):
                    
                    # OpenRouter Settings (conditional)
                    with gr.Group(visible=True) as openrouter_settings:
                        gr.Markdown("### OpenRouter Settings")
                        with gr.Row():
                            openrouter_model = gr.Dropdown(
                                choices=[
                                    "Nemotron Nano 12B VL (FREE)",
                                    "Gemini 2.0 Flash Lite ($0.08/1K pages)",
                                    "Qwen 2.5-VL 32B ($0.05/1K pages)",
                                    "Qwen 2.5-VL 72B ($0.15/1K pages)",
                                    "Mistral Pixtral Large ($2/1K pages)"
                                ],
                                value="Gemini 2.0 Flash Lite ($0.08/1K pages)",
                                label="Model",
                                info="FREE model recommended for most use cases"
                            )
                        with gr.Row():
                            openrouter_api_key = gr.Textbox(
                                label="OpenRouter API Key",
                                type="password",
                                placeholder="sk-or-v1-...",
                                info="Get free key at https://openrouter.ai",
                                value=os.getenv("OPENROUTER_API_KEY", "")
                            )
                        cost_estimate = gr.Markdown("**Estimated Cost:** FREE", elem_classes="cost-display")
                    
                    # RapidOCR Settings (conditional)
                    with gr.Group(visible=False) as rapidocr_settings:
                        gr.Markdown("### Local OCR Settings")
                        gr.Markdown("*Digital PDFs use pymupdf4llm automatically. These settings apply to scanned documents.*")
                        with gr.Row():
                            dpi_slider = gr.Slider(minimum=150, maximum=600, value=300, step=50, label="DPI")
                            ocr_lang = gr.Dropdown(
                                choices=["en", "ch_sim", "ch_tra", "ja", "ko", "ru"],
                                value="en",
                                label="OCR Language",
                                info="For scanned documents (6 languages supported)"
                            )
                    
                    # Common Advanced Options
                    gr.Markdown("### Processing Options")
                    with gr.Row():
                        use_cache = gr.Checkbox(value=True, label="Use Cache")
                        page_start = gr.Number(label="Start Page", precision=0, minimum=0, value=1)
                        page_end = gr.Number(label="End Page", precision=0, minimum=0)
                    
                    # Theme Toggle
                    gr.Markdown("### Appearance")
                    with gr.Row():
                        theme_btn = gr.Button("Toggle Dark/Light Mode", elem_classes="theme-btn")

            
            # Status Card (PERSISTENT - ALWAYS VISIBLE)
            gr.Markdown("## Status")
            with gr.Column(elem_id="status-card"):
                with gr.Row():
                    stats_display = gr.Markdown("**Stats:** N/A")
                    quality_display = gr.Markdown("**Quality:** N/A")
                
                with gr.Row():
                    metadata_display = gr.Markdown("**Metadata:** N/A")
            
            # Results Section
            gr.Markdown("## Results")
            
            # Status display in Results section
            status_box = gr.Markdown("**Status:** Upload a file to get started")
            
            with gr.Tabs():
                with gr.TabItem("Preview"):
                    output_md_view = gr.Markdown(elem_classes="output-markdown")
                with gr.TabItem("Raw Code"):
                    output_raw_text = gr.TextArea(label="Markdown Source", lines=18)
                    copy_btn = gr.Button("Copy to Clipboard", elem_classes="copy-btn", size="sm")
                with gr.TabItem("Extracted Images"):
                    image_gallery = gr.Gallery(label="Images from PDF", columns=3, height=400, value=[])

            download_btn = gr.File(label="Download", interactive=False)

    # Events
    # Settings toggle
    ocr_engine.change(
        fn=toggle_settings,
        inputs=[ocr_engine],
        outputs=[openrouter_settings, rapidocr_settings]
    )
    
    # Dynamic cost estimation
    openrouter_model.change(
        fn=estimate_cost,
        inputs=[openrouter_model],
        outputs=[cost_estimate]
    )
    
    process_btn.click(
        fn=process_upload,
        inputs=[file_input, export_format, dpi_slider, ocr_lang, page_start, page_end, use_cache, 
                ocr_engine, openrouter_model, openrouter_api_key],
        outputs=[output_md_view, output_raw_text, download_btn, status_box, stats_display, stats_display, 
                 metadata_display, image_gallery, quality_display]
    )
    
    # Clear button
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[file_input, output_md_view, output_raw_text, download_btn, status_box, 
                 stats_display, quality_display, metadata_display, image_gallery]
    )
    
    copy_btn.click(fn=None, js="copyToClipboard")
    theme_btn.click(fn=None, js="() => { document.body.classList.toggle('dark'); return []; }")

if __name__ == "__main__":
    print("=" * 70)
    print("DocFlow - Production Ready PDF to Markdown Converter")
    print("=" * 70)
    print("Web UI: http://localhost:7860")
    print("API Server: Run 'python api.py' for REST API on port 8000")
    print("=" * 70)
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate", spacing_size="sm", radius_size="sm"),
        css=custom_css,
        js=copy_js
    )
