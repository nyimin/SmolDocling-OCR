import fitz # PyMuPDF
import os
import cleaner # New cleaner module
from collections import Counter

def get_most_common_size(blocks):
    """Estimate body text size to detect headings."""
    sizes = []
    for b in blocks:
        if "lines" not in b: continue
        for l in b["lines"]:
            for s in l["spans"]:
                sizes.append(round(s["size"], 1))
    if not sizes: return 11
    return Counter(sizes).most_common(1)[0][0]

def convert_fast(file_path):
    """
    Converts a file to Markdown using PyMuPDF (fitz) with custom layout logic.
    - Uses 'dict' mode for text (Avoiding AssertionError).
    - Uses 'find_tables' for tables.
    - Merges content based on vertical position (y-axis).
    """
    try:
        doc = fitz.open(file_path)
        full_text = ""
        
        for page in doc:
            # 1. Extract Tables
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]
            
            # 2. Extract Text Blocks
            text_blocks = page.get_text("dict")["blocks"]
            body_size = get_most_common_size(text_blocks)
            
            elements = []
            
            # Process Tables
            for i, tab in enumerate(tables):
                elements.append({
                    "type": "table",
                    "bbox": tab.bbox,
                    "content": tab.to_markdown()
                })
            
            # Process Text Blocks (with Collision Detection)
            for b in text_blocks:
                if "lines" not in b: continue
                
                # Check intersection with any table
                is_table_content = False
                b_rect = fitz.Rect(b["bbox"])
                for t_rect in table_bboxes:
                    # If intersection area is > 60% of block area, it's inside the table
                    if b_rect.intersect(t_rect).get_area() > 0.6 * b_rect.get_area():
                        is_table_content = True
                        break
                
                if is_table_content:
                    continue
                
                # ... existing text formatting logic ...
                block_content = ""
                is_header = False
                
                try:
                    first_span = b["lines"][0]["spans"][0]
                    if first_span["size"] > body_size * 1.05 and first_span["size"] < body_size * 2.5:
                        is_header = True
                        if first_span["size"] > body_size * 1.5:
                            block_content += "# "
                        else:
                            block_content += "## "
                except (IndexError, KeyError):
                    # Block has no spans or lines - skip header detection
                    pass

                for l in b["lines"]:
                    line_text = ""
                    for s in l["spans"]:
                        line_text += s["text"]
                    
                    clean_line = line_text.strip()
                    if clean_line.startswith(("•", "-", "*")) and not is_header:
                        block_content += "- " + clean_line.lstrip("•-* ") + "\n"
                    else:
                        block_content += line_text + " "
                
                block_content = block_content.strip()
                if block_content:
                    elements.append({
                        "type": "text",
                        "bbox": b["bbox"],
                        "content": block_content
                    })

            # 3. Sort by vertical position (y0)
            elements.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
            
            # 4. Render Page
            page_text = ""
            for el in elements:
                page_text += el["content"] + "\n\n"
            
            full_text += page_text + "\n--- Page Break ---\n\n"
            
        doc.close()
        
        # Apply cleaner
        full_text = cleaner.merge_hyphenated_words(full_text)
        
        return full_text
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Fallback to pure text if everything explodes
        return f"Error in Fast Mode (Tables): {str(e)}"
