import re
import statistics
from collections import Counter

def merge_hyphenated_words(text):
    """
    Fixes words broken by hyphens at line ends.
    Example: "This is a bro- \n ken sentence." -> "This is a broken sentence."
    """
    # Regex for word ending with hyphen, optional whitespace/newline, starting next word
    # distinct from dashes used as punctuation. Strict: second part must be lowercase.
    pattern = r'([a-zA-Z]+)-\s*\n\s*([a-z]+)'
    return re.sub(pattern, r'\1\2', text)

def detect_and_tag_headers_footers(pages_elements, threshold=0.6, use_patterns=True, use_position=True):
    """
    Identifies and tags headers/footers using pattern matching and position analysis.
    
    Industrial-standard approach:
    - Pattern matching for variable content (Page N, Chapter N, dates)
    - Position-based filtering (top/bottom 10% of page)
    - Repetition analysis (appears in 60%+ of pages)
    - Tags elements instead of removing (Tag-Don't-Remove strategy)
    
    Args:
        pages_elements: List of Lists. Each inner list contains elements for a page.
                        Element: {'y': float, 'type': 'text'|'table', 'content': str, 'bbox': tuple}
        threshold: Fraction of pages a text must appear in to be considered header/footer (0.0-1.0)
        use_patterns: Enable regex pattern matching for common header/footer formats
        use_position: Enable position-based filtering (top/bottom regions)
    
    Returns:
        Pages with elements tagged with 'semantic_role': 'header' or 'footer'
    """
    if len(pages_elements) < 3:
        return pages_elements
    
    # Common header/footer patterns (compiled for performance)
    patterns = []
    if use_patterns:
        patterns = [
            # Page numbers
            (re.compile(r'^Page\s+\d+', re.IGNORECASE), 'page_number'),
            (re.compile(r'^\d+\s+of\s+\d+$', re.IGNORECASE), 'page_number'),
            (re.compile(r'^-\s*\d+\s*-$'), 'page_number'),
            (re.compile(r'^\[\s*\d+\s*\]$'), 'page_number'),
            (re.compile(r'^\d+$'), 'page_number'),  # Standalone number
            
            # Chapters and sections
            (re.compile(r'^Chapter\s+\d+', re.IGNORECASE), 'chapter'),
            (re.compile(r'^Section\s+\d+\.?\d*', re.IGNORECASE), 'section'),
            (re.compile(r'^Part\s+\d+', re.IGNORECASE), 'part'),
            
            # Dates
            (re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}'), 'date'),
            (re.compile(r'\w+\s+\d{1,2},\s+\d{4}'), 'date'),
            (re.compile(r'\d{4}-\d{2}-\d{2}'), 'date'),
            
            # Common footer text
            (re.compile(r'Copyright\s*©', re.IGNORECASE), 'copyright'),
            (re.compile(r'Confidential', re.IGNORECASE), 'confidential'),
            (re.compile(r'Internal\s+Use\s+Only', re.IGNORECASE), 'internal'),
            (re.compile(r'Draft', re.IGNORECASE), 'draft'),
            (re.compile(r'All\s+Rights\s+Reserved', re.IGNORECASE), 'copyright'),
        ]
    
    # Calculate page height for position-based filtering
    page_heights = []
    if use_position:
        for page in pages_elements:
            if page:
                max_y = max([elem.get('bbox', (0, 0, 0, 0))[3] if 'bbox' in elem else elem.get('y', 0) 
                           for elem in page], default=1000)
                page_heights.append(max_y)
            else:
                page_heights.append(1000)  # Default height
    
    # Collect candidates with normalized text and position info
    candidates = Counter()
    position_info = {}  # Track where each text appears (top/bottom)
    total_pages = len(pages_elements)
    
    for page_idx, page in enumerate(pages_elements):
        page_height = page_heights[page_idx] if use_position else 1000
        
        for elem in page:
            if elem.get('type') != 'text':
                continue
            
            clean_text = elem.get('content', '').strip()
            if len(clean_text) < 3:  # Ignore tiny artifacts
                continue
            
            # Normalize for matching (lowercase, collapse whitespace)
            normalized = ' '.join(clean_text.lower().split())
            
            # Check if matches any pattern
            pattern_match = False
            if use_patterns:
                for pattern, pattern_type in patterns:
                    if pattern.search(clean_text):
                        pattern_match = True
                        break
            
            # Position analysis
            elem_y = elem.get('bbox', (0, 0, 0, 0))[1] if 'bbox' in elem else elem.get('y', 0)
            in_header_zone = elem_y < page_height * 0.1  # Top 10%
            in_footer_zone = elem_y > page_height * 0.9  # Bottom 10%
            
            # Count occurrences
            if pattern_match or in_header_zone or in_footer_zone:
                candidates[normalized] += 1
                
                # Track position
                if normalized not in position_info:
                    position_info[normalized] = {'header': 0, 'footer': 0, 'middle': 0}
                
                if in_header_zone:
                    position_info[normalized]['header'] += 1
                elif in_footer_zone:
                    position_info[normalized]['footer'] += 1
                else:
                    position_info[normalized]['middle'] += 1
    
    # Identify repeaters (appear in threshold% of pages)
    repeaters = {}
    for text, count in candidates.items():
        if count / total_pages >= threshold:
            # Determine if header or footer based on position
            pos = position_info.get(text, {})
            if pos.get('header', 0) > pos.get('footer', 0):
                repeaters[text] = 'header'
            else:
                repeaters[text] = 'footer'
    
    # Tag elements (don't remove)
    tagged_pages = []
    for page in pages_elements:
        tagged_page = []
        for elem in page:
            elem_copy = elem.copy()  # Don't modify original
            
            if elem.get('type') == 'text':
                clean_text = elem.get('content', '').strip()
                normalized = ' '.join(clean_text.lower().split())
                
                if normalized in repeaters:
                    # Tag as header or footer
                    elem_copy['semantic_role'] = repeaters[normalized]
            
            tagged_page.append(elem_copy)
        tagged_pages.append(tagged_page)
    
    return tagged_pages


# Legacy function for backward compatibility
def detect_and_remove_headers_footers(pages_elements, threshold=0.6):
    """
    DEPRECATED: Use detect_and_tag_headers_footers() instead.
    
    Legacy function that removes headers/footers (old behavior).
    Kept for backward compatibility.
    """
    tagged_pages = detect_and_tag_headers_footers(pages_elements, threshold)
    
    # Remove tagged elements
    cleaned_pages = []
    for page in tagged_pages:
        cleaned_page = []
        for elem in page:
            if elem.get('semantic_role') not in ('header', 'footer'):
                cleaned_page.append(elem)
        cleaned_pages.append(cleaned_page)
    
    return cleaned_pages


def defragment_text(text):
    """
    Merges lines that appear to be part of the same paragraph.
    Simple heuristic: If a line doesn't end with [.!?], merge with next.
    """
    lines = text.split('\n')
    merged = []
    current_line = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_line:
                merged.append(current_line)
                current_line = ""
            merged.append("") # Preserve empty lines as paragraph breaks
            continue
            
        if not current_line:
            current_line = line
        else:
            # Check if previous line ended with sentence punctuation
            if current_line.endswith(('.', '?', '!', ':')):
                merged.append(current_line)
                current_line = line
            else:
                # Merge
                current_line += " " + line
    
    if current_line:
        merged.append(current_line)
        
    return '\n'.join(merged)


def normalize_markdown(text: str) -> str:
    """
    Normalize Markdown output to ensure consistent styling and remove artifacts.
    
    Operations:
    1. Normalize Unicode dash characters to standard hyphens
    2. Remove excessive indentation from list items
    3. Standardize line breaks (max 2 consecutive newlines)
    4. Convert asteroid bullets (*) to hyphens (-)
    5. Ensure blank lines before headers
    6. Remove empty semantic tags
    
    Args:
        text (str): Raw markdown text
        
    Returns:
        str: Normalized markdown text
    """
    if not text:
        return ""

    # 1. Normalize Unicode dash characters to standard hyphen
    # This prevents minus signs (−) from being misinterpreted
    dash_replacements = {
        '\u2212': '-',  # MINUS SIGN (commonly used in PDFs)
        '\u2013': '-',  # EN DASH
        '\u2014': '-',  # EM DASH
        '\u2015': '-',  # HORIZONTAL BAR
    }
    for unicode_dash, standard_dash in dash_replacements.items():
        text = text.replace(unicode_dash, standard_dash)
    
    # 2. Remove excessive indentation from list items
    # Only remove indents from top-level lists (at start or after blank lines)
    # This prevents lists from being rendered as code blocks (4+ spaces = code)
    # while preserving intentional nested list indentation
    # Pattern: Match list items at start of text or after blank line with 2-4 space indent
    text = re.sub(r'(^|\n\n)[ ]{2,4}([-*+])\s', r'\1\2 ', text, flags=re.MULTILINE)
    
    # 3. Standardize line breaks (max 2 newlines)
    # Replace 3 or more newlines with 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 4. Standardize list bullets (convert * start to -)
    # Only matches * at start of line or after whitespace
    text = re.sub(r'^(\s*)\* ', r'\1- ', text, flags=re.MULTILINE)
    
    # 5. Ensure blank lines before headers
    # If a header (#) is preceded by a non-newline char and a single newline, add another newline
    text = re.sub(r'([^\n])\n(#{1,6} )', r'\1\n\n\2', text)
    
    # 6. Remove empty semantic tags (e.g., <!-- role:artifact -->\n<!-- /role -->)
    # This regex looks for tags with only whitespace content
    text = re.sub(r'<!-- role:\w+ -->\s*<!-- /role -->', '', text)
    
    return text.strip()
