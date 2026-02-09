"""
Semantic Annotator Module for DocFlow

Provides centralized semantic annotation engine for RAG-optimized Markdown output.
Ensures consistent semantic role annotations across all extraction engines.
"""

from typing import Dict, Any, List, Optional, Tuple
import re


class PageContext:
    """Context information for semantic analysis."""
    
    def __init__(self, elements: List[Dict[str, Any]], page_num: int, page_height: float = 800):
        """
        Initialize page context.
        
        Args:
            elements: All elements on the page
            page_num: Page number (1-indexed)
            page_height: Page height in pixels
        """
        self.elements = elements
        self.page_num = page_num
        self.page_height = page_height
        self.avg_font_size = self._calculate_avg_font_size()
        self.avg_text_length = self._calculate_avg_text_length()
    
    def _calculate_avg_font_size(self) -> float:
        """Calculate average font size for heading detection."""
        sizes = [e.get('font_size', 12) for e in self.elements 
                if e.get('type') == 'text' and e.get('font_size')]
        return sum(sizes) / len(sizes) if sizes else 12.0
    
    def _calculate_avg_text_length(self) -> float:
        """Calculate average text length for semantic analysis."""
        lengths = [len(e.get('content', '')) for e in self.elements 
                  if e.get('type') == 'text']
        return sum(lengths) / len(lengths) if lengths else 50.0


class SemanticAnnotator:
    """
    Centralized semantic annotation engine.
    
    Provides consistent semantic role annotations for all document elements
    following the RAG-optimized Markdown schema v2.0.
    """
    
    # Semantic role classification patterns
    HEADING_PATTERNS = [
        r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS
        r'^\d+\.\s+[A-Z]',  # "1. Introduction"
        r'^\d+\.\d+\s+[A-Z]',  # "1.1 Subsection"
        r'^Chapter\s+\d+',  # "Chapter 1"
        r'^Section\s+\d+',  # "Section 1"
        r'^Part\s+[IVX]+',  # "Part I"
        r'^Abstract$',
        r'^Introduction$',
        r'^Conclusion$',
        r'^References$',
        r'^Bibliography$',
        r'^Acknowledgements?$',
        r'^Appendix\s*[A-Z]?$',
    ]
    
    LIST_PATTERNS = [
        (r'^\s*[\-\*\+]\s+', 'unordered'),  # Unordered: "- item"
        (r'^\s*\d+\.\s+', 'ordered'),  # Ordered: "1. item"
        (r'^\s*\d+\)\s+', 'ordered'),  # Ordered: "1) item"
        (r'^\s*[a-z]\)\s+', 'ordered'),  # Lettered: "a) item"
        (r'^\s*[ivx]+\.\s+', 'ordered'),  # Roman: "i. item"
        (r'^\s*•\s+', 'unordered'),  # Bullet
        (r'^\s*○\s+', 'unordered'),  # Circle bullet
        (r'^\s*■\s+', 'unordered'),  # Square bullet
    ]
    
    CAPTION_PATTERNS = [
        r'^[Tt][Aa][Bb][Ll][Ee]\s*\d+',
        r'^[Tt][Bb][Ll]\.?\s*\d+',
        r'^[Ff][Ii][Gg][Uu][Rr][Ee]\s*\d+',
        r'^[Ff][Ii][Gg]\.?\s*\d+',
        r'^[Cc][Hh][Aa][Rr][Tt]\s*\d+',
        r'^[Gg][Rr][Aa][Pp][Hh]\s*\d+',
        r'^[Dd][Ii][Aa][Gg][Rr][Aa][Mm]\s*\d+',
        r'^[Ii][Mm][Aa][Gg][Ee]\s*\d+',
    ]
    
    FOOTNOTE_PATTERNS = [
        r'^\[\d+\]\s*',  # "[1] Footnote text"
        r'^\(\d+\)\s*',  # "(1) Footnote text"
        r'^\d+\.\s*(?=[A-Z])',  # "1. Footnote text" (starts with capital)
        r'^[*†‡§]\s*',  # Symbol footnotes
    ]
    
    EQUATION_PATTERNS = [
        r'^\$\$.*\$\$$',  # $$...$$
        r'^\\\[.*\\\]$',  # \[...\]
        r'^\\begin\{equation\}',  # LaTeX equation
        r'=\s*[a-zA-Z0-9\+\-\*\/\^\(\)]+\s*$',  # Simple equation
    ]
    
    def __init__(self):
        """Initialize semantic annotator with compiled patterns."""
        self.compiled_heading_patterns = [re.compile(p) for p in self.HEADING_PATTERNS]
        self.compiled_list_patterns = [(re.compile(p), t) for p, t in self.LIST_PATTERNS]
        self.compiled_caption_patterns = [re.compile(p) for p in self.CAPTION_PATTERNS]
        self.compiled_footnote_patterns = [re.compile(p) for p in self.FOOTNOTE_PATTERNS]
        self.compiled_equation_patterns = [re.compile(p) for p in self.EQUATION_PATTERNS]
    
    def annotate_element(self, element: Dict[str, Any], 
                        context: PageContext) -> str:
        """
        Generate standardized semantic annotation for an element.
        
        Args:
            element: Text/table/figure element with keys:
                - type: 'text' | 'table' | 'figure'
                - content: Element content
                - confidence: OCR confidence (0.0-1.0)
                - bbox: Bounding box (x0, y0, x1, y1)
                - font_size: Font size (for text)
                - reading_order: Reading order index
            context: Page-level context
        
        Returns:
            Markdown with semantic annotations
        """
        elem_type = element.get('type', 'text')
        
        if elem_type == 'table':
            return self._annotate_table(element, context)
        elif elem_type == 'figure':
            return self._annotate_figure(element, context)
        else:
            return self._annotate_text(element, context)
    
    def _annotate_text(self, element: Dict[str, Any], 
                      context: PageContext) -> str:
        """Annotate text element with semantic role."""
        content = element.get('content', '').strip()
        confidence = element.get('confidence', 1.0)
        try:
            confidence = float(confidence) if isinstance(confidence, str) else confidence
        except (ValueError, TypeError):
            confidence = 1.0
        reading_order = element.get('reading_order')
        
        if not content:
            return ""
        
        # Classify semantic role
        noise_type = element.get('noise_type')
        if noise_type:
            role = noise_type
            attributes = {}
        else:
            role, attributes = self._classify_text_role(element, context)
        
        # Build annotation
        output = ""
        
        # Add reading order annotation for multi-column
        if reading_order is not None:
            output += f"<!-- reading-order:{reading_order} -->\n"
        
        # Add role annotation
        output += f"<!-- role:{role}"
        if attributes:
            output += " " + " ".join(f"{k}:{v}" for k, v in attributes.items())
        output += " -->\n"
        
        # Add confidence marker if low
        if confidence < 0.7:
            output += f"<!-- confidence:{confidence:.2f} -->\n"
        
        # Format content based on role
        if role == 'heading':
            level = attributes.get('level', 2)
            formatted_content = f"{'#' * level} {content}"
        elif role == 'equation':
            display = attributes.get('display', 'block')
            if display == 'block':
                formatted_content = f"$$\n{content}\n$$"
            else:
                formatted_content = f"${content}$"
        elif confidence < 0.5:
            formatted_content = f"[low-confidence: {content}]"
        elif confidence < 0.7:
            formatted_content = f"[uncertain: {content}]"
        else:
            formatted_content = content
        
        return output + formatted_content + "\n"
    
    def _classify_text_role(self, element: Dict[str, Any], 
                           context: PageContext) -> Tuple[str, Dict[str, Any]]:
        """
        Classify semantic role of text element.
        
        Returns:
            (role, attributes) tuple
        """
        content = element.get('content', '').strip()
        font_size = element.get('font_size')
        bbox = element.get('bbox')
        
        # Check for heading by font size
        if font_size and font_size > context.avg_font_size * 1.3:
            # Calculate level based on size ratio
            ratio = font_size / context.avg_font_size
            if ratio > 1.8:
                level = 1
            elif ratio > 1.5:
                level = 2
            elif ratio > 1.3:
                level = 3
            else:
                level = 4
            return ('heading', {'level': level})
        
        # Check for heading by pattern
        for pattern in self.compiled_heading_patterns:
            if pattern.match(content):
                # Determine level based on pattern
                if 'chapter' in pattern.pattern.lower():
                    level = 1
                elif re.match(r'^\d+\.\d+', content):
                    level = 3
                elif re.match(r'^\d+\.', content):
                    level = 2
                else:
                    level = 2
                return ('heading', {'level': level})
        
        # Check for short lines that might be headings
        if len(content) < 60 and content and content[0].isupper():
            # Short line starting with capital, no ending punctuation
            if not content[-1] in '.!?,;:':
                # Could be a heading, use position heuristic
                if bbox and context.page_height:
                    y_ratio = bbox[1] / context.page_height
                    if y_ratio < 0.15:  # Top 15% of page
                        return ('heading', {'level': 2})
        
        # Check for list
        for pattern, list_type in self.compiled_list_patterns:
            if pattern.match(content):
                return ('list_item', {'type': list_type})
        
        # Check for caption
        for pattern in self.compiled_caption_patterns:
            if pattern.match(content):
                caption_type = 'table' if 'table' in content.lower() or 'tbl' in content.lower() else 'figure'
                return ('caption', {'for': caption_type})
        
        # Check for footnote
        for pattern in self.compiled_footnote_patterns:
            match = pattern.match(content)
            if match:
                # Extract footnote ID
                id_match = re.search(r'\d+', content[:10])
                footnote_id = id_match.group() if id_match else '1'
                return ('footnote', {'id': footnote_id})
        
        # Check for equation
        for pattern in self.compiled_equation_patterns:
            if pattern.match(content):
                display = 'block' if len(content) > 20 else 'inline'
                return ('equation', {'display': display})
        
        # Check position for footnote (bottom of page)
        if bbox and context.page_height:
            y_ratio = bbox[1] / context.page_height
            if y_ratio > 0.85 and len(content) < 200:
                # Bottom 15% of page, short text - might be footnote
                if re.match(r'^\d', content):
                    id_match = re.match(r'^(\d+)', content)
                    return ('footnote', {'id': id_match.group(1) if id_match else '1'})
        
        # Default to paragraph
        return ('paragraph', {})
    
    def _annotate_table(self, element: Dict[str, Any], 
                       context: PageContext) -> str:
        """Annotate table element."""
        content = element.get('content', '')
        caption = element.get('caption', '')
        reading_order = element.get('reading_order')
        
        output = ""
        
        # Add reading order if present
        if reading_order is not None:
            output += f"<!-- reading-order:{reading_order} -->\n"
        
        # Add table annotation
        output += "<!-- role:table"
        if caption:
            # Escape quotes in caption
            caption_escaped = caption.replace('"', '\\"')
            output += f' caption:"{caption_escaped}"'
        output += " -->\n"
        
        # Add caption as separate element if present
        if caption:
            output += f"\n**{caption}**\n\n"
        
        output += content + "\n"
        
        return output
    
    def _annotate_figure(self, element: Dict[str, Any], 
                        context: PageContext) -> str:
        """Annotate figure element."""
        caption = element.get('caption', '')
        alt_text = element.get('alt', element.get('description', 'Figure'))
        reading_order = element.get('reading_order')
        
        output = ""
        
        # Add reading order if present
        if reading_order is not None:
            output += f"<!-- reading-order:{reading_order} -->\n"
        
        # Add figure annotation
        output += "<!-- role:figure"
        if caption:
            caption_escaped = caption.replace('"', '\\"')
            output += f' caption:"{caption_escaped}"'
        output += " -->\n"
        
        # Markdown image syntax
        output += f"![{alt_text}](image)\n"
        
        # Add caption if present
        if caption:
            output += f"\n*{caption}*\n"
        
        return output
    
    def annotate_page(self, elements: List[Dict[str, Any]], 
                     page_num: int, page_height: float = 800) -> str:
        """
        Annotate all elements on a page.
        
        Args:
            elements: List of page elements
            page_num: Page number (1-indexed)
            page_height: Page height in pixels
        
        Returns:
            Annotated Markdown for the page
        """
        context = PageContext(elements, page_num, page_height)
        
        output = f"\n<!-- page:{page_num} -->\n\n"
        
        # Sort elements by reading order if available, otherwise by Y position
        sorted_elements = sorted(
            elements, 
            key=lambda e: (e.get('reading_order', 0), e.get('y', 0))
        )
        
        for elem in sorted_elements:
            annotated = self.annotate_element(elem, context)
            if annotated:
                output += annotated + "\n"
        
        return output
    
    def detect_document_structure(self, all_pages_elements: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze document structure across all pages.
        
        Args:
            all_pages_elements: List of element lists, one per page
        
        Returns:
            Document structure analysis:
            - has_toc: Whether document has table of contents
            - heading_hierarchy: Detected heading structure
            - section_count: Number of major sections
            - has_footnotes: Whether document has footnotes
            - has_equations: Whether document has equations
        """
        structure = {
            'has_toc': False,
            'heading_hierarchy': [],
            'section_count': 0,
            'has_footnotes': False,
            'has_equations': False,
            'has_tables': False,
            'has_figures': False,
        }
        
        headings = []
        
        for page_num, elements in enumerate(all_pages_elements, 1):
            context = PageContext(elements, page_num)
            
            for elem in elements:
                if elem.get('type') == 'table':
                    structure['has_tables'] = True
                elif elem.get('type') == 'figure':
                    structure['has_figures'] = True
                elif elem.get('type') == 'text':
                    role, attrs = self._classify_text_role(elem, context)
                    
                    if role == 'heading':
                        headings.append({
                            'text': elem.get('content', '')[:50],
                            'level': attrs.get('level', 2),
                            'page': page_num
                        })
                    elif role == 'footnote':
                        structure['has_footnotes'] = True
                    elif role == 'equation':
                        structure['has_equations'] = True
        
        # Analyze heading structure
        structure['heading_hierarchy'] = headings
        structure['section_count'] = sum(1 for h in headings if h['level'] <= 2)
        
        # Detect TOC (multiple headings on first 1-2 pages followed by page numbers)
        first_pages_headings = [h for h in headings if h['page'] <= 2]
        if len(first_pages_headings) > 5:
            structure['has_toc'] = True
        
        return structure


# Convenience function
def annotate_document(all_pages_elements: List[List[Dict[str, Any]]]) -> str:
    """
    Annotate entire document with semantic roles.
    
    Args:
        all_pages_elements: List of element lists, one per page
    
    Returns:
        Complete annotated Markdown
    """
    annotator = SemanticAnnotator()
    output = ""
    
    for page_num, elements in enumerate(all_pages_elements, 1):
        output += annotator.annotate_page(elements, page_num)
    
    return output


# Example usage
if __name__ == "__main__":
    # Sample elements
    sample_elements = [
        {'type': 'text', 'content': 'Introduction', 'font_size': 18, 'y': 100, 'confidence': 0.95},
        {'type': 'text', 'content': 'This is the first paragraph of the document.', 'font_size': 12, 'y': 150, 'confidence': 0.92},
        {'type': 'text', 'content': '• First bullet point', 'font_size': 12, 'y': 200, 'confidence': 0.88},
        {'type': 'text', 'content': '• Second bullet point', 'font_size': 12, 'y': 230, 'confidence': 0.90},
        {'type': 'table', 'content': '| A | B |\n|---|---|\n| 1 | 2 |', 'y': 300, 'caption': 'Table 1: Sample Data'},
    ]
    
    annotator = SemanticAnnotator()
    result = annotator.annotate_page(sample_elements, page_num=1)
    print(result)
