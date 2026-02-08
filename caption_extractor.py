"""
Caption Extractor Module for DocFlow

Extracts and links captions to tables and figures.
"""

from typing import Dict, List, Any, Optional, Tuple
import re


class CaptionExtractor:
    """
    Extract and link captions to tables and figures.
    
    Detects:
    - Table captions (above or below tables)
    - Figure captions (below figures)
    - Numbered references (Table 1, Figure 2, etc.)
    """
    
    # Caption patterns - using character classes for case insensitivity
    TABLE_CAPTION_PATTERNS = [
        r'^[Tt][Aa][Bb][Ll][Ee]\s*(\d+)[\s:\.]*(.*)$',
        r'^[Tt][Bb][Ll]\.?\s*(\d+)[\s:\.]*(.*)$',
    ]
    
    FIGURE_CAPTION_PATTERNS = [
        r'^[Ff][Ii][Gg][Uu][Rr][Ee]\s*(\d+)[\s:\.]*(.*)$',
        r'^[Ff][Ii][Gg]\.?\s*(\d+)[\s:\.]*(.*)$',
        r'^[Ii][Mm][Aa][Gg][Ee]\s*(\d+)[\s:\.]*(.*)$',
        r'^[Cc][Hh][Aa][Rr][Tt]\s*(\d+)[\s:\.]*(.*)$',
        r'^[Dd][Ii][Aa][Gg][Rr][Aa][Mm]\s*(\d+)[\s:\.]*(.*)$',
        r'^[Gg][Rr][Aa][Pp][Hh]\s*(\d+)[\s:\.]*(.*)$',
    ]
    
    # Proximity threshold (pixels)
    PROXIMITY_THRESHOLD = 100
    
    def __init__(self):
        self.compiled_table_patterns = [re.compile(p) for p in self.TABLE_CAPTION_PATTERNS]
        self.compiled_figure_patterns = [re.compile(p) for p in self.FIGURE_CAPTION_PATTERNS]
        self.extracted_captions: List[Dict[str, Any]] = []
    
    def extract(self, pages_elements: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """
        Extract captions and link to elements.
        
        Args:
            pages_elements: Document pages
        
        Returns:
            Updated pages with caption annotations
        """
        self.extracted_captions = []
        result = []
        
        for page_num, elements in enumerate(pages_elements, 1):
            result.append(self._process_page(elements, page_num))
        
        return result
    
    def _process_page(self, elements: List[Dict[str, Any]], page_num: int) -> List[Dict[str, Any]]:
        """Process single page for caption extraction."""
        # Find tables and figures
        tables = [(i, e) for i, e in enumerate(elements) if e.get('type') == 'table']
        figures = [(i, e) for i, e in enumerate(elements) if e.get('type') == 'figure']
        
        # Find potential captions
        captions = []
        caption_indices = set()
        
        for i, elem in enumerate(elements):
            if elem.get('type') != 'text':
                continue
            
            content = elem.get('content', '').strip()
            caption_info = self._parse_caption(content)
            
            if caption_info:
                captions.append((i, elem, caption_info))
                caption_indices.add(i)
        
        # Link captions to tables/figures
        for idx, elem, cap_info in captions:
            target = None
            
            if cap_info['type'] == 'table':
                target = self._find_nearest_element(elem, tables, elements)
            else:
                target = self._find_nearest_element(elem, figures, elements)
            
            if target:
                target_idx, target_elem = target
                # Add caption to target
                target_elem['caption'] = cap_info['text']
                target_elem['caption_number'] = cap_info['number']
                
                self.extracted_captions.append({
                    'page': page_num,
                    'type': cap_info['type'],
                    'number': cap_info['number'],
                    'text': cap_info['text']
                })
        
        # Remove standalone caption elements (they're now linked)
        return [e for i, e in enumerate(elements) if i not in caption_indices or e.get('type') != 'text']
    
    def _parse_caption(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse text to extract caption info."""
        # Check table captions
        for pattern in self.compiled_table_patterns:
            match = pattern.match(text)
            if match:
                return {
                    'type': 'table',
                    'number': int(match.group(1)),
                    'text': match.group(2).strip() if match.group(2) else ''
                }
        
        # Check figure captions
        for pattern in self.compiled_figure_patterns:
            match = pattern.match(text)
            if match:
                return {
                    'type': 'figure',
                    'number': int(match.group(1)),
                    'text': match.group(2).strip() if match.group(2) else ''
                }
        
        return None
    
    def _find_nearest_element(self, caption_elem: Dict, 
                             candidates: List[Tuple[int, Dict]], 
                             all_elements: List[Dict]) -> Optional[Tuple[int, Dict]]:
        """Find nearest table/figure to caption."""
        if not candidates:
            return None
        
        caption_y = self._get_y_position(caption_elem)
        
        best = None
        best_dist = float('inf')
        
        for idx, target in candidates:
            target_y = self._get_y_position(target)
            dist = abs(target_y - caption_y)
            
            if dist < best_dist and dist < self.PROXIMITY_THRESHOLD:
                best_dist = dist
                best = (idx, target)
        
        return best
    
    def _get_y_position(self, elem: Dict) -> float:
        """Get Y position of element."""
        if 'y' in elem:
            return elem['y']
        if 'bbox' in elem and elem['bbox']:
            return elem['bbox'][1]
        return 0
    
    def get_caption_report(self) -> Dict[str, Any]:
        """Get extraction report."""
        return {
            'total_extracted': len(self.extracted_captions),
            'tables': len([c for c in self.extracted_captions if c['type'] == 'table']),
            'figures': len([c for c in self.extracted_captions if c['type'] == 'figure']),
            'captions': self.extracted_captions
        }


class FootnoteLinker:
    """
    Link footnote references to their content.
    
    Detects:
    - Superscript footnote references (¹, ², ³)
    - Bracketed references ([1], [2])
    - Footnote content at page bottom
    """
    
    REFERENCE_PATTERNS = [
        (r'\[(\d+)\]', 'bracket'),
        (r'\((\d+)\)', 'paren'),
        (r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+', 'superscript'),
    ]
    
    FOOTNOTE_CONTENT_PATTERNS = [
        r'^\[(\d+)\]\s*(.+)$',
        r'^\((\d+)\)\s*(.+)$',
        r'^(\d+)\.\s+(.+)$',
        r'^[¹²³⁴⁵⁶⁷⁸⁹⁰]\s*(.+)$',
    ]
    
    def __init__(self, page_bottom_threshold: float = 0.8):
        """
        Initialize footnote linker.
        
        Args:
            page_bottom_threshold: Y position ratio for footnote area
        """
        self.page_bottom_threshold = page_bottom_threshold
        self.footnotes: List[Dict[str, Any]] = []
    
    def link(self, pages_elements: List[List[Dict[str, Any]]], 
            page_heights: Optional[List[float]] = None) -> List[List[Dict[str, Any]]]:
        """
        Link footnote references to content.
        
        Args:
            pages_elements: Document pages
            page_heights: Optional page heights for position calculation
        
        Returns:
            Updated pages with footnote annotations
        """
        self.footnotes = []
        result = []
        
        for page_num, elements in enumerate(pages_elements, 1):
            page_height = page_heights[page_num - 1] if page_heights else 800
            result.append(self._process_page(elements, page_num, page_height))
        
        return result
    
    def _process_page(self, elements: List[Dict[str, Any]], 
                     page_num: int, page_height: float) -> List[Dict[str, Any]]:
        """Process page for footnote linking."""
        # Identify footnote content (bottom of page)
        footnote_content = {}
        footnote_indices = set()
        
        for i, elem in enumerate(elements):
            if elem.get('type') != 'text':
                continue
            
            y_pos = self._get_y_position(elem)
            if y_pos / page_height < self.page_bottom_threshold:
                continue
            
            # Check if it's footnote content
            content = elem.get('content', '').strip()
            for pattern in self.FOOTNOTE_CONTENT_PATTERNS:
                match = re.match(pattern, content)
                if match:
                    fn_id = match.group(1) if match.lastindex >= 1 else '1'
                    fn_text = match.group(2) if match.lastindex >= 2 else content
                    
                    footnote_content[fn_id] = fn_text
                    footnote_indices.add(i)
                    
                    self.footnotes.append({
                        'page': page_num,
                        'id': fn_id,
                        'content': fn_text
                    })
                    break
        
        # Mark footnote elements with role annotation
        for i, elem in enumerate(elements):
            if i in footnote_indices:
                elem['semantic_role'] = 'footnote'
                # Extract ID if present
                content = elem.get('content', '').strip()
                match = re.match(r'^\[?(\d+)\]?', content)
                if match:
                    elem['footnote_id'] = match.group(1)
        
        return elements
    
    def _get_y_position(self, elem: Dict) -> float:
        """Get Y position."""
        if 'y' in elem:
            return elem['y']
        if 'bbox' in elem and elem['bbox']:
            return elem['bbox'][1]
        return 0
    
    def get_footnote_report(self) -> Dict[str, Any]:
        """Get footnote extraction report."""
        return {
            'total_footnotes': len(self.footnotes),
            'by_page': {},
            'footnotes': self.footnotes
        }


def extract_captions_and_footnotes(pages_elements: List[List[Dict[str, Any]]],
                                   page_heights: Optional[List[float]] = None) -> Tuple[List[List[Dict]], Dict]:
    """
    Extract captions and link footnotes.
    
    Args:
        pages_elements: Document pages
        page_heights: Optional page heights
    
    Returns:
        (processed_pages, combined_report)
    """
    caption_extractor = CaptionExtractor()
    footnote_linker = FootnoteLinker()
    
    # Extract captions first
    pages_with_captions = caption_extractor.extract(pages_elements)
    
    # Then link footnotes
    final_pages = footnote_linker.link(pages_with_captions, page_heights)
    
    report = {
        'captions': caption_extractor.get_caption_report(),
        'footnotes': footnote_linker.get_footnote_report()
    }
    
    return final_pages, report


# Example usage
if __name__ == "__main__":
    sample_pages = [
        [
            {'type': 'text', 'content': 'Table 1: Sample Data', 'y': 90},
            {'type': 'table', 'content': '| A | B |\n|---|---|', 'y': 100},
            {'type': 'text', 'content': 'This is body text with a footnote [1].', 'y': 200},
            {'type': 'text', 'content': '[1] This is the footnote content.', 'y': 700},
        ],
    ]
    
    result, report = extract_captions_and_footnotes(sample_pages, [800])
    
    print("Caption Report:", report['captions'])
    print("Footnote Report:", report['footnotes'])
