"""
Adaptive Noise Filter Module for DocFlow

Provides comprehensive noise reduction for OCR output including:
- Header/footer detection with regex-based number masking
- Watermark text detection
- Page number pattern filtering
- OCR artifact detection and removal
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import re
from collections import Counter


class AdaptiveNoiseFilter:
    """
    Adaptive noise filter for document cleanup.
    
    Detects and removes:
    - Repetitive headers and footers
    - Watermarks and confidentiality notices
    - Page numbers in various formats
    - OCR artifacts (garbled text, stray characters)
    """
    
    # Page number patterns
    PAGE_NUMBER_PATTERNS = [
        r'^Page\s*\d+\s*(?:of\s*\d+)?$',
        r'^\d+\s*/\s*\d+$',
        r'^-\s*\d+\s*-$',
        r'^\[\s*\d+\s*\]$',
        r'^\d+$',
        r'^(?:p|pg|page)\.?\s*\d+$',
    ]
    
    # Watermark patterns
    WATERMARK_PATTERNS = [
        r'(?i)^confidential\s*-?\s*(?:internal|external)?$',
        r'(?i)^draft\s*(?:copy)?$',
        r'(?i)^do\s+not\s+(?:copy|distribute)$',
        r'(?i)^for\s+internal\s+use\s+only$',
        r'(?i)^proprietary\s+(?:and\s+)?confidential$',
        r'(?i)^copy(?:right)?\s*©?\s*\d{4}',
        r'(?i)^all\s+rights\s+reserved\.?$',
        r'(?i)^www\.[a-zA-Z0-9]+\.[a-z]{2,}$',
        r'(?i)^sample\s*$',
        r'(?i)^preview\s*$',
        r'(?i)^watermark\s*$',
    ]
    
    # OCR artifact patterns
    ARTIFACT_PATTERNS = [
        r'^[|_\-=+]{3,}$',  # Lines of special chars
        r'^[\s\.\,\!\?\;\:]+$',  # Just punctuation
        r'^[^\w\s]{2,}$',  # Multiple special characters
        r'^\s+$',  # Whitespace only
        r'^[a-zA-Z]{1,2}$',  # Single/double letter artifacts
        r'^[0-9]{1,2}$',  # Single/double digit artifacts (handled by page numbers)
    ]
    
    def __init__(self, 
                 header_footer_threshold: float = 0.5,
                 min_pages_for_detection: int = 3):
        """
        Initialize noise filter.
        
        Args:
            header_footer_threshold: Min occurrence ratio for header/footer detection
            min_pages_for_detection: Min pages needed for pattern detection
        """
        self.header_footer_threshold = header_footer_threshold
        self.min_pages_for_detection = min_pages_for_detection
        
        self.compiled_page_patterns = [re.compile(p, re.IGNORECASE) for p in self.PAGE_NUMBER_PATTERNS]
        self.compiled_watermark_patterns = [re.compile(p) for p in self.WATERMARK_PATTERNS]
        self.compiled_artifact_patterns = [re.compile(p) for p in self.ARTIFACT_PATTERNS]
        
        self.detected_noise: Dict[str, List[str]] = {
            'headers': [],
            'footers': [],
            'watermarks': [],
            'page_numbers': [],
            'artifacts': []
        }
    
    def filter(self, pages_elements: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """
        Apply all noise filters to document.
        
        Args:
            pages_elements: List of element lists, one per page
        
        Returns:
            Cleaned pages_elements
        """
        self.detected_noise = {k: [] for k in self.detected_noise}
        
        if len(pages_elements) < self.min_pages_for_detection:
            # Too few pages, only apply single-page filters
            return [self._filter_single_page(page) for page in pages_elements]
        
        # 1. Detect cross-page patterns
        repeaters = self._detect_repeating_content(pages_elements)
        
        # 2. Filter each page
        cleaned_pages = []
        for page in pages_elements:
            cleaned_page = []
            for elem in page:
                if elem.get('type') == 'text':
                    content = elem.get('content', '').strip()
                    
                    # Check all filters
                    if self._is_header_footer(content, repeaters):
                        continue
                    if self._is_page_number(content):
                        self.detected_noise['page_numbers'].append(content)
                        continue
                    if self._is_watermark(content):
                        self.detected_noise['watermarks'].append(content)
                        continue
                    if self._is_artifact(content, elem):
                        self.detected_noise['artifacts'].append(content)
                        continue
                
                cleaned_page.append(elem)
            
            cleaned_pages.append(cleaned_page)
        
        return cleaned_pages
    
    def _detect_repeating_content(self, pages: List[List[Dict[str, Any]]]) -> Dict[str, Set[str]]:
        """Detect content that repeats across pages."""
        total_pages = len(pages)
        
        # Collect candidates from top/bottom of each page
        top_candidates = Counter()
        bottom_candidates = Counter()
        
        for page in pages:
            text_elements = [e for e in page if e.get('type') == 'text']
            if not text_elements:
                continue
            
            # Sort by Y position
            sorted_elems = sorted(text_elements, key=lambda e: e.get('y', e.get('bbox', [0, 0])[1] if e.get('bbox') else 0))
            
            # Top 3 elements (potential headers)
            for elem in sorted_elems[:3]:
                content = self._normalize_for_matching(elem.get('content', ''))
                if content and len(content) < 100:
                    top_candidates[content] += 1
            
            # Bottom 3 elements (potential footers)
            for elem in sorted_elems[-3:]:
                content = self._normalize_for_matching(elem.get('content', ''))
                if content and len(content) < 100:
                    bottom_candidates[content] += 1
        
        # Find repeaters
        threshold = total_pages * self.header_footer_threshold
        
        headers = {text for text, count in top_candidates.items() if count >= threshold}
        footers = {text for text, count in bottom_candidates.items() if count >= threshold}
        
        self.detected_noise['headers'] = list(headers)
        self.detected_noise['footers'] = list(footers)
        
        return {'headers': headers, 'footers': footers}
    
    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for pattern matching (replace numbers with placeholders)."""
        # Replace page numbers with placeholder
        normalized = re.sub(r'\b\d+\b', '[NUM]', text.strip())
        return normalized
    
    def _is_header_footer(self, content: str, repeaters: Dict[str, Set[str]]) -> bool:
        """Check if content matches detected header/footer patterns."""
        normalized = self._normalize_for_matching(content)
        return normalized in repeaters['headers'] or normalized in repeaters['footers']
    
    def _is_page_number(self, content: str) -> bool:
        """Check if content is a page number."""
        content = content.strip()
        for pattern in self.compiled_page_patterns:
            if pattern.match(content):
                return True
        return False
    
    def _is_watermark(self, content: str) -> bool:
        """Check if content is a watermark."""
        content = content.strip()
        for pattern in self.compiled_watermark_patterns:
            if pattern.match(content):
                return True
        return False
    
    def _is_artifact(self, content: str, element: Dict) -> bool:
        """Check if content is an OCR artifact."""
        content = content.strip()
        
        # Check patterns
        for pattern in self.compiled_artifact_patterns:
            if pattern.match(content):
                return True
        
        # Check confidence - very low confidence often indicates artifacts
        confidence = element.get('confidence', 1.0)
        if confidence < 0.3 and len(content) < 10:
            return True
        
        # Check for garbled text (unusual character combinations)
        if self._is_garbled(content):
            return True
        
        return False
    
    def _is_garbled(self, text: str) -> bool:
        """Detect garbled OCR output."""
        if len(text) < 3:
            return False
        
        # High ratio of special characters
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_count / len(text) > 0.5:
            return True
        
        # Too many consecutive consonants (common OCR error)
        if re.search(r'[bcdfghjklmnpqrstvwxz]{5,}', text, re.IGNORECASE):
            return True
        
        # Repeating character patterns
        if re.search(r'(.)\1{4,}', text):
            return True
        
        return False
    
    def _filter_single_page(self, page: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a single page without cross-page pattern detection."""
        cleaned = []
        for elem in page:
            if elem.get('type') == 'text':
                content = elem.get('content', '').strip()
                
                if self._is_page_number(content):
                    continue
                if self._is_watermark(content):
                    continue
                if self._is_artifact(content, elem):
                    continue
            
            cleaned.append(elem)
        
        return cleaned
    
    def get_noise_report(self) -> Dict[str, Any]:
        """Get report of detected noise."""
        return {
            'headers_detected': len(self.detected_noise['headers']),
            'footers_detected': len(self.detected_noise['footers']),
            'watermarks_removed': len(self.detected_noise['watermarks']),
            'page_numbers_removed': len(self.detected_noise['page_numbers']),
            'artifacts_removed': len(self.detected_noise['artifacts']),
            'details': self.detected_noise
        }


def merge_hyphenated_words(text: str) -> str:
    """
    Merge hyphenated words split across lines.
    
    Handles: "docu-\nment" -> "document"
    """
    # Pattern: word ending with hyphen followed by newline and continuation
    pattern = r'(\w+)-\s*\n\s*(\w+)'
    return re.sub(pattern, r'\1\2', text)


def clean_ocr_artifacts(text: str) -> str:
    """
    Clean common OCR artifacts from text.
    
    Removes:
    - Stray special characters
    - Multiple consecutive spaces
    - Empty lines with only whitespace
    """
    # Remove stray special characters at line starts
    text = re.sub(r'^[|_\-=+]+\s*', '', text, flags=re.MULTILINE)
    
    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Remove empty lines
    text = re.sub(r'\n\s*\n{2,}', '\n\n', text)
    
    return text.strip()


def filter_document(pages_elements: List[List[Dict[str, Any]]]) -> Tuple[List[List[Dict[str, Any]]], Dict]:
    """
    Convenience function to filter entire document.
    
    Args:
        pages_elements: Document pages with elements
    
    Returns:
        (cleaned_pages, noise_report)
    """
    filter_obj = AdaptiveNoiseFilter()
    cleaned = filter_obj.filter(pages_elements)
    report = filter_obj.get_noise_report()
    return cleaned, report


# Example usage
if __name__ == "__main__":
    # Sample pages with noise
    sample_pages = [
        [
            {'type': 'text', 'content': 'Company Name Inc.', 'y': 10},
            {'type': 'text', 'content': '# Introduction', 'y': 100, 'font_size': 18},
            {'type': 'text', 'content': 'This is the content.', 'y': 150},
            {'type': 'text', 'content': 'Page 1 of 3', 'y': 700},
            {'type': 'text', 'content': 'CONFIDENTIAL', 'y': 750},
        ],
        [
            {'type': 'text', 'content': 'Company Name Inc.', 'y': 10},
            {'type': 'text', 'content': '## Methods', 'y': 100, 'font_size': 16},
            {'type': 'text', 'content': 'More content here.', 'y': 150},
            {'type': 'text', 'content': '|||---===', 'y': 300, 'confidence': 0.2},
            {'type': 'text', 'content': 'Page 2 of 3', 'y': 700},
            {'type': 'text', 'content': 'CONFIDENTIAL', 'y': 750},
        ],
        [
            {'type': 'text', 'content': 'Company Name Inc.', 'y': 10},
            {'type': 'text', 'content': '## Results', 'y': 100, 'font_size': 16},
            {'type': 'text', 'content': 'Final content.', 'y': 150},
            {'type': 'text', 'content': 'Page 3 of 3', 'y': 700},
            {'type': 'text', 'content': 'CONFIDENTIAL', 'y': 750},
        ],
    ]
    
    cleaned, report = filter_document(sample_pages)
    
    print("Noise Report:")
    for k, v in report.items():
        if k != 'details':
            print(f"  {k}: {v}")
    
    print("\nCleaned Content:")
    for i, page in enumerate(cleaned, 1):
        print(f"\nPage {i}:")
        for elem in page:
            print(f"  - {elem.get('content', '')[:50]}")
