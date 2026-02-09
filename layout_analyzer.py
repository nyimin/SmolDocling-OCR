"""
Layout Analysis Module for RapidOCR

Provides column detection, reading order determination, and semantic role classification
for multi-column document layouts.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class LayoutAnalyzer:
    """Analyzes document layout for proper reading order and structure."""
    
    def __init__(self, column_gap_threshold: int = 50):
        """
        Initialize layout analyzer.
        
        Args:
            column_gap_threshold: Minimum horizontal gap (pixels) to detect column boundaries
        """
        self.column_gap_threshold = column_gap_threshold
    
    def detect_columns(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect columns in a page by clustering text boxes by X-position.
        
        Uses gap-based clustering: if horizontal gap between elements > threshold,
        they belong to different columns.
        
        Args:
            elements: List of text elements with bounding boxes
                     Each element should have: {'bbox': (x0, y0, x1, y1), ...}
        
        Returns:
            List of column definitions: [{'id': 0, 'x_min': 10, 'x_max': 300}, ...]
        """
        if not elements:
            return []
        
        # Extract X positions (left edges)
        x_positions = []
        for elem in elements:
            if 'bbox' in elem:
                x_positions.append(elem['bbox'][0])  # x0 (left edge)
        
        if not x_positions:
            return []
        
        # Sort X positions
        x_sorted = sorted(set(x_positions))
        
        # Detect gaps
        columns = []
        current_column = {'id': 0, 'x_min': x_sorted[0], 'x_max': x_sorted[0]}
        
        for i in range(1, len(x_sorted)):
            gap = x_sorted[i] - x_sorted[i-1]
            
            if gap > self.column_gap_threshold:
                # Large gap detected - new column
                columns.append(current_column)
                current_column = {
                    'id': len(columns),
                    'x_min': x_sorted[i],
                    'x_max': x_sorted[i]
                }
            else:
                # Same column - extend range
                current_column['x_max'] = x_sorted[i]
        
        # Add last column
        columns.append(current_column)
        
        return columns
    
    def assign_column_ids(self, elements: List[Dict[str, Any]], 
                         columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assign column ID to each element based on its X position.
        
        Args:
            elements: List of text elements
            columns: List of column definitions from detect_columns()
        
        Returns:
            Elements with 'column_id' field added
        """
        for elem in elements:
            if 'bbox' not in elem:
                elem['column_id'] = 0
                continue
            
            x_center = (elem['bbox'][0] + elem['bbox'][2]) / 2
            
            # Find which column this element belongs to
            assigned = False
            for col in columns:
                if col['x_min'] <= x_center <= col['x_max']:
                    elem['column_id'] = col['id']
                    assigned = True
                    break
            
            if not assigned:
                # Assign to nearest column
                distances = [abs(x_center - (col['x_min'] + col['x_max']) / 2) 
                           for col in columns]
                elem['column_id'] = columns[np.argmin(distances)]['id']
        
        return elements
    
    def xy_cut_sort(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort elements by reading order using XY-cut algorithm.
        
        Reading order: left-to-right columns, then top-to-bottom within each column.
        
        Args:
            elements: List of elements with 'column_id' and 'bbox' fields
        
        Returns:
            Sorted list of elements with 'reading_order' field added
        """
        if not elements:
            return []
        
        # Group by column
        columns_dict = {}
        for elem in elements:
            col_id = elem.get('column_id', 0)
            if col_id not in columns_dict:
                columns_dict[col_id] = []
            columns_dict[col_id].append(elem)
        
        # Sort columns left-to-right, then elements top-to-bottom within each column
        sorted_elements = []
        reading_order = 1
        
        for col_id in sorted(columns_dict.keys()):
            col_elements = columns_dict[col_id]
            
            # Sort by Y position (top to bottom)
            col_elements_sorted = sorted(col_elements, 
                                        key=lambda e: e['bbox'][1] if 'bbox' in e else 0)
            
            for elem in col_elements_sorted:
                elem['reading_order'] = reading_order
                sorted_elements.append(elem)
                reading_order += 1
        
        return sorted_elements
    
    def classify_semantic_role(self, elem: Dict[str, Any], 
                              page_elements: List[Dict[str, Any]]) -> str:
        """
        Classify the semantic role of a text element.
        
        Args:
            elem: Text element to classify
            page_elements: All elements on the page (for context)
        
        Returns:
            Semantic role: 'heading', 'paragraph', 'list_item', 'caption', 'footnote'
        """
        text = elem.get('text', '').strip()
        bbox = elem.get('bbox', (0, 0, 0, 0))
        
        if not text:
            return 'paragraph'
        
        # Calculate text metrics
        text_height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
        text_width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
        
        # Calculate average text height for the page
        avg_height = np.mean([e['bbox'][3] - e['bbox'][1] 
                             for e in page_elements 
                             if 'bbox' in e and len(e['bbox']) >= 4]) if page_elements else 12.0
        
        # Heading detection heuristics
        # 1. Short text (< 100 chars)
        # 2. Larger font size (height > average * 1.2)
        # 3. Often at top of page or after large vertical gap
        
        if len(text) < 100:
            if text_height > avg_height * 1.2:
                return 'heading'
        
        # List item detection
        # Starts with bullet (•, -, *, ◦) or number (1., 2., etc.)
        if text.startswith(('•', '-', '*', '◦', '○', '▪', '▫')):
            return 'list_item'
        
        if len(text) > 0 and text[0].isdigit():
            # Check for numbered list pattern: "1. ", "1) ", etc.
            if len(text) > 2 and text[1:3] in ('. ', ') ', ': '):
                return 'list_item'
        
        # Caption detection
        # Usually short text near tables/figures
        # Often starts with "Figure", "Table", "Fig.", etc.
        caption_keywords = ['figure', 'fig.', 'table', 'chart', 'diagram', 'image']
        if any(text.lower().startswith(kw) for kw in caption_keywords):
            return 'caption'
        
        # Footnote detection
        # Small text at bottom of page
        # Often starts with superscript number or symbol
        page_height = max([e['bbox'][3] for e in page_elements 
                          if 'bbox' in e and len(e['bbox']) >= 4], default=1000)
        
        if bbox[1] > page_height * 0.85:  # Bottom 15% of page
            if text_height < avg_height * 0.8:  # Smaller font
                return 'footnote'
        
        # Default to paragraph
        return 'paragraph'
    
    def filter_by_confidence(self, elements: List[Dict[str, Any]], 
                            threshold: float = 0.7) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter elements by OCR confidence score.
        
        Args:
            elements: List of elements with 'confidence' field
            threshold: Minimum confidence score (0.0-1.0)
        
        Returns:
            Tuple of (high_confidence_elements, low_confidence_elements)
        """
        high_conf = []
        low_conf = []
        
        for elem in elements:
            confidence = elem.get('confidence', 1.0)
            # Convert to float if it's a string (RapidOCR sometimes returns string)
            try:
                confidence = float(confidence) if isinstance(confidence, str) else confidence
            except (ValueError, TypeError):
                confidence = 1.0  # Default to high confidence if conversion fails
            
            if confidence >= threshold:
                high_conf.append(elem)
            else:
                # Mark as uncertain
                elem['uncertain'] = True
                low_conf.append(elem)
        
        return high_conf, low_conf
    
    def analyze_page_layout(self, elements: List[Dict[str, Any]], 
                           confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Comprehensive page layout analysis.
        
        Args:
            elements: List of OCR text elements with bounding boxes
            confidence_threshold: Minimum OCR confidence to accept
        
        Returns:
            Analysis results with sorted elements and metadata
        """
        if not elements:
            return {
                'elements': [],
                'columns': [],
                'column_count': 0,
                'reading_order_applied': False
            }
        
        # 1. Filter by confidence
        high_conf, low_conf = self.filter_by_confidence(elements, confidence_threshold)
        
        # 2. Detect columns
        columns = self.detect_columns(high_conf)
        
        # 3. Assign column IDs
        high_conf = self.assign_column_ids(high_conf, columns)
        
        # 4. Apply XY-cut sorting
        sorted_elements = self.xy_cut_sort(high_conf)
        
        # 5. Classify semantic roles
        for elem in sorted_elements:
            elem['semantic_role'] = self.classify_semantic_role(elem, sorted_elements)
        
        # 6. Add low-confidence elements at the end (marked as uncertain)
        for elem in low_conf:
            elem['reading_order'] = len(sorted_elements) + 1
            elem['semantic_role'] = 'paragraph'
            sorted_elements.append(elem)
        
        return {
            'elements': sorted_elements,
            'columns': columns,
            'column_count': len(columns),
            'reading_order_applied': True,
            'high_confidence_count': len(high_conf),
            'low_confidence_count': len(low_conf)
        }


# Example usage
if __name__ == "__main__":
    # Sample elements (simulating RapidOCR output)
    sample_elements = [
        {'text': 'Column 1 Title', 'bbox': (50, 100, 250, 130), 'confidence': 0.95},
        {'text': 'Column 1 content line 1', 'bbox': (50, 150, 250, 170), 'confidence': 0.92},
        {'text': 'Column 2 Title', 'bbox': (350, 100, 550, 130), 'confidence': 0.93},
        {'text': 'Column 2 content line 1', 'bbox': (350, 150, 550, 170), 'confidence': 0.88},
        {'text': 'Column 1 content line 2', 'bbox': (50, 180, 250, 200), 'confidence': 0.65},  # Low confidence
    ]
    
    analyzer = LayoutAnalyzer(column_gap_threshold=50)
    result = analyzer.analyze_page_layout(sample_elements, confidence_threshold=0.7)
    
    print(f"Detected {result['column_count']} columns")
    print(f"High confidence: {result['high_confidence_count']}")
    print(f"Low confidence: {result['low_confidence_count']}")
    print("\nReading order:")
    for elem in result['elements']:
        print(f"  {elem['reading_order']}: [{elem['semantic_role']}] {elem['text'][:50]}")
