"""
Confidence Tracker Module for DocFlow

Tracks and aggregates OCR confidence scores throughout the document processing pipeline.
Provides comprehensive confidence metrics for quality assessment and YAML frontmatter.
"""

from typing import Dict, List, Any, Optional, Tuple
import statistics
from dataclasses import dataclass, field


@dataclass
class ConfidenceRecord:
    """Record of a single confidence measurement."""
    page_num: int
    element_id: str
    confidence: float
    element_type: str = 'text'
    source: str = 'ocr'  # 'ocr', 'llm', 'heuristic'


class ConfidenceTracker:
    """
    Track and aggregate confidence scores throughout the document processing pipeline.
    
    Provides end-to-end confidence tracking for:
    - Per-element OCR confidence
    - Per-page aggregated confidence
    - Document-level confidence statistics
    - Low-confidence region identification
    """
    
    def __init__(self, low_threshold: float = 0.7, critical_threshold: float = 0.5):
        """
        Initialize confidence tracker.
        
        Args:
            low_threshold: Threshold below which text is marked as uncertain (default: 0.7)
            critical_threshold: Threshold below which text is marked as low-confidence (default: 0.5)
        """
        self.low_threshold = low_threshold
        self.critical_threshold = critical_threshold
        
        self.records: List[ConfidenceRecord] = []
        self.page_confidences: Dict[int, List[float]] = {}
        self._element_counter = 0
    
    def add_element(self, page_num: int, confidence: float, 
                   element_type: str = 'text', source: str = 'ocr',
                   element_id: Optional[str] = None) -> str:
        """
        Record element-level confidence.
        
        Args:
            page_num: Page number (1-indexed)
            confidence: Confidence score (0.0-1.0)
            element_type: 'text' | 'table' | 'figure'
            source: Confidence source ('ocr', 'llm', 'heuristic')
            element_id: Optional unique element identifier
        
        Returns:
            Generated or provided element_id
        """
        if element_id is None:
            self._element_counter += 1
            element_id = f"elem_{page_num}_{self._element_counter}"
        
        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))
        
        record = ConfidenceRecord(
            page_num=page_num,
            element_id=element_id,
            confidence=confidence,
            element_type=element_type,
            source=source
        )
        self.records.append(record)
        
        # Track per-page
        if page_num not in self.page_confidences:
            self.page_confidences[page_num] = []
        self.page_confidences[page_num].append(confidence)
        
        return element_id
    
    def add_batch(self, page_num: int, elements: List[Dict[str, Any]], 
                 confidence_key: str = 'confidence') -> None:
        """
        Add multiple elements from a batch.
        
        Args:
            page_num: Page number
            elements: List of elements with confidence scores
            confidence_key: Key to extract confidence from each element
        """
        for elem in elements:
            confidence = elem.get(confidence_key, 1.0)
            elem_type = elem.get('type', 'text')
            self.add_element(page_num, confidence, elem_type)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate aggregate confidence metrics.
        
        Returns:
            Dictionary with:
            - avg: Average confidence
            - min: Minimum confidence
            - max: Maximum confidence
            - median: Median confidence
            - std_dev: Standard deviation
            - uncertain_count: Elements with confidence < low_threshold
            - uncertain_percentage: Percentage uncertain
            - low_confidence_count: Elements with confidence < critical_threshold
            - low_confidence_percentage: Percentage low confidence
            - per_page: Per-page statistics
            - by_type: Statistics grouped by element type
        """
        if not self.records:
            return self._empty_statistics()
        
        scores = [r.confidence for r in self.records]
        
        return {
            'avg': round(statistics.mean(scores), 4),
            'min': round(min(scores), 4),
            'max': round(max(scores), 4),
            'median': round(statistics.median(scores), 4),
            'std_dev': round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            'total_elements': len(scores),
            'uncertain_count': sum(1 for s in scores if s < self.low_threshold),
            'uncertain_percentage': round(
                sum(1 for s in scores if s < self.low_threshold) / len(scores) * 100, 2
            ),
            'low_confidence_count': sum(1 for s in scores if s < self.critical_threshold),
            'low_confidence_percentage': round(
                sum(1 for s in scores if s < self.critical_threshold) / len(scores) * 100, 2
            ),
            'per_page': self._get_per_page_statistics(),
            'by_type': self._get_by_type_statistics(),
            'by_source': self._get_by_source_statistics()
        }
    
    def _get_per_page_statistics(self) -> Dict[int, Dict[str, float]]:
        """Calculate per-page confidence statistics."""
        per_page = {}
        
        for page_num, scores in self.page_confidences.items():
            if scores:
                per_page[page_num] = {
                    'avg': round(statistics.mean(scores), 4),
                    'min': round(min(scores), 4),
                    'max': round(max(scores), 4),
                    'element_count': len(scores),
                    'uncertain_count': sum(1 for s in scores if s < self.low_threshold)
                }
        
        return per_page
    
    def _get_by_type_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics grouped by element type."""
        by_type: Dict[str, List[float]] = {}
        
        for record in self.records:
            if record.element_type not in by_type:
                by_type[record.element_type] = []
            by_type[record.element_type].append(record.confidence)
        
        result = {}
        for elem_type, scores in by_type.items():
            result[elem_type] = {
                'avg': round(statistics.mean(scores), 4),
                'min': round(min(scores), 4),
                'count': len(scores)
            }
        
        return result
    
    def _get_by_source_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics grouped by confidence source."""
        by_source: Dict[str, List[float]] = {}
        
        for record in self.records:
            if record.source not in by_source:
                by_source[record.source] = []
            by_source[record.source].append(record.confidence)
        
        result = {}
        for source, scores in by_source.items():
            result[source] = {
                'avg': round(statistics.mean(scores), 4),
                'count': len(scores)
            }
        
        return result
    
    def _empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics when no data available."""
        return {
            'avg': 1.0,
            'min': 1.0,
            'max': 1.0,
            'median': 1.0,
            'std_dev': 0.0,
            'total_elements': 0,
            'uncertain_count': 0,
            'uncertain_percentage': 0.0,
            'low_confidence_count': 0,
            'low_confidence_percentage': 0.0,
            'per_page': {},
            'by_type': {},
            'by_source': {}
        }
    
    def get_low_confidence_regions(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get list of low-confidence regions.
        
        Args:
            threshold: Confidence threshold (default: self.low_threshold)
        
        Returns:
            List of low-confidence elements with metadata
        """
        if threshold is None:
            threshold = self.low_threshold
        
        return [
            {
                'page': r.page_num,
                'element_id': r.element_id,
                'confidence': r.confidence,
                'type': r.element_type,
                'source': r.source
            }
            for r in self.records if r.confidence < threshold
        ]
    
    def get_page_quality_summary(self) -> List[Dict[str, Any]]:
        """
        Get page-by-page quality summary.
        
        Returns:
            List of page summaries with quality indicators
        """
        summaries = []
        
        for page_num in sorted(self.page_confidences.keys()):
            scores = self.page_confidences[page_num]
            avg_conf = statistics.mean(scores) if scores else 1.0
            
            # Determine quality level
            if avg_conf >= 0.9:
                quality = 'excellent'
            elif avg_conf >= 0.8:
                quality = 'good'
            elif avg_conf >= 0.7:
                quality = 'acceptable'
            elif avg_conf >= 0.5:
                quality = 'poor'
            else:
                quality = 'very_poor'
            
            summaries.append({
                'page': page_num,
                'avg_confidence': round(avg_conf, 4),
                'quality': quality,
                'element_count': len(scores),
                'uncertain_count': sum(1 for s in scores if s < self.low_threshold),
                'needs_review': avg_conf < self.low_threshold
            })
        
        return summaries
    
    def get_overall_quality_score(self) -> float:
        """
        Calculate overall quality score (0.0-1.0).
        
        Quality score is a weighted combination of:
        - Average confidence (60%)
        - Minimum confidence (20%)
        - Percentage of uncertain elements (20%)
        
        Returns:
            Overall quality score
        """
        if not self.records:
            return 1.0
        
        stats = self.get_statistics()
        
        # Component scores
        avg_score = stats['avg']  # Already 0-1
        min_score = stats['min']  # Already 0-1
        uncertainty_score = 1.0 - (stats['uncertain_percentage'] / 100.0)
        
        # Weighted combination
        quality = (
            0.6 * avg_score +
            0.2 * min_score +
            0.2 * uncertainty_score
        )
        
        return round(max(0.0, min(1.0, quality)), 4)
    
    def to_yaml_dict(self) -> Dict[str, Any]:
        """
        Convert confidence statistics to YAML-friendly dictionary.
        
        Returns:
            Dictionary suitable for YAML frontmatter
        """
        stats = self.get_statistics()
        
        return {
            'confidence_score': self.get_overall_quality_score(),
            'confidence_avg': stats['avg'],
            'confidence_min': stats['min'],
            'uncertain_regions': stats['uncertain_count'],
            'uncertain_percentage': stats['uncertain_percentage'],
            'low_confidence_regions': stats['low_confidence_count'],
        }
    
    def reset(self) -> None:
        """Reset all tracked data."""
        self.records = []
        self.page_confidences = {}
        self._element_counter = 0


# Convenience function
def track_confidence(elements_by_page: List[List[Dict[str, Any]]],
                    confidence_key: str = 'confidence') -> Dict[str, Any]:
    """
    Track confidence for a complete document.
    
    Args:
        elements_by_page: List of element lists, one per page
        confidence_key: Key to extract confidence from each element
    
    Returns:
        Confidence statistics dictionary
    """
    tracker = ConfidenceTracker()
    
    for page_num, elements in enumerate(elements_by_page, 1):
        tracker.add_batch(page_num, elements, confidence_key)
    
    return tracker.get_statistics()


# Example usage
if __name__ == "__main__":
    tracker = ConfidenceTracker()
    
    # Simulate adding elements
    tracker.add_element(1, 0.95, 'text', 'ocr')
    tracker.add_element(1, 0.88, 'text', 'ocr')
    tracker.add_element(1, 0.65, 'text', 'ocr')  # Uncertain
    tracker.add_element(1, 0.92, 'table', 'heuristic')
    tracker.add_element(2, 0.78, 'text', 'ocr')
    tracker.add_element(2, 0.45, 'text', 'ocr')  # Low confidence
    tracker.add_element(2, 0.91, 'text', 'ocr')
    
    print("Statistics:")
    stats = tracker.get_statistics()
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    print("\nPage Quality Summary:")
    for page in tracker.get_page_quality_summary():
        print(f"  Page {page['page']}: {page['quality']} (avg: {page['avg_confidence']:.2f})")
    
    print(f"\nOverall Quality Score: {tracker.get_overall_quality_score():.2f}")
    
    print("\nLow Confidence Regions:")
    for region in tracker.get_low_confidence_regions():
        print(f"  Page {region['page']}: {region['element_id']} ({region['confidence']:.2f})")
    
    print("\nYAML Dict:")
    print(tracker.to_yaml_dict())
