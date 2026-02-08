"""
Enhanced Pipeline Module for DocFlow

Orchestrates all quality enhancement modules in a unified pipeline.
Integrates with existing structure_engine.py extraction functions.
"""

from typing import Dict, List, Any, Optional, Tuple
import traceback

# Import all enhancement modules
try:
    from semantic_annotator import SemanticAnnotator, PageContext
    from confidence_tracker import ConfidenceTracker
    from schema_enforcer import SchemaEnforcer
    from language_detector import LanguageDetector
    from noise_filter import AdaptiveNoiseFilter, merge_hyphenated_words
    from caption_extractor import CaptionExtractor, FootnoteLinker
    from validation_framework import ValidationFramework, QualityGate
    ENHANCEMENTS_AVAILABLE = True
    print("Quality enhancement modules loaded successfully.")
except ImportError as e:
    print(f"Warning: Enhancement modules not available: {e}")
    ENHANCEMENTS_AVAILABLE = False


class EnhancedPipeline:
    """
    Enhanced document processing pipeline.
    
    Stages:
    1. Noise Reduction - Remove headers, footers, watermarks, artifacts
    2. Caption Extraction - Link captions to tables/figures
    3. Footnote Linking - Connect footnote references
    4. Semantic Annotation - Add role annotations to all elements
    5. Confidence Tracking - Aggregate OCR confidence scores
    6. Language Detection - Auto-detect document language
    7. Schema Enforcement - Ensure Schema v2.0 compliance
    8. Validation - Run quality checks
    9. Quality Gate - Enforce thresholds, trigger fallbacks
    """
    
    def __init__(self, 
                 enable_noise_filter: bool = True,
                 enable_caption_extraction: bool = True,
                 enable_semantic_annotation: bool = True,
                 quality_threshold: float = 0.6):
        """
        Initialize enhanced pipeline.
        
        Args:
            enable_noise_filter: Enable noise reduction
            enable_caption_extraction: Enable caption/footnote extraction
            enable_semantic_annotation: Enable semantic annotations
            quality_threshold: Minimum quality score for gate
        """
        self.enable_noise_filter = enable_noise_filter
        self.enable_caption_extraction = enable_caption_extraction
        self.enable_semantic_annotation = enable_semantic_annotation
        self.quality_threshold = quality_threshold
        
        # Initialize components
        self.noise_filter = AdaptiveNoiseFilter() if ENHANCEMENTS_AVAILABLE else None
        self.caption_extractor = CaptionExtractor() if ENHANCEMENTS_AVAILABLE else None
        self.footnote_linker = FootnoteLinker() if ENHANCEMENTS_AVAILABLE else None
        self.semantic_annotator = SemanticAnnotator() if ENHANCEMENTS_AVAILABLE else None
        self.confidence_tracker = ConfidenceTracker() if ENHANCEMENTS_AVAILABLE else None
        self.language_detector = LanguageDetector() if ENHANCEMENTS_AVAILABLE else None
        self.schema_enforcer = SchemaEnforcer() if ENHANCEMENTS_AVAILABLE else None
        self.validator = ValidationFramework() if ENHANCEMENTS_AVAILABLE else None
        self.quality_gate = QualityGate(min_quality_score=quality_threshold) if ENHANCEMENTS_AVAILABLE else None
        
        self.last_report: Dict[str, Any] = {}
    
    def process_elements(self, pages_elements: List[List[Dict[str, Any]]],
                        page_heights: Optional[List[float]] = None) -> Tuple[List[List[Dict]], Dict]:
        """
        Process document elements through enhancement pipeline.
        
        Args:
            pages_elements: List of element lists per page
            page_heights: Optional page heights for position calculations
        
        Returns:
            (processed_elements, pipeline_report)
        """
        if not ENHANCEMENTS_AVAILABLE:
            return pages_elements, {'status': 'enhancements_unavailable'}
        
        report = {
            'stages': {},
            'quality': {},
            'gate': {}
        }
        
        current_elements = pages_elements
        
        # Stage 1: Noise Reduction
        if self.enable_noise_filter and self.noise_filter:
            try:
                current_elements = self.noise_filter.filter(current_elements)
                report['stages']['noise_reduction'] = self.noise_filter.get_noise_report()
            except Exception as e:
                report['stages']['noise_reduction'] = {'error': str(e)}
        
        # Stage 2: Caption Extraction
        if self.enable_caption_extraction and self.caption_extractor:
            try:
                current_elements = self.caption_extractor.extract(current_elements)
                report['stages']['caption_extraction'] = self.caption_extractor.get_caption_report()
            except Exception as e:
                report['stages']['caption_extraction'] = {'error': str(e)}
        
        # Stage 3: Footnote Linking
        if self.enable_caption_extraction and self.footnote_linker:
            try:
                current_elements = self.footnote_linker.link(current_elements, page_heights)
                report['stages']['footnote_linking'] = self.footnote_linker.get_footnote_report()
            except Exception as e:
                report['stages']['footnote_linking'] = {'error': str(e)}
        
        # Stage 4: Confidence Tracking
        if self.confidence_tracker:
            try:
                self.confidence_tracker.reset()
                for page_num, elements in enumerate(current_elements, 1):
                    self.confidence_tracker.add_batch(page_num, elements)
                report['quality']['confidence'] = self.confidence_tracker.get_statistics()
                report['quality']['score'] = self.confidence_tracker.get_overall_quality_score()
            except Exception as e:
                report['quality']['confidence'] = {'error': str(e)}
        
        self.last_report = report
        return current_elements, report
    
    def render_markdown(self, pages_elements: List[List[Dict[str, Any]]],
                       metadata: Dict[str, Any]) -> str:
        """
        Render elements as annotated Markdown.
        
        Args:
            pages_elements: Processed elements
            metadata: Document metadata
        
        Returns:
            Schema-compliant Markdown string
        """
        if not ENHANCEMENTS_AVAILABLE or not self.semantic_annotator:
            return self._fallback_render(pages_elements)
        
        output = ""
        
        # Render each page
        for page_num, elements in enumerate(pages_elements, 1):
            page_height = 800  # Default
            output += self.semantic_annotator.annotate_page(elements, page_num, page_height)
        
        # Merge hyphenated words
        output = merge_hyphenated_words(output)
        
        # Enforce schema compliance
        if self.schema_enforcer:
            # Enhance metadata with quality info
            if self.confidence_tracker:
                metadata.update(self.confidence_tracker.to_yaml_dict())
            
            # Detect language
            if self.language_detector:
                lang_result = self.language_detector.detect(output)
                metadata['language'] = lang_result['language']
            
            output = self.schema_enforcer.enforce(output, metadata)
        
        return output
    
    def validate_output(self, markdown_text: str) -> Dict[str, Any]:
        """
        Validate final Markdown output.
        
        Args:
            markdown_text: Rendered Markdown
        
        Returns:
            Validation report with quality gate result
        """
        if not ENHANCEMENTS_AVAILABLE:
            return {'status': 'enhancements_unavailable', 'passed': True}
        
        # Run validation
        validation_report = {}
        if self.validator:
            validation_report = self.validator.validate(markdown_text)
        
        # Check quality gate
        gate_result = {}
        if self.quality_gate:
            gate_result = self.quality_gate.check(markdown_text, validation_report)
        
        return {
            'validation': validation_report,
            'gate': gate_result,
            'passed': gate_result.get('passed', True),
            'suggested_action': gate_result.get('suggested_action', 'accept')
        }
    
    def _fallback_render(self, pages_elements: List[List[Dict[str, Any]]]) -> str:
        """Fallback rendering without enhancements."""
        output = ""
        for page_num, elements in enumerate(pages_elements, 1):
            output += f"\n<!-- page:{page_num} -->\n\n"
            for elem in elements:
                if elem.get('type') == 'text':
                    output += elem.get('content', '') + "\n\n"
                elif elem.get('type') == 'table':
                    output += elem.get('content', '') + "\n\n"
        return output
    
    def get_last_report(self) -> Dict[str, Any]:
        """Get report from last processing run."""
        return self.last_report


def process_document_enhanced(pages_elements: List[List[Dict[str, Any]]],
                              metadata: Dict[str, Any],
                              page_heights: Optional[List[float]] = None,
                              **options) -> Tuple[str, Dict]:
    """
    Process document with full enhancement pipeline.
    
    Args:
        pages_elements: Document elements by page
        metadata: Document metadata
        page_heights: Optional page heights
        **options: Pipeline options
    
    Returns:
        (markdown_output, processing_report)
    """
    pipeline = EnhancedPipeline(**options)
    
    # Process elements
    processed, element_report = pipeline.process_elements(pages_elements, page_heights)
    
    # Render to Markdown
    markdown = pipeline.render_markdown(processed, metadata)
    
    # Validate
    validation_result = pipeline.validate_output(markdown)
    
    # Combine reports
    full_report = {
        **element_report,
        **validation_result,
        'metadata': metadata
    }
    
    return markdown, full_report


# Example usage
if __name__ == "__main__":
    sample_pages = [
        [
            {'type': 'text', 'content': 'Company Header', 'y': 10, 'confidence': 0.95},
            {'type': 'text', 'content': '# Introduction', 'y': 100, 'font_size': 18, 'confidence': 0.92},
            {'type': 'text', 'content': 'This is the first paragraph.', 'y': 150, 'confidence': 0.88},
            {'type': 'table', 'content': '| A | B |\n|---|---|\n| 1 | 2 |', 'y': 250},
            {'type': 'text', 'content': 'Page 1', 'y': 700, 'confidence': 0.99},
        ],
        [
            {'type': 'text', 'content': 'Company Header', 'y': 10, 'confidence': 0.95},
            {'type': 'text', 'content': '## Methods', 'y': 100, 'font_size': 16, 'confidence': 0.90},
            {'type': 'text', 'content': 'Methodology description here.', 'y': 150, 'confidence': 0.85},
            {'type': 'text', 'content': 'Page 2', 'y': 700, 'confidence': 0.99},
        ],
    ]
    
    metadata = {
        'source_file': 'sample.pdf',
        'document_id': 'test123',
        'pages': 2,
        'extraction_method': 'RapidOCR'
    }
    
    markdown, report = process_document_enhanced(sample_pages, metadata)
    
    print("Generated Markdown:")
    print("=" * 50)
    print(markdown[:800])
    
    print("\nValidation Result:")
    print(f"  Passed: {report.get('passed', 'N/A')}")
    print(f"  Action: {report.get('suggested_action', 'N/A')}")
