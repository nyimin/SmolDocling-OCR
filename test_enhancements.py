"""
Test Suite for DocFlow Quality Enhancement Modules

Comprehensive tests for all Phase 1-5 enhancements.
"""

import unittest
import sys
import os

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestSemanticAnnotator(unittest.TestCase):
    """Tests for semantic_annotator.py"""
    
    def setUp(self):
        from semantic_annotator import SemanticAnnotator, PageContext
        self.annotator = SemanticAnnotator()
    
    def test_heading_detection_by_font_size(self):
        """Test heading detection based on font size."""
        element = {'type': 'text', 'content': 'Introduction', 'font_size': 20, 'confidence': 0.95}
        elements = [element, {'type': 'text', 'content': 'Body text', 'font_size': 12}]
        from semantic_annotator import PageContext
        context = PageContext(elements, page_num=1)
        
        result = self.annotator.annotate_element(element, context)
        self.assertIn('role:heading', result)
    
    def test_list_detection(self):
        """Test list item detection."""
        element = {'type': 'text', 'content': '• First item', 'font_size': 12, 'confidence': 0.9}
        from semantic_annotator import PageContext
        context = PageContext([element], page_num=1)
        
        result = self.annotator.annotate_element(element, context)
        self.assertIn('list_item', result)
    
    def test_caption_detection(self):
        """Test table/figure caption detection."""
        element = {'type': 'text', 'content': 'Table 1: Sample data', 'font_size': 12, 'confidence': 0.9}
        from semantic_annotator import PageContext
        context = PageContext([element], page_num=1)
        
        result = self.annotator.annotate_element(element, context)
        self.assertIn('caption', result)


class TestConfidenceTracker(unittest.TestCase):
    """Tests for confidence_tracker.py"""
    
    def setUp(self):
        from confidence_tracker import ConfidenceTracker
        self.tracker = ConfidenceTracker()
    
    def test_add_element(self):
        """Test adding element confidence."""
        self.tracker.add_element(1, 0.92, 'text')
        stats = self.tracker.get_statistics()
        
        self.assertEqual(stats['total_elements'], 1)
        self.assertEqual(stats['avg'], 0.92)
    
    def test_quality_score(self):
        """Test overall quality score calculation."""
        self.tracker.add_element(1, 0.95)
        self.tracker.add_element(1, 0.88)
        self.tracker.add_element(2, 0.72)
        
        score = self.tracker.get_overall_quality_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_low_confidence_detection(self):
        """Test low confidence region detection."""
        self.tracker.add_element(1, 0.95)
        self.tracker.add_element(1, 0.45)  # Low
        self.tracker.add_element(2, 0.62)  # Uncertain
        
        low_regions = self.tracker.get_low_confidence_regions(threshold=0.7)
        self.assertEqual(len(low_regions), 2)


class TestSchemaEnforcer(unittest.TestCase):
    """Tests for schema_enforcer.py"""
    
    def setUp(self):
        from schema_enforcer import SchemaEnforcer
        self.enforcer = SchemaEnforcer()
    
    def test_add_frontmatter(self):
        """Test frontmatter addition."""
        content = "# Heading\n\nContent here."
        metadata = {'source_file': 'test.pdf', 'document_id': 'abc123'}
        
        result = self.enforcer.enforce(content, metadata)
        
        self.assertTrue(result.startswith('---'))
        self.assertIn('source_file:', result)
        self.assertIn('document_id:', result)
    
    def test_page_marker_addition(self):
        """Test page marker is added."""
        content = "# Heading\n\nContent here."
        metadata = {'source_file': 'test.pdf', 'document_id': 'abc123'}
        
        result = self.enforcer.enforce(content, metadata)
        
        self.assertIn('<!-- page:1 -->', result)


class TestNoiseFilter(unittest.TestCase):
    """Tests for noise_filter.py"""
    
    def setUp(self):
        from noise_filter import AdaptiveNoiseFilter
        self.filter = AdaptiveNoiseFilter()
    
    def test_page_number_removal(self):
        """Test page number detection and removal."""
        pages = [[
            {'type': 'text', 'content': 'Content', 'y': 100},
            {'type': 'text', 'content': 'Page 1', 'y': 700},
        ]]
        
        result = self.filter.filter(pages)
        
        contents = [e['content'] for e in result[0]]
        self.assertNotIn('Page 1', contents)
    
    def test_watermark_removal(self):
        """Test watermark detection and removal."""
        pages = [[
            {'type': 'text', 'content': 'Content', 'y': 100},
            {'type': 'text', 'content': 'CONFIDENTIAL', 'y': 50},
        ]]
        
        result = self.filter.filter(pages)
        
        contents = [e['content'] for e in result[0]]
        self.assertNotIn('CONFIDENTIAL', contents)
    
    def test_header_footer_detection(self):
        """Test repetitive header/footer removal."""
        pages = [
            [{'type': 'text', 'content': 'Company Inc.', 'y': 10}, 
             {'type': 'text', 'content': 'Page content 1', 'y': 200}],
            [{'type': 'text', 'content': 'Company Inc.', 'y': 10}, 
             {'type': 'text', 'content': 'Page content 2', 'y': 200}],
            [{'type': 'text', 'content': 'Company Inc.', 'y': 10}, 
             {'type': 'text', 'content': 'Page content 3', 'y': 200}],
        ]
        
        result = self.filter.filter(pages)
        
        report = self.filter.get_noise_report()
        self.assertGreater(report['headers_detected'], 0)


class TestValidationFramework(unittest.TestCase):
    """Tests for validation_framework.py"""
    
    def setUp(self):
        from validation_framework import ValidationFramework, QualityGate
        self.validator = ValidationFramework()
        self.gate = QualityGate()
    
    def test_missing_frontmatter_detection(self):
        """Test detection of missing frontmatter."""
        content = "# Heading\n\nContent without frontmatter."
        
        report = self.validator.validate(content)
        
        self.assertFalse(report['is_valid'])
        self.assertGreater(report['errors'], 0)
    
    def test_hallucination_detection(self):
        """Test hallucination phrase detection."""
        content = """---
document:
  source_file: test.pdf
---
Based on the image, the document shows several items.
"""
        report = self.validator.validate(content)
        
        self.assertTrue(report['hallucination_detected'])
    
    def test_quality_gate_pass(self):
        """Test quality gate with good content."""
        content = """---
document:
  source_file: test.pdf
  document_id: abc123
quality:
  confidence_score: 0.92
---

<!-- page:1 -->

<!-- role:heading level:1 -->
# Introduction

This is clean content without issues.
"""
        report = self.validator.validate(content)
        gate_result = self.gate.check(content, report)
        
        self.assertTrue(gate_result['passed'])


class TestEnhancedPipeline(unittest.TestCase):
    """Tests for enhanced_pipeline.py"""
    
    def setUp(self):
        from enhanced_pipeline import EnhancedPipeline
        self.pipeline = EnhancedPipeline()
    
    def test_full_pipeline_processing(self):
        """Test end-to-end pipeline."""
        pages = [
            [
                {'type': 'text', 'content': 'Introduction', 'y': 100, 'font_size': 18, 'confidence': 0.95},
                {'type': 'text', 'content': 'Body paragraph.', 'y': 150, 'font_size': 12, 'confidence': 0.88},
            ]
        ]
        
        processed, report = self.pipeline.process_elements(pages)
        
        self.assertEqual(len(processed), 1)
        self.assertIn('stages', report)
    
    def test_markdown_rendering(self):
        """Test Markdown output generation."""
        pages = [[
            {'type': 'text', 'content': 'Test heading', 'font_size': 18, 'confidence': 0.9, 'y': 100},
        ]]
        
        metadata = {'source_file': 'test.pdf', 'document_id': 'test123'}
        
        processed, _ = self.pipeline.process_elements(pages)
        markdown = self.pipeline.render_markdown(processed, metadata)
        
        self.assertTrue(markdown.startswith('---'))
        self.assertIn('page:1', markdown)


class TestLanguageDetector(unittest.TestCase):
    """Tests for language_detector.py"""
    
    def setUp(self):
        from language_detector import LanguageDetector
        self.detector = LanguageDetector()
    
    def test_english_detection(self):
        """Test English language detection."""
        text = "The quick brown fox jumps over the lazy dog."
        result = self.detector.detect(text)
        
        self.assertEqual(result['language'], 'en')
    
    def test_german_detection(self):
        """Test German language detection."""
        text = "Der schnelle braune Fuchs springt über den faulen Hund."
        result = self.detector.detect(text)
        
        self.assertEqual(result['language'], 'de')
    
    def test_myanmar_script_detection(self):
        """Test Myanmar script detection."""
        text = "မြန်မာဘာသာစကား နမူနာ စာသား"
        result = self.detector.detect(text)
        
        self.assertEqual(result['language'], 'my')
        self.assertEqual(result['script'], 'myanmar')


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
