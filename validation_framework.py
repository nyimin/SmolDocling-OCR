"""
Validation Framework Module for DocFlow

Unified validation framework combining all validation checks.
"""

from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'schema', 'content', 'format', 'hallucination'
    message: str
    line: Optional[int] = None
    context: Optional[str] = None


class ValidationFramework:
    """
    Unified validation framework for Markdown output.
    
    Combines:
    - Schema validation
    - Content quality checks
    - Hallucination detection
    - Format compliance
    """
    
    # Hallucination markers
    HALLUCINATION_PHRASES = [
        r"(?i)based on (the|this) (image|document|page)",
        r"(?i)as (shown|seen|visible) in",
        r"(?i)it (appears|seems) that",
        r"(?i)I can (see|observe) that",
        r"(?i)the (image|document) shows",
        r"(?i)looking at (the|this)",
        r"(?i)from what I can (see|tell)",
        r"(?i)this (appears|seems) to be",
    ]
    
    def __init__(self, strict: bool = False):
        """
        Initialize validation framework.
        
        Args:
            strict: If True, warnings become errors
        """
        self.strict = strict
        self.issues: List[ValidationIssue] = []
        self.compiled_hallucination = [re.compile(p) for p in self.HALLUCINATION_PHRASES]
    
    def validate(self, markdown_text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Args:
            markdown_text: Markdown content to validate
            metadata: Optional document metadata
        
        Returns:
            Validation report
        """
        self.issues = []
        
        # Schema validation
        self._validate_schema(markdown_text)
        
        # Content quality
        self._validate_content(markdown_text)
        
        # Hallucination detection
        self._detect_hallucinations(markdown_text)
        
        # Format compliance
        self._validate_format(markdown_text)
        
        # Calculate scores
        return self._generate_report(markdown_text, metadata)
    
    def _validate_schema(self, text: str):
        """Validate schema compliance."""
        # Check frontmatter
        if not text.strip().startswith('---'):
            self.issues.append(ValidationIssue(
                severity='error',
                category='schema',
                message='Missing YAML frontmatter'
            ))
        else:
            # Validate frontmatter structure
            fm_end = text.find('---', 3)
            if fm_end == -1:
                self.issues.append(ValidationIssue(
                    severity='error',
                    category='schema',
                    message='Incomplete YAML frontmatter (missing closing ---)'
                ))
            else:
                fm = text[3:fm_end]
                if 'document:' not in fm:
                    self.issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message='Missing document section in frontmatter'
                    ))
                if 'source_file:' not in fm:
                    self.issues.append(ValidationIssue(
                        severity='warning',
                        category='schema',
                        message='Missing source_file in frontmatter'
                    ))
                if 'quality:' not in fm:
                    self.issues.append(ValidationIssue(
                        severity='warning',
                        category='schema',
                        message='Missing quality section in frontmatter'
                    ))
        
        # Check page markers
        page_markers = re.findall(r'<!--\s*page:\s*(\d+)\s*-->', text)
        if not page_markers:
            self.issues.append(ValidationIssue(
                severity='warning',
                category='schema',
                message='No page markers found'
            ))
        else:
            # Check sequence
            pages = [int(p) for p in page_markers]
            expected = list(range(1, max(pages) + 1))
            missing = set(expected) - set(pages)
            if missing:
                self.issues.append(ValidationIssue(
                    severity='warning',
                    category='schema',
                    message=f'Missing page markers: {sorted(missing)}'
                ))
        
        # Check semantic annotations
        role_annotations = len(re.findall(r'<!--\s*role:\w+', text))
        headings = len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE))
        tables = len(re.findall(r'^\|', text, re.MULTILINE)) // 2
        
        if role_annotations < (headings + tables) * 0.5:
            self.issues.append(ValidationIssue(
                severity='warning',
                category='schema',
                message=f'Low annotation coverage: {role_annotations} annotations for {headings} headings and {tables} tables'
            ))
    
    def _validate_content(self, text: str):
        """Validate content quality."""
        lines = text.split('\n')
        
        # Strip frontmatter for content analysis
        content_start = 0
        if text.startswith('---'):
            fm_end = text.find('---', 3)
            if fm_end > 0:
                content_start = text.find('\n', fm_end) + 1
        
        content = text[content_start:]
        content_lines = content.split('\n')
        
        # Check for empty content
        non_empty = [l for l in content_lines if l.strip() and not l.strip().startswith('<!--')]
        if len(non_empty) < 3:
            self.issues.append(ValidationIssue(
                severity='error',
                category='content',
                message='Document appears to have very little content'
            ))
        
        # Check for uncertain text markers
        uncertain = len(re.findall(r'\[uncertain:', text))
        low_conf = len(re.findall(r'\[low-confidence:', text))
        
        if uncertain > len(non_empty) * 0.3:
            self.issues.append(ValidationIssue(
                severity='warning',
                category='content',
                message=f'High uncertainty: {uncertain} uncertain regions'
            ))
        
        if low_conf > len(non_empty) * 0.1:
            self.issues.append(ValidationIssue(
                severity='warning',
                category='content',
                message=f'Many low-confidence regions: {low_conf}'
            ))
        
        # Check heading structure
        heading_levels = re.findall(r'^(#{1,6})\s+', text, re.MULTILINE)
        if heading_levels:
            first_level = len(heading_levels[0])
            if first_level != 1:
                self.issues.append(ValidationIssue(
                    severity='info',
                    category='content',
                    message=f'Document does not start with H1 (starts with H{first_level})'
                ))
    
    def _detect_hallucinations(self, text: str):
        """Detect potential LLM hallucinations."""
        for i, line in enumerate(text.split('\n'), 1):
            for pattern in self.compiled_hallucination:
                if pattern.search(line):
                    self.issues.append(ValidationIssue(
                        severity='error',
                        category='hallucination',
                        message='Potential hallucination detected',
                        line=i,
                        context=line[:80]
                    ))
                    break
    
    def _validate_format(self, text: str):
        """Validate Markdown formatting."""
        lines = text.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for very long lines
            if len(line) > 500 and not line.startswith('|'):
                self.issues.append(ValidationIssue(
                    severity='info',
                    category='format',
                    message=f'Very long line ({len(line)} chars)',
                    line=i
                ))
            
            # Check table formatting
            if line.strip().startswith('|') and not line.strip().endswith('|'):
                self.issues.append(ValidationIssue(
                    severity='warning',
                    category='format',
                    message='Incomplete table row',
                    line=i
                ))
    
    def _generate_report(self, text: str, metadata: Optional[Dict]) -> Dict[str, Any]:
        """Generate validation report."""
        errors = [i for i in self.issues if i.severity == 'error']
        warnings = [i for i in self.issues if i.severity == 'warning']
        infos = [i for i in self.issues if i.severity == 'info']
        
        # Calculate quality score
        base_score = 1.0
        base_score -= len(errors) * 0.15
        base_score -= len(warnings) * 0.05
        base_score -= len(infos) * 0.01
        quality_score = max(0.0, min(1.0, base_score))
        
        # Check for hallucinations
        hallucination_issues = [i for i in self.issues if i.category == 'hallucination']
        
        return {
            'is_valid': len(errors) == 0 or not self.strict,
            'quality_score': round(quality_score, 4),
            'errors': len(errors),
            'warnings': len(warnings),
            'infos': len(infos),
            'hallucination_detected': len(hallucination_issues) > 0,
            'hallucination_count': len(hallucination_issues),
            'issues': [
                {
                    'severity': i.severity,
                    'category': i.category,
                    'message': i.message,
                    'line': i.line,
                    'context': i.context
                }
                for i in self.issues
            ],
            'summary': {
                'schema_issues': len([i for i in self.issues if i.category == 'schema']),
                'content_issues': len([i for i in self.issues if i.category == 'content']),
                'format_issues': len([i for i in self.issues if i.category == 'format']),
            }
        }
    
    def get_issues(self) -> List[ValidationIssue]:
        """Get list of validation issues."""
        return self.issues


class QualityGate:
    """
    Quality gate for document processing.
    
    Enforces minimum quality thresholds and triggers fallback mechanisms.
    """
    
    def __init__(self,
                 min_quality_score: float = 0.6,
                 max_hallucinations: int = 0,
                 max_errors: int = 2,
                 min_content_lines: int = 5):
        """
        Initialize quality gate.
        
        Args:
            min_quality_score: Minimum acceptable quality score
            max_hallucinations: Maximum allowed hallucination markers
            max_errors: Maximum allowed validation errors
            min_content_lines: Minimum content lines required
        """
        self.min_quality_score = min_quality_score
        self.max_hallucinations = max_hallucinations
        self.max_errors = max_errors
        self.min_content_lines = min_content_lines
    
    def check(self, markdown_text: str, validation_report: Dict) -> Dict[str, Any]:
        """
        Check if document passes quality gate.
        
        Args:
            markdown_text: Processed Markdown
            validation_report: Output from ValidationFramework.validate()
        
        Returns:
            Gate result with pass/fail and recommendations
        """
        failures = []
        recommendations = []
        
        # Check quality score
        if validation_report['quality_score'] < self.min_quality_score:
            failures.append(f"Quality score {validation_report['quality_score']:.2f} < {self.min_quality_score}")
            recommendations.append("Consider using a different extraction engine")
        
        # Check hallucinations
        if validation_report['hallucination_count'] > self.max_hallucinations:
            failures.append(f"{validation_report['hallucination_count']} hallucinations detected")
            recommendations.append("Try local OCR instead of cloud LLM")
        
        # Check errors
        if validation_report['errors'] > self.max_errors:
            failures.append(f"{validation_report['errors']} validation errors")
            recommendations.append("Review and fix schema compliance issues")
        
        # Check content
        content_lines = len([l for l in markdown_text.split('\n') 
                           if l.strip() and not l.startswith('---') and not l.startswith('<!--')])
        if content_lines < self.min_content_lines:
            failures.append(f"Only {content_lines} content lines (min: {self.min_content_lines})")
            recommendations.append("Document may need OCR instead of text extraction")
        
        passed = len(failures) == 0
        
        return {
            'passed': passed,
            'failures': failures,
            'recommendations': recommendations,
            'suggested_action': self._suggest_action(failures, validation_report),
            'quality_level': self._get_quality_level(validation_report['quality_score'])
        }
    
    def _suggest_action(self, failures: List[str], report: Dict) -> str:
        """Suggest action based on failures."""
        if not failures:
            return 'accept'
        
        if report.get('hallucination_count', 0) > 0:
            return 'fallback_to_local_ocr'
        
        if report['quality_score'] < 0.3:
            return 'reject_and_retry'
        
        if report['quality_score'] < 0.5:
            return 'fallback_to_alternative'
        
        return 'accept_with_warnings'
    
    def _get_quality_level(self, score: float) -> str:
        """Classify quality level."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'good'
        elif score >= 0.6:
            return 'acceptable'
        elif score >= 0.4:
            return 'poor'
        else:
            return 'unacceptable'


def validate_output(markdown_text: str, metadata: Optional[Dict] = None,
                   strict: bool = False) -> Dict[str, Any]:
    """
    Validate Markdown output.
    
    Args:
        markdown_text: Content to validate
        metadata: Optional metadata
        strict: Strict mode
    
    Returns:
        Validation report
    """
    framework = ValidationFramework(strict=strict)
    return framework.validate(markdown_text, metadata)


def check_quality_gate(markdown_text: str, 
                       validation_report: Optional[Dict] = None,
                       **thresholds) -> Dict[str, Any]:
    """
    Check quality gate.
    
    Args:
        markdown_text: Content to check
        validation_report: Pre-computed validation report
        **thresholds: Override default thresholds
    
    Returns:
        Quality gate result
    """
    if validation_report is None:
        validation_report = validate_output(markdown_text)
    
    gate = QualityGate(**thresholds)
    return gate.check(markdown_text, validation_report)


# Example usage
if __name__ == "__main__":
    sample = """---
document:
  source_file: "test.pdf"
  document_id: "abc123"
quality:
  confidence_score: 0.85
---

<!-- page:1 -->

<!-- role:heading level:1 -->
# Introduction

This is sample content. Based on the image, it shows a document.

<!-- role:table -->
| A | B |
|---|---|
| 1 | 2
"""
    
    report = validate_output(sample)
    print("Validation Report:")
    print(f"  Valid: {report['is_valid']}")
    print(f"  Quality: {report['quality_score']}")
    print(f"  Errors: {report['errors']}")
    print(f"  Hallucinations: {report['hallucination_count']}")
    
    gate_result = check_quality_gate(sample, report)
    print(f"\nQuality Gate:")
    print(f"  Passed: {gate_result['passed']}")
    print(f"  Action: {gate_result['suggested_action']}")
    for f in gate_result['failures']:
        print(f"  - {f}")
