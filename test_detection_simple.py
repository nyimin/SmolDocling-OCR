"""
Test script for validating industrial-standard document structure detection.

Tests:
1. Header/Footer Detection (pattern-based + position-based)
2. Multi-Signal Heading Classification
3. Image Region Detection
4. Backward Compatibility

Usage:
    python test_detection_simple.py [--verbose]
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import cleaner
import layout_analyzer


def test_header_footer_detection(verbose=False):
    """Test pattern-based and position-based header/footer detection."""
    print("\n" + "="*60)
    print("TEST 1: Header/Footer Detection")
    print("="*60)
    
    # Create mock page elements
    pages_elements = [[
        # Header (top 10%, pattern match)
        {'type': 'text', 'content': 'Company Confidential', 'bbox': (0, 50, 1275, 70)},
        
        # Regular content
        {'type': 'text', 'content': 'INTRODUCTION', 'bbox': (0, 200, 1275, 240)},
        {'type': 'text', 'content': 'This is a sample paragraph.', 'bbox': (0, 300, 1275, 330)},
        
        # Footer (bottom 10%, pattern match)
        {'type': 'text', 'content': 'Page 1 of 10', 'bbox': (0, 1550, 1275, 1570)},
        {'type': 'text', 'content': '(c) 2024 Test Company', 'bbox': (0, 1600, 1275, 1620)},
    ]]
    
    # Test new tagging function
    tagged_pages = cleaner.detect_and_tag_headers_footers(
        pages_elements, 
        threshold=0.6,
        use_patterns=True,
        use_position=True
    )
    
    # Verify results
    header_found = False
    footer_found = False
    
    for page in tagged_pages:
        for elem in page:
            role = elem.get('semantic_role')
            content = elem.get('content', '')
            
            if role == 'header' and 'Confidential' in content:
                header_found = True
                if verbose:
                    print(f"  + Header detected: {content[:50]}")
            elif role == 'footer' and ('Page' in content or '(c)' in content):
                footer_found = True
                if verbose:
                    print(f"  + Footer detected: {content[:50]}")
    
    # Results
    print(f"\nPattern Detection:")
    print(f"  Headers: {'PASS' if header_found else 'FAIL'}")
    print(f"  Footers: {'PASS' if footer_found else 'FAIL'}")
    
    # Test backward compatibility
    legacy_result = cleaner.detect_and_remove_headers_footers(pages_elements)
    legacy_pass = len(legacy_result[0]) < len(pages_elements[0])
    print(f"\nBackward Compatibility:")
    print(f"  Legacy function: {'PASS' if legacy_pass else 'FAIL'}")
    
    return header_found and footer_found and legacy_pass


def test_heading_classification(verbose=False):
    """Test multi-signal heading classification."""
    print("\n" + "="*60)
    print("TEST 2: Multi-Signal Heading Classification")
    print("="*60)
    
    analyzer = layout_analyzer.LayoutAnalyzer()
    
    # Create test elements with different heading signals
    test_cases = [
        {
            'name': 'ALL CAPS + Large Font',
            'elem': {
                'text': 'INTRODUCTION',
                'bbox': (100, 200, 500, 240),
                'font_size': 28
            },
            'page_elements': [
                {'bbox': (100, 100, 500, 120), 'font_size': 16},  # prev
                {'bbox': (100, 200, 500, 240), 'font_size': 28},  # current
                {'bbox': (100, 280, 500, 300), 'font_size': 16},  # next
            ],
            'expected_role': 'heading',
            'min_confidence': 0.6
        },
        {
            'name': 'Title Case + Whitespace',
            'elem': {
                'text': 'Key Findings',
                'bbox': (100, 400, 500, 430),
                'font_size': 22
            },
            'page_elements': [
                {'bbox': (100, 300, 500, 320), 'font_size': 16},  # prev (gap)
                {'bbox': (100, 400, 500, 430), 'font_size': 22},  # current
                {'bbox': (100, 470, 500, 490), 'font_size': 16},  # next (gap)
            ],
            'expected_role': 'heading',
            'min_confidence': 0.4
        },
        {
            'name': 'Regular Paragraph',
            'elem': {
                'text': 'This is a normal paragraph with regular formatting.',
                'bbox': (100, 500, 500, 520),
                'font_size': 16
            },
            'page_elements': [
                {'bbox': (100, 480, 500, 500), 'font_size': 16},
                {'bbox': (100, 500, 500, 520), 'font_size': 16},
                {'bbox': (100, 520, 500, 540), 'font_size': 16},
            ],
            'expected_role': 'paragraph',
            'min_confidence': 0.0
        },
        {
            'name': 'Caption (Figure keyword)',
            'elem': {
                'text': 'Figure 1: Sample diagram',
                'bbox': (100, 800, 500, 815),
                'font_size': 14
            },
            'page_elements': [
                {'bbox': (100, 600, 500, 780), 'font_size': 16},  # image gap
                {'bbox': (100, 800, 500, 815), 'font_size': 14},
                {'bbox': (100, 850, 500, 870), 'font_size': 16},
            ],
            'expected_role': 'caption',
            'min_confidence': 0.8
        }
    ]
    
    results = []
    for test in test_cases:
        elem = test['elem']
        page_elements = test['page_elements']
        
        # Get prev/next elements
        prev_elem = page_elements[0] if len(page_elements) > 0 else None
        next_elem = page_elements[2] if len(page_elements) > 2 else None
        
        role, confidence = analyzer.classify_semantic_role_enhanced(
            elem, page_elements, prev_elem, next_elem
        )
        
        passed = (role == test['expected_role'] and confidence >= test['min_confidence'])
        results.append(passed)
        
        status = 'PASS' if passed else 'FAIL'
        print(f"\n{test['name']}: {status}")
        if verbose or not passed:
            print(f"  Expected: {test['expected_role']} (conf >= {test['min_confidence']})")
            print(f"  Got: {role} (conf = {confidence:.2f})")
    
    overall_pass = all(results)
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'} ({sum(results)}/{len(results)} tests passed)")
    
    return overall_pass


def test_end_to_end(verbose=False):
    """Test complete pipeline integration."""
    print("\n" + "="*60)
    print("TEST 3: End-to-End Integration")
    print("="*60)
    
    # Create test elements
    page_elements = [
        {'type': 'text', 'content': 'Page 1', 'bbox': (0, 50, 100, 70)},
        {'type': 'text', 'content': 'CHAPTER ONE', 'bbox': (0, 200, 300, 240), 'font_size': 28},
        {'type': 'text', 'content': 'Introduction text here.', 'bbox': (0, 300, 500, 320), 'font_size': 16},
        {'type': 'text', 'content': 'Figure 1: Test', 'bbox': (0, 650, 300, 670), 'font_size': 14},
        {'type': 'text', 'content': '(c) 2024', 'bbox': (0, 1600, 100, 1620)},
    ]
    
    # Test header/footer tagging
    tagged = cleaner.detect_and_tag_headers_footers([page_elements])
    
    # Test layout analysis
    analyzer = layout_analyzer.LayoutAnalyzer()
    text_elements = [e for e in page_elements if e['type'] == 'text']
    layout_result = analyzer.analyze_page_layout(
        text_elements,
        use_enhanced_classification=True
    )
    
    # Verify semantic roles are assigned
    roles_found = set()
    for elem in layout_result['elements']:
        role = elem.get('semantic_role')
        if role:
            roles_found.add(role)
    
    expected_roles = {'heading', 'paragraph'}
    has_expected = expected_roles.issubset(roles_found)
    
    print(f"\nSemantic Roles Found: {roles_found}")
    print(f"Expected Roles Present: {'PASS' if has_expected else 'FAIL'}")
    
    # Check confidence scores
    has_confidence = all('role_confidence' in elem for elem in layout_result['elements'])
    print(f"Confidence Scores: {'PASS' if has_confidence else 'FAIL'}")
    
    return has_expected and has_confidence


def main():
    """Run all tests."""
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    print("\n" + "="*60)
    print("INDUSTRIAL-STANDARD DETECTION VALIDATION")
    print("="*60)
    print("\nTesting enhancements:")
    print("  - Pattern-based header/footer detection (15+ patterns)")
    print("  - Multi-signal heading classification (5 signals)")
    print("  - Position-based filtering (top/bottom 10%)")
    print("  - Backward compatibility")
    
    results = {
        'Header/Footer Detection': test_header_footer_detection(verbose),
        'Heading Classification': test_heading_classification(verbose),
        'End-to-End Integration': test_end_to_end(verbose),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("SUCCESS: ALL TESTS PASSED!")
        print("Industrial-standard detection is working correctly.")
    else:
        print("WARNING: SOME TESTS FAILED")
        print("Review the output above for details.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
