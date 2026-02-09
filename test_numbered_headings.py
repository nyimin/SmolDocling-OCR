"""Test for numbered heading vs numbered list distinction"""
import sys
sys.path.insert(0, '.')

import layout_analyzer

# Create analyzer
a = layout_analyzer.LayoutAnalyzer()

# Test cases
test_cases = [
    # Short numbered items (should be list_item)
    {'text': '1. Buy milk', 'bbox': (0, 0, 100, 20)},
    {'text': '10. Call dentist', 'bbox': (0, 0, 100, 20)},
    {'text': '2. Review document', 'bbox': (0, 0, 100, 20)},
    
    # Long numbered items (should be heading)
    {'text': '1. For investments in sectors listed in a notification to be issued by the Commission', 'bbox': (0, 0, 400, 30)},
    {'text': '2. Income tax exemptions shall only be granted to sectors that the Commission has specified', 'bbox': (0, 0, 400, 30)},
    {'text': '10. Right to deduct expenses from assessable income incurred for research and development', 'bbox': (0, 0, 400, 30)},
]

page_elems = [{'bbox': (0, 0, 100, 20)}]

print("Numbered Heading vs List Item Test:")
print("=" * 80)
for tc in test_cases:
    role, conf = a.classify_semantic_role_enhanced(tc, page_elems)
    text_preview = tc['text'][:60] + "..." if len(tc['text']) > 60 else tc['text']
    expected = "list_item" if len(tc['text']) < 80 else "heading"
    status = "OK" if role == expected else "FAIL"
    print(f"{status:4} | {text_preview:65} -> {role:12} (conf={conf:.2f})")

print("=" * 80)
