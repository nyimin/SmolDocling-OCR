"""Quick test for list detection improvements"""
import sys
sys.path.insert(0, '.')

import layout_analyzer

# Create analyzer
a = layout_analyzer.LayoutAnalyzer()

# Test cases
test_cases = [
    {'text': '1. First item', 'bbox': (0, 0, 100, 20)},
    {'text': '10. Tenth item', 'bbox': (0, 0, 100, 20)},
    {'text': '100. Hundredth item', 'bbox': (0, 0, 100, 20)},
    {'text': 'a. Lettered list', 'bbox': (0, 0, 100, 20)},
    {'text': '- Bullet point', 'bbox': (0, 0, 100, 20)},
    {'text': 'HEADING TEXT', 'bbox': (0, 0, 100, 30)},
    {'text': 'Regular paragraph text', 'bbox': (0, 0, 100, 20)},
]

page_elems = [{'bbox': (0, 0, 100, 20)}]

print("List Detection Test Results:")
print("=" * 60)
for tc in test_cases:
    role, conf = a.classify_semantic_role_enhanced(tc, page_elems)
    status = "OK" if (role == 'list_item' and tc['text'][0] in '1a-') or (role != 'list_item' and tc['text'][0] not in '1a-') else "FAIL"
    print(f"{status:4} | {tc['text'][:25]:25} -> {role:12} (conf={conf:.2f})")

print("=" * 60)
