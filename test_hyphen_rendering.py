"""
Test script to reproduce the hyphen-to-code rendering issue.

This script tests how different dash/hyphen characters are being processed
and rendered in the Markdown output.
"""

import re

# Sample text with different dash characters
test_cases = [
    ("Regular hyphen", "- Item 1\n- Item 2"),
    ("Minus sign (Unicode)", "− Item 1\n− Item 2"),
    ("Indented minus sign", "  − Item 1\n  − Item 2"),
    ("4-space indented minus", "    − Item 1\n    − Item 2"),
    ("Mixed dashes", "- Item 1\n− Item 2\n– Item 3"),
]

print("=" * 60)
print("HYPHEN RENDERING TEST")
print("=" * 60)

for name, text in test_cases:
    print(f"\n{name}:")
    print(f"Input: {repr(text)}")
    print(f"Rendered:\n{text}")
    print(f"Unicode codepoints: {[hex(ord(c)) for c in text if c in '-−–—']}")
    
    # Check if it would be interpreted as code block
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('    '):
            print(f"  ⚠️ Line {i+1} starts with 4 spaces - will render as CODE BLOCK")
        elif line.startswith('  '):
            print(f"  ℹ️ Line {i+1} starts with 2 spaces - indented but not code")

print("\n" + "=" * 60)
print("NORMALIZATION TEST")
print("=" * 60)

# Test the normalization from cleaner.py
def normalize_dashes(text):
    """Convert all Unicode dash variants to standard hyphen."""
    # Unicode dash characters to normalize
    dash_chars = {
        '\u2212': '-',  # MINUS SIGN
        '\u2013': '-',  # EN DASH
        '\u2014': '-',  # EM DASH
        '\u2015': '-',  # HORIZONTAL BAR
    }
    
    for unicode_dash, standard_dash in dash_chars.items():
        text = text.replace(unicode_dash, standard_dash)
    
    return text

for name, text in test_cases:
    normalized = normalize_dashes(text)
    print(f"\n{name}:")
    print(f"Before: {repr(text)}")
    print(f"After:  {repr(normalized)}")
    if text != normalized:
        print(f"✅ Changed")
    else:
        print(f"⏭️ No change")

print("\n" + "=" * 60)
print("INDENTATION REMOVAL TEST")
print("=" * 60)

def remove_list_indentation(text):
    """Remove leading spaces before list markers."""
    # Match lines that start with spaces followed by a list marker
    pattern = r'^(\s+)([-−–—*+])\s'
    return re.sub(pattern, r'\2 ', text, flags=re.MULTILINE)

for name, text in test_cases:
    cleaned = remove_list_indentation(text)
    print(f"\n{name}:")
    print(f"Before: {repr(text)}")
    print(f"After:  {repr(cleaned)}")
    if text != cleaned:
        print(f"✅ Indentation removed")
    else:
        print(f"⏭️ No indentation to remove")
