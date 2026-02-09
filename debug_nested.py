"""Quick debug script to test the nested list behavior."""

from cleaner import normalize_markdown

# Test case that's failing
input_text = "- Item 1\n  - Nested item\n- Item 2"
result = normalize_markdown(input_text)

print("Input:")
print(repr(input_text))
print("\nOutput:")
print(repr(result))
print("\nRendered:")
print(result)
print("\nChecking for '  - Nested':")
print(f"Found: {'  - Nested' in result}")
print(f"Actual nested line: {repr([line for line in result.split('\\n') if 'Nested' in line])}")
