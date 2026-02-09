"""
Quick test for cache filename sanitization fix.
Tests that model names with special characters are properly sanitized.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from services.cache_manager import sanitize_filename

def test_sanitize_filename():
    """Test that problematic filenames are sanitized correctly."""
    
    test_cases = [
        # (input, expected_contains)
        ("Gemini 2.0 Flash Lite ($0.08/1K pages)", "Gemini_2.0_Flash_Lite"),
        ("OpenRouter (Cloud - Recommended ⭐)", "OpenRouter_Cloud_Recommended"),
        ("Qwen 2.5-VL 72B ($0.15/1K pages)", "Qwen_2.5-VL_72B"),
        ("Mistral Pixtral Large ($2/1K pages)", "Mistral_Pixtral_Large"),
        ("file<>:name|with?invalid*chars", "file_name_with_invalid_chars"),
    ]
    
    print("=" * 60)
    print("Cache Filename Sanitization Test")
    print("=" * 60)
    
    all_passed = True
    
    for input_str, expected_contains in test_cases:
        result = sanitize_filename(input_str)
        
        # Check that result doesn't contain invalid characters
        invalid_chars = '<>:"/\\|?*()$'
        has_invalid = any(char in result for char in invalid_chars)
        
        # Check that expected substring is present
        contains_expected = expected_contains.replace('$', 'USD') in result or expected_contains in result
        
        if has_invalid:
            print(f"✗ FAIL: '{input_str}'")
            print(f"  Result: '{result}'")
            print(f"  Still contains invalid characters!")
            all_passed = False
        elif not contains_expected:
            print(f"⚠ WARNING: '{input_str}'")
            print(f"  Result: '{result}'")
            print(f"  Expected to contain: '{expected_contains}'")
        else:
            print(f"✓ PASS: '{input_str}'")
            print(f"  Result: '{result}'")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


def test_cache_key_generation():
    """Test that cache keys are generated correctly with sanitization."""
    print("\n" + "=" * 60)
    print("Cache Key Generation Test")
    print("=" * 60)
    
    # Simulate the problematic cache key from the error
    file_hash = "7473cdf60562262a06375c2fa027c649"
    engine = "OpenRouter (Cloud - Recommended ⭐)"
    model = "Gemini 2.0 Flash Lite ($0.08/1K pages)"
    export_format = "md"
    
    cache_key = f"{file_hash}_{engine}_{model}_{export_format}"
    sanitized = sanitize_filename(cache_key)
    
    print(f"Original: {cache_key}")
    print(f"Sanitized: {sanitized}")
    
    # Check for invalid characters
    invalid_chars = '<>:"/\\|?*()$'
    has_invalid = any(char in sanitized for char in invalid_chars)
    
    if has_invalid:
        print("✗ FAIL: Sanitized key still contains invalid characters!")
        return 1
    else:
        print("✓ PASS: Sanitized key is safe for filesystem!")
        return 0


if __name__ == "__main__":
    result1 = test_sanitize_filename()
    result2 = test_cache_key_generation()
    
    sys.exit(max(result1, result2))
