"""
Integration test for list normalization fixes.

Tests:
1. Unicode dash normalization (− → -)
2. Indented list cleanup (remove 2-4 space indents)
3. Combined fix (indented Unicode dashes)
"""

from cleaner import normalize_markdown

def test_unicode_dash_normalization():
    """Test that Unicode dash variants are converted to standard hyphens."""
    input_text = "− Item 1\n− Item 2\n– Item 3\n— Item 4"
    expected = "- Item 1\n- Item 2\n- Item 3\n- Item 4"
    result = normalize_markdown(input_text)
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✅ Unicode dash normalization test passed")

def test_indented_list_cleanup():
    """Test that excessive indentation is removed from list items."""
    input_text = "    - Item 1\n    - Item 2"
    result = normalize_markdown(input_text)
    # Should NOT start with 4 spaces (would be code block)
    assert not result.startswith('    '), f"List should not have 4-space indentation. Got: {repr(result)}"
    # Should be proper list
    assert result.startswith('- '), f"Should start with hyphen bullet. Got: {repr(result)}"
    print("✅ Indented list cleanup test passed")

def test_combined_fix():
    """Test the real-world scenario: indented Unicode minus signs."""
    # This is what causes the code block rendering issue
    input_text = "    − Goods imported or exported\n    − Motor vehicles imported"
    result = normalize_markdown(input_text)
    
    # Should convert Unicode minus to hyphen
    assert '−' not in result, f"Unicode minus should be converted. Got: {repr(result)}"
    
    # Should remove 4-space indentation
    assert not result.startswith('    '), f"Should not have 4-space indent. Got: {repr(result)}"
    
    # Should be proper Markdown list
    lines = result.split('\n')
    for line in lines:
        if line.strip():
            assert line.startswith('- '), f"Each line should start with '- '. Got: {repr(line)}"
    
    print("✅ Combined fix test passed")
    print(f"   Input:  {repr(input_text)}")
    print(f"   Output: {repr(result)}")

if __name__ == "__main__":
    print("=" * 60)
    print("LIST NORMALIZATION INTEGRATION TESTS")
    print("=" * 60)
    print()
    
    try:
        test_unicode_dash_normalization()
        test_indented_list_cleanup()
        test_combined_fix()
        
        print()
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("📝 Note: This fix removes 2-4 space indentation from lists")
        print("   to prevent code-block rendering. Nested lists may need")
        print("   manual adjustment after extraction.")
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise
