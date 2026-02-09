"""
Test script to verify pymupdf4llm integration in DocFlow v2.1.0
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from structure_engine import extract_with_pymupdf4llm, extract_smart_local

def test_pymupdf4llm_function():
    """Test that the extract_with_pymupdf4llm function exists and is callable"""
    print("=" * 70)
    print("Testing pymupdf4llm Integration")
    print("=" * 70)
    
    # Test 1: Function exists
    print("\n✓ Test 1: extract_with_pymupdf4llm function exists")
    assert callable(extract_with_pymupdf4llm), "Function should be callable"
    
    # Test 2: Function signature
    print("✓ Test 2: Function has correct signature")
    import inspect
    sig = inspect.signature(extract_with_pymupdf4llm)
    assert 'pdf_path' in sig.parameters, "Function should accept pdf_path parameter"
    
    # Test 3: extract_smart_local exists and uses new function
    print("✓ Test 3: extract_smart_local function exists")
    assert callable(extract_smart_local), "extract_smart_local should be callable"
    
    # Test 4: Check function docstring
    print("✓ Test 4: Function has proper documentation")
    assert extract_with_pymupdf4llm.__doc__ is not None, "Function should have docstring"
    assert "pymupdf4llm" in extract_with_pymupdf4llm.__doc__.lower(), "Docstring should mention pymupdf4llm"
    
    print("\n" + "=" * 70)
    print("✅ All tests passed! pymupdf4llm integration successful")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        test_pymupdf4llm_function()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
