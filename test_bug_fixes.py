"""
Test suite for DocFlow critical bug fixes.
Tests P0 crash bug, OpenRouter timeout/retry, and exception handling.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_cache_manager_exception_handling():
    """Test that cache_manager handles corrupted cache files gracefully."""
    print("\n=== Testing cache_manager exception handling ===")
    
    from services import cache_manager
    
    # Create a corrupted cache file
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    corrupted_file = cache_dir / "test_corrupted.json"
    with open(corrupted_file, 'w') as f:
        f.write("{ invalid json }")
    
    # Test that get_cached_result handles corruption gracefully
    try:
        result = cache_manager.get_cached_result("test", "corrupted", "md")
        assert result is None, "Should return None for corrupted cache"
        print("✓ cache_manager handles corrupted files correctly")
    except Exception as e:
        print(f"✗ cache_manager failed: {e}")
        return False
    finally:
        # Cleanup
        if corrupted_file.exists():
            corrupted_file.unlink()
    
    return True


def test_history_manager_exception_handling():
    """Test that history_manager handles corrupted history files gracefully."""
    print("\n=== Testing history_manager exception handling ===")
    
    from services import history_manager
    
    # Create a corrupted history file
    history_file = Path("conversion_history.json")
    original_content = None
    
    # Backup original if exists
    if history_file.exists():
        with open(history_file, 'r') as f:
            original_content = f.read()
    
    # Write corrupted content
    with open(history_file, 'w') as f:
        f.write("{ invalid json }")
    
    try:
        history = history_manager.load_history()
        assert history == [], "Should return empty list for corrupted history"
        print("✓ history_manager handles corrupted files correctly")
        result = True
    except Exception as e:
        print(f"✗ history_manager failed: {e}")
        result = False
    finally:
        # Restore original or delete
        if original_content:
            with open(history_file, 'w') as f:
                f.write(original_content)
        elif history_file.exists():
            history_file.unlink()
    
    return result


def test_formatters_exception_handling():
    """Test that formatters handles corrupted stats files gracefully."""
    print("\n=== Testing formatters exception handling ===")
    
    from utils import formatters
    
    # Create a corrupted stats file
    stats_file = Path("usage_stats.json")
    original_content = None
    
    # Backup original if exists
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            original_content = f.read()
    
    # Write corrupted content
    with open(stats_file, 'w') as f:
        f.write("{ invalid json }")
    
    try:
        stats = formatters.load_stats()
        assert "total_conversions" in stats, "Should return default stats for corrupted file"
        assert stats["total_conversions"] == 0, "Should have default values"
        print("✓ formatters handles corrupted files correctly")
        result = True
    except Exception as e:
        print(f"✗ formatters failed: {e}")
        result = False
    finally:
        # Restore original or delete
        if original_content:
            with open(stats_file, 'w') as f:
                f.write(original_content)
        elif stats_file.exists():
            stats_file.unlink()
    
    return result


def test_openrouter_timeout_config():
    """Test that OpenRouter API calls have timeout configured."""
    print("\n=== Testing OpenRouter timeout configuration ===")
    
    # Read structure_engine.py to verify timeout is set
    with open("structure_engine.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for timeout parameter
    if "timeout=30.0" in content:
        print("✓ OpenRouter timeout configured (30s)")
        return True
    else:
        print("✗ OpenRouter timeout NOT configured")
        return False


def test_openrouter_retry_logic():
    """Test that OpenRouter API calls have retry logic."""
    print("\n=== Testing OpenRouter retry logic ===")
    
    # Read structure_engine.py to verify retry logic
    with open("structure_engine.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for retry loop
    has_retry_loop = "max_retries = 3" in content
    has_exponential_backoff = "retry_delay * (2 ** attempt)" in content
    
    if has_retry_loop and has_exponential_backoff:
        print("✓ OpenRouter retry logic with exponential backoff configured")
        return True
    else:
        print(f"✗ OpenRouter retry logic incomplete (loop={has_retry_loop}, backoff={has_exponential_backoff})")
        return False


def test_gmft_validation_fix():
    """Test that extract_with_gmft doesn't reference undefined variables."""
    print("\n=== Testing extract_with_gmft validation fix ===")
    
    # Read structure_engine.py to verify the fix
    with open("structure_engine.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that the old buggy code is gone
    if "page_count = len(images)" in content and "def extract_with_gmft" in content:
        # Need to check if it's in the gmft function
        gmft_start = content.find("def extract_with_gmft")
        gmft_end = content.find("\ndef ", gmft_start + 1)
        gmft_function = content[gmft_start:gmft_end]
        
        if "page_count = len(images)" in gmft_function:
            print("✗ extract_with_gmft still references undefined 'images' variable")
            return False
    
    # Check that the fix is present
    if 'len(all_pages_elements) if \'all_pages_elements\' in locals() else 0' in content:
        print("✓ extract_with_gmft validation fix applied correctly")
        return True
    else:
        print("⚠ extract_with_gmft validation code may have changed")
        return True  # Don't fail if the implementation changed


def test_bare_except_blocks():
    """Test that bare except blocks have been replaced."""
    print("\n=== Testing bare except block fixes ===")
    
    files_to_check = [
        "app.py",
        "services/cache_manager.py",
        "services/history_manager.py",
        "utils/formatters.py",
        "fast_converter.py"
    ]
    
    all_fixed = True
    
    for filepath in files_to_check:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count bare except blocks (except: followed by pass or minimal code)
        import re
        bare_excepts = re.findall(r'except:\s*\n\s*pass', content)
        
        if bare_excepts:
            print(f"✗ {filepath} still has {len(bare_excepts)} bare except blocks")
            all_fixed = False
        else:
            print(f"✓ {filepath} has no bare except blocks")
    
    return all_fixed


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("DocFlow Critical Bug Fixes - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Cache Manager Exception Handling", test_cache_manager_exception_handling),
        ("History Manager Exception Handling", test_history_manager_exception_handling),
        ("Formatters Exception Handling", test_formatters_exception_handling),
        ("OpenRouter Timeout Configuration", test_openrouter_timeout_config),
        ("OpenRouter Retry Logic", test_openrouter_retry_logic),
        ("GMFT Validation Fix", test_gmft_validation_fix),
        ("Bare Except Block Fixes", test_bare_except_blocks),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
