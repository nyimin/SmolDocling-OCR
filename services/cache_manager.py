"""
Cache management service for DocFlow.
Handles file hashing, cache storage, and retrieval.
"""

import os
import json
import hashlib
from datetime import datetime

# Constants
CACHE_DIR = "cache"

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)


def get_file_hash(file_path):
    """Generate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while buf:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def sanitize_filename(filename):
    """
    Sanitize a string to be safe for use as a filename.
    Removes or replaces characters that are invalid on Windows/Unix filesystems.
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Replace other problematic characters
    filename = filename.replace('$', 'USD')  # Dollar signs
    filename = filename.replace('(', '_')
    filename = filename.replace(')', '_')
    filename = filename.replace(' ', '_')
    filename = filename.replace(',', '_')
    
    # Remove consecutive underscores
    while '__' in filename:
        filename = filename.replace('__', '_')
    
    # Trim underscores from start/end
    filename = filename.strip('_')
    
    # Limit length to avoid path length issues (Windows has 260 char limit)
    max_length = 200
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    return filename


def get_cached_result(file_hash, cache_key, export_format):
    """Retrieve cached conversion result if available."""
    full_cache_key = f"{file_hash}_{cache_key}_{export_format}"
    full_cache_key = sanitize_filename(full_cache_key)  # Sanitize to avoid invalid filename characters
    cache_path = os.path.join(CACHE_DIR, f"{full_cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            # Cache file is corrupted or unreadable - log and return None
            print(f"Warning: Failed to read cache file {cache_path}: {type(e).__name__} - {e}")
            return None
    return None


def save_to_cache(file_hash, cache_key, export_format, markdown_text, method_used):
    """Save conversion result to cache."""
    full_cache_key = f"{file_hash}_{cache_key}_{export_format}"
    full_cache_key = sanitize_filename(full_cache_key)  # Sanitize to avoid invalid filename characters
    cache_path = os.path.join(CACHE_DIR, f"{full_cache_key}.json")
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump({
            "markdown": markdown_text,
            "method": method_used,
            "cached_at": datetime.now().isoformat()
        }, f)


def clear_cache():
    """Clear all cached files."""
    if os.path.exists(CACHE_DIR):
        for file in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        return f"Cache cleared successfully"
    return "Cache directory not found"


def get_cache_size():
    """Get total size of cache directory in MB."""
    total_size = 0
    if os.path.exists(CACHE_DIR):
        for file in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size / (1024 * 1024)  # Convert to MB
