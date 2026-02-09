"""
History management service for DocFlow.
Handles conversion history tracking and display.
"""

import os
import json
from datetime import datetime

# Constants
HISTORY_FILE = "conversion_history.json"
MAX_HISTORY = 10


def load_history():
    """Load conversion history from file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            # History file is corrupted or unreadable - log and return empty list
            print(f"Warning: Failed to load history file {HISTORY_FILE}: {type(e).__name__} - {e}")
            return []
    return []


def save_history(entry):
    """Save a new conversion entry to history."""
    history = load_history()
    history.insert(0, entry)
    history = history[:MAX_HISTORY]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


def get_history_display():
    """Generate formatted history display for UI."""
    history = load_history()
    if not history:
        return "**Recent Conversions**\n\nNo history yet."
    
    lines = ["**Recent Conversions**\n"]
    for entry in history[:5]:
        timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%m/%d %H:%M')
        lines.append(f"- **{entry['filename']}** ({timestamp})")
        lines.append(f"  {entry['words']:,} words • {entry['method']}")
    
    return "\n".join(lines)


def clear_history():
    """Clear all conversion history."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return "History cleared"
