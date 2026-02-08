from markitdown import MarkItDown
import os

def convert_fast(file_path):
    """
    Converts a file to Markdown using Microsoft MarkItDown.
    This is extremely fast but does not perform OCR or complex layout analysis.
    """
    try:
        md = MarkItDown()
        result = md.convert(file_path)
        return result.text_content
    except Exception as e:
        return f"Error in Fast Mode: {str(e)}"
