# Changelog

All notable changes to this project will be documented in this file.

## [v2.1.1] - 2026-02-09

### 🔧 Fixes

- **Hyphen Rendering Fix**: Fixed issue where lists with Unicode minus signs (`−`) and indentation were rendered as code blocks. Now explicitly converts Unicode dashes to standard hyphens.
- **List Normalization**: Implemented intelligent indent cleanup (removing 2-4 space top-level indentation) to prevent accidental code block rendering while preserving nested list structure.
- **Digital PDF Consistency**: Applied `normalize_markdown` to the `pymupdf4llm` extraction path, ensuring consistent formatting rules for both digital and scanned documents.

## [v2.1.0] - 2026-02-09

### 🚀 pymupdf4llm Migration

- **Simplified Architecture**: Replaced GMFT with `pymupdf4llm` for digital PDF extraction, reducing code complexity from ~140 lines to ~40 lines while improving quality.
- **LLM-Optimized Extraction**: Digital PDFs now use `pymupdf4llm` which is specifically optimized for LLM/RAG workflows with superior table handling and document structure preservation.
- **Auto-Detection**: The "Local OCR" mode now automatically detects digital vs scanned PDFs:
  - **Digital PDFs**: Uses `pymupdf4llm` for fast, high-quality extraction
  - **Scanned PDFs**: Falls back to RapidOCR with layout-aware processing

### ✨ UI Improvements

- **Clearer Labels**: Updated OCR engine dropdown to "Cloud OCR (OpenRouter)" and "Local OCR (Auto-Detect Digital/Scan)" to better communicate the workflow.
- **Informative Settings**: Added explanatory text in the Local OCR settings section clarifying that digital PDFs use pymupdf4llm automatically.
- **Progress Display Fix**: Fixed the nested progress callback issue that caused progress indicators to appear in multiple windows simultaneously.

### 🔧 Technical Changes

- **Dependencies**: Removed `gmft` dependency, reducing installation size and complexity.
- **Function Signatures**: Updated `toggle_settings()` to match new dropdown labels.
- **Progress Handling**: Simplified progress callback passing to prevent UI clutter.

---

## [v2.0.0] - 2026-02-09

### 🚀 High-Fidelity Pipeline (Phase 1-4)

- **Tag-Don't-Remove Strategy**: Shifted from destructive cleaning to semantic tagging. Noise elements (headers, footers, watermarks) are now preserved and tagged (e.g., `<!-- role:header -->`) for downstream filtering, ensuring 100% content fidelity.
- **Normalization Engine**: Added `cleaner.normalize_markdown` to deterministically standardize spacing, list bullets, and line breaks before validation.
- **Data-Driven Validation**: Validation now uses PDF text layer word counts (when available) for accurate "Completeness" scoring, replacing generic page-based heuristics.
- **OpenRouter Integration**: Full support for OpenRouter Vision models (Qwen 2.5-VL, Nemotron) with "Balanced", "Quality", and "Fast" presets.
- **Myanmar Support**: Verified OCR support for Myanmar language via OpenRouter models.

### ✨ Improvements

- **Prompt Engineering**: Updated System Prompt to enforce strict Markdown styling (lists, tables, spacing).
- **Metadata Extraction**: Enhanced `metadata_extractor.py` to count words in PDF text layers.
- **Validation Rules**: Whitelisted `header`, `footer`, `page_number` roles to prevent false positives in quality reports.
- **Input Resolution**: Increased PDF rendering DPI to 300 for higher OCR accuracy.

### 🔧 Fixes

- Fixed hyphenation merging to be less aggressive.
- Fixed page number misclassification by using spatial heuristics.
- Fixed validation errors for "invalid roles" on legitimate noise elements.
