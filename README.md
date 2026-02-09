# DocFlow: Effortless PDF & Image to Markdown Converter

A powerful, optimized application that intelligently converts PDFs and images to clean, structured Markdown. It features a **Hybrid Pipeline** that automatically selects the best tool for the job:

- **🌐 Cloud OCR (OpenRouter)**: High-quality OCR using vision models. Recommended for difficult documents and 100+ languages including **Myanmar** 🇲🇲.
- **🤖 Local OCR (Auto-Detect)**: **(NEW v2.1)** Automatically detects digital vs scanned PDFs:
  - **Digital PDFs**: Uses `pymupdf4llm` for LLM-optimized extraction with superior table handling
  - **Scanned PDFs**: Falls back to RapidOCR with layout-aware processing
- **✨ Smart Cleaning**: "Tag-Don't-Remove" strategy: Tags noise (headers, footers, watermarks) with semantic roles instead of deleting them, ensuring 100% content preservation for high-fidelity RAG.

---

## ✨ Features

- **RAG-Ready Output**: **(NEW)** Generates Markdown enriched with semantic annotations (headings, tables, lists, captions) and page markers for optimal chunking and retrieval.
- **Smart Formatting**: **(FIX)** Automatically normalizes Unicode characters and fixes list indentation to ensure clean, valid Markdown output.
- **Layout-Aware OCR**: **(NEW)** Intelligent column detection and XY-cut reading order for multi-column documents (academic papers, newspapers).
- **Quality Validation**: **(NEW)** Built-in validation layer that assesses output quality, detects hallucinations, and provides transparency with 0.0-1.0 quality scores.
- **Enhanced Metadata**: **(NEW)** Every extraction includes detailed metadata (page counts, detected columns, OCR confidence) and YAML frontmatter.
- **🆕 v2.0 Quality Pipeline**: Advanced enhancement pipeline with noise reduction, caption extraction, footnote linking, and schema enforcement.
- **Cloud-First OCR**: Priority support for OpenRouter's vision models with specialized RAG-optimized prompts.
- **Myanmar 🇲🇲 Support**: High-accuracy OCR for Myanmar language via OpenRouter.
- **Auto-Detection**: Intelligently switches between extraction methods based on file content.
- **Smart Fallback**: Automatically falls back to local RapidOCR with layout analysis if cloud extraction fails.
- **Modern UI**: Clean, responsive interface with real-time quality metrics and cost estimation.

---

### 🆕 v2.0 Quality Enhancement Modules

DocFlow v2.0 introduces a comprehensive quality enhancement pipeline:

| Module                    | Purpose                                                                       |
| ------------------------- | ----------------------------------------------------------------------------- |
| `semantic_annotator.py`   | Semantic role classification (headings, lists, captions)                      |
| `confidence_tracker.py`   | End-to-end OCR confidence tracking and aggregation                            |
| `schema_enforcer.py`      | RAG-optimized Markdown Schema v2.0 compliance                                 |
| `language_detector.py`    | Auto-detects document language (100+ languages)                               |
| `noise_filter.py`         | Removes headers/footers, watermarks, page numbers, artifacts                  |
| `caption_extractor.py`    | Extracts and links captions to tables/figures                                 |
| `cleaner.py`              | Deterministic Markdown normalization (spacing, lists, bullet standardization) |
| `validation_framework.py` | Unified validation with hallucination detection                               |
| `enhanced_pipeline.py`    | Orchestrates all modules in a staged pipeline                                 |

---

## 🚀 Getting Started

### Option 1: Docker (Recommended)

1.  **Build the Image**:

    ```bash
    docker build -t docflow .
    ```

2.  **Run the Container**:

    ```bash
    docker run -p 7860:7860 docflow
    ```

3.  **Open the UI**: Go to `http://localhost:7860`

---

### Option 2: Python CLI (Direct Usage)

**Prerequisites**:

```bash
pip install -r requirements.txt
```

**Usage**:

```bash
python app.py
```

---

## ⚙️ OCR Engines

| Engine             | Best For                              | Languages      | Privacy    | Cost                        |
| :----------------- | :------------------------------------ | :------------- | :--------- | :-------------------------- |
| **OpenRouter** ⭐  | Scanned PDFs, Images, Complex Layouts | 100+ (inc. 🇲🇲) | Cloud API  | **FREE** (Nemotron) or Paid |
| **pymupdf4llm** 🆕 | Digital PDFs (LLM/RAG optimized)      | All            | 100% Local | FREE                        |
| **RapidOCR**       | Scanned Documents, Offline Use        | 6 Languages    | 100% Local | FREE                        |

### Recommended Cloud Models (via OpenRouter)

1. **Nemotron Nano 12B VL (FREE)**: Best for general use and default choice.
2. **Qwen 2.5-VL 72B**: Best for maximum accuracy on complex documents.
3. **Gemini 2.0 Flash Lite**: Ultra-fast and cost-effective.

---

## 🛠️ Configuration

To use OpenRouter, you'll need an API key from [openrouter.ai](https://openrouter.ai/). Enter it in the **⚙️ Settings** panel in the UI.

---

## 📜 License

DocFlow is released under the MIT License.
