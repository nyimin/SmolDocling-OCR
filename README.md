# SmolDocling: Hybrid PDF & Image to Markdown Converter

A lightweight, CPU-optimized application that intelligently converts PDFs and images to Markdown. It features a **Hybrid Pipeline** that automatically selects the best tool for the job:

- **⚡ Fast Mode**: Uses `MarkItDown` (Microsoft) for instant text extraction from digital PDFs.
- **🐢 Accurate Mode**: Uses `SmolDocling-256M` (HuggingFace) for high-quality OCR and layout analysis on scanned documents and images.

## Features

- **🚀 Smart Auto-Detection**: Automatically detects if a PDF is digital text or a scan and switches modes instantly.
- **💻 CPU Optimized**: Designed to run efficiently on standard CPUs without requiring a GPU.
- **🐳 Docker Ready**: dedicated container with a clean web UI (Gradio).
- **📝 Markdown Output**: Clean, structured Markdown perfect for LLM context or documentation.

---

## Option 1: Docker (Recommended)

Build and run the application in a container. This ensures all dependencies are isolated.

1.  **Build the Image**:

    ```bash
    docker build -t smol-docling .
    ```

2.  **Run the Container**:

    ```bash
    docker run -p 7860:7860 smol-docling
    ```

3.  **Open the UI**:
    Go to `http://localhost:7860` in your browser.

    _You will see a "Conversion Mode" option. Leave it on **Auto** for best results, or manually select **Fast** (Text) or **Accurate** (OCR)._

---

## Option 2: Python CLI (Direct Usage)

Run the script directly in your terminal.

**Prerequisites**:

```bash
pip install -r requirements.txt
```

**Usage**:

```bash
# Start the Web UI
python app.py

# Run backend OCR directly (Legacy Mode)
python smoldocling.py my_document.pdf -o output.md
```
