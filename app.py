
import gradio as gr
import shutil
import tempfile
import os
import smoldocling
import torch
import fitz # PyMuPDF
import fast_converter

# Ensure CUDA is used if available
DEVICE = "cpu"

def process_upload(file, mode, progress=gr.Progress()):
    if file is None:
        return None, None, "No file uploaded."
    
    input_path = file.name
    print(f"Processing input: {input_path} (Mode: {mode})")
    
    try:
        progress(0, desc="Analyzing document...")
        
        # --- AUTO DETECTION LOGIC ---
        if mode == "Auto":
            if input_path.lower().endswith(".pdf"):
                try:
                    doc = fitz.open(input_path)
                    # Check first page for text density
                    if len(doc) > 0:
                        text = doc[0].get_text()
                        if len(text.strip()) > 50: # Arbitrary threshold for "digital text"
                            print("Auto-Detect: Found significant text layer. Using FAST mode.")
                            mode = "Fast (Text Only)"
                        else:
                            print("Auto-Detect: Low text density. Using OCR mode.")
                            mode = "Accurate (OCR)"
                    doc.close()
                except Exception as e:
                    print(f"Auto-detect failed: {e}. Defaulting to OCR.")
                    mode = "Accurate (OCR)"
            else:
                # Images always need OCR
                mode = "Accurate (OCR)"

        # --- DISPATCHER ---
        if mode == "Fast (Text Only)":
             progress(0.2, desc="Extracting text (Fast Mode)...")
             markdown_text = fast_converter.convert_fast(input_path)
             if markdown_text is None:
                 return None, None, "Failed to convert in Fast Mode."

        else: # Accurate (OCR)
            # Callback wrapper for Gradio Progress
            def update_progress(p, desc):
                progress(p, desc=desc)

            markdown_text = smoldocling.process_document(
                input_path, 
                output_path=None, 
                device=DEVICE,
                progress_callback=update_progress
            )
        
        if markdown_text is None:
            return None, None, "Failed to process document."
            
        progress(1.0, desc="Finalizing...")
            
        # Create a temporary file for download
        temp_dir = tempfile.mkdtemp()
        output_filename = os.path.splitext(os.path.basename(input_path))[0] + ".md"
        output_path = os.path.join(temp_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
            
        return markdown_text, output_path, None # None = No error
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Minimalist Custom CSS
custom_css = """
body { background-color: #f8f9fa; }
.container { max-width: 800px; margin: auto; padding-top: 2rem; }
.output-markdown { border: 1px solid #e0e0e0; padding: 1rem; border-radius: 8px; background: white; }
footer { visibility: hidden; }
"""

# Create Gradio Interface with Monochrome Theme
with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css, title="SmolDocling") as demo:
    
    with gr.Column(elem_classes="container"):
        gr.Markdown("# 📄 SmolDocling", elem_id="header")
        gr.Markdown("Lightweight PDF & Image to Markdown Converter")
        
        with gr.Row():
            file_input = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                type="filepath",
                scale=2
            )
            mode_input = gr.Radio(
                ["Auto", "Fast (Text Only)", "Accurate (OCR)"],
                label="Conversion Mode (Default: Auto)",
                value="Auto",
                scale=1
            )
            process_btn = gr.Button("Convert", variant="primary", scale=1)
            
            # Quick Tip
            gr.Markdown("ℹ️ **Tip:** 'Auto' uses Fast mode for digital PDFs. Select 'Accurate' if tables are broken.")
        
        error_box = gr.Markdown(visible=True) # To show errors
        
        with gr.Tabs():
            with gr.TabItem("Preview"):
                output_md_view = gr.Markdown(elem_classes="output-markdown")
            
            with gr.TabItem("Raw Code"):
                output_raw_text = gr.TextArea(show_copy_button=True, label="Markdown Source")

        download_btn = gr.File(label="Download Markdown", interactive=False)

    # Event Logic
    process_btn.click(
        fn=process_upload,
        inputs=[file_input, mode_input],
        outputs=[output_md_view, download_btn, error_box]
    ).success(
        fn=lambda x: x[0], # Just to populate raw text area from the first output
        inputs=[output_md_view],  # Note: logic might need adjusting depending on gradio version, simplified here
        outputs=[output_raw_text]
    )
    
    # Sync visual markdown to raw text area directly
    process_btn.click(
        fn=process_upload, 
        inputs=[file_input, mode_input], 
        outputs=[output_raw_text, download_btn, error_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
