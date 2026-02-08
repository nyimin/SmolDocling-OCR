
import argparse
import os
import sys
import torch
import warnings
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Suppress warnings
warnings.filterwarnings("ignore")

# Try to import PyMuPDF
try:
    import fitz
except ImportError:
    fitz = None

# Try to import Docling Core
try:
    from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
    from docling_core.types.doc import ImageRefMode
except ImportError:
    print("Error: docling-core is not installed. Please install it via pip.")
    sys.exit(1)

def load_model(device):
    """Load the SmolDocling-256M model and processor."""
    model_id = "ds4sd/SmolDocling-256M-preview"
    # Use float32 for CPU
    dtype = torch.float32
    
    print(f"Loading model: {model_id} on {device} ({dtype})...")
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
    except Exception as e:
        print(f"Warning: Failed to load. Error: {e}")
        # Fallback ensuring CPU/Standard precision if the above fails
        model = AutoModelForVision2Seq.from_pretrained(model_id).to("cpu")
        processor = AutoProcessor.from_pretrained(model_id)
        
    return model, processor

def pdf_to_images(pdf_path):
    """Convert a PDF file to a list of PIL Images using PyMuPDF."""
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is not installed. Run `pip install pymupdf`.")
    
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    images = []
    
    for i, page in enumerate(doc):
        # Render page to an image (pixmap) at 150 DPI (usually good for OCR)
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        
    print(f"Extracted {len(images)} pages.")
    return images

def process_document(input_path, output_path=None, device="cpu", progress_callback=None):
    """Main processing function."""
    
    # 1. Load Images
    images = []
    if input_path.lower().endswith(".pdf"):
        images = pdf_to_images(input_path)
    else:
        try:
            img = Image.open(input_path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Error loading image: {e}")
            return

    # 2. Load Model
    model, processor = load_model(device)
    
    # 3. Process Pages Sequentially
    all_doctags = []
    all_images = [] # We need to keep images for DocTagsDocument reconstruction
    
    print("Processing pages...")
    total_pages = len(images)
    for i, image in enumerate(images):
        print(f"  - Converting Page {i+1}/{total_pages}...")
        
        if progress_callback:
            progress_callback((i + 1) / total_pages, f"Converting Page {i+1} of {total_pages}...")
        
        # Prepare inputs
        # Standard chat template for SmolDocling
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this page to docling."}
                ]
            }
        ]
        
        # Format input
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False, # Deterministic generation
            )
        
        # Decode
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up 'Assistant:' prompt if present in output (common with chat models)
        if "Assistant:" in generated_text:
            generated_text = generated_text.split("Assistant:", 1)[1].strip()
            
        all_doctags.append(generated_text)
        all_images.append(image)
        
        # Explicitly free memory
        del inputs, generated_ids

    # 4. Structure Document
    print("Structuring document...")
    # Create DocTagsDocument from the list of (doctag_string, image) tuples
    # Note: docling_core expects the exact generated string which contains XML-like tags
    try:
        # Construct the iterator of tuples expected by from_doctags_and_image_pairs
        # It expects: Iterable[Tuple[str, Image.Image]]
        pairs = zip(all_doctags, all_images)
        
        doc_tags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            pairs
        )
        
        doc = DoclingDocument(name=os.path.basename(input_path))
        doc.load_from_doctags(doc_tags_doc)
        
        # Export
        markdown_output = doc.export_to_markdown()
        
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_output)
            print(f"\nSuccess! Markdown saved to: {output_path}")
        else:
            print("\n--- Output ---\n")
            # print(markdown_output) # Suppress printing entire content for app usage scenarios
            
        return markdown_output
            
    except Exception as e:
        print(f"Error constructing document structure: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="SmolDocling Converter (Lightweight Windows Version)")
    parser.add_argument("input", help="Input file (PDF or Image)")
    parser.add_argument("--output", "-o", help="Output file (Markdown)")
    parser.add_argument("--device", default="cpu", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
        
    process_document(args.input, args.output, args.device)

if __name__ == "__main__":
    main()
