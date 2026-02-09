import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
import os
import time
load_dotenv()

# Test configuration
pdf_path = os.path.join('Data', 'Optimal_Sizing_of_a_Wind_PV_Grid_Connected_Hybrid_System_for_Base_Load_Helsinki_Case.pdf')
# Use the "cheap" model (Gemini Flash Lite) as "free" Nemotron is providing poor results
MODEL_TO_TEST = "cheap" 

print('='*60)
print('Testing CloudOCR (OpenRouter) Engine')
print('='*60)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("❌ ERROR: OPENROUTER_API_KEY not found in environment variables.")
    print("Please check your .env file.")
    sys.exit(1)
else:
    print(f"✅ API Key found: {api_key[:5]}...{api_key[-4:]}")

if not os.path.exists(pdf_path):
    print(f"⚠️ Test file not found at {pdf_path}. Please provide a valid file.")
    # Try to find any PDF in Data folder
    if os.path.exists('Data'):
        pdfs = [f for f in os.listdir('Data') if f.endswith('.pdf')]
        if pdfs:
            pdf_path = os.path.join('Data', pdfs[0])
            print(f"Using alternative file: {pdf_path}")
        else:
            print("No PDFs found in Data/ directory.")
            sys.exit(1)
    else:
        sys.exit(1)

try:
    import structure_engine
    print(f"Initiating extraction with model: {structure_engine.OPENROUTER_MODELS[MODEL_TO_TEST]['name']}")
    
    start_time = time.time()
    
    # Simulate what app.py does
    result, metadata = structure_engine.extract_with_openrouter(
        pdf_path, 
        model=MODEL_TO_TEST, 
        api_key=api_key
    )
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Time elapsed: {elapsed:.2f}s")
    
    if "Error" in result:
        print(f"❌ Extraction Failed: {result}")
        print(f"Metadata: {metadata}")
    elif not result:
        print("❌ Result is EMPTY (None or empty string).")
        print(f"Metadata received: {metadata}")
    else:
        print(f"✅ SUCCESS: Extracted {len(result)} characters")
        print(f"Metadata: {metadata}")
        
        output_file = "test_openrouter_output.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
            
        print(f"\n✅ Full output saved to: {os.path.abspath(output_file)}")
        print("\n--- Snippet ---")
        print(result[:500] + "...")
        
except Exception as e:
    print(f"❌ Exception occurred: {e}")
    import traceback
    traceback.print_exc()
