import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
import os
import numpy as np
load_dotenv()

# Test file path - Updated to use relative path safely
# Adjust this path if running from a different directory
pdf_path = os.path.join('Data', 'Optimal_Sizing_of_a_Wind_PV_Grid_Connected_Hybrid_System_for_Base_Load_Helsinki_Case.pdf')

if not os.path.exists(pdf_path):
    print(f"WARNING: Test file not found at {pdf_path}")
    print("Please update the 'pdf_path' variable in the script to point to a valid PDF.")
    sys.exit(1)

print('='*60)
print('Testing RapidOCR (Local) Engine - Final Verification')
print('='*60)
try:
    import structure_engine
    
    # Run extraction with enhanced mode enabled to verify full pipeline
    print(f"Processing: {pdf_path}")
    result = structure_engine.extract_with_rapidocr(pdf_path, dpi=300, lang='en', enhanced_mode=True)
    
    if result:
        print(f'\nSUCCESS: Extracted {len(result)} characters')
        
        # Check for error messages that shouldn't be in the text
        if 'Enhanced pipeline failed' in result:
             print('FAILURE: Error message found in output text (Pipeine Fallback occurred)')
        else:
             print('PASSED: No error messages in output (Enhanced Pipeline successful)')
             
        # Print a snippet
        print('\n--- Extracted Content Snippet ---')
        print(result[:500] + '...')
        print('---------------------------------')
             
    else:
        print('FAILED: No result returned')
        
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
