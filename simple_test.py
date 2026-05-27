"""Simple test without complex imports."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Test the HF model directly
try:
    from backend.model_wrapper_hf import HuggingFaceDeepfakeModel
    print("HF Model wrapper imported successfully")
    
    # Try to initialize (this will download the model)
    print("Initializing HF model...")
    model = HuggingFaceDeepfakeModel()
    print("HF Model initialized successfully")
    
except Exception as e:
    print(f"HF Model failed: {e}")
    import traceback
    traceback.print_exc()