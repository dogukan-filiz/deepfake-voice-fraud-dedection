"""Test HF model loading and basic functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import json

# Test HF model loading
try:
    from backend.model_wrapper import get_model, get_model_status
    print("Testing HF model loading...")
    
    # Get model status (should load it)
    status = get_model_status()
    print("Model status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Try to get model instance
    model = get_model()
    print(f"\nModel loaded successfully: {type(model).__name__}")
    
    if hasattr(model, 'model_id'):
        print(f"Model ID: {model.model_id}")
    else:
        print("Model ID: Not available")
    
    # Test basic prediction with dummy data
    import numpy as np
    t = np.arange(16000) / 16000.0
    test_waveform = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    
    features = {
        "waveform": test_waveform,
        "sr": 16000,
        "spectral_residual_score": 0.5
    }
    
    result = model.predict(features)
    print(f"\nTest prediction successful!")
    print(f"Result: {json.dumps(result, indent=2, default=str)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()