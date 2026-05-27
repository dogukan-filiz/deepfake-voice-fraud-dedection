"""Quick test script for HF model integration."""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

import numpy as np
from backend.model_wrapper import get_model, get_model_status

# Test 1: Model status
print("=== Model Status ===")
status = get_model_status()
for k, v in status.items():
    print(f"{k}: {v}")

print("\n=== Loading Model ===")
model = get_model()
print(f"Model loaded: {type(model).__name__}")

# Test 2: Simple inference
print("\n=== Test Inference ===")
# Generate a simple test signal (1 second sine wave at 1000Hz)
t = np.arange(16000) / 16000.0
test_waveform = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

features = {
    "waveform": test_waveform,
    "sr": 16000,
    "spectral_residual_score": 0.5
}

try:
    result = model.predict(features)
    print("Prediction successful!")
    for k, v in result.items():
        print(f"{k}: {v}")
except Exception as e:
    print(f"Prediction failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")