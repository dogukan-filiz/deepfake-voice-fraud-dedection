import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.model_wrapper_tf import get_model, get_model_status

print("🧪 TensorFlow Model Test")
print("=" * 30)

# Test model status
print("📊 Model Status:")
status = get_model_status()
print(f"  Loaded: {status.get('loaded', False)}")
print(f"  Type: {status.get('type', 'Unknown')}")
print(f"  Path: {status.get('model_path', 'Unknown')}")

if status.get('loaded'):
    print("✅ Model loaded successfully")
    
    # Test prediction
    print("\n🎯 Test Prediction:")
    import numpy as np
    
    # Create test audio
    t = np.arange(16000) / 16000.0
    sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    
    features = {
        "waveform": sine,
        "sr": 16000,
    }
    
    try:
        model = get_model()
        result = model.predict(features)
        print(f"  P(Real): {result['p_real']:.4f}")
        print(f"  P(Fake): {result['p_fake']:.4f}")
        print("✅ Prediction successful")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
else:
    print("❌ Model failed to load")
    print(f"  Error: {status.get('error', 'Unknown error')}")