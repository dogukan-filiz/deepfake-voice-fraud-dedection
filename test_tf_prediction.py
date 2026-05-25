#!/usr/bin/env python3
"""
Test TensorFlow model wrapper with different input formats
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.model_wrapper_tf import get_model

def test_model_prediction():
    """Test model prediction with different input formats"""
    
    print("🧪 Testing TensorFlow model prediction...")
    
    # Get model
    model = get_model()
    print(f"✅ Model loaded: {model.is_loaded}")
    
    if not model.is_loaded:
        print("❌ Model not loaded")
        return False
    
    # Test 1: Direct numpy array
    print("\n🎵 Test 1: Direct numpy array")
    try:
        audio_data = np.random.randn(46080).astype(np.float32) * 0.1
        result = model.predict(audio_data)
        print(f"✅ Direct array prediction: {result['p_real']:.4f}")
    except Exception as e:
        print(f"❌ Direct array failed: {e}")
    
    # Test 2: Dictionary format (PyTorch style)
    print("\n🎵 Test 2: Dictionary format")
    try:
        audio_dict = {
            "waveform": np.random.randn(46080).astype(np.float32) * 0.1,
            "sr": 16000
        }
        result = model.predict(audio_dict)
        print(f"✅ Dictionary prediction: {result['p_real']:.4f}")
    except Exception as e:
        print(f"❌ Dictionary failed: {e}")
    
    # Test 3: Different length audio
    print("\n🎵 Test 3: Different length audio")
    try:
        short_audio = np.random.randn(20000).astype(np.float32) * 0.1
        result = model.predict(short_audio)
        print(f"✅ Short audio prediction: {result['p_real']:.4f}")
    except Exception as e:
        print(f"❌ Short audio failed: {e}")
    
    print("\n🎉 Model prediction tests completed!")

if __name__ == "__main__":
    test_model_prediction()