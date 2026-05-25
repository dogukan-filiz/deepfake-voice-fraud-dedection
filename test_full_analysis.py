#!/usr/bin/env python3
"""
Test full backend API with audio analysis
"""

import requests
import json
import numpy as np
import io
import soundfile as sf

def test_backend_analysis():
    """Test the full backend analysis endpoint"""
    
    print("🧪 Testing full backend analysis...")
    
    # Create test audio data
    t = np.linspace(0, 3, 48000)  # 3 seconds at 16kHz
    audio_data = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.1
    
    # Convert to WAV format
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_data, 16000, format='WAV')
    wav_buffer.seek(0)
    
    # Test the analysis endpoint
    try:
        files = {'file': ('test.wav', wav_buffer, 'audio/wav')}
        response = requests.post("http://127.0.0.1:8010/analyze", files=files, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"📊 Result: {json.dumps(result, indent=2)}")
            
            # Check expected fields
            expected_fields = ['cagri_id', 'authenticity_score', 'is_suspected_fraud', 'p_real', 'p_fake', 'spectral_residual']
            missing_fields = [field for field in expected_fields if field not in result]
            
            if missing_fields:
                print(f"⚠️  Missing fields: {missing_fields}")
            else:
                print("✅ All expected fields present")
            
            return True
        else:
            print(f"❌ Analysis failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Full Backend Analysis Test")
    print("=" * 30)
    
    success = test_backend_analysis()
    
    if success:
        print("\n🎉 Full backend analysis works!")
        print("💡 You can now upload audio files through the frontend")
    else:
        print("\n❌ Backend analysis still has issues")
        print("💡 Check the error messages above")