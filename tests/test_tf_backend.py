#!/usr/bin/env python3
"""
Test script for TensorFlow backend API endpoints
"""

import requests
import json
import numpy as np
import io

def test_health_endpoint():
    """Test the /health endpoint"""
    try:
        # Try both localhost and 127.0.0.1
        for host in ["http://127.0.0.1:8010/health", "http://localhost:8010/health"]:
            try:
                response = requests.get(host, timeout=5)
                if response.status_code == 200:
                    print("✅ Health endpoint works!")
                    print(f"Response: {json.dumps(response.json(), indent=2)}")
                    return True
                else:
                    print(f"❌ Health endpoint failed with status {response.status_code} for {host}")
            except:
                continue
        
        print("❌ Cannot connect to health endpoint on any host")
        return False
    except Exception as e:
        print(f"❌ Health endpoint test error: {e}")
        return False

def test_analysis_endpoint():
    """Test the /analyze endpoint with dummy audio"""
    try:
        # Try both localhost and 127.0.0.1
        for host in ["http://127.0.0.1:8010/analyze", "http://localhost:8010/analyze"]:
            try:
                # Create dummy audio data
                t = np.linspace(0, 2.88, 46080)  # 2.88 seconds at 16kHz
                audio_data = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
                
                # Convert to bytes (WAV format)
                import soundfile as sf
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, audio_data, 16000, format='WAV')
                wav_buffer.seek(0)
                
                # Upload file
                files = {'file': ('test.wav', wav_buffer, 'audio/wav')}
                response = requests.post(host, files=files, timeout=10)
                
                if response.status_code == 200:
                    print("✅ Analysis endpoint works!")
                    result = response.json()
                    print(f"Result: {json.dumps(result, indent=2)}")
                    return True
                else:
                    print(f"❌ Analysis endpoint failed with status {response.status_code} for {host}")
                    
            except Exception as e:
                print(f"❌ Analysis endpoint test failed for {host}: {e}")
                continue
        
        print("❌ Analysis endpoint failed on all hosts")
        return False
            
    except Exception as e:
        print(f"❌ Analysis endpoint test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing TensorFlow backend API...")
    
    # Wait a moment for server to start
    import time
    time.sleep(2)
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    
    if health_ok:
        # Test analysis endpoint
        analysis_ok = test_analysis_endpoint()
        
        if analysis_ok:
            print("🎉 All tests passed! TensorFlow backend is working correctly.")
        else:
            print("❌ Analysis test failed.")
    else:
        print("❌ Health test failed. Backend may not be running.")