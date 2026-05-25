#!/usr/bin/env python3
"""
Simple test for TensorFlow backend endpoints
"""

import requests
import json

def test_backend():
    """Test the backend with localhost"""
    
    # Test with localhost
    try:
        print("🔍 Testing with localhost...")
        response = requests.get("http://localhost:8010/health", timeout=5)
        print(f"✅ Success! Status: {response.status_code}")
        print(f"📊 Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Localhost test failed: {e}")
        
    # Test with 127.0.0.1
    try:
        print("🔍 Testing with 127.0.0.1...")
        response = requests.get("http://127.0.0.1:8010/health", timeout=5)
        print(f"✅ Success! Status: {response.status_code}")
        print(f"📊 Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ 127.0.0.1 test failed: {e}")
        
    return False

if __name__ == "__main__":
    print("🧪 Testing TensorFlow backend connection...")
    success = test_backend()
    
    if success:
        print("🎉 Backend is accessible!")
    else:
        print("❌ Backend is not accessible.")
        print("💡 Make sure the server is running with:")
        print("   cd D:\\Workspace\\deepfake-voice-fraud-dedection")
        print("   .venv\\Scripts\\python.exe -m uvicorn backend.main_tf:app --host 127.0.0.1 --port 8010")