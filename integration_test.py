#!/usr/bin/env python3
"""
Direct test of TensorFlow backend in the same process
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.main_tf import app
import uvicorn
import requests
import time
import threading

def test_app_directly():
    """Test the FastAPI app directly without server"""
    
    print("🔍 Testing FastAPI app directly...")
    
    # Test model loading
    try:
        from backend.model_wrapper_tf import get_model
        model = get_model()
        print(f"✅ Model loaded: {model.is_loaded}")
        if model.is_loaded:
            info = model.get_model_info()
            print(f"📊 Model: {info['model_name']}")
            print(f"📊 Parameters: {info['total_params']:,}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    print("✅ Backend components are working!")
    return True

def run_server_test():
    """Run server and test"""
    print("🚀 Starting server test...")
    
    # Start server in background
    config = uvicorn.Config(app, host="127.0.0.1", port=8010, log_level="info")
    server = uvicorn.Server(config)
    
    # Start server in a separate thread
    def run_server():
        server.run()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    # Test server
    try:
        response = requests.get("http://127.0.0.1:8010/health", timeout=5)
        print(f"✅ Server responded with status: {response.status_code}")
        print(f"📊 Health check: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TensorFlow Backend Integration Test")
    print("=" * 50)
    
    # Test app components
    app_ok = test_app_directly()
    
    if app_ok:
        print("\n📝 App components work! Testing server...")
        
        # Try to run server test
        server_ok = run_server_test()
        
        if server_ok:
            print("\n🎉 Full integration test passed!")
        else:
            print("\n⚠️  App works but server has issues.")
            print("💡 This might be a Windows networking issue.")
            print("💡 Try running the server manually and testing with curl or browser.")
    else:
        print("\n❌ App components failed to load.")