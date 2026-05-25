#!/usr/bin/env python3
"""
Direct Python test for TensorFlow backend
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.main_tf import app
import uvicorn
import threading
import time
import requests

def start_server():
    """Start the server in a separate thread"""
    config = uvicorn.Config(app, host="127.0.0.1", port=8010, log_level="info")
    server = uvicorn.Server(config)
    server.run_in_thread()

def test_connection():
    """Test connection to the server"""
    time.sleep(2)  # Wait for server to start
    
    try:
        print("🔍 Testing connection to server...")
        
        # Test health endpoint
        response = requests.get("http://127.0.0.1:8010/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ Health endpoint works!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Health endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting TensorFlow backend test...")
    
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Test connection
    success = test_connection()
    
    if success:
        print("🎉 Server is working!")
    else:
        print("❌ Server test failed!")
        
    # Keep server running for manual testing
    print("📝 Server is running. Press Ctrl+C to stop.")
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("🛑 Server stopped.")