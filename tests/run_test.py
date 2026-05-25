import subprocess
import sys
import os
import time
import threading

def start_backend():
    """Start backend server in background"""
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "backend.main_tf:app", 
        "--host", "127.0.0.1", 
        "--port", "8010"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 Backend server starting...")
    time.sleep(8)  # Wait for server to start
    
    # Test if server is running
    try:
        import requests
        response = requests.get("http://127.0.0.1:8010/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running!")
            return backend_process
        else:
            print(f"❌ Backend server returned status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Backend server failed to start: {e}")
        return None

def run_test():
    """Run the comprehensive test"""
    try:
        # Import and run test
        from comprehensive_test import main
        main()
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🧪 Starting Deepfake Detection Test System")
    print("=" * 50)
    
    # Start backend
    backend_process = start_backend()
    
    if backend_process:
        print("\n📁 Running comprehensive test...")
        run_test()
        
        # Clean up
        print("\n🛑 Shutting down backend...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ Test completed!")
    else:
        print("❌ Failed to start backend server")