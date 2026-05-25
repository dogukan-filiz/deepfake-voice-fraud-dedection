import subprocess
import sys
import os
import time

# Start backend server
backend_process = subprocess.Popen([
    sys.executable, "-m", "uvicorn", 
    "backend.main_tf:app", 
    "--host", "127.0.0.1", 
    "--port", "8010"
], cwd=os.path.dirname(os.path.abspath(__file__)))

print("🚀 Backend server starting...")
time.sleep(5)  # Wait for server to start

# Test if server is running
import requests
try:
    response = requests.get("http://127.0.0.1:8010/health", timeout=5)
    print("✅ Backend server is running!")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"❌ Backend server failed: {e}")
    backend_process.terminate()
    sys.exit(1)

print("\n🧪 Starting test...")
# Import and run test
from simple_test import main
main()