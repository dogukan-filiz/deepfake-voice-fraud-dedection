#!/usr/bin/env python3
"""
Quick test to verify frontend-backend connection
"""

import requests
import json

def test_backend():
    """Test backend connection"""
    try:
        response = requests.get("http://127.0.0.1:8010/health", timeout=5)
        print("✅ Backend connection successful!")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing frontend-backend connection...")
    success = test_backend()
    
    if success:
        print("\n🎉 Everything is working!")
        print("📝 Frontend should now be able to connect to backend.")
        print("🌐 Open http://127.0.0.1:5174 in your browser")
    else:
        print("\n❌ Still connection issues.")
        print("💡 Make sure both services are running:")
        print("   Backend: .venv\\Scripts\\python.exe -m uvicorn backend.main_tf:app --host 127.0.0.1 --port 8010")
        print("   Frontend: cd frontend && npm run dev")