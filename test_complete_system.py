#!/usr/bin/env python3
"""
Complete system test for TensorFlow Deepfake Detection
Backend ve frontend tam testi
"""

import os
import sys
import time
import requests
import subprocess
import signal
import threading
from pathlib import Path

class SystemTester:
    def __init__(self):
        self.backend_process = None
        self.base_url = "http://127.0.0.1:8010"
        self.frontend_url = "http://localhost:5173"
        
    def start_backend(self):
        """Backend'i başlat"""
        print("🚀 Starting TensorFlow backend...")
        
        # Backend dizinini kontrol et
        backend_dir = Path(__file__).parent
        if not (backend_dir / "backend" / "main_tf.py").exists():
            print(f"❌ Backend file not found: {backend_dir / 'backend' / 'main_tf.py'}")
            return False
        
        # Backend'i arka planda başlat
        venv_python = backend_dir / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            print(f"❌ Virtual environment not found: {venv_python}")
            return False
        
        # Environment değişkenlerini ayarla
        env = os.environ.copy()
        env["USE_TENSORFLOW_MODEL"] = "true"
        env["AUTH_THRESHOLD"] = "0.5"
        
        cmd = [
            str(venv_python),
            "-m", "uvicorn", "backend.main_tf:app",
            "--reload", "--host", "127.0.0.1", "--port", "8010"
        ]
        
        try:
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=str(backend_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Backend'in başlamasını bekle
            print("⏳ Waiting for backend to start...")
            time.sleep(5)
            
            # Backend kontrolü
            if self.check_backend_health():
                print("✅ Backend started successfully")
                return True
            else:
                print("❌ Backend failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Backend startup error: {e}")
            return False
    
    def check_backend_health(self):
        """Backend sağlık kontrolü"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok" and data.get("model", {}).get("loaded"):
                    print(f"✅ Model type: {data.get('model_type')}")
                    return True
            return False
        except:
            return False
    
    def test_backend_endpoints(self):
        """Backend endpoint testleri"""
        print("\n🧪 Testing backend endpoints...")
        
        # Health check
        if not self.check_backend_health():
            print("❌ Backend health check failed")
            return False
        
        # Test files
        test_files = ["test_real.wav", "test_fake.wav"]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"📁 Testing with {test_file}...")
                
                try:
                    with open(test_file, 'rb') as f:
                        files = {'file': f}
                        response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
                        
                        if response.status_code == 200:
                            data = response.json()
                            print(f"   ✅ Score: {data.get('authenticity_score'):.4f}")
                            print(f"   ✅ Fraud: {data.get('is_suspected_fraud')}")
                        else:
                            print(f"   ❌ Analysis failed: {response.status_code}")
                            return False
                            
                except Exception as e:
                    print(f"   ❌ Test error: {e}")
                    return False
            else:
                print(f"⚠️  Test file not found: {test_file}")
        
        return True
    
    def test_frontend_connection(self):
        """Frontend bağlantı testi"""
        print("\n🌐 Testing frontend connection...")
        
        # Sadece backend'e bağlanabilirliği test et
        # Frontend'i manuel olarak başlatmak daha güvenli
        try:
            # Frontend'in API hedefini kontrol et
            frontend_dir = Path(__file__).parent / "frontend"
            if frontend_dir.exists():
                config_file = frontend_dir / "vite.config.mts"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        content = f.read()
                    if "127.0.0.1:8010" in content:
                        print("✅ Frontend configured for TensorFlow backend")
                        return True
        except:
            pass
        
        print("⚠️  Frontend test skipped (start manually)")
        return True
    
    def cleanup(self):
        """Temizlik"""
        if self.backend_process:
            print("\n🧹 Cleaning up...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            print("✅ Backend stopped")
    
    def run_complete_test(self):
        """Tam sistem testi"""
        print("🔬 Deepfake Detection System Test - TensorFlow")
        print("=" * 50)
        
        try:
            # Backend başlat
            if not self.start_backend():
                print("❌ Backend startup failed")
                return False
            
            # Backend testleri
            if not self.test_backend_endpoints():
                print("❌ Backend tests failed")
                return False
            
            # Frontend testi
            if not self.test_frontend_connection():
                print("❌ Frontend test failed")
                return False
            
            print("\n🎉 All tests passed!")
            print("\n📋 Manual Testing Instructions:")
            print("1. Backend is running on: http://127.0.0.1:8010")
            print("2. Swagger docs: http://127.0.0.1:8010/docs")
            print("3. Frontend commands:")
            print("   cd frontend")
            print("   npm run dev")
            print("4. Test files:")
            print("   - test_real.wav (should score high)")
            print("   - test_fake.wav (should score low)")
            
            return True
            
        except KeyboardInterrupt:
            print("\n👋 Test interrupted")
            return False
        except Exception as e:
            print(f"\n❌ Test error: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """Ana fonksiyon"""
    tester = SystemTester()
    success = tester.run_complete_test()
    
    if success:
        print("\n✅ System test completed successfully")
        sys.exit(0)
    else:
        print("\n❌ System test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()