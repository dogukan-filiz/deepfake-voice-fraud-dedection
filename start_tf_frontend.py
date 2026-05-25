#!/usr/bin/env python3
"""
Frontend setup for TensorFlow backend
"""

import os
import subprocess

def setup_frontend():
    """Frontend'i TensorFlow backend ile yapılandır"""
    
    print("🔧 Setting up frontend for TensorFlow backend...")
    
    # Frontend dizinine git
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    
    if not os.path.exists(frontend_dir):
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return False
    
    # Backend hedeflerini ayarla
    os.environ["VITE_API_TARGET"] = "http://127.0.0.1:8010"
    os.environ["VITE_WS_TARGET"] = "ws://127.0.0.1:8010"
    
    print(f"✅ API Target: {os.environ['VITE_API_TARGET']}")
    print(f"✅ WebSocket Target: {os.environ['VITE_WS_TARGET']}")
    
    # Vite config'u güncellemek için frontend dizininde çalış
    original_dir = os.getcwd()
    try:
        os.chdir(frontend_dir)
        
        # Bağımlılıkları yükle
        print("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "ci"], check=True, capture_output=True)
        
        # Vite config'unu doğrula
        config_file = "vite.config.mts"
        if os.path.exists(config_file):
            print(f"✅ Frontend config found: {config_file}")
            
            # Config'u oku ve API hedeflerini kontrol et
            with open(config_file, 'r') as f:
                content = f.read()
                
            if "127.0.0.1:8010" in content:
                print("✅ Frontend configured for TensorFlow backend (port 8010)")
            else:
                print("⚠️  Frontend config may need manual update for port 8010")
        else:
            print(f"⚠️  Config file not found: {config_file}")
        
        print("✅ Frontend setup completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend setup failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Frontend setup error: {e}")
        return False
    finally:
        os.chdir(original_dir)

def start_frontend():
    """Frontend'i başlat"""
    
    print("🚀 Starting frontend...")
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    
    if not os.path.exists(frontend_dir):
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return False
    
    original_dir = os.getcwd()
    try:
        os.chdir(frontend_dir)
        
        # Frontend'i başlat
        print("Frontend starting on http://localhost:5173")
        print("Press Ctrl+C to stop")
        print()
        
        subprocess.run(["npm", "run", "dev"])
        
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped")
    except Exception as e:
        print(f"❌ Frontend error: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    if setup_frontend():
        start_frontend()
    else:
        print("❌ Frontend setup failed")