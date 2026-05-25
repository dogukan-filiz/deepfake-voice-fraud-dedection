import h5py
import json
import numpy as np

def analyze_h5_file(file_path):
    """H5 dosyasının yapısını detaylı incele"""
    print(f"Analiz ediliyor: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n=== Ana Anahtarlar ===")
            print(list(f.keys()))
            
            # Model metadata ve config ara
            def find_metadata(group, prefix=""):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        if 'config' in key.lower():
                            print(f"\n🔍 {prefix}{key} - MODEL CONFIG BULUNDU!")
                        print(f"{prefix}{key}: shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, h5py.Group):
                        if 'config' in key.lower():
                            print(f"\n🔍 {prefix}{key} - MODEL CONFIG GRUBU BULUNDU!")
                        find_metadata(item, prefix + "  ")
            
            find_metadata(f)
            
            # JSON config ara
            def find_json(group, prefix=""):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        if item.dtype.kind == 'S' or item.dtype.kind == 'O':  # String veya object
                            try:
                                content = item[()]
                                if isinstance(content, bytes):
                                    content = content.decode('utf-8')
                                if 'model' in content.lower() or 'config' in content.lower():
                                    print(f"\n🔍 STRING CONFIG BULUNDU: {key}")
                                    print(f"İçerik: {content[:200]}...")
                            except:
                                pass
                    elif isinstance(item, h5py.Group):
                        find_json(item, prefix + "  ")
            
            find_json(f)
            
    except Exception as e:
        print(f"Hata: {e}")

def check_model_config(file_path):
    """Model config dosyası var mı kontrol et"""
    config_paths = [
        file_path.replace('.h5', '_config.json'),
        file_path.replace('.h5', '.config.json'),
        'model_config.json',
        'config.json'
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            print(f"✅ Config bulundu: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"Config içerik: {config}")
                return config_path, config
    
    print("❌ Config dosyası bulunamadı")
    return None, None

if __name__ == "__main__":
    import os
    
    model_path = "C:\\Users\\DOGUKAN\\Downloads\\ckpt.h5"
    project_model_path = "D:\\Workspace\\deepfake-voice-fraud-dedection\\models\\tf_conformer\\model.h5"
    
    print("=== Model Dosyası Analizi ===")
    analyze_h5_file(model_path)
    print("\n" + "="*50)
    analyze_h5_file(project_model_path)
    
    print("\n=== Config Dosyası Kontrol ===")
    check_model_config(model_path)
    check_model_config(project_model_path)