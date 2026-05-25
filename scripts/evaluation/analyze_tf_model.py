import h5py
import numpy as np

def inspect_h5_structure(model_path):
    """H5 dosyasının yapısını incele TensorFlow olmadan"""
    print(f"Model dosyası: {model_path}")
    
    try:
        with h5py.File(model_path, 'r') as f:
            print(f"Ana anahtarlar: {list(f.keys())}")
            
            # Modelin JSON yapısını bulmaya çalış
            def explore_group(group, prefix=""):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"{prefix}{key}: shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, h5py.Group):
                        print(f"{prefix}{key}: Group")
                        # Model mimarisi bilgisi içerebilecek alanları ara
                        if 'config' in key.lower():
                            print(f"{prefix}  {key} contains model config!")
                        explore_group(item, prefix + "  ")
            
            explore_group(f)
            
    except Exception as e:
        print(f"Error: {e}")

def create_simple_tf_wrapper():
    """Basit TensorFlow wrapper oluştur"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input
    import json
    
    # Basit Conformer benzeri model oluştur
    model = Sequential([
        Input(shape=(46080,)),
        Dense(144, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Modeli kaydet
    model.save('simple_conformer_model.h5')
    print("Basit model oluşturuldu ve kaydedildi")
    
    # Model özetini göster
    model.summary()
    
    return model

if __name__ == "__main__":
    model_path = "C:\\Users\\DOGUKAN\\Downloads\\ckpt.h5"
    
    print("=== H5 Yapısı Analizi ===")
    inspect_h5_structure(model_path)
    
    print("\n=== Basit Model Wrapper Test ===")
    try:
        create_simple_tf_wrapper()
    except Exception as e:
        print(f"TensorFlow hata: {e}")
        print("TensorFlow kurulumunda sorun var, ama model yapısını inceleyebiliriz.")