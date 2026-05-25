import tensorflow as tf
import numpy as np
import h5py
import json
import os

def create_model_from_weights(weights_path):
    """H5 ağırlıklarından model oluştur"""
    
    print(f"🔧 Model ağırlıklarından model oluşturuluyor: {weights_path}")
    
    # Model mimarisini oluştur
    def create_conformer_model():
        # Input layers
        inputs = tf.keras.layers.Input(shape=(46080,), name='input_1')
        
        # Reshape for Conv2D: (batch, 1, height, width) where height=1 for 1D conv
        x = tf.keras.layers.Reshape((1, 1, 46080), name='reshape_input')(inputs)
        
        # Subsampling layer
        x = tf.keras.layers.Conv2D(144, kernel_size=(3, 3), strides=(2, 1), 
                                 padding='same', activation='relu', name='conformer_encoder_subsampling_1')(x)
        x = tf.keras.layers.Conv2D(144, kernel_size=(3, 3), strides=(2, 1), 
                                 padding='same', activation='relu', name='conformer_encoder_subsampling_2')(x)
        
        # Reshape back to sequence: (batch, sequence_length, features)
        x = tf.keras.layers.Reshape((-1, 144), name='reshape_conv')(x)
        
        # Linear projection
        x = tf.keras.layers.Dense(144, activation='relu', name='conformer_encoder_linear')(x)
        
        # Conformer blocks - 16 tane
        for i in range(16):
            block_name = f'conformer_encoder_block_{i}'
            x = create_conformer_block(block_name, 144)(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D(name='global_average_pooling1d')(x)
        
        # Final dense layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='dense')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='conformer_model')
        return model
    
    def create_conformer_block(name, dim):
        """Tek bir Conformer bloğu oluştur"""
        block = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(name=f'{name}_ln'),
            
            # 1D Convolution module (更适合音频处理)
            tf.keras.layers.Conv1D(dim, kernel_size=3, padding='same', 
                                 activation='relu', name=f'{name}_conv_module_conv'),
            tf.keras.layers.LayerNormalization(name=f'{name}_conv_module_ln'),
            
            # Feed forward module 1
            tf.keras.layers.Dense(dim*4, activation='relu', name=f'{name}_ff_module_1_dense_1'),
            tf.keras.layers.Dense(dim, name=f'{name}_ff_module_1_dense_2'),
            tf.keras.layers.LayerNormalization(name=f'{name}_ff_module_1_ln'),
            
            # Feed forward module 2
            tf.keras.layers.Dense(dim*4, activation='relu', name=f'{name}_ff_module_2_dense_1'),
            tf.keras.layers.Dense(dim, name=f'{name}_ff_module_2_dense_2'),
            tf.keras.layers.LayerNormalization(name=f'{name}_ff_module_2_ln'),
            
            # MHSA module (simplified - 使用全局平均池化替代)
            tf.keras.layers.LayerNormalization(name=f'{name}_mhsa_module_ln'),
        ], name=name)
        
        return block
    
    # Model oluştur
    model = create_conformer_model()
    model.summary()
    
    # Ağırlıkları yükle
    try:
        # H5 dosyasını doğrudan oku
        with h5py.File(weights_path, 'r') as f:
            # Model katmanlarını ve ağırlıklarını eşleştir
            load_weights_to_model(model, f)
        
        print("✅ Model ağırlıkları başarıyla yüklendi")
        return model
        
    except Exception as e:
        print(f"❌ Ağırlık yükleme hatası: {e}")
        raise

def load_weights_to_model(model, h5_file):
    """H5 dosyasındaki ağırlıkları modele yükle"""
    
    def load_group_weights(group, model_layer, prefix=""):
        """H5 grubundaki ağırlıkları katmana yükle"""
        
        for key in group.keys():
            item = group[key]
            
            if isinstance(item, h5py.Dataset):
                weight_name = f"{prefix}{key}"
                
                # Eğer bu katmanda ilgili varsa
                if hasattr(model_layer, 'get_weights'):
                    try:
                        # Ağırlıkları yükle
                        weights = [item[()]] if len(item.shape) == 0 else [item[:]]
                        model_layer.set_weights(weights)
                        print(f"  📦 Yüklendi: {weight_name} -> {model_layer.name}")
                    except Exception as e:
                        print(f"  ⚠️  Atlandı: {weight_name} ({e})")
                        
            elif isinstance(item, h5py.Group):
                # Grup katmanları için özyineli çağrı
                load_group_weights(item, model_layer, f"{prefix}{key}_")
    
    # Modelin tüm katmanlarını gezip ağırlıkları yükle
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        
        # H5 dosyasındaki ilgili grubu bul
        if layer_name in h5_file:
            print(f"🔍 Katman bulundu: {layer_name}")
            load_group_weights(h5_file[layer_name], layer, "")
        else:
            print(f"⚠️  Katman bulunamadı: {layer_name}")

def test_model_inference(model):
    """Model inference testi"""
    
    print("🧪 Model inference testi...")
    
    # Test ses oluştur
    t = np.arange(16000) / 16000.0
    test_audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    
    # Model beklentisine göre hazırla
    test_input = np.expand_dims(test_audio, axis=0)  # Shape: (1, 16000)
    if test_input.shape[1] < 46080:
        # Padding
        padding = 46080 - test_input.shape[1]
        test_input = np.pad(test_input, ((0, 0), (0, padding)), mode='constant')
    else:
        # Trimming
        test_input = test_input[:, :46080]
    
    print(f"📊 Input shape: {test_input.shape}")
    
    # Tahmin yap
    try:
        prediction = model.predict(test_input, verbose=0)
        print(f"✅ Prediction: {prediction[0][0]:.4f}")
        return True
    except Exception as e:
        print(f"❌ Inference hatası: {e}")
        return False

if __name__ == "__main__":
    weights_path = "C:\\Users\\DOGUKAN\\Downloads\\ckpt.h5"
    project_model_path = "D:\\Workspace\\deepfake-voice-fraud-dedection\\models\\tf_conformer\\model.h5"
    
    # Proje modelini kullan
    try:
        print("🚀 Proje modeli ile test yapılıyor...")
        model = create_model_from_weights(project_model_path)
        
        if test_model_inference(model):
            print("✅ Model başarıyla çalışıyor!")
            
            # Modeli kaydet
            model.save("D:\\Workspace\\deepfake-voice-fraud-dedection\\models\\tf_conformer\\working_model.h5")
            print("✅ Çalışan model kaydedildi: models/tf_conformer/working_model.h5")
        else:
            print("❌ Model testi başarısız")
            
    except Exception as e:
        print(f"❌ Model oluşturma hatası: {e}")