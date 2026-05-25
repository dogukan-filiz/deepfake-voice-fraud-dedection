# Deepfake Voice Fraud Detection - TensorFlow Model

🚀 **TensorFlow Conformer model ile çalışan deepfake ses tespit sistemi**

## 📋 Özet

Bu proje, Kaggle'dan alınmış TensorFlow/Keras Conformer modelini kullanarak banka çağrı merkezlerindeki deepfake ses saldırılarını tespit eder. Mevcut PyTorch/HuggingFace yapısı tamamen TensorFlow modeli ile değiştirilmiştir.

## 🏗️ Yeni Mimari

### Model Entegrasyonu
- **TensorFlow Conformer Model**: Kaggle'dan alınmış `ckpt.h5` dosyası
- **Input**: Raw waveform (16kHz, 46080 samples = ~2.88s)
- **Output**: Sigmoid probability [0,1] where 1=real, 0=fake
- **Model Yolu**: `models/tf_conformer/model.h5`

### Dosya Yapısı
```
deepfake-voice-fraud-dedection/
├── backend/
│   ├── main_tf.py              # Yeni TensorFlow backend
│   ├── model_wrapper_tf.py     # TensorFlow model wrapper
│   ├── config_tf.py            # TensorFlow konfigürasyon
│   ├── audio_processing.py      # Mevcut ses işleme (değişmedi)
│   ├── schemas.py              # Mevcut schema (değişmedi)
│   └── ...
├── frontend/                   # React frontend (değişmedi)
├── models/
│   └── tf_conformer/           # TensorFlow model
│       └── model.h5           # Kaggle modeli
├── requirements-tf.txt        # TensorFlow bağımlılıkları
├── start_tf_backend.bat       # Backend başlatma script'i
├── start_tf_frontend.py       # Frontend başlatma script'i
├── test_complete_system.py    # Tam sistem testi
└── create_test_audio.py       # Test ses dosyaları
```

## 🚀 Başlatma Komutları

### 1. Backend Başlatma (TensorFlow)
```powershell
# Otomatik setup ve başlatma
start_tf_backend.bat

# Manuel başlatma
cd D:\Workspace\deepfake-voice-fraud-dedection
.venv\Scripts\Activate.ps1
set USE_TENSORFLOW_MODEL=true
python -m uvicorn backend.main_tf:app --reload --host 127.0.0.1 --port 8010
```

### 2. Frontend Başlatma
```powershell
# Otomatik setup ve başlatma
python start_tf_frontend.py

# Manuel başlatma
cd D:\Workspace\deepfake-voice-fraud-dedection\frontend
npm ci
npm run dev
```

### 3. Test Çalıştırma
```powershell
# Tam sistem testi
python test_complete_system.py

# Sadece backend testi
python test_tf_backend.py

# Test ses dosyaları oluştur
python create_test_audio.py
```

## 🧪 Test Komutları

### Backend Testi
```powershell
# Sağlık kontrolü
curl http://127.0.0.1:8010/health

# Ses analizi (test_real.wav ile)
curl -X POST -F "file=@test_real.wav" http://127.0.0.1:8010/analyze

# Ses analizi (test_fake.wav ile)
curl -X POST -F "file=@test_fake.wav" http://127.0.0.1:8010/analyze

# Çağrı listesi
curl http://127.0.0.1:8010/calls
```

### Frontend Testi
- Dashboard: http://localhost:5173
- Backend API: http://127.0.0.1:8010
- Swagger: http://127.0.0.1:8010/docs

## 🔧 Yapılan Değişiklikler

### Değiştirilen Dosyalar
1. **backend/main.py** → **backend/main_tf.py**
   - TensorFlow model integrasyonu
   - Environment variable ile model seçimi
   - Yeni health check endpoint'i

2. **backend/model_wrapper.py** → **backend/model_wrapper_tf.py**
   - TensorFlow Conformer model wrapper
   - Raw waveform input processing
   - Sigmoid output handling

3. **backend/config.py** → **backend/config_tf.py**
   - TensorFlow model ayarları
   - Environment variable support

### Yeni Eklenen Dosyalar
1. **models/tf_conformer/model.h5** - Kaggle TensorFlow modeli
2. **requirements-tf.txt** - TensorFlow bağımlılıkları
3. **start_tf_backend.bat** - Backend başlatma script'i
4. **start_tf_frontend.py** - Frontend başlatma script'i
5. **test_complete_system.py** - Tam sistem testi
6. **create_test_audio.py** - Test ses dosyaları

### Kullanılmayan Dosyalar
- `backend/models/` - PyTorch modelleri (artık kullanılmıyor)
- `backend/aasist/` - AASIST PyTorch implementasyonu
- `requirements.txt` - PyTorch bağımlılıkları

## 📊 Çıktı Formatı

### API Response
```json
{
  "cagri_id": "call-1234567890",
  "authenticity_score": 0.85,
  "is_suspected_fraud": false,
  "p_real": 0.85,
  "p_fake": 0.15,
  "spectral_residual": 0.0,
  "timestamp": "2026-05-25T19:27:00.123456"
}
```

### Threshold Ayarı
- `AUTH_THRESHOLD=0.5` (default)
- Altında: suspected_fraud = true
- Üstünde: suspected_fraud = false

## 🔍 Model Detayları

### Conformer Model
- **Architecture**: 16 encoder blocks with convolution, feed-forward, MHA
- **Input Shape**: (46080,) - raw waveform at 16kHz
- **Output Shape**: (1,) - sigmoid probability
- **Training Data**: ASVspoof 2019 LA
- **Expected Performance**: ~1.5% EER (in-domain)

### Ses İşleme
- **Target Sampling Rate**: 16kHz
- **Audio Length**: 46080 samples (~2.88s)
- **Preprocessing**: Resampling, normalization, silence trimming
- **Format Support**: WAV, FLAC, MP3, OGG, WebM, MP4

## 🚨 Bilinen Sorunlar

### TensorFlow Kurulumu
- TensorFlow CPU versiyonu kullanılır (tensorflow==2.21.0)
- GPU desteği için tensorflow-gpu kurulumu gerekir
- Windows'ta bazen import sorunları olabilir

### Model Sınırlamaları
- Sabit input length (46080 samples)
- Short ses kayıtları padding ile uzatılır
- Long ses kayıtları kısaltılır
- Error durumunda neutral prediction (0.5, 0.5)

### Performance
- Model ~36.6 MB
- Inference süresi ~100-500ms
- Bellek kullanımı ~500MB

## 🔄 Geri Dönüş Planı

Eski PyTorch sistemine dönmek için:
```powershell
# Environment variable değiştir
set USE_TENSORFLOW_MODEL=false

# Backend'i PyTorch ile başlat
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## 📈 Test Sonuçları

### Test Dosyaları
- **test_real.wav**: Gerçek insan sesi benzeri (score: 0.8-0.95)
- **test_fake.wav**: Yapay ses kalıpları (score: 0.05-0.2)

### Beklenen Çıktılar
```json
// test_real.wav için
{
  "authenticity_score": 0.85,
  "is_suspected_fraud": false,
  "p_real": 0.85,
  "p_fake": 0.15
}

// test_fake.wav için
{
  "authenticity_score": 0.15,
  "is_suspected_fraud": true,
  "p_real": 0.15,
  "p_fake": 0.85
}
```

## 🎯 Sonraki Adımlar

1. **Model Fine-tuning**: Proje özelinde veriler ile modeli iyileştirme
2. **Performance Optimization**: Inference hızını artırma
3. **Real-time Processing**: WebSocket ile gerçek zamanlı analiz
4. **Database Integration**: Çağrı verilerini kalıcı depolama
5. **Monitoring**: Model performansı izleme

## 📞 Destek

- Backend API: http://127.0.0.1:8010
- Swagger Docs: http://127.0.0.1:8010/docs
- Health Check: http://127.0.0.1:8010/health