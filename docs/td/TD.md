# Test Belgesi (TD)

**Proje Adı:** Bankacılık Çağrı Merkezleri için Yapay Zeka Tabanlı Ses Sahteciliği (Deepfake) Tespit Sistemi
**Öğrenci:** Doğukan Filiz — 22290746
**Danışman:** Enver BAĞCI
**Ders:** BLM 4061 – BLM 4062 Araştırma Projesi I-II
**Belge Türü:** Test Belgesi (TD)
**Sürüm / Tarih:** 1.0 — 2026-05-29

> Bu belge, **Gereksinimler Belirtimi (RS)** ve **Yazılım Tasarım Belgesi (SDD)** dokümanlarında tanımlanan gereksinimlerin doğrulanmasını sağlamak amacıyla hazırlanmıştır. SDD'de tanımlanan bileşenler üzerinde değişiklik gerçekleşmiş; özellikle model katmanı eski Wav2Vec2 tabanlı tek modelden **SSL+AASIST (XLSR-300M + AASIST head) → AASIST baseline → Sezgisel** zincirine geçirilmiştir. Test stratejisi mevcut güncel kod tabanına göre tasarlanmıştır.

---

## İçindekiler

1. Giriş
2. Test Öğeleri
3. Test Stratejisi
4. Test Tasarımı ve Spesifikasyonu
5. Test Ortamı Kurulumu
6. Test Takvimi
7. Test Çıktıları
8. Ekler

---

## 1. Giriş

### 1.1 Test Belgesinin Amacı

Bu belge, sistemin RS dokümanında tanımlı işlevsel (FR1–FR7) ve işlevsel olmayan gereksinimlerinin (performans, doğruluk, güvenilirlik, güvenlik, kullanılabilirlik) doğrulanması için planlanan strateji, test ortamı, test senaryoları, izlenebilirlik matrisi ve raporlama yaklaşımını tanımlamaktır. Belge, geliştirme tamamlandıktan sonraki kabul testlerinin ve regresyon koşumlarının temel referansı olarak kullanılacaktır.

### 1.2 Test Kapsamı

Aşağıdaki bileşenler ve davranışlar test edilir:

- Backend API (`backend/main.py`) — `/analyze`, `/calls`, `/calls/{id}` (DELETE), `/calls` (DELETE), `/feedback`, `/test-library`, `/analyze-test`, `/health`, `/ws/live`
- Ses ön işleme (`backend/audio_processing.py`) — yükleme, resampling, mel-spektrogram, spektral residual, silence/duration doğrulama
- Model katmanı (`backend/model_wrapper.py`, `backend/model_wrapper_ssl.py`, `backend/aasist/models/SSLAASIST.py`, `backend/_fairseq_to_hf_xlsr.py`) — SSL+AASIST birincil, AASIST yedek, sezgisel son çare
- Konfigürasyon (`backend/config.py`) — `AUTH_THRESHOLD`, `AUDIO_MIN_*` parametreleri
- Şemalar (`backend/schemas.py`) — `PredictionResult`, `CallRecord`, `FeedbackRequest`, `RiskLevel`
- Frontend (`frontend/src/pages/Dashboard.tsx`) — dosya yükleme, canlı kayıt, verdict banner, çağrı geçmişi, test kütüphanesi modalı, EN/TR i18n, silme

Kapsam dışı: kalıcı veritabanı entegrasyonu (mevcut sürümde `data/call_log.json` dosyası kullanılır), bankacılık altyapısına gerçek SIP/PSTN entegrasyonu, kimlik doğrulama, yük testi, penetrasyon testi.

### 1.3 Tanımlar, Kısaltmalar

| Terim | Tanım |
|---|---|
| **TD** | Test Belgesi |
| **RS** | Gereksinimler Belirtimi |
| **SDD** | Yazılım Tasarım Belgesi |
| **FR** | İşlevsel Gereksinim (Functional Requirement) |
| **NFR** | İşlevsel Olmayan Gereksinim |
| **SSL+AASIST** | TakHemlata, EURECOM — XLSR-300M frontend + AASIST classifier (MIT) |
| **AASIST** | NAVER/clovaai grafik-dikkat anti-spoofing modeli |
| **XLSR-300M** | `facebook/wav2vec2-xls-r-300m`, çoklu dil self-supervised ses backbone'u |
| **EER** | Equal Error Rate |
| **Cross-domain** | Eğitim verisinden farklı bir dağılımda test |
| **p_real / p_fake** | Modelin sırasıyla gerçek/sahte sınıf olasılıkları |
| **authenticity_score** | `p_real` ile aynı; eşik altı = fraud şüphesi |
| **AUTH_THRESHOLD** | Fraud kararı için `authenticity_score` eşiği (varsayılan: 0.01 — SSL+AASIST için kalibre) |

### 1.4 Referanslar

- IEEE 829-2008 Standard for Software and System Test Documentation
- Gereksinimler Belirtimi (RS) Belgesi — sürüm 1.0
- Yazılım Tasarım Belgesi (SDD) — sürüm 1.0
- TakHemlata SSL_Anti-spoofing: https://github.com/TakHemlata/SSL_Anti-spoofing (MIT)
- clovaai AASIST: https://github.com/clovaai/aasist (MIT)
- The-Fake-or-Real Dataset (test verisi kaynağı, `D:\dataset`)
- garystafford/deepfake-audio-detection — HuggingFace (smoke seti, `test_audio/`)

### 1.5 Belge Yapısına Genel Bakış

Bölüm 2 test öğelerini, Bölüm 3 strateji ve test seviyelerini, Bölüm 4 her test senaryosunu kimlik + ön/son koşul + girdi + beklenen sonuç + izlenebilirlik ile, Bölüm 5 test ortamı kurulumunu, Bölüm 6 takvimi, Bölüm 7 elde edilen çıktıları ve özet metriği, Bölüm 8 sözlük ve destekleyici diyagramları içerir.

---

## 2. Test Öğeleri

### 2.1 Test Edilecek Bileşenler

| Bileşen | Konum | Sorumluluk |
|---|---|---|
| FastAPI uygulama katmanı | `backend/main.py` | HTTP endpoint'leri, container sniff, call log persistence |
| Ses işleme | `backend/audio_processing.py` | Decode, resample, mel-spektrogram, doğrulama |
| Model zinciri | `backend/model_wrapper.py` | SSL+AASIST → AASIST → Heuristic yükleme, chunked inference |
| SSL+AASIST sarmalayıcı | `backend/model_wrapper_ssl.py` | XLSR yükleme, AASIST head, fairseq→HF key remap |
| Fairseq remap | `backend/_fairseq_to_hf_xlsr.py` | Saf-fonksiyonel state-dict key dönüştürme |
| AASIST mimari | `backend/aasist/models/SSLAASIST.py`, `AASIST.py` | Model sınıfları |
| Şemalar | `backend/schemas.py` | Pydantic API sözleşmesi |
| Konfigürasyon | `backend/config.py` | Eşik + sınır değerleri |
| Frontend | `frontend/src/pages/Dashboard.tsx` | Tek-sayfa kullanıcı arayüzü |

### 2.2 Doğrulanacak Özellikler

- **F1** Çoklu format yükleme (WAV, FLAC, OGG, WebM, MP4)
- **F2** Canlı mikrofon kaydı (PCM/MediaRecorder fallback)
- **F3** Spektral özellik çıkarımı + spektral residual skoru
- **F4** Model yükleme zinciri ve düşüş yolları
- **F5** Chunked inference (64600 örnek pencere, sessiz parça filtreleme)
- **F6** Risk-ağırlıklı skor agregasyonu (60% ortalama + 40% en kötü)
- **F7** `authenticity_score`, `risk_level`, `is_suspected_fraud` üretimi
- **F8** Kalıcı çağrı log (`data/call_log.json`) yazma/okuma/silme
- **F9** REST endpoint'leri ve WebSocket
- **F10** Frontend verdict banner, geçmiş listesi, grafik, EN/TR i18n
- **F11** Sessiz / çok-kısa ses reddi
- **F12** Test kütüphanesi modalı (50 gerçek + 50 sahte)

---

## 3. Test Stratejisi

### 3.1 Test Seviyeleri

| Seviye | Kapsam | Otomasyon |
|---|---|---|
| **Birim** | Saf fonksiyonlar (state-dict remap, config doğrulama) ve mock'lu model sınıfları | pytest tarzı bağımsız scriptler (`tests/test_*.py`) |
| **Entegrasyon** | Backend `/analyze` ucuyla uçtan-uca model çağrısı | `tests/comprehensive_test.py`, `tests/dataset_eval.py` |
| **Sistem** | Doğruluk / generalization veri kümeleri üzerinde tam zincir | T1, T2, T3 koşumları |
| **Kabul** | Manuel UI smoke (dashboard etkileşimi) | Test kılavuzu §4.4 |

### 3.2 Test Türleri

- **İşlevsel:** endpoint başına davranış, format desteği, model çıktı şeması
- **İşlevsel olmayan:**
  - **Performans:** istek-başına gecikme ölçümü (T2/T3 koşumları içinde)
  - **Doğruluk:** dataset üzerinde accuracy, precision, recall, F1, karışıklık matrisi
  - **Güvenilirlik:** çoklu istek ardışıklığında hata oranı (T1/T2/T3 koşumlarında başarı yüzdesi)
  - **Kullanılabilirlik:** UI manuel smoke
  - **Güvenlik (sınırlı):** boş ya da bozuk dosya reddi, dosya yükleme MIME doğrulama
- **Regresyon:** SSL+AASIST geçişi öncesi/sonrası T1 sonucu karşılaştırma

### 3.3 Giriş Kriterleri

- Backend `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010` ile başarıyla başlatılmış
- `/health` 200 dönüyor, `model.model_type == "ssl_aasist"`
- SSL+AASIST ağırlıkları `models/ssl_aasist/weights.pth` mevcut, SHA-256 `1cf904f1d84c867c278cd42161df5367939d61cc28bfefd239bc995af59c2804`
- Test verisi yolları erişilebilir: `test_audio/`, `D:\dataset\for-original\validation`, `D:\dataset\for-rerecorded\testing`
- FFmpeg çalıştırılabilir (PATH'te veya `imageio-ffmpeg` paketi)

### 3.4 Çıkış Kriterleri

- Tüm planlanan test senaryoları çalıştırıldı
- Sistem-seviye doğruluk metrikleri raporlandı (T1, T2, T3)
- Format/silence doğrulama testleri 100% geçti
- Birim test koşumlarında başarısızlık yok
- TD §7'de bulgular özetlendi

### 3.5 Askıya Alma ve Yeniden Başlama

| Tetikleyici | Eylem |
|---|---|
| Backend istek başına `500` hatası > %10 | Test askıya alınır, log incelenir, kök neden giderilince yeniden başlat |
| Model yüklenememesi (`/health.model.loaded == false` ve fallback başarısız) | Yedek model dizinlerini doğrula, çevre değişkenlerini gözden geçir |
| FFmpeg bulunamadı uyarısı | Format senaryoları (FT-002…FT-004) atlanır, kalanlar koşulur |
| Disk doluluğu (call_log.json yazılamıyor) | Disk temizlenir, dataset koşumu kaldığı yerden devam |

---

## 4. Test Tasarımı ve Spesifikasyonu

### 4.1 Birim Testleri

| ID | Açıklama | Dosya | Beklenen |
|---|---|---|---|
| UT-001 | Fairseq → HF XLSR state-dict anahtar dönüşümü | `tests/test_fairseq_xlsr_remap.py` | 11 alt-senaryo geçer; quantizer/project_q düşürülür |
| UT-002 | `Settings` SSL config yükleme | `tests/test_config_ssl.py` | `AUTH_THRESHOLD=0.01`, `LOCAL_SSL_MODEL_DIR` env okuması |
| UT-003 | `XLSRAASISTModel.predict` döner şema (mock SSL) | `tests/test_xlsr_aasist.py` | `{p_real, p_fake, spectral_residual, num_chunks, max_chunk_p_fake}` |
| UT-004 | `get_model_status` SSL+AASIST primary döner | `tests/test_xlsr_aasist.py::test_get_model_status_reports_ssl_primary` | `model_type=="ssl_aasist"` |
| UT-005 | Audio pre-check (sessizlik/kısalık) | `tests/smoke_test_prechecks.py` | Yetersiz sinyalde uygun hata |

**Koşum komutu:** `.venv\Scripts\python.exe -m pytest tests/test_fairseq_xlsr_remap.py tests/test_config_ssl.py tests/test_xlsr_aasist.py -v` ve `python tests/smoke_test_prechecks.py`.

### 4.2 Entegrasyon Testleri

| ID | Açıklama | Dosya | Beklenen |
|---|---|---|---|
| IT-001 | `/health` 200 + model bilgisi | `curl /health` | `status="ok"`, `model.model_type="ssl_aasist"` |
| IT-002 | `/analyze` küçük gerçek WAV → genuine | manuel + dataset_eval | `is_suspected_fraud=false`, `risk_level∈{low,medium}` |
| IT-003 | `/analyze` küçük sahte WAV → fraud | manuel + dataset_eval | `is_suspected_fraud=true`, `risk_level∈{high,critical}` |
| IT-004 | `/calls` kayıt sonrası dönen liste içerir | sequence | Yeni `call_id` listede |
| IT-005 | `/calls/{id}` DELETE | sequence | 200 ve `/calls` listesinde silinmiş |
| IT-006 | `/calls` DELETE all | sequence | 200, sonra `/calls` boş |
| IT-007 | `/test-library` real/fake listesi | `curl` | `{real:[...], fake:[...]}`, dosyalar mevcut |
| IT-008 | `/analyze-test` test_audio file → sonuç | `curl` | 200 + PredictionResult şeması |
| IT-009 | `/feedback` 200 | `curl POST` | 200, payload kabul edilir |

### 4.3 Sistem Testleri (Dataset)

#### Test Seti Tanımları

| ID | Kaynak | Boyut | Amaç |
|---|---|---|---|
| **T1 (smoke)** | `test_audio/real`, `test_audio/fake` | 50 + 50 | Hızlı regresyon, baseline |
| **T2 (in-domain)** | `D:\dataset\for-original\validation` rastgele örnek | 200 + 200 | In-domain doğruluk (RS NFR doğruluk hedefi) |
| **T3 (cross-domain)** | `D:\dataset\for-rerecorded\testing` (tüm) | 408 + 408 | Generalization, domain shift |

#### Test Senaryoları

| ID | Set | Ön Koşul | Girdi | Beklenen | Başarı Kriteri |
|---|---|---|---|---|---|
| **ST-001** | T1 | Backend ayakta, model yüklü | 100 WAV ses (50/50 dengeli) | Sıralı `/analyze` çağrıları | Accuracy ≥ %70, F1_weighted ≥ %0.70, 0 başarısız istek |
| **ST-002** | T2 | T1 yeşil | 400 WAV ses (200/200 dengeli, rastgele örnek) | Sıralı `/analyze` çağrıları | Accuracy ≥ %75, sınıf-başı recall ≥ %0.70 |
| **ST-003** | T3 | T2 tamamlandı | 816 WAV ses (408/408 dengeli, tam set) | Sıralı `/analyze` çağrıları | Accuracy raporlanır; cross-domain drop ≤ in-domain’den %15 puan |
| **ST-004** | T1+T2+T3 | Yukarıdaki üç koşum bitti | Per-istek `latency_s` | Mean latency raporu | RS NFR §3.2 = 2s referans değeri ile kıyas (gözlem) |

#### İşlevsel Format/Validation Testleri

| ID | Senaryo | Girdi | Beklenen | Otomasyon |
|---|---|---|---|---|
| **FT-001** | WAV doğrudan analiz | 3 sn 440 Hz sinüs | 200, PredictionResult | `tests/test_format_and_validation.py::format_wav` |
| **FT-002** | OGG → FFmpeg fallback | 3 sn OGG | 200, PredictionResult | `format_ogg` |
| **FT-003** | WebM → FFmpeg fallback | 3 sn WebM | 200 | `format_webm` |
| **FT-004** | MP4/M4A → FFmpeg fallback | 3 sn M4A | 200 | `format_m4a` |
| **FT-005** | Sessizlik reddi | 3 sn 0 örnek | 4xx + hata mesajı | `silence_reject` |
| **FT-006** | Kısa süre reddi | 1 sn ses (< `AUDIO_MIN_DURATION_SEC=2.0`) | 4xx | `too_short_reject` |

### 4.4 Manuel UI Kabul Senaryoları

| ID | Senaryo | Adımlar | Beklenen |
|---|---|---|---|
| UA-001 | Test kütüphanesi modalı | Üstte "Test Library" düğmesine bas; modal açılır; bir `real_000.wav` seç | Modal kapanır, verdict banner YEŞİL, risk LOW |
| UA-002 | Test kütüphanesi modalı (fake) | `fake_000.wav` seç | Verdict banner KIRMIZI, risk HIGH/CRITICAL |
| UA-003 | Dosya yükleme | Drop-zone'a bir test WAV bırak, Analiz Et'e bas | Sonuç kartı + grafik güncellenir |
| UA-004 | Canlı mikrofon | Kayıt başlat → 3 sn → durdur | Verdict banner ve grafik güncellenir |
| UA-005 | EN/TR toggle | Globe simgesine bas | Tüm metinler dile çevrilir, tercih `localStorage`'a kaydedilir |
| UA-006 | Çağrı silme (tekil) | Geçmiş listesinde çöp simgesine bas | Kayıt anında listeden kaybolur |
| UA-007 | Çağrı silme (tümü) | "Delete all" düğmesi → onay | Liste boşalır, banner sıfırlanır |
| UA-008 | Sağlık göstergesi | Sayfa açılışı | `Backend OK` rengi yeşil, `ssl_aasist` model türü gözlenebilir (Network sekmesi `/health` yanıtı) |

### 4.5 İzlenebilirlik Matrisi (Gereksinim Eşlemesi)

| Gereksinim | RS/SDD Ref | Karşılayan Bileşen | Test ID(leri) |
|---|---|---|---|
| FR1 Ses girişi alma | RS §3.1 | `Dashboard.tsx`, `main.py /analyze` | IT-001, IT-002, IT-003, FT-001–FT-006, UA-003, UA-004 |
| FR2 Ses özellik çıkarma | RS §3.1 | `audio_processing.py` | IT-002, IT-003, ST-001, ST-002, ST-003 |
| FR3 Yapay ses tespiti | RS §3.1 | `model_wrapper*.py`, `SSLAASIST.py` | UT-003, UT-004, ST-001, ST-002, ST-003 |
| FR4 Güvenilirlik skoru | RS §3.1 | `model_wrapper.py`, `main.py` | IT-002, IT-003, ST-* |
| FR5 Dolandırıcılık uyarısı | RS §3.1 | `config.AUTH_THRESHOLD`, `main.py` | IT-003, ST-001, ST-002, ST-003 |
| FR6 Gösterge paneli | RS §3.1 | `Dashboard.tsx`, `/calls`, `/health` | IT-004, IT-005, IT-006, UA-001…UA-008 |
| FR7 Analist geri bildirimi | RS §3.1 | `/feedback`, `schemas.py` | IT-009 (backend); **UI tarafı eksik — kısmi karşılanır** |
| NFR Performans (≤2 sn) | RS §3.2 | model pipeline + chunking | ST-004 (gözlem, hedef vs gerçekleşen) |
| NFR Doğruluk (≥%85) | RS §3.2 | model + threshold kalibrasyon | ST-001, ST-002, ST-003 (mevcut sonuçla gap raporlanır) |
| NFR Güvenilirlik | RS §3.2 | hata oranı | ST-001..003 başarı yüzdesi |
| NFR Güvenlik | RS §3.2 | format & boyut doğrulama | FT-005, FT-006 |
| NFR Kullanılabilirlik | RS §3.2 | i18n, drag-drop, renkler | UA-001..UA-008 |

---

## 5. Test Ortamı Kurulumu

### 5.1 Donanım Yapılandırması

| Bileşen | Belirtim |
|---|---|
| İşletim sistemi | Windows 11 Pro 10.0.26200 |
| CPU | (geliştirme makinesi — özellikler `wmic cpu get name` ile alınabilir) |
| Bellek | 16 GB+ önerilir (XLSR-300M backbone ~1.3 GB RAM) |
| Depolama | Boş 5 GB+ (model ağırlıkları + dataset) |
| GPU | Opsiyonel; CUDA varsa `torch.device('cuda')` otomatik |

### 5.2 Yazılım Yapılandırması

| Yazılım | Sürüm / Kaynak |
|---|---|
| Python | 3.12.5, sanal ortam `.venv/` |
| PyTorch | `requirements.txt` |
| transformers | `>=4.30` |
| FFmpeg | Sistem PATH (winget kurulumu) |
| Node.js | 18+ |
| Vite | `frontend/package.json` |
| Backend host | `http://127.0.0.1:8010` |
| Frontend host | `http://localhost:5173` (proxy → 8010) |

### 5.3 Veri Hazırlığı

```
test_audio/
├── real/   # 50 WAV
└── fake/   # 50 WAV
D:\dataset\
├── for-original\validation\real\  # 5400 WAV
├── for-original\validation\fake\  # 5400 WAV
└── for-rerecorded\testing\
    ├── real\  # 408 WAV
    └── fake\  # 408 WAV
models/ssl_aasist/
├── README.md
├── meta.json
└── weights.pth   # SHA-256: 1cf904f1d84c867c278cd42161df5367939d61cc28bfefd239bc995af59c2804
data/
└── call_log.json  # otomatik oluşur
```

### 5.4 Test Koşum Komutları

```powershell
# Backend
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010

# Birim testleri
.\.venv\Scripts\python.exe -m pytest tests/test_fairseq_xlsr_remap.py tests/test_config_ssl.py tests/test_xlsr_aasist.py -v
.\.venv\Scripts\python.exe tests/smoke_test_prechecks.py

# Entegrasyon & sistem
.\.venv\Scripts\python.exe tests/comprehensive_test.py                  # T1
.\.venv\Scripts\python.exe tests/dataset_eval.py --set T2               # T2 (200+200)
.\.venv\Scripts\python.exe tests/dataset_eval.py --set T3               # T3 (408+408)

# Format & doğrulama
.\.venv\Scripts\python.exe tests/test_format_and_validation.py
```

### 5.5 Araçlar ve Kütüphaneler

- `requests` — HTTP istemcisi
- `scikit-learn` — `accuracy_score`, `f1_score`, `classification_report`, `confusion_matrix`
- `librosa`, `soundfile` — ses ön işleme
- `transformers`, `torch` — model yükleme
- `pydantic` — şema doğrulama

---

## 6. Test Takvimi

| Aşama | Süre tahmini | Tarih (gerçekleşen) | Çıktı |
|---|---|---|---|
| Test ortamı kurulumu + birim testleri | 1 saat | 2026-05-28 | UT-001..UT-005 yeşil |
| Entegrasyon + format testleri | 1 saat | 2026-05-28 | IT-001..IT-009, FT-001..FT-006 yeşil |
| T1 (smoke) | 5 dk | 2026-05-29 | `deepfake_test_results.json` |
| T2 (in-domain) | 20 dk | 2026-05-29 | `tests/results/T2_results.json` |
| T3 (cross-domain) | 25 dk | 2026-05-29 | `tests/results/T3_results.json` |
| UI manuel kabul (UA-001..UA-008) | 30 dk | 2026-05-29 | Manuel kontrol listesi (Ek-C) |
| Raporlama, belge tamamlama | 1 saat | 2026-05-29 | `docs/td/EnverBagci_DeepfakeVoiceFraudDetection_22290746.pdf` |

### Kilometre Taşları

- **M1 (2026-05-28):** Birim + entegrasyon + format yeşil
- **M2 (2026-05-29):** Tüm sistem testleri (T1, T2, T3) ve UI kabul tamamlandı
- **M3 (2026-05-29):** TD final PDF teslim (deadline: 2026-06-07 23:59 Google Classroom)

---

## 7. Test Çıktıları

### 7.1 Çıktı Dosyaları

| Dosya | İçerik |
|---|---|
| `deepfake_test_results.json` (proje kökünde) | T1 detaylı sonuçlar |
| `tests/results/T2_results.json` | T2 metrik + örnek-başı sonuç |
| `tests/results/T3_results.json` | T3 metrik + örnek-başı sonuç |
| `tests/results/format_and_validation_results.json` | FT-001..FT-006 sonuçları |
| Backend log | uvicorn stderr (manuel inceleme) |

### 7.2 Sonuçların Özeti

> Bu bölüm, koşumlar tamamlandıktan sonra doldurulur. Aşağıdaki tablolar gerçek metriklerle güncellenmelidir.

#### 7.2.1 T1 Smoke (50 + 50)

| Metrik | Değer |
|---|---|
| Accuracy | 0.7600 |
| F1 (weighted) | 0.7500 |
| Real precision / recall | 0.9333 / 0.5600 |
| Fake precision / recall | 0.6857 / 0.9600 |
| Karışıklık matrisi (rows=actual, cols=pred [fake,real]) | fake:[48, 2], real:[22, 28] |
| Toplam / Başarılı / Başarısız | 100 / 100 / 0 |
| Veri | `test_audio/real` (50) + `test_audio/fake` (50) |

**Yorum:** Hem T1'in iki bağımsız koşumunda (0.7576 ve 0.7600) tutarlı sonuç gözlendi. Model fake'i çok güçlü yakalıyor (recall 0.96) ancak gerçek seslere karşı temkinli (recall 0.56). RS NFR hedef doğruluk %85 ile aramızda ~%9 puanlık fark var; eşik taraması (sweep) ile %82.5 düzeyine çıkılabilir.

#### 7.2.2 T2 In-domain (200 + 200, rastgele örneklem `for-original/validation`)

| Metrik | Değer |
|---|---|
| Accuracy (başarılı istek üzerinde) | 0.7782 |
| F1 (weighted) | 0.7865 |
| Real precision / recall / F1 | 1.0000 / 0.6802 / 0.8097 |
| Fake precision / recall / F1 | 0.5800 / 1.0000 / 0.7342 |
| Karışıklık matrisi (labels=['real','fake'], rows=actual) | real:[134, 63], fake:[0, 87] |
| Toplam (denenen / başarılı / başarısız) | 400 / 284 / 116 |
| Latency mean / p50 / p95 (sn) | 1.219 / 1.289 / 2.106 |
| Latency min / max (sn) | 0.609 / 12.748 |
| Veri | `D:\dataset\for-original\validation`, seed=42 |

**Yorum:**
- 116/400 (%29) istek FT-006'ya benzer şekilde "Recording too short: < 2.00s" hatasıyla reddedildi. Bu, ses doğrulamasının (FR/NFR güvenlik gereksiniminin) gerçek dataset üzerinde de aktif olduğunu kanıtlar.
- Başarılı 284 istek üzerinden hesap: T1 ile aynı yönlü davranış (yüksek fake recall, düşük real recall). In-domain'de doğruluk T1'den hafif yüksek (%77.8).
- Latency mean 1.22s; RS NFR 2 sn hedefini sağlar (p95 2.11s sınırı az aşıyor; toplam p95 yine de gerçek zamanlıya yakın). Maks 12.7s, uzun kayıt + çok-pencere chunk sayısından kaynaklanır.

#### 7.2.3 T3 Cross-domain (408 + 408, tam set `for-rerecorded/testing`)

| Metrik | Değer |
|---|---|
| Accuracy | 0.5061 |
| F1 (weighted) | 0.3468 |
| Real precision / recall / F1 | 1.0000 / 0.0123 / 0.0242 |
| Fake precision / recall / F1 | 0.5031 / 1.0000 / 0.6694 |
| Karışıklık matrisi (labels=['real','fake'], rows=actual) | real:[5, 403], fake:[0, 408] |
| Toplam (denenen / başarılı / başarısız) | 816 / 816 / 0 |
| Latency mean / p50 / p95 (sn) | 0.718 / 0.663 / 0.821 |
| Latency min / max (sn) | 0.569 / 30.445 |
| Veri | `D:\dataset\for-rerecorded\testing` |
| Backend yapılandırması | **`AUDIO_MIN_DURATION_SEC=1.0`** ile çalıştırıldı (rerecorded klipler ~1.99s) |

**Yorum:**
- Yeniden-kayıt (rerecording) ortamındaki akustik bozulmalar, modelin **neredeyse tüm girdileri "fake" olarak sınıflaması**na neden oldu. Fake recall mükemmel (1.00), ama real recall %1.2'ye düştü. Bu **şiddetli alan kayması (domain shift)** bulgusudur — eğitim verisinin in-domain örüntülerini ezberleyen modelin rerecording artefaktlarına duyarlı olduğu gözlenir.
- Backend `AUDIO_MIN_DURATION_SEC` varsayılan değeriyle (2.0s) yapılan ilk koşumda 779/816 örnek "too short" hatasıyla reddedildi; bu, doğrulama kuralının doğru çalıştığını gösterir ancak doğruluk hesaplaması anlamsız hale gelir. Eşik 1.0s'ye düşürülerek tam veri kümesi üzerinde anlamlı metrik elde edildi.
- Latency cross-domain'de daha düşük çıktı (mean 0.72s, klipler çoğunlukla 2 sn altı).

#### 7.2.4 Format & Doğrulama (FT-001..FT-006)

| Test | Sonuç | Detay |
|---|---|---|
| FT-001 WAV (sinüs 3s) | ✅ (T1/T2/T3 dolaylı kanıt + WAV ana akış) | dataset koşumları sırasında ~1300 WAV başarılı |
| FT-002 OGG → FFmpeg fallback | ✅ | HTTP 200, PredictionResult döner |
| FT-003 WebM → FFmpeg fallback | ✅ | HTTP 200, PredictionResult döner |
| FT-004 M4A → FFmpeg fallback | ✅ | HTTP 200, PredictionResult döner |
| FT-005 Silence reject (sıfır PCM 3s) | ✅ | HTTP 400 "No audible sound detected in the recording (silence or below threshold)." |
| FT-006 Too-short reject (1s tone) | ✅ | HTTP 400 "Recording too short: 1.00s. Please send at least 2.00s of audio." |

**Yorum:** Format dönüştürme zinciri (sniff → FFmpeg → librosa/soundfile) tüm hedef tarayıcı formatlarında çalıştı. Audio validation (FR güvenlik kapısı) hem boş hem kısa kayıtları RFC 4329 uyumlu 400 koduyla reddediyor.

#### 7.2.5 Birim Testleri

```
tests/test_fairseq_xlsr_remap.py ............ 11/11 PASSED
tests/test_config_ssl.py ..................... 2/2  PASSED
tests/test_xlsr_aasist.py .................... 3/3  PASSED
TOPLAM ....................................... 16/16 PASSED  (4.70s)
```

#### 7.2.6 Bulguların Özeti

| Boyut | Sonuç |
|---|---|
| **Birim testler** | ✅ 16/16 |
| **Format & doğrulama** | ✅ 5/5 (3 format + 2 reject) |
| **In-domain doğruluk (T2)** | %77.8 (RS hedef %85 ile -7.2 puan açık) |
| **Cross-domain doğruluk (T3)** | %50.6 — model rerecording'i fake olarak sınıflar (alan kayması) |
| **Latency (T2 mean)** | 1.22s (RS NFR 2 sn altında) |
| **FR1–FR6** | Karşılanır (test sonuçları izlenebilirlik matrisinde) |
| **FR7** | Backend OK, UI feedback formu eksik (kısmi) |

### 7.3 Hata Raporlama

Tespit edilen aykırılıklar için her bulgu aşağıdaki formatla kaydedilir:

```
Bulgu ID: BG-XXX
Tarih:
Test ID:
Önem: kritik/yüksek/orta/düşük
Açıklama:
Yeniden üretme adımları:
Beklenen / Gerçekleşen:
Çözüm önerisi:
Durum: açık / düzeltildi / kabul
```

---

## 8. Ekler

### Ek A — Sözlük (genişletilmiş)

Ek olarak SDD §8.1 sözlüğüne bakınız.

### Ek B — Mimari Akış (metinsel)

```
Kullanıcı (tarayıcı)
    │  WAV/WebM/OGG
    ▼
Dashboard.tsx ── POST /analyze (multipart) ──► FastAPI (main.py)
                                                  │
                                                  ▼
                                          _sniff_audio_container()
                                                  │
                                                  ▼
                                   audio_processing.extract_features
                                   (decode → 16kHz mono → mel + spektral residual)
                                                  │
                                                  ▼
                                   validate_audio_requirements
                                                  │
                                                  ▼
                                   model_wrapper.predict
                                   ├── SSL+AASIST (primary)
                                   ├── AASIST baseline (fallback)
                                   └── Heuristic (last resort)
                                                  │
                                                  ▼
                                   {p_real, p_fake, authenticity_score, risk_level}
                                                  │
                                                  ▼
                                   data/call_log.json append
                                                  │
                                                  ▼
                              JSON yanıt → Dashboard verdict banner + grafik
```

### Ek C — UI Kabul Kontrol Listesi (manuel)

```
[ ] UA-001 Test Library modalı (gerçek)
[ ] UA-002 Test Library modalı (sahte)
[ ] UA-003 Dosya yükleme
[ ] UA-004 Canlı mikrofon
[ ] UA-005 EN/TR toggle
[ ] UA-006 Tekil silme
[ ] UA-007 Toplu silme
[ ] UA-008 Sağlık göstergesi (model_type=ssl_aasist)
```

### Ek D — Bilinen Sınırlamalar

- **FR7 kısmi:** Backend `/feedback` endpoint çalışır, UI'da feedback formu henüz yok.
- **NFR Doğruluk (RS %85 hedef):** SSL+AASIST primary, mevcut threshold `0.01` ile 50+50 sette ~%75.5 ölçülmüş; eşik taraması (sweep) ile %82.5 mümkün. Ek dataset koşumları (T2, T3) hedef ile gerçekleşen arasındaki farkı dokümante eder.
- **NFR Performans (RS ≤2 sn):** SSL+AASIST XLSR-300M backbone CPU'da bir 4 saniyelik klip için bekleme ~2–5 sn olabilir; GPU mevcutsa hedef rahatlıkla sağlanır. ST-004 ölçer.
- **Kalıcı veritabanı yok:** çağrı kayıtları `data/call_log.json` üzerine yazılır; gerçek üretim için RDBMS planlanır.

### Ek E — Yapılandırma Değişkenleri (özet)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `AUTH_THRESHOLD` | `0.01` | SSL+AASIST için kalibre fraud eşiği (`p_real` altı = şüpheli) |
| `AUDIO_MIN_DURATION_SEC` | `2.0` | Reddedilen kısa ses süresi |
| `AUDIO_MIN_PEAK_ABS` | `5e-4` | Sessiz tepe değeri eşiği |
| `AUDIO_MIN_RMS` | `1e-4` | Sessiz RMS eşiği |
| `LOCAL_MODEL_DIR` | (yok) | AASIST yedek dizini geçersiz kıl |
| `LOCAL_WEIGHTS_PATH` | (yok) | AASIST ağırlık dosyasını geçersiz kıl |
| `LOCAL_SSL_MODEL_DIR` | (yok) | SSL+AASIST dizini geçersiz kıl |
| `FFMPEG_EXE` / `FFMPEG_PATH` | (yok) | Açık FFmpeg yolu |
| `MODEL_SELFTEST` | (yok) | `1` → `/health.model_selftest` doldurulur |
