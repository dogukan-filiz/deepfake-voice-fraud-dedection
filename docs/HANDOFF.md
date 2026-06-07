# HANDOFF — MongoDB Persistence for Analysis Outputs

> Son güncelleme: 2026-06-06. Sonraki agent bu dosyayı okuyup devam edebilir.
> Proje: `D:\Workspace\deepfake-voice-fraud-dedection` (Windows local). MacBook'ta da çalışacak.

---

## Goal

Danışman hocanın geliştirme önerisi #1: **ses analizi çıktılarını bir veritabanında sakla.**
Karar: **MongoDB** kullanılacak (kullanıcı seçti). Hem Windows hem MacBook'ta çalışmalı.
Hedef: "git sync yapar yapmaz çalışan uygulama" — Mongo yoksa bile uygulama çökmemeli (graceful fallback).

---

## Current Progress

- DB seçimi yapıldı: **MongoDB** (SQLite ve Atlas cloud elendi; kullanıcı local MongoDB istedi).
- Henüz kod yazılmadı. Mevcut storage analiz edildi (aşağıda entegrasyon noktaları).
- Bu handoff oluşturuldu.

---

## Mevcut Sistem — saklama bu an nasıl çalışıyor

Şu an çıktılar JSON dosyada tutuluyor: `data/call_log.json`.

- `backend/main.py:66` — `_CALL_LOG_FILE = .../data/call_log.json`
- `backend/main.py:69` `_load_call_log()` — açılışta JSON oku → `list[CallRecord]`
- `backend/main.py:79` `_save_call_log(log)` — tüm listeyi her seferinde JSON'a yeniden yaz
- `backend/main.py:87` `_CALL_LOG: list[CallRecord]` — bellekte global liste

**Sorun:** her analizde tüm dosya yeniden yazılıyor, eşzamanlılık güvenliği yok, sorgu yok.

### Veri modeli (`backend/schemas.py`)

`CallRecord` (saklanan kayıt):
```python
class CallRecord(BaseModel):
    call_id: str
    authenticity_score: float
    is_suspected_fraud: bool
    risk_level: RiskLevel = RiskLevel.MEDIUM   # low|medium|high|critical
    timestamp: datetime
    notes: Optional[str] = None
```
`PredictionResult` = API yanıtı (daha geniş: p_real, p_fake, spectral_residual, num_chunks, max_chunk_p_fake). NOT: şu an sadece `CallRecord` alanları saklanıyor; PredictionResult'taki ekstra skorlar (p_real/p_fake/spectral_residual) DB'ye yazılmıyor. Hoca için iyi geliştirme: bu skorları da sakla → schema genişlet.

### Değiştirilecek 5 storage noktası (hepsi backend/main.py)

| Yer | Satır | İşlem |
|---|---|---|
| `/analyze` | ~196-197 | `_CALL_LOG.append(record)` + `_save_call_log` → DB insert |
| `/analyze-test` | ~398-407 | aynı insert |
| `GET /calls` | ~213-216 | son N kayıt (reversed, limit) → DB find sorted desc |
| `POST /feedback` | ~219-243 | call_id ile bul, notes güncelle → DB update_one |
| `DELETE /calls/{call_id}` | ~246-254 | tek sil → DB delete_one |
| `DELETE /calls` | ~257+ | hepsini sil → DB delete_many |

---

## What Worked

- Mevcut `.env` config full_run_002 modeliyle çalışıyor (XLSR+AASIST head). Backend ayağa kalkıyor.
- `.env` UTF-8 **NO BOM** olmalı (pydantic-settings BOM'da patlıyor). Yazarken:
  `python -c "open('.env','w',encoding='utf-8',newline='\n').write(c)"`
- config.py geçerli anahtar: `MODEL_BACKEND` (MODEL_NAME DEĞİL — eski .env'de MODEL_NAME vardı, crash etti).

## What Didn't Work / Tuzaklar

- `.env`-de `MODEL_NAME=...` → `ValidationError: model_name Extra inputs are not permitted`. Doğrusu `MODEL_BACKEND`.
- `Out-File -Encoding UTF8` / `Set-Content -Encoding UTF8` → BOM ekler → pydantic crash.
- Bash tool'da Windows path backslash bozuluyor; forward slash kullan (`.venv/Scripts/python.exe`).
- `gh` ve bazen `python` json pipe Bash'te sorunlu; ToolSearch→WebFetch veya curl daha güvenilir.

---

## MongoDB Implementation Plan (Next Steps)

### 1. Kurulum (her iki makinede)
- **Windows:** MongoDB Community Server MSI installer → Windows service olarak çalışır (otomatik başlar). Port 27017.
- **MacBook:** `brew tap mongodb/brew && brew install mongodb-community && brew services start mongodb-community`. Port 27017.
- Doğrula: `mongosh --eval "db.runCommand({ping:1})"`.

### 2. Python driver
- **pymongo** (sync) öner — endpoint'ler async ama düşük yükte sorun yok, en basit. (Alternatif: `motor` async, daha karmaşık.)
- `pip install pymongo` → `requirements.txt`'e ekle.

### 3. Config (`backend/config.py` + `.env` + `.env.example`)
Yeni alanlar:
```python
MONGODB_URI: str = "mongodb://localhost:27017"
MONGODB_DB: str = "deepfake_fraud"
MONGODB_COLLECTION: str = "calls"
PERSISTENCE_BACKEND: str = "mongo"   # "mongo" | "json"  (fallback switch)
```

### 4. Repository katmanı — `backend/storage.py` (YENİ dosya)
- `CallStore` interface: `add(record)`, `list(limit)`, `update_notes(call_id, notes)`, `delete(call_id)`, `delete_all()`.
- İki implementasyon: `MongoCallStore`, `JsonCallStore` (mevcut JSON mantığını taşı).
- **GRACEFUL FALLBACK (kritik):** açılışta Mongo'ya ping at; bağlanamazsa otomatik `JsonCallStore`'a düş + uyarı logla. Böylece "Mongo açık değil" demo'yu çökertmez. "git sync sonrası çalışsın" hedefi için zorunlu.
- `call_id` üzerine unique index oluştur.

### 5. main.py entegrasyonu
- `_CALL_LOG` / `_load_call_log` / `_save_call_log` yerine `store = get_call_store()` kullan.
- Yukarıdaki 5 storage noktasını store metodlarına çevir.

### 6. Migration
- İlk çalıştırmada `data/call_log.json` varsa içeriği Mongo'ya aktar (call_id ile upsert, idempotent). Bir kez çalış, sonra atla.

### 7. (Opsiyonel, hoca puanı) Schema genişlet
- `CallRecord`'a `p_real`, `p_fake`, `spectral_residual`, `model_backend`, `threshold` ekle → DB'ye tam skor seti yaz. Dashboard'da gösterilebilir.

### 8. Test / doğrulama
- Backend başlat, bir analiz yap, `mongosh` ile `db.calls.find()` kayıt görünmeli.
- Mongo'yu durdur → backend yine çalışmalı (JSON fallback).
- `GET /calls`, `/feedback`, `DELETE` uçları MongoDB ile çalışmalı.

---

## Sabit Kurallar (bu proje)
- İzinsiz training YOK. İzinsiz backend/frontend mantık değişikliği YOK (bu görev onaylı feature).
- `test_audio` + `microphone_benchmark` HOLDOUT.
- `models/ssl_aasist/weights.pth` üzerine YAZMA.
- full_run_002/003/004 artifact'larını SİLME.
- Commit'lerde co-author EKLEME.

## Aktif Model Config (referans)
`.env` şu an (full_run_002):
```
MODEL_BACKEND=ssl_aasist
AUTH_THRESHOLD=0.49
LOCAL_SSL_MODEL_DIR=models/ssl_aasist
LOCAL_SSL_WEIGHTS_PATH=training_runs/full_run_002/best_head.pth
CALL_CHANNEL_MODE=false
```
Benchmark: test_audio %100, FoR-original %83.2, FoR-rerecorded %55.5.

## Git
- origin: `https://github.com/dogukan-filiz/deepfake-voice-fraud-dedection.git`
- HEAD: `22d939e` (colleague: dashboard progress bar). Push GCM gerektirir.
- Push öncesi her zaman `git fetch` + ahead/behind kontrol. Local commit yoksa `git pull --ff-only` güvenli.

---

## Fresh start için
Yeni session'da bu dosyayı ver: `docs/HANDOFF.md`
