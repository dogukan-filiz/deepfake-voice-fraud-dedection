# HANDOFF — Persistence + Analysis History + Spoken Result

> Son güncelleme: 2026-06-07 (macOS). Önceki agent'ın bıraktığı MongoDB görevi + 2 ek
> danışman görevi tamamlandı. Bu dosya neyin yapıldığını ve nasıl çalıştırılacağını anlatır.
> Repo (mac): `/Users/dogukanfiliz/Documents/Workspace/deepfake-voice-fraud-dedection`.

---

## Tamamlanan İşler (3 danışman görevi)

### 1. MongoDB persistence — TAMAM ✅  (commit `e54f4c8`)
Analiz çıktıları artık MongoDB'de. JSON dosya tek başına kullanılmıyor.

- **`backend/storage.py` (YENİ):** `CallStore` ABC + `MongoCallStore` + `JsonCallStore`.
  `get_call_store()` factory: `PERSISTENCE_BACKEND=mongo` ise Mongo'ya 1.5s ping;
  ulaşılamazsa otomatik `JsonCallStore`'a düşer (**graceful fallback** — Mongo kapalıyken
  uygulama çökmez, `data/call_log.json`'a yazar). Metodlar: `add/list/get/update_notes/
  delete/delete_all/count/latest`.
- **`backend/main.py`:** eski `_CALL_LOG`/JSON global bloğu kaldırıldı → `store = get_call_store()`.
  6 storage noktası (analyze, analyze-test, GET /calls, feedback, DELETE tek, DELETE hepsi) +
  WS loop store metodlarına çevrildi.
- **`backend/config.py`:** `PERSISTENCE_BACKEND`, `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`.
- **`backend/schemas.py` — CallRecord genişletildi:** `p_real, p_fake, spectral_residual,
  model_backend, threshold` (hepsi Optional, eski kayıt uyumu). Tam skor seti DB'ye yazılıyor.
- **`requirements.txt`:** `pymongo>=4.6` (4.17 kurulu).
- Migration YOK (karar): Mongo boş başladı, eski JSON aktarılmadı.
- DB adı: `deepfake_fraud`, koleksiyon: `calls`. `call_id` unique index.

### 2. Analiz Geçmişi modal + ses oynatma — TAMAM ✅  (commit `e54f4c8`)
- Ana ekrandaki inline "recent analyses" kaldırıldı → nav'da **"Geçmiş"** butonu + modal.
  Modal DB'den (`/calls`) çeker: dosya adı, risk badge, fraud verdict, skor, zaman,
  `<audio controls>` oynatıcı, tekil + toplu sil.
- **Ses saklama:** upload/mic sesleri `data/audio/<call_id>.<ext>` diske yazılır
  (`_store_audio_bytes`), `CallRecord.audio_path`+`audio_filename`'e kaydedilir.
  test-library dosyaları yerinde referanslanır (kopya yok).
- **`GET /calls/{id}/audio`** → `FileResponse` (path-traversal guard `_resolve_audio_path`,
  sadece `data/audio` veya `test_audio` altı). Silince `data/audio` dosyası da silinir,
  `test_audio` orijinaline dokunulmaz.
- **`GET /test-library/audio?category=&filename=`** → test kütüphanesi dosyasını analiz etmeden
  önizleme. Frontend Test Library modalında her dosyaya **play/pause** butonu eklendi.

### 3. Sesli sonuç (TTS) — TAMAM ✅  (commit `1dbe01d`)
- Analiz sonrası doğruluk yüzdesi **otomatik sesli okunur** (`window.speechSynthesis`,
  sıfır backend). Result kartında **hoparlör butonu** ile tekrar.
- Dil UI locale'e göre: TR sesi varsa Türkçe, yoksa İngilizce'ye düşer.
  Metin: TR *"Doğruluk oranı yüzde X. Sahtecilik şüphesi var/yok."* / EN karşılığı.
- `pickBestVoice`: Premium/Enhanced/Siri/Neural sesleri otomatik tercih, yoksa
  Yelda/Samantha, yoksa ilk eşleşen.
- Sadece frontend: `Dashboard.tsx`, `i18n.ts`, `styles.css`.

---

## Çalıştırma (mac, güncel)

### Tek komut (önerilen)
```bash
cd /Users/dogukanfiliz/Documents/Workspace/deepfake-voice-fraud-dedection
bash run_mac.sh
```
`run_mac.sh` artık: `.env` kontrol → venv+deps → **MongoDB başlat + ping bekle (15s)** →
backend `:8010` (arka plan) → frontend Vite. Mongo başlamazsa uyarır + JSON fallback'e devam eder.

### Manuel
```bash
# MongoDB (brew service, login'de oto-başlar; gerekirse:)
brew services start mongodb-community
# backend
.venv/bin/python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
# frontend
cd frontend && npm run dev
```

### Kayıtları görme
- **Compass (GUI):** `open -a "MongoDB Compass"` → `mongodb://localhost:27017` → `deepfake_fraud` > `calls`.
- **CLI:** `mongosh deepfake_fraud --eval "db.calls.find().pretty()"`

### Demo notları
- **Sesli okuma için Safari kullan** — Chrome bazı sürümlerde Enhanced sesleri
  `getVoices()`'ta listelemez. Safari sistem Enhanced seslerini gösterir.
- Enhanced sesler kurulu: Samantha (Enhanced) en_US, Yelda (Enhanced) tr_TR.
  Yenisi gerekirse: System Settings → Accessibility → Spoken Content → System Voice → Manage Voices.

---

## Ortam Kurulumu (yeni makinede)
- **MongoDB:** `brew tap mongodb/brew && brew install mongodb-community mongosh && brew services start mongodb-community`
- **Compass (opsiyonel GUI):** `brew install --cask mongodb-compass`
- **Python deps:** `pip install -r requirements.txt` (pymongo dahil)

---

## Doğrulama (yapıldı, hepsi geçti)
- Mongo yolu: analiz → `db.calls` kaydı + tam skor seti; /calls, /feedback, DELETE, 404.
- Audio: upload → `data/audio/*.wav` + GET 200; test-library referans + GET 200; sil → dosya gider, 404.
- Library preview endpoint: 200; bad category 400; path traversal 404.
- Fallback: Mongo durdurulunca backend ayağa kalkar, JSON'a yazar.
- Frontend `tsc`: Dashboard/i18n 0 yeni hata (baseline 1 pre-existing i18n as-const hatası).

---

## Sabit Kurallar (bu proje)
- İzinsiz training YOK. İzinsiz model/inference mantık değişikliği YOK.
- `models/ssl_aasist/weights.pth` üzerine YAZMA. full_run_002/003/004 artifact'larını SİLME.
- `test_audio` + `microphone_benchmark` HOLDOUT (test seti).
- Commit'lerde co-author EKLEME.
- `.env` UTF-8 NO BOM (pydantic BOM'da patlar). Geçerli anahtar `MODEL_BACKEND` (MODEL_NAME değil).

## Aktif Model Config (referans, .env)
```
MODEL_BACKEND=ssl_aasist
AUTH_THRESHOLD=0.50
LOCAL_SSL_MODEL_DIR=models/ssl_aasist
LOCAL_SSL_WEIGHTS_PATH=training_runs/full_run_004/best_head.pth
CALL_CHANNEL_MODE=false
```

## Git
- origin: `https://github.com/dogukan-filiz/deepfake-voice-fraud-dedection.git`
- `main` HEAD: `1dbe01d`. 3 feature merge edildi (e54f4c8 persistence+history, 1dbe01d TTS).
- Push öncesi `git fetch` + ahead/behind kontrol.
