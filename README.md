# deepfake-voice-fraud-dedection

Basit bir demo full-stack proje:

- **Backend:** FastAPI (ses dosyasi yukle / canli kayit analizi)
- **Frontend:** Vite + React dashboard

## Gereksinimler

- Windows 10/11
- Python 3.11+ (projede `.venv` kullaniliyor)
- Node.js 18+ (npm dahil)

> Not: Tarayici kayitlari `webm/ogg/mp4` gelebilir. Bazi formatlarda donusum icin **FFmpeg** gerekli olabilir.
> Bu projede ayrica `imageio-ffmpeg` ile (sistem FFmpeg yoksa) otomatik FFmpeg binary fallback vardir.

## Kurulum

### 1) Python bagimliliklari

Proje kokunde:

```powershell
.
# (Istege bagli) venv yoksa olustur
python -m venv .venv

# venv aktif et
.\.venv\Scripts\Activate.ps1

# bagimliliklari kur
pip install -r requirements.txt
```

### 2) Frontend bagimliliklari

```powershell
cd frontend
npm ci
```

## Calistirma

### Backend (FastAPI)

Onerilen: **proje kokunde** calistirin:

```powershell
\.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
```

Eger yanlislikla `backend/` klasoru icindeysen (prompt'ta `...\backend>` goruyorsan), su komut calisir:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Kontrol:

- http://127.0.0.1:8010/health
- Swagger: http://127.0.0.1:8010/docs

### Frontend (Vite)

`frontend/` icinde:

```powershell
npm run dev
```

Dashboard:

- http://localhost:5173

Vite dev server, `vite.config.mts` icindeki proxy ile `/api/*` isteklerini backend'e yonlendirir.

## Model (Yerel)

- **Architecture:** AASIST (raw-waveform graph attention, ~297k params, trained on ASVspoof 2019 LA; in-domain EER ~1.5%)
- Backend, varsayilan olarak once `models/aasist_baseline` (eger varsa), yoksa `models/aasist_finetuned` altindan model yuklemeyi dener.
- Farkli bir klasor kullanmak icin ortam degiskeni verin: `LOCAL_MODEL_DIR=...`

Metadata uretimi:

```powershell
python prepare_librispeech_metadata.py --dataset_root "C:\Users\DOGUKAN\OneDrive\Masaüstü\dataset" --output_csv data/metadata.csv
```

Bu script `for-original` ve `for-rerecorded` altindaki `training`, `validation` ve `testing` klasorlerini tarar, `real=0` ve `fake=1` etiketlerini metadata'ya yazar.

Egitim (metadata'daki split kolonunu kullanir, varsa testing setini de raporlar):
- Legacy training script available at `legacy/train_wav2vec2.py` for reference.
- Training is performed externally on Kaggle using the asvspoof-pipeline (out of scope for this repo).

Modelin her seye ayni sinifi vermesi gibi durumlari hizli teshis etmek icin:

```powershell
$env:MODEL_SELFTEST='1'
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
```

Sonra `GET /health` icinde `model_selftest` alanina bakabilirsiniz.

## Ayarlar

Backend esik degeri `AUTH_THRESHOLD` ile ayarlanir:

- Ortam degiskeni: `AUTH_THRESHOLD=0.5`
- (Opsiyonel) `.env` dosyasi: proje kokune koyabilirsin

## Sorun Giderme

- **Backend baslarken `WinError 10013` (port acilamiyor):**
	- 8000 portunu baska bir proses kullaniyor veya Windows tarafinda engellenmis olabilir.
	- 8000'i kullanan prosesi bulup kapat:

		```powershell
		$c = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
		if ($c) { Stop-Process -Id $c.OwningProcess -Force }
		```

	- Hala oluyorsa baska bir portla baslat (ornegin 8010):

		```powershell
		python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
		```

	- Frontend proxy'yi bu porta yonlendirmek icin (ayni terminalde):

		```powershell
		cd frontend
		$env:VITE_API_TARGET='http://127.0.0.1:8010'
		$env:VITE_WS_TARGET='ws://127.0.0.1:8010'
		npm run dev
		```

- **`/analyze` 400 ve format hatasi:** FFmpeg kurulu olmayabilir. Windows icin FFmpeg kurup `ffmpeg` komutunun PATH'te oldugunu dogrula.
- **Model ilk istekte gec acilir:** Transformers modeli ilk defa indirildiginde zaman alabilir.