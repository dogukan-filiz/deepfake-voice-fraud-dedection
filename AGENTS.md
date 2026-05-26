# AGENTS.md

**Read CLAUDE.md first for full architecture, commands, and design decisions.**

## Project Overview

**Deepfake Voice Fraud Detection System** - Full-stack FastAPI backend + React frontend for detecting synthetic voice attacks in bank call centers.

## Critical Commands

### Backend (FastAPI)
**From project root (recommended):**
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
```

**From backend directory (fallback):**
```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Health checks:**
- http://127.0.0.1:8010/health (includes model diagnostics)
- Swagger: http://127.0.0.1:8010/docs

### Frontend (Vite + React)
```powershell
cd frontend
npm run dev
```
Dashboard: http://localhost:5173

### Model Training
Training is performed externally on Kaggle using the asvspoof-pipeline (out of scope for this repo).
Legacy training script available at `legacy/train_wav2vec2.py` for reference.

### Metadata Generation
```powershell
python scripts/prepare_librispeech_metadata.py --dataset_root "C:\path\to\dataset" --output_csv data/metadata.csv
```

## Configuration

### Environment Variables
- `AUTH_THRESHOLD=0.5` - Authentication threshold (lower = more sensitive)
- `LOCAL_MODEL_DIR=...` - Override model directory
- `VITE_API_TARGET=http://127.0.0.1:8010` - Frontend backend target
- `VITE_WS_TARGET=ws://127.0.0.1:8010` - WebSocket target
- `MODEL_SELFTEST=1` - Enable model self-diagnostic on health check

### Model Loading Priority
1. `models/aasist_baseline` (default; in-domain EER ~1.5%)
2. `models/aasist_finetuned` (fallback; worse in-domain, see meta.json)
3. `LOCAL_MODEL_DIR` (if set, bypasses priority list)

## Development Workflow

### Port Conflicts (Windows)
**Critical:** If you get `WinError 10013` (port access denied):

```powershell
# Find and kill process using port 8000
$c = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($c) { Stop-Process -Id $c.OwningProcess -Force }

# Use alternative port for backend
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010

# Update frontend proxy targets
cd frontend
$env:VITE_API_TARGET='http://127.0.0.1:8010'
$env:VITE_WS_TARGET='ws://127.0.0.1:8010'
npm run dev
```

### Audio Processing
- **Supported formats:** WAV, FLAC, MP3, OGG, WebM, MP4
- **Automatic format detection** with FFmpeg fallback
- **Browser recordings** (webm/ogg/mp4) always use FFmpeg
- **WAV/FLAC files** try direct reading first, then FFmpeg if needed

### Model Debugging
**Essential:** Set `MODEL_SELFTEST=1` to run lightweight model validation on `/health` endpoint. Helps detect cases where model always outputs the same class.

## Setup Requirements

### Python Dependencies
```powershell
# Project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Frontend Dependencies
```powershell
cd frontend
npm ci
```

### System Requirements
- **Windows 10/11** (tested environment)
- Python 3.11+ (required for transformers compatibility)
- Node.js 18+
- FFmpeg (optional, fallback via `imageio-ffmpeg`)

## Testing Commands

### Single Test Run
```powershell
# Test backend health
curl http://127.0.0.1:8010/health

# Test model self-diagnostic
$env:MODEL_SELFTEST='1'
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
```

### Integration Test
1. Start backend on port 8010
2. Start frontend with proxy environment variables
3. Upload test audio files via dashboard
4. Check `/calls` endpoint for analysis results

## Architecture Notes

### Model System
- **Architecture:** AASIST (raw-waveform graph attention, ~297k params, trained on ASVspoof 2019 LA; in-domain EER ~1.5%)
- **Location:** `models/` directory with custom PyTorch format
- **Required files:** `model_config.json`, `weights.pth`, `meta.json`
- **Model wrapper:** `backend/model_wrapper.py` - handles loading and inference

### Backend Entry Points
- **Main:** `backend/main.py` (handles import paths from root or subdir)
- **Audio processing:** `backend/audio_processing.py` - format conversion and feature extraction
- **Model inference:** `backend/model_wrapper.py` - model loading and prediction

### Frontend Proxy
- **Config:** `frontend/vite.config.mts` (auto-detects backend port)
- **Behavior:** Redirects `/api/*` to backend and `/ws/*` to WebSocket

### File Structure Notes
- **Training data:** `data/` directory with metadata.csv
- **Models:** `models/` directory (multiple versions supported)
- **Audio artifacts:** Excluded from git (`.gitignore` line 29)
- **VS Code tasks:** `.vscode/tasks.json` for easy development setup

## Critical Windows-Specific Issues

### Port Management
- **8000 port conflicts** are common on Windows
- **8010** is the recommended alternative port
- **Process cleanup:** Use `Get-NetTCPConnection` + `Stop-Process` for persistent processes

### Audio Processing
- **FFmpeg dependencies:** Windows PATH issues are common
- **Fallback:** `imageio-ffmpeg` provides automatic fallback
- **Explicit override:** Set `FFMPEG_EXE` environment variable if needed

### Model Loading
- **Windows file paths:** Use backslashes (`\`) consistently
- **Path resolution:** Model wrapper handles both absolute and relative paths
- **Cache issues:** Transformers models may download on first run