# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Bank call-center deepfake voice fraud detection system. Full-stack: FastAPI backend runs AASIST model inference on uploaded/recorded audio, React dashboard visualizes results.

## Commands

### Run Backend (from project root)
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
```

### Run Frontend
```powershell
cd frontend && npm run dev
```

### Full Stack (VS Code task)
Run "Dev: Full stack (backend + frontend)" task. Launches both with matching port env vars.

### Run Tests
```powershell
# Smoke test for audio pre-checks (silence/duration rejection)
python tests/smoke_test_prechecks.py

# Integration test (requires running backend on port 8010)
python tests/comprehensive_test.py
```

### Model Self-Test
```powershell
$env:MODEL_SELFTEST='1'
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
# Then GET /health and check model_selftest field
```

### Metadata Generation
```powershell
python scripts/prepare_librispeech_metadata.py --dataset_root "C:\path\to\dataset" --output_csv data/metadata.csv
```

## Architecture

### Detection Pipeline (request flow)

1. **Audio upload**: `POST /analyze` receives file bytes
2. **Container sniffing**: `_sniff_audio_container()` detects WAV/FLAC/OGG/WebM/MP4 from magic bytes
3. **Feature extraction** via `extract_features()`:
   - Loads audio (librosa/soundfile, or FFmpeg fallback for browser formats)
   - Resamples to 16 kHz mono
   - Computes mel-spectrogram (64 mels, 1024 FFT, 256 hop)
   - Computes spectral anomaly score (spectral flatness + temporal consistency + HF residual ratio)
4. **Audio validation**: `validate_audio_requirements()` rejects silence and clips under 2 seconds
5. **Model inference** via `DeepfakeVoiceModel.predict()`:
   - Trims silence (librosa, top_db=30)
   - Chunks audio into 64600-sample windows (4.04s, non-overlapping)
   - Filters silent chunks (below -30dB of loudest)
   - Runs AASIST per chunk, aggregates with risk-weighted blend (60% mean + 40% worst-case)
   - Adjusts score using spectral anomaly when above 0.6
   - Returns `{p_real, p_fake, spectral_residual, num_chunks, max_chunk_p_fake}`
6. **Fraud decision**: `authenticity_score = p_real`; fraud if below `AUTH_THRESHOLD` (default 0.5)
7. **Risk classification**: LOW (>=0.75), MEDIUM (>=0.50), HIGH (>=0.25), CRITICAL (<0.25)

### Model System

**Primary:** AASIST, raw-waveform graph attention network (~297k params, ASVspoof 2019 LA).
**Fallback:** `HeuristicFallbackModel`, sigmoid mapping on spectral anomaly score (when AASIST fails to load).

Model loading priority:
1. `LOCAL_MODEL_DIR` env var (if set)
2. `models/aasist_finetuned` (preferred if `best_finetuned.pth` exists at repo root)
3. `models/aasist_baseline` (default)

Required model directory contents: `model_config.json`, `weights.pth`, optionally `meta.json`.

AASIST model code imported from `backend/aasist/` which is added to `sys.path` at runtime.

### Backend Modules

| Module | Role |
|--------|------|
| `backend/main.py` | FastAPI app, endpoints, container sniffing |
| `backend/audio_processing.py` | Load/resample audio, mel-spectrogram, spectral anomaly, AASIST prep |
| `backend/model_wrapper.py` | AASIST model loading, chunked inference, risk-weighted aggregation, fallback |
| `backend/schemas.py` | Pydantic models (PredictionResult, CallRecord, FeedbackRequest, RiskLevel) |
| `backend/config.py` | Settings via pydantic-settings (threshold, audio limits) |
| `legacy/inference.py` | Legacy HuggingFace wav2vec2 inference (unused) |
| `legacy/tf_experimental/` | TensorFlow Conformer variant (inactive) |

### Frontend

Single-page React app (Vite + TypeScript). One component: `Dashboard.tsx`.
- File upload analysis
- Live microphone recording (PCM capture with client-side WAV encoding as fallback)
- Real-time call log with recharts line chart
- Color-coded risk levels (green/yellow/orange/red)
- Polls `/api/calls` every 5s, `/api/health` every 10s

Vite proxy: `/api/*` goes to backend (default port 8010), `/ws/*` goes to WebSocket.

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/analyze` | Upload audio file, get fraud prediction with risk level |
| GET | `/calls` | List recent call records (in-memory) |
| POST | `/feedback` | Analyst feedback on a call (false positive, confirmed fraud) |
| WS | `/ws/live` | WebSocket for real-time updates |
| GET | `/health` | System status, FFmpeg availability, model diagnostics |

## Project Structure

```
backend/                  # FastAPI application (active)
  aasist/                 # AASIST training framework (NAVER, MIT license)
  main.py                 # App entry point
  audio_processing.py     # Feature extraction pipeline
  model_wrapper.py        # AASIST inference wrapper
  schemas.py              # Pydantic models
  config.py               # Settings
frontend/                 # Vite + React dashboard
  src/pages/Dashboard.tsx  # Single-page UI
models/                   # Model weights (gitignored, local only)
tests/                    # All test scripts (smoke, integration, backend)
scripts/                  # Data prep and conversion utilities
  evaluation/             # Model evaluation and analysis scripts
legacy/                   # Dead/superseded code
  tf_experimental/        # TensorFlow Conformer variant (inactive)
  train_wav2vec2.py       # Old wav2vec2 training script
  inference.py            # Old HuggingFace inference
docs/                     # Session logs, TF architecture docs
```

## Key Design Decisions

- **All-English codebase**: identifiers, API fields, error messages, and UI text are in English.
- **In-memory call log**: no database; `_CALL_LOG` list resets on restart.
- **Dual import paths**: `backend/main.py` handles being run from project root (`backend.X`) or from within `backend/` (`X` directly).
- **FFmpeg fallback chain**: explicit env var, then system PATH, then `imageio-ffmpeg` package.
- **Audio chunking**: 64600 samples (AASIST training window), non-overlapping, silent chunks filtered.
- **Risk-weighted aggregation**: 60% mean + 40% worst-case chunk score prevents single-fake-chunk dilution.
- **Spectral anomaly adjustment**: high anomaly scores (>0.6) nudge the fraud probability upward.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTH_THRESHOLD` | `0.5` | Fraud detection threshold (below = suspected fraud) |
| `LOCAL_MODEL_DIR` | - | Override model directory path |
| `LOCAL_WEIGHTS_PATH` | - | Override weights file path |
| `MODEL_SELFTEST` | - | Set `1` to enable model diagnostic on `/health` |
| `FFMPEG_EXE` / `FFMPEG_PATH` | - | Explicit FFmpeg binary path |
| `VITE_API_TARGET` | `http://127.0.0.1:8010` | Frontend proxy target |
| `VITE_WS_TARGET` | derived from API target | WebSocket proxy target |
| `AUDIO_MIN_DURATION_SEC` | `2.0` | Minimum audio duration to accept |

## Windows-Specific Notes

- Port 8000 conflicts are common; use 8010.
- `WinError 10013` means port held by stale process. Kill via `Get-NetTCPConnection`.
- Model paths use backslashes; `_resolve_model_dir` handles both.
- Training is done externally (Kaggle); only inference runs locally.
