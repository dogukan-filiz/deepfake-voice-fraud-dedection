#!/usr/bin/env bash
# One-shot launcher for macOS — full_run_002 digital deepfake detection.
# Usage:  bash run_mac.sh
# After git pull, this sets up .env, installs deps, starts backend + frontend.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> Repo: $ROOT"

# --- 1. .env (full_run_002 digital mode, pinned to ssl_aasist) ---
if [ ! -f .env ]; then
  cat > .env <<'EOF'
MODEL_BACKEND=ssl_aasist
AUTH_THRESHOLD=0.49
LOCAL_SSL_MODEL_DIR=models/ssl_aasist
LOCAL_SSL_WEIGHTS_PATH=training_runs/full_run_002/best_head.pth
CALL_CHANNEL_MODE=false
EOF
  echo "==> .env created (full_run_002, threshold 0.49)"
else
  echo "==> .env exists, leaving as-is"
fi

# --- 2. base model check ---
if [ ! -f models/ssl_aasist/weights.pth ]; then
  echo "!! MISSING: models/ssl_aasist/weights.pth (TakHemlata base, 1.2GB)"
  echo "   Copy it from the Windows machine or download per models/ssl_aasist/README.md"
  exit 1
fi
echo "==> base weights.pth present"

# --- 3. python venv + deps ---
if [ ! -d .venv ]; then
  echo "==> creating venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "==> installing python deps (cached if already present)"
pip install -q -r requirements.txt

# --- 4. ffmpeg note (only needed for mp3 uploads; wav works without) ---
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "   note: system ffmpeg not found. WAV uploads work; for MP3 run: brew install ffmpeg"
fi

# --- 5. start backend (background) ---
echo "==> starting backend on http://127.0.0.1:8010"
mkdir -p .tmp
nohup .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010 \
  > .tmp/backend.log 2>&1 &
echo "    backend PID $! (log: .tmp/backend.log)"

# --- 6. frontend ---
cd frontend
if [ ! -d node_modules ]; then
  echo "==> npm install (first run)"
  npm install
fi
echo "==> starting frontend (Vite) — open the printed localhost URL"
npm run dev
