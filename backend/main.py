from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

try:
    from backend.audio_processing import (
        extract_features,
        _resolve_ffmpeg_executable as resolve_ffmpeg_executable,
        validate_audio_requirements,
    )
    from backend.model_wrapper import get_model, get_model_status
    from backend.schemas import PredictionResult, CallRecord, FeedbackRequest, calculate_risk_level
    from backend.config import settings
    from backend.call_channel import normalize_audio as _normalize_audio
    from backend.preprocess import preprocess_waveform as _preprocess_waveform
    from backend.storage import get_call_store
    _UVICORN_APP_PATH = "backend.main:app"
except ModuleNotFoundError:
    from audio_processing import (
        extract_features,
        _resolve_ffmpeg_executable as resolve_ffmpeg_executable,
        validate_audio_requirements,
    )
    from model_wrapper import get_model, get_model_status
    from schemas import PredictionResult, CallRecord, FeedbackRequest, calculate_risk_level
    from config import settings
    from call_channel import normalize_audio as _normalize_audio
    from preprocess import preprocess_waveform as _preprocess_waveform
    from storage import get_call_store
    _UVICORN_APP_PATH = "main:app"


def _apply_call_normalization(features: dict) -> dict:
    """Prepare decoded audio for inference.

    Always applies the OOD preprocessing pipeline (DC removal, high-pass,
    silence trim, RMS loudness normalization) so uploads / mic recordings /
    voice notes land closer to the training distribution. Optionally applies
    telephony call-channel normalization on top when CALL_CHANNEL_MODE is set.
    """
    waveform = features["waveform"]
    sr = int(features["sr"])
    waveform = _preprocess_waveform(waveform, sr)
    if settings.CALL_CHANNEL_MODE:
        waveform = _normalize_audio(waveform, sr, profile=settings.CALL_PROFILE)
    return {**features, "waveform": waveform, "sr": sr}


app = FastAPI(title="Bank Call Center Deepfake Voice Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


store = get_call_store()

THRESHOLD = settings.AUTH_THRESHOLD

from pathlib import Path as _Path
from fastapi.responses import FileResponse

_REPO_ROOT = _Path(__file__).resolve().parents[1]
_AUDIO_DIR = _REPO_ROOT / "data" / "audio"
_TEST_AUDIO_DIR = _REPO_ROOT / "test_audio"

# Containers we can persist + serve for later playback (browser-playable).
_AUDIO_MEDIA_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
    "webm": "audio/webm",
    "mp4": "audio/mp4",
}


def _store_audio_bytes(call_id: str, container: str, raw_bytes: bytes) -> str:
    """Persist uploaded/recorded audio for later playback.

    Returns the stored path relative to the repo root.
    """
    ext = container if container in _AUDIO_MEDIA_TYPES else "bin"
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    dest = _AUDIO_DIR / f"{call_id}.{ext}"
    dest.write_bytes(raw_bytes)
    return str(dest.relative_to(_REPO_ROOT))


def _resolve_audio_path(rel_path: str) -> _Path | None:
    """Resolve a stored audio path, only if it lives under an allowed dir."""
    candidate = (_REPO_ROOT / rel_path).resolve()
    for allowed in (_AUDIO_DIR.resolve(), _TEST_AUDIO_DIR.resolve()):
        try:
            candidate.relative_to(allowed)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _sniff_audio_container(raw_bytes: bytes) -> str:
    """Detect audio container format from magic bytes."""
    head = raw_bytes[:16]
    if len(head) >= 12 and head[0:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
        return "wav"
    if head.startswith(b"fLaC"):
        return "flac"
    if head.startswith(b"OggS"):
        return "ogg"
    if head.startswith(b"ID3") or head[:2] == b"\xff\xfb":
        return "mp3"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    if len(head) >= 8 and raw_bytes[4:8] == b"ftyp":
        return "mp4"
    return "unknown"


@app.post("/analyze", response_model=PredictionResult)
async def analyze_call(file: UploadFile = File(...)):
    """Analyze an audio file and return authenticity score with fraud risk assessment."""

    raw_bytes = await file.read()

    print("[ANALYZE] content_type=", file.content_type, "size=", len(raw_bytes))

    mime = (file.content_type or "").lower()
    container = _sniff_audio_container(raw_bytes)
    browser_like = any(
        mime.startswith(prefix)
        for prefix in (
            "audio/webm",
            "video/webm",
            "audio/ogg",
            "video/ogg",
            "audio/mp4",
            "video/mp4",
            "audio/x-m4a",
        )
    )
    prefer_ffmpeg = browser_like or container in {"webm", "ogg", "mp4"}

    direct_err: str | None = None
    if not prefer_ffmpeg:
        try:
            features = extract_features(raw_bytes, use_ffmpeg=False)
        except ValueError as e:
            direct_err = str(e)
            features = None
    else:
        features = None

    if features is None:
        try:
            features = extract_features(raw_bytes, use_ffmpeg=True)
        except ValueError as e:
            if container in {"wav", "flac"} and direct_err:
                if resolve_ffmpeg_executable() is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Could not read this WAV/FLAC file directly (possibly unsupported codec or corrupted). "
                            "Try re-exporting as PCM WAV 16-bit, or install FFmpeg and add it to your PATH. "
                            f"Details: {direct_err}"
                        ),
                    )
            raise HTTPException(status_code=400, detail=str(e))

    try:
        validate_audio_requirements(
            features["waveform"],
            int(features["sr"]),
            min_duration_sec=float(settings.AUDIO_MIN_DURATION_SEC),
            min_peak_abs=float(settings.AUDIO_MIN_PEAK_ABS),
            min_rms=float(settings.AUDIO_MIN_RMS),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    features = _apply_call_normalization(features)

    model = get_model()
    model_out = model.predict(features)

    p_real = model_out["p_real"]
    p_fake = model_out["p_fake"]
    spectral_resid = model_out["spectral_residual"]
    num_chunks = int(model_out.get("num_chunks", 0))
    max_chunk_p_fake = model_out.get("max_chunk_p_fake", p_fake)

    authenticity_score = float(p_real)
    is_suspected_fraud = authenticity_score < THRESHOLD
    risk = calculate_risk_level(authenticity_score)

    call_id = f"call-{datetime.now(timezone.utc).timestamp()}"
    now = datetime.now(timezone.utc)

    audio_path = _store_audio_bytes(call_id, container, raw_bytes)
    audio_filename = file.filename or f"{call_id}.{container}"

    record = CallRecord(
        call_id=call_id,
        authenticity_score=authenticity_score,
        is_suspected_fraud=is_suspected_fraud,
        risk_level=risk,
        timestamp=now,
        p_real=float(p_real),
        p_fake=float(p_fake),
        spectral_residual=float(spectral_resid),
        model_backend=get_model_status().get("model_type"),
        threshold=THRESHOLD,
        audio_path=audio_path,
        audio_filename=audio_filename,
    )
    store.add(record)

    return PredictionResult(
        call_id=call_id,
        authenticity_score=authenticity_score,
        is_suspected_fraud=is_suspected_fraud,
        risk_level=risk,
        p_real=p_real,
        p_fake=p_fake,
        spectral_residual=spectral_resid,
        num_chunks=num_chunks,
        max_chunk_p_fake=max_chunk_p_fake,
        timestamp=now,
    )


@app.get("/calls", response_model=List[CallRecord])
async def list_calls(limit: int = 50):
    """List recent call analysis records for the dashboard."""
    return store.list(limit)


@app.get("/calls/{call_id}/audio")
async def get_call_audio(call_id: str):
    """Stream the stored audio for a call so it can be replayed in the UI."""
    record = store.get(call_id)
    if record is None or not record.audio_path:
        raise HTTPException(status_code=404, detail="Audio not found")

    resolved = _resolve_audio_path(record.audio_path)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Audio file missing")

    ext = resolved.suffix.lstrip(".").lower()
    media_type = _AUDIO_MEDIA_TYPES.get(ext, "application/octet-stream")
    return FileResponse(
        resolved,
        media_type=media_type,
        filename=record.audio_filename or resolved.name,
    )


@app.post("/feedback")
async def add_feedback(feedback: FeedbackRequest):
    """Accept analyst feedback on a previous call analysis."""
    note_parts = []
    if feedback.is_false_positive:
        note_parts.append("false_positive")
    if feedback.is_confirmed_fraud:
        note_parts.append("confirmed_fraud")
    if feedback.description:
        note_parts.append(feedback.description)

    if note_parts:
        store.update_notes(feedback.call_id, "; ".join(note_parts))

    return {"status": "ok"}


def _delete_stored_audio(record: CallRecord | None) -> None:
    """Remove a record's saved audio, but never touch test-library originals."""
    if record is None or not record.audio_path:
        return
    candidate = (_REPO_ROOT / record.audio_path).resolve()
    try:
        candidate.relative_to(_AUDIO_DIR.resolve())
    except ValueError:
        return  # not under data/audio (e.g. test_audio) -> leave it
    candidate.unlink(missing_ok=True)


@app.delete("/calls/{call_id}")
async def delete_call(call_id: str):
    """Delete a single call record (and its saved audio, if any)."""
    record = store.get(call_id)
    if store.delete(call_id):
        _delete_stored_audio(record)
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Call not found")


@app.delete("/calls")
async def delete_all_calls():
    """Delete all call records and their saved audio files."""
    for record in store.list(10_000):
        _delete_stored_audio(record)
    store.delete_all()
    return {"status": "ok"}


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket endpoint for real-time call monitoring."""
    await websocket.accept()
    try:
        last_len = 0
        while True:
            current_len = store.count()
            if current_len != last_len:
                last_len = current_len
                latest = store.latest()
                if latest is not None:
                    await websocket.send_json(latest.model_dump(mode="json"))
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health_check():
    ffmpeg_exe = resolve_ffmpeg_executable()
    model_status = get_model_status()
    payload = {
        "status": "ok",
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "ffmpeg_available": ffmpeg_exe is not None,
        "ffmpeg_exe": ffmpeg_exe,
        "model": model_status,
        "pid": os.getpid(),
        "module_file": __file__,
        "threshold": THRESHOLD,
        "call_channel_mode": settings.CALL_CHANNEL_MODE,
        "call_profile": settings.CALL_PROFILE,
        "active_weights_path": str(settings.LOCAL_SSL_WEIGHTS_PATH or "default (model_dir/weights.pth)"),
        "sample_rate": 16000,
    }

    if os.getenv("MODEL_SELFTEST", "").strip() in {"1", "true", "True", "YES", "yes"}:
        try:
            import numpy as np
            t = np.arange(16000) / 16000.0
            sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
            out = get_model().predict({"waveform": sine, "sr": 16000})
            ok = (abs(out["p_real"] + out["p_fake"] - 1.0) < 1e-3
                  and out["p_real"] != out["p_fake"])
            payload["model_selftest"] = {
                "ran": True,
                "architecture": "AASIST",
                "model_dir": str(getattr(get_model(), "model_dir", "N/A")),
                "sine_p_real": out["p_real"],
                "sine_p_fake": out["p_fake"],
                "ok": ok,
            }
        except Exception as exc:
            payload["model_selftest"] = {"ran": True, "ok": False, "error": str(exc)}

    return payload


from pathlib import Path as _Path

def _test_audio_root() -> _Path:
    return _Path(__file__).resolve().parents[1] / "test_audio"


@app.get("/test-library")
async def list_test_library():
    """List available test audio files grouped by category."""
    root = _test_audio_root()
    result = {}
    for category in ("real", "fake"):
        d = root / category
        if d.is_dir():
            files = sorted(
                f.name for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
            )
            result[category] = files
        else:
            result[category] = []
    return result


@app.get("/test-library/audio")
async def get_test_library_audio(category: str, filename: str):
    """Stream a test-library file for preview playback (no analysis)."""
    if category not in ("real", "fake"):
        raise HTTPException(status_code=400, detail="Category must be 'real' or 'fake'")

    safe_name = _Path(filename).name
    filepath = _test_audio_root() / category / safe_name
    resolved = _resolve_audio_path(str(filepath.resolve().relative_to(_REPO_ROOT))) if filepath.is_file() else None
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"File not found: {category}/{safe_name}")

    ext = resolved.suffix.lstrip(".").lower()
    media_type = _AUDIO_MEDIA_TYPES.get(ext, "application/octet-stream")
    return FileResponse(resolved, media_type=media_type, filename=safe_name)


@app.post("/analyze-test", response_model=PredictionResult)
async def analyze_test_file(category: str, filename: str):
    """Analyze a file from the test library."""
    if category not in ("real", "fake"):
        raise HTTPException(status_code=400, detail="Category must be 'real' or 'fake'")

    safe_name = _Path(filename).name
    filepath = _test_audio_root() / category / safe_name
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {category}/{safe_name}")

    raw_bytes = filepath.read_bytes()

    try:
        features = extract_features(raw_bytes, use_ffmpeg=False)
    except ValueError:
        try:
            features = extract_features(raw_bytes, use_ffmpeg=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    try:
        validate_audio_requirements(
            features["waveform"],
            int(features["sr"]),
            min_duration_sec=float(settings.AUDIO_MIN_DURATION_SEC),
            min_peak_abs=float(settings.AUDIO_MIN_PEAK_ABS),
            min_rms=float(settings.AUDIO_MIN_RMS),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    features = _apply_call_normalization(features)

    model = get_model()
    model_out = model.predict(features)

    p_real = model_out["p_real"]
    p_fake = model_out["p_fake"]
    spectral_resid = model_out["spectral_residual"]
    num_chunks = int(model_out.get("num_chunks", 0))
    max_chunk_p_fake = model_out.get("max_chunk_p_fake", p_fake)

    authenticity_score = float(p_real)
    is_suspected_fraud = authenticity_score < THRESHOLD
    risk = calculate_risk_level(authenticity_score)

    call_id = f"test-{category}-{safe_name}-{datetime.now(timezone.utc).timestamp()}"
    now = datetime.now(timezone.utc)

    # Reference the existing test-library file in place (no copy needed).
    audio_path = str(filepath.resolve().relative_to(_REPO_ROOT))

    record = CallRecord(
        call_id=call_id,
        authenticity_score=authenticity_score,
        is_suspected_fraud=is_suspected_fraud,
        risk_level=risk,
        timestamp=now,
        p_real=float(p_real),
        p_fake=float(p_fake),
        spectral_residual=float(spectral_resid),
        model_backend=get_model_status().get("model_type"),
        threshold=THRESHOLD,
        audio_path=audio_path,
        audio_filename=safe_name,
    )
    store.add(record)

    return PredictionResult(
        call_id=call_id,
        authenticity_score=authenticity_score,
        is_suspected_fraud=is_suspected_fraud,
        risk_level=risk,
        p_real=p_real,
        p_fake=p_fake,
        spectral_residual=spectral_resid,
        num_chunks=num_chunks,
        max_chunk_p_fake=max_chunk_p_fake,
        timestamp=now,
    )


if __name__ == "__main__":
    uvicorn.run(_UVICORN_APP_PATH, host="0.0.0.0", port=8000, reload=True)
