from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import sys

# Model seçimine göre ilgili modülleri import et
if os.getenv("USE_TENSORFLOW_MODEL", "true").lower() == "true":
    # TensorFlow model kullan
    from backend.audio_processing import (
        ozellik_cikar,
        _resolve_ffmpeg_executable as resolve_ffmpeg_executable,
        validate_audio_requirements,
    )
    from backend.model_wrapper_tf import get_model, get_model_status
    from backend.schemas import TahminSonucu, CagriKaydi, FeedbackIstegi
    from backend.config_tf import ayarlar
    _UVICORN_APP_PATH = "backend.main_tf:app"
    _MODEL_TYPE = "TensorFlow"
else:
    # PyTorch model kullan (mevcut sistem)
    from backend.audio_processing import (
        ozellik_cikar,
        _resolve_ffmpeg_executable as resolve_ffmpeg_executable,
        validate_audio_requirements,
    )
    from backend.model_wrapper import get_model, get_model_status
    from backend.schemas import TahminSonucu, CagriKaydi, FeedbackIstegi
    from backend.config import ayarlar
    _UVICORN_APP_PATH = "backend.main_tf:app"
    _MODEL_TYPE = "PyTorch"

app = FastAPI(title="Banka Cagri Merkezi Deepfake Ses Tespit API - TensorFlow")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cagri listesi
_CALL_LOG: list[CagriKaydi] = []

# Esik deger
THRESHOLD = ayarlar.AUTH_THRESHOLD


def _sniff_audio_container(raw_bytes: bytes) -> str:
    """Gelen sesin konteynerini tespit et."""
    head = raw_bytes[:16]
    if len(head) >= 12 and head[0:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
        return "wav"
    if head.startswith(b"fLaC"):
        return "flac"
    if head.startswith(b"OggS"):
        return "ogg"
    if head.startswith(b"ID3") or raw_bytes[:2] == b"\xff\xfb":
        return "mp3"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    if len(head) >= 8 and raw_bytes[4:8] == b"ftyp":
        return "mp4"
    return "unknown"


@app.post("/analyze", response_model=TahminSonucu)
async def analyze_call(file: UploadFile = File(...)):
    """Ses dosyasini analiz et, guvenilirlik skoru uret."""
    
    raw_bytes = await file.read()
    
    print(f"[ANALYZE] content_type={file.content_type}, size={len(raw_bytes)}, model={_MODEL_TYPE}")
    
    # Audio processing
    mime = (file.content_type or "").lower()
    container = _sniff_audio_container(raw_bytes)
    browser_like = any(
        mime.startswith(prefix)
        for prefix in (
            "audio/webm", "video/webm", "audio/ogg", "video/ogg",
            "audio/mp4", "video/mp4", "audio/x-m4a",
        )
    )
    prefer_ffmpeg = browser_like or container in {"webm", "ogg", "mp4"}
    
    direct_err: str | None = None
    features = None  # Initialize features variable
    
    if not prefer_ffmpeg:
        try:
            features = ozellik_cikar(raw_bytes, kullan_ffmpeg=False)
        except ValueError as e:
            direct_err = str(e)
            features = None
    
    if features is None:
        try:
            features = ozellik_cikar(raw_bytes, kullan_ffmpeg=True)
        except ValueError as e:
            if container in {"wav", "flac"} and direct_err:
                if resolve_ffmpeg_executable() is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "WAV/FLAC dosyasi dogrudan okunamadi. "
                            "Cozum: dosyayi PCM WAV olarak tekrar export edin veya FFmpeg kurun. "
                            f"Detay: {direct_err}"
                        ),
                    )
            raise HTTPException(status_code=400, detail=str(e))
    
    # Ensure features is not None before proceeding
    if features is None:
        raise HTTPException(status_code=400, detail="Ses islenemedi. Lutfen farkli bir dosya deneyin.")
    
    # Audio validation
    try:
        validate_audio_requirements(
            features["waveform"],
            int(features["sr"]),
            min_duration_sec=float(ayarlar.AUDIO_MIN_DURATION_SEC),
            min_peak_abs=float(ayarlar.AUDIO_MIN_PEAK_ABS),
            min_rms=float(ayarlar.AUDIO_MIN_RMS),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Model prediction
    model = get_model()
    model_out = model.predict(features)
    
    p_real = model_out["p_real"]
    p_fake = model_out["p_fake"]
    spectral_resid = model_out["spectral_residual"]
    
    authenticity_score = float(p_real)
    is_suspected_fraud = authenticity_score < THRESHOLD
    
    cagri_id = f"call-{datetime.utcnow().timestamp()}"
    now = datetime.utcnow()
    
    kayit = CagriKaydi(
        cagri_id=cagri_id,
        authenticity_score=authenticity_score,
        is_suspected_fraud=is_suspected_fraud,
        timestamp=now,
    )
    _CALL_LOG.append(kayit)
    
    return TahminSonucu(
        cagri_id=cagri_id,
        authenticity_score=authenticity_score,
        is_suspected_fraud=is_suspected_fraud,
        p_real=p_real,
        p_fake=p_fake,
        spectral_residual=spectral_resid,
        timestamp=now,
    )


@app.get("/calls", response_model=List[CagriKaydi])
async def list_calls(limit: int = 50):
    """Son cagri kayitlarini listele."""
    return list(reversed(_CALL_LOG))[:limit]


@app.post("/feedback")
async def add_feedback(feedback: FeedbackIstegi):
    """Analist geri bildirimi al."""
    
    for idx, kayit in enumerate(_CALL_LOG):
        if kayit.cagri_id == feedback.cagri_id:
            not_parcalari = []
            if feedback.is_false_positive:
                not_parcalari.append("false_positive")
            if feedback.is_confirmed_fraud:
                not_parcalari.append("confirmed_fraud")
            if feedback.aciklama:
                not_parcalari.append(feedback.aciklama)
            
            _CALL_LOG[idx] = CagriKaydi(
                cagri_id=kayit.cagri_id,
                authenticity_score=kayit.authenticity_score,
                is_suspected_fraud=kayit.is_suspected_fraud,
                timestamp=kayit.timestamp,
                notlar="; ".join(not_parcalari) if not_parcalari else kayit.notlar,
            )
            break
    
    return {"status": "ok"}


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Gercek zamanli izleme icin WebSocket."""
    
    await websocket.accept()
    try:
        last_len = 0
        while True:
            if len(_CALL_LOG) != last_len:
                last_len = len(_CALL_LOG)
                if _CALL_LOG:
                    son = _CALL_LOG[-1]
                    await websocket.send_json(son.dict())
            await websocket.receive_text()  # ping/pong icin bekleme
    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health_check():
    """System health check."""
    
    ffmpeg_exe = resolve_ffmpeg_executable()
    model_status = get_model_status()
    
    payload = {
        "status": "ok",
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "ffmpeg_available": ffmpeg_exe is not None,
        "ffmpeg_exe": ffmpeg_exe,
        "model": model_status,
        "model_type": _MODEL_TYPE,
        "auth_threshold": THRESHOLD,
        "pid": os.getpid(),
        "module_file": __file__,
    }
    
    # Optional self-test
    if os.getenv("MODEL_SELFTEST", "").strip() in {"1", "true", "True", "YES", "yes"}:
        try:
            import numpy as np
            # Create test audio
            t = np.arange(16000) / 16000.0
            sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
            
            # Get features and run prediction
            features = {
                "waveform": sine,
                "sr": 16000,
            }
            model = get_model()
            out = model.predict(features)
            
            ok = (abs(out["p_real"] + out["p_fake"] - 1.0) < 1e-3
                  and 0.0 <= out["p_real"] <= 1.0
                  and 0.0 <= out["p_fake"] <= 1.0)
            
            payload["model_selftest"] = {
                "ran": True,
                "architecture": "TensorFlow Conformer",
                "model_path": str(model_status.get("model_path", "")),
                "sine_p_real": out["p_real"],
                "sine_p_fake": out["p_fake"],
                "ok": ok,
            }
        except Exception as exc:
            payload["model_selftest"] = {"ran": True, "ok": False, "error": str(exc)}
    
    return payload


if __name__ == "__main__":
    print(f"Starting Deepfake Detection API with {_MODEL_TYPE} model")
    uvicorn.run(_UVICORN_APP_PATH, host="0.0.0.0", port=8010, reload=True)