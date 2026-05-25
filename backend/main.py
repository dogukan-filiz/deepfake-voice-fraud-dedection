from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import sys

try:
    # Proje kokunden calistirma (onerilen): `python -m uvicorn backend.main:app ...`
    from backend.audio_processing import (
        ozellik_cikar,
        _resolve_ffmpeg_executable as resolve_ffmpeg_executable,
        validate_audio_requirements,
    )
    from backend.model_wrapper import get_model, get_model_status
    from backend.schemas import TahminSonucu, CagriKaydi, FeedbackIstegi
    from backend.config import ayarlar
    _UVICORN_APP_PATH = "backend.main:app"
except ModuleNotFoundError:
    # `backend/` klasorunun icinden calistirma: `python -m uvicorn main:app ...`
    from audio_processing import (
        ozellik_cikar,
        _resolve_ffmpeg_executable as resolve_ffmpeg_executable,
        validate_audio_requirements,
    )
    from model_wrapper import get_model, get_model_status
    from schemas import TahminSonucu, CagriKaydi, FeedbackIstegi
    from config import ayarlar
    _UVICORN_APP_PATH = "main:app"


app = FastAPI(title="Banka Cagri Merkezi Deepfake Ses Tespit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Basit bir bellek ici cagri listesi (gercek sistemde DB kullan)
_CALL_LOG: list[CagriKaydi] = []

# Esik deger (gereksinime gore: skor dusukse riskli kabul edilecek)
THRESHOLD = ayarlar.AUTH_THRESHOLD


def _sniff_audio_container(raw_bytes: bytes) -> str:
    """Gelen sesin konteynerini kaba bir sekilde tespit et.

    Amac: WAV/FLAC gibi dosyalar icin gereksiz FFmpeg fallback'e dusmemek ve
    webm/ogg/mp4 gibi tarayici formatlarinda dogru yonlendirme yapmak.
    """
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
    # ISO BMFF: ....ftyp
    if len(head) >= 8 and raw_bytes[4:8] == b"ftyp":
        return "mp4"
    return "unknown"


@app.post("/analyze", response_model=TahminSonucu)
async def analyze_call(file: UploadFile = File(...)):
    """Ses dosyasini analiz et, guvenilirlik skoru ve dolandiricilik riski uret.

    FR1, FR2, FR3, FR4, FR5 karsilanir.
    """

    raw_bytes = await file.read()

    # DEBUG: istemciden gelen dosya turunu logla
    print("[ANALYZE] content_type=", file.content_type, "size=", len(raw_bytes))

    # 1) Ozellik cikarimi (mel-spektrogram + spectral residual)
    # - WAV/FLAC genelde dogrudan okunabilir (soundfile).
    # - Tarayici kayitlari webm/ogg/mp4/m4a gibi formatlarda gelebilir; bunlar icin FFmpeg gerekebilir.
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
            features = ozellik_cikar(raw_bytes, kullan_ffmpeg=False)
        except ValueError as e:
            direct_err = str(e)
            features = None
    else:
        features = None

    if features is None:
        # FFmpeg fallback (varsa)
        try:
            features = ozellik_cikar(raw_bytes, kullan_ffmpeg=True)
        except ValueError as e:
            # WAV/FLAC gibi gorunen ama PCM olmayan (ADPCM vb.) dosyalar Windows'ta okunamayabilir.
            if container in {"wav", "flac"} and direct_err:
                if resolve_ffmpeg_executable() is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "WAV/FLAC dosyasi dogrudan okunamadi (muhtemelen desteklenmeyen codec / bozuk dosya). "
                            "Cozum: dosyayi PCM WAV (16-bit) olarak tekrar export edin veya FFmpeg kurup PATH'e ekleyin. "
                            f"Detay: {direct_err}"
                        ),
                    )
            raise HTTPException(status_code=400, detail=str(e))

    # 1.5) Pre-checks: duration >= 2s and not silent (no VAD, just basic amplitude/energy)
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

    # 2) Model tahmini
    model = get_model()
    model_out = model.tahmin_et(features)

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
    """Son cagri kayitlarini listele (dashboard icin)."""
    return list(reversed(_CALL_LOG))[:limit]


@app.post("/feedback")
async def add_feedback(feedback: FeedbackIstegi):
    """Analist geri bildirimi al (FR7).

    Simdilik sadece log'a not dusuyoruz; gercek projede DB'ye yazilip
    yeniden egitim pipeline'ina girdi olarak kullanilir.
    """
    # Burada sadece basit bir ornek: ilgili cagri kaydini bul ve not ekle.
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
    """Gercek zamanli izleme icin basit WebSocket (FR1, FR5, FR6).

    Frontend bu kanaldan periyodik olarak son cagriyi cekebilir veya
    ileride buraya direkt streaming baglanabilir.
    """
    await websocket.accept()
    try:
        last_len = 0
        while True:
            # Bu ornek, her yeni cagri eklenince butun log'u gonderiyor.
            # Gercekte sadece son kaydi gondermek daha mantikli.
            if len(_CALL_LOG) != last_len:
                last_len = len(_CALL_LOG)
                if _CALL_LOG:
                    son = _CALL_LOG[-1]
                    await websocket.send_json(son.dict())
            await websocket.receive_text()  # ping/pong icin basit bekleme
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
    }

    # Optional, lightweight self-test (enable via env var) to help debug cases where
    # the model always outputs the same class.
    if os.getenv("MODEL_SELFTEST", "").strip() in {"1", "true", "True", "YES", "yes"}:
        try:
            import numpy as np
            t = np.arange(16000) / 16000.0
            sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
            out = get_model().tahmin_et({"waveform": sine, "sr": 16000})
            ok = (abs(out["p_real"] + out["p_fake"] - 1.0) < 1e-3
                  and out["p_real"] != out["p_fake"])
            payload["model_selftest"] = {
                "ran": True,
                "architecture": "AASIST",
                "model_dir": str(get_model().model_dir),
                "sine_p_real": out["p_real"],
                "sine_p_fake": out["p_fake"],
                "ok": ok,
            }
        except Exception as exc:
            payload["model_selftest"] = {"ran": True, "ok": False, "error": str(exc)}

    return payload


if __name__ == "__main__":
    uvicorn.run(_UVICORN_APP_PATH, host="0.0.0.0", port=8000, reload=True)
