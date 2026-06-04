"""Call-channel normalization utility for backend inference.

Thin wrapper around scripts/normalize_call_channel.py.
Operates on in-memory numpy arrays (no file I/O).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from normalize_call_channel import (  # noqa: E402
    rms_normalize,
    trim_silence,
    bandpass_filter,
    g711_codec_sim,
    opus_codec_sim,
    MODES,
)

TARGET_SR = 16000


def normalize_audio(audio: np.ndarray, sr: int, profile: str = "narrowband_g711") -> np.ndarray:
    """Apply call-channel normalization pipeline to an in-memory waveform.

    Args:
        audio: float32 mono array.
        sr: sample rate (must be 16000 — resample before calling if needed).
        profile: one of narrowband_g711, wideband_opus, bypass.

    Returns:
        Normalized float32 array @ 16kHz.
    """
    if profile == "bypass":
        return audio.astype(np.float32)

    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    audio = audio.astype(np.float32)
    cfg = MODES[profile]

    audio = rms_normalize(audio)
    audio = trim_silence(audio, sr=sr)

    if cfg.get("bandpass"):
        lo, hi = cfg["bandpass"]
        audio = bandpass_filter(audio, sr=sr, lo_hz=lo, hi_hz=hi)

    codec = cfg.get("codec")
    if codec == "g711":
        audio = g711_codec_sim(audio, sr=sr)
    elif codec == "opus_12k":
        audio = opus_codec_sim(audio, sr=sr, bitrate="12k")

    audio = rms_normalize(audio)
    return audio
