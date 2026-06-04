"""Inference-side preprocessing pipeline.

Pulls incoming audio (uploads, browser mic, WhatsApp voice notes) toward the
training distribution of the deployed model. The training/test corpus is
loudness-normalized at source (RMS ~0.0666, peak ~0.5, 16 kHz mono), so audio
arriving at other loudness levels or with DC offset / rumble is out-of-domain
and inflates false fraud flags.

Ops (in order):
    1. DC offset removal
    2. Gentle high-pass (50 Hz) - removes rumble/handling noise mic input carries
    3. Silence trim (librosa, top_db=35) - leading/trailing only
    4. RMS loudness normalization to the corpus target, with peak ceiling

Codec damage (Opus etc.) is NOT recoverable here; that requires augmented
training (see docs/superpowers/specs/2026-06-04-ood-preprocess-design.md).
"""
from __future__ import annotations

import numpy as np

TARGET_SR = 16000
# Measured over test_audio/ 50+50 corpus (proxy for training distribution):
# RMS mean=0.0667 median=0.0666 p10=0.0536 p90=0.0798; peak median=0.499.
TARGET_RMS = 0.0666
PEAK_CEILING = 0.95
HIGHPASS_HZ = 50.0
TRIM_TOP_DB = 35.0


def _remove_dc(audio: np.ndarray) -> np.ndarray:
    return audio - float(np.mean(audio))


def _highpass(audio: np.ndarray, sr: int, cutoff_hz: float = HIGHPASS_HZ) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def _trim_silence(audio: np.ndarray, top_db: float = TRIM_TOP_DB) -> np.ndarray:
    import librosa

    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    # Never return empty audio - fall back to original if trim ate everything.
    return trimmed if trimmed.size > 0 else audio


def _rms_normalize(audio: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 1e-8:
        return audio
    out = audio * (target_rms / rms)
    peak = float(np.max(np.abs(out)))
    if peak > PEAK_CEILING:
        out = out * (PEAK_CEILING / peak)
    return out


def preprocess_waveform(audio: np.ndarray, sr: int) -> np.ndarray:
    """Normalize a decoded waveform toward the training distribution.

    Args:
        audio: float mono waveform.
        sr: sample rate; must already be TARGET_SR (resampling happens in
            audio_processing.extract_features before this point).

    Returns:
        float32 waveform, same sample rate.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = _remove_dc(audio)
    audio = _highpass(audio, sr)
    audio = _trim_silence(audio)
    audio = _rms_normalize(audio)
    return audio.astype(np.float32)
