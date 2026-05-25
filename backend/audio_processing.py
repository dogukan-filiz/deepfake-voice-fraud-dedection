import io
import os
import subprocess
import shutil
import tempfile
from typing import Tuple

import librosa
import numpy as np
from scipy import ndimage
import soundfile as sf

TARGET_SR = 16000
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def validate_audio_requirements(
    waveform: np.ndarray,
    sr: int,
    *,
    min_duration_sec: float = 2.0,
    min_peak_abs: float = 5e-4,
    min_rms: float = 1e-4,
) -> None:
    """Reject audio that is too short or effectively silent."""
    wf = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wf.size == 0 or sr <= 0:
        raise ValueError("Audio recording could not be read or is empty.")

    duration = float(wf.size) / float(sr)
    if duration < float(min_duration_sec):
        raise ValueError(
            f"Recording too short: {duration:.2f}s. Please send at least {float(min_duration_sec):.2f}s of audio."
        )

    peak = float(np.max(np.abs(wf)))
    rms = float(np.sqrt(np.mean(np.square(wf))))
    if peak < float(min_peak_abs) and rms < float(min_rms):
        raise ValueError("No audible sound detected in the recording (silence or below threshold).")


def _resolve_ffmpeg_executable() -> str | None:
    explicit = (os.getenv("FFMPEG_EXE") or os.getenv("FFMPEG_PATH") or "").strip().strip('"')
    if explicit:
        if os.path.isfile(explicit):
            return explicit

    exe = shutil.which("ffmpeg")
    if exe and os.path.isfile(exe):
        return exe

    try:
        import imageio_ffmpeg  # type: ignore
        p = imageio_ffmpeg.get_ffmpeg_exe()
        return p if p and os.path.isfile(p) else None
    except Exception:
        return None


def _ffmpeg_to_wav(raw_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Convert audio bytes to WAV via FFmpeg subprocess."""
    ffmpeg_exe = _resolve_ffmpeg_executable()
    if ffmpeg_exe is None:
        raise ValueError(
            "This audio format requires FFmpeg but it was not found. "
            "Options: (1) set FFMPEG_EXE to point to ffmpeg.exe, "
            "(2) install FFmpeg and add it to PATH, or "
            "(3) install the imageio-ffmpeg package."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.bin")
        out_path = os.path.join(tmpdir, "output.wav")

        with open(in_path, "wb") as f:
            f.write(raw_bytes)

        cmd = [
            ffmpeg_exe, "-y", "-i", in_path,
            "-ac", "1",
            "-ar", str(TARGET_SR),
            out_path,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            raise ValueError(
                "FFmpeg conversion failed. Is FFmpeg installed and is the file a valid audio format?"
            ) from e

        data, sr = sf.read(out_path, always_2d=False)
        waveform = np.asarray(data, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return waveform, sr


def load_and_preprocess(raw_bytes: bytes, *, use_ffmpeg: bool = False) -> Tuple[np.ndarray, int]:
    """Read audio from raw bytes, convert to mono, resample to target rate."""
    if use_ffmpeg:
        waveform, sr = _ffmpeg_to_wav(raw_bytes)
    else:
        audio_buf = io.BytesIO(raw_bytes)
        try:
            waveform, sr = librosa.load(audio_buf, sr=None, mono=True)
        except Exception:
            try:
                audio_buf.seek(0)
                data, sr = sf.read(audio_buf, always_2d=False)
                waveform = np.asarray(data, dtype=np.float32)
                if waveform.ndim == 2:
                    waveform = waveform.mean(axis=1)
            except Exception as e:
                raise ValueError(f"Unsupported or corrupted audio format: {e}") from e

    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return waveform.astype(np.float32), sr


def compute_mel_spectrogram(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Compute mel-spectrogram in dB scale."""
    S = librosa.feature.melspectrogram(
        y=waveform, sr=sr,
        n_fft=1024, hop_length=256, n_mels=64,
        fmin=20, fmax=8000, power=2.0,
    )
    return librosa.power_to_db(S, ref=np.max)


def compute_spectral_anomaly_score(mel_spec: np.ndarray) -> float:
    """Spectral anomaly score for detecting synthetic voice artifacts.

    Combines three sub-scores that capture common AI-voice signatures:
    1. Spectral flatness: synthetic voices produce unnaturally uniform energy
    2. Temporal consistency: AI-generated speech shows less natural variation
    3. High-frequency residual: vocoder artifacts concentrate in upper bands
    """
    mu = mel_spec.mean(axis=1, keepdims=True)
    sigma = mel_spec.std(axis=1, keepdims=True) + 1e-6
    z_norm = (mel_spec - mu) / sigma

    mel_linear = np.power(10.0, mel_spec / 10.0)
    mel_linear = np.clip(mel_linear, 1e-10, None)
    geo_mean = np.exp(np.mean(np.log(mel_linear), axis=0))
    arith_mean = np.mean(mel_linear, axis=0) + 1e-10
    flatness = geo_mean / arith_mean
    flatness_score = float(np.mean(flatness))

    if z_norm.shape[1] > 1:
        deltas = np.diff(z_norm, axis=1)
        temporal_var = float(np.mean(np.var(deltas, axis=1)))
        temporal_score = float(np.exp(-temporal_var))
    else:
        temporal_score = 0.5

    n_bands = mel_spec.shape[0]
    hf_start = int(n_bands * 0.75)
    blur = ndimage.gaussian_filter(z_norm, sigma=1.0)
    residual = z_norm - blur
    hf_residual = float(np.mean(np.abs(residual[hf_start:])))
    lf_residual = float(np.mean(np.abs(residual[:hf_start]))) + 1e-6
    hf_ratio = hf_residual / lf_residual

    combined = (0.35 * flatness_score
                + 0.35 * temporal_score
                + 0.30 * float(np.tanh(hf_ratio)))
    return float(np.clip(combined, 0.0, 1.0))


def prepare_for_aasist(audio_np: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz mono, pad/trim to 64600 samples, return float32."""
    if audio_np.ndim != 1:
        audio_np = audio_np.reshape(-1)
    if sr != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
    audio_np = audio_np.astype(np.float32)
    target = 64600
    if audio_np.shape[0] >= target:
        return audio_np[:target]
    n = int(target / audio_np.shape[0]) + 1
    return np.tile(audio_np, n)[:target]


def extract_features(raw_bytes: bytes, *, use_ffmpeg: bool = False) -> dict:
    """Extract all features needed for model inference."""
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File too large: {len(raw_bytes) / (1024*1024):.1f} MB. "
            f"Maximum allowed is {MAX_UPLOAD_BYTES // (1024*1024)} MB."
        )
    if len(raw_bytes) < 44:
        raise ValueError("File too small or empty - not a valid audio file.")
    waveform, sr = load_and_preprocess(raw_bytes, use_ffmpeg=use_ffmpeg)
    mel = compute_mel_spectrogram(waveform, sr)
    spectral_resid = compute_spectral_anomaly_score(mel)

    return {
        "waveform": waveform,
        "sr": sr,
        "mel_spectrogram": mel,
        "spectral_residual_score": spectral_resid,
    }
