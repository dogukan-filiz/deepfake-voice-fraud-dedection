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


def validate_audio_requirements(
    waveform: np.ndarray,
    sr: int,
    *,
    min_duration_sec: float = 2.0,
    min_peak_abs: float = 5e-4,
    min_rms: float = 1e-4,
) -> None:
    """Validate basic audio requirements before running model inference.

    Requirements:
    - Reject if duration is below `min_duration_sec`
    - Reject if the recording is effectively silent (no sound), WITHOUT VAD:
      uses simple amplitude/energy thresholds on the waveform.
    """

    wf = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wf.size == 0 or sr <= 0:
        raise ValueError("Ses kaydi okunamadi veya bos.")

    duration = float(wf.size) / float(sr)
    if duration < float(min_duration_sec):
        raise ValueError(
            f"Ses kaydi cok kisa: {duration:.2f}s. En az {float(min_duration_sec):.2f}s kayit gonderin."
        )

    peak = float(np.max(np.abs(wf)))
    rms = float(np.sqrt(np.mean(np.square(wf))))
    if peak < float(min_peak_abs) and rms < float(min_rms):
        raise ValueError("Ses kaydinda ses yok (tamamen sessiz/duyulamaz seviye).")


def _resolve_ffmpeg_executable() -> str | None:
    # 0) Allow explicit override (useful on Windows when PATH isn't updated)
    explicit = (os.getenv("FFMPEG_EXE") or os.getenv("FFMPEG_PATH") or "").strip().strip('"')
    if explicit:
        p = explicit
        if os.path.isfile(p):
            return p

    # 1) System PATH
    exe = shutil.which("ffmpeg")
    if exe and os.path.isfile(exe):
        return exe

    # Fallback: `imageio-ffmpeg` paketinin sagladigi ffmpeg binary'si.
    try:
        import imageio_ffmpeg  # type: ignore

        p = imageio_ffmpeg.get_ffmpeg_exe()
        return p if p and os.path.isfile(p) else None
    except Exception:
        return None


def _ffmpeg_to_wav(raw_bytes: bytes) -> Tuple[np.ndarray, int]:
    """FFmpeg kullanarak gelen sesi gecici wav'e cevir ve oku.

    Bu fonksiyon mp4/aac gibi formatlar icin kullanilir. FFmpeg'in sistemde
    kurulu olmasi ve `ffmpeg` komutunun PATH'te bulunmasi gerekir.
    """
    ffmpeg_exe = _resolve_ffmpeg_executable()
    if ffmpeg_exe is None:
        raise ValueError(
            "Bu ses formati icin FFmpeg gerekiyor fakat bulunamadi. "
            "Cozum: (1) `FFMPEG_EXE` ile ffmpeg.exe yolunu verin, veya (2) FFmpeg kurup PATH'e ekleyin (winget/choco), "
            "veya (3) `imageio-ffmpeg` paketini kurun."
        )

    # Gecici dizinde giris ve cikis dosyalari olustur
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.bin")
        out_path = os.path.join(tmpdir, "output.wav")

        with open(in_path, "wb") as f:
            f.write(raw_bytes)

        cmd = [
            ffmpeg_exe,
            "-y",  # uzerine yaz
            "-i",
            in_path,
            "-ac",
            "1",  # mono
            "-ar",
            str(TARGET_SR),  # ornekleme orani
            out_path,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            raise ValueError(
                "FFmpeg ile ses donusumu basarisiz. FFmpeg kurulu/erisilebilir mi ve dosya formati gecerli mi?"
            ) from e

        # Donusen wav'i oku
        data, sr = sf.read(out_path, always_2d=False)
        waveform = np.asarray(data, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return waveform, sr


def yukle_ve_on_isle(raw_bytes: bytes, *, kullan_ffmpeg: bool = False) -> Tuple[np.ndarray, int]:
    """Dosya baytlarindan sesi oku, mono'ya cevir ve ornekleme oranini sabitle.

    `kullan_ffmpeg=True` ise once FFmpeg ile wav'e donusturup oyle okur.
    Aksi halde once librosa, sonra soundfile dener.
    """
    if kullan_ffmpeg:
        waveform, sr = _ffmpeg_to_wav(raw_bytes)
    else:
        audio_buf = io.BytesIO(raw_bytes)

        try:
            waveform, sr = librosa.load(audio_buf, sr=None, mono=True)
        except Exception:
            # Librosa okuyamazsa soundfile ile dene
            try:
                audio_buf.seek(0)
                data, sr = sf.read(audio_buf, always_2d=False)
                waveform = np.asarray(data, dtype=np.float32)
                # stereo ise mono'ya cevir
                if waveform.ndim == 2:
                    waveform = waveform.mean(axis=1)
            except Exception as e:
                raise ValueError(f"Ses formati desteklenmiyor veya bozuk: {e}") from e

    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return waveform.astype(np.float32), sr


def mel_spektrogram(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Mel-spektrogram (dB) hesapla."""
    S = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        fmin=20,
        fmax=8000,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def spectral_residual_anomali_skoru(mel_spec: np.ndarray) -> float:
    """Cok kaba bir spectral residual benzeri anomali skoru.

    Burada amac, AI seslerde sik gorulen duzlesmis / dogal olmayan enerji
    dagilimlarini yakalayabilecek bir "patern farki" metriği cikarmak.
    Bu gercek projede daha gelismis bir modul ile degistirilebilir.
    """
    # 1) Z-axis normalize (her frekans icin z-score)
    mu = mel_spec.mean(axis=1, keepdims=True)
    sigma = mel_spec.std(axis=1, keepdims=True) + 1e-6
    z_norm = (mel_spec - mu) / sigma

    # 2) Basit bir blur ve residual
    blur = ndimage.gaussian_filter(z_norm, sigma=1.0)
    residual = z_norm - blur

    # 3) Residual'in genliginden bir skor
    resid_energy = np.mean(np.abs(residual))

    # Skoru 0-1 araligina kabaca sikistiralim (heuristik)
    score = float(np.tanh(resid_energy))
    return score


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


def ozellik_cikar(raw_bytes: bytes, *, kullan_ffmpeg: bool = False) -> dict:
    """Model icin gerekli tum ozellikleri tek fonksiyonda hazirla."""
    waveform, sr = yukle_ve_on_isle(raw_bytes, kullan_ffmpeg=kullan_ffmpeg)
    mel = mel_spektrogram(waveform, sr)
    spectral_resid = spectral_residual_anomali_skoru(mel)

    return {
        "waveform": waveform,
        "sr": sr,
        "mel_spectrogram": mel,
        "spectral_residual_score": spectral_resid,
    }
