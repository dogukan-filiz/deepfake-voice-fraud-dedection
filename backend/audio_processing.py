import io
import os
import subprocess
import tempfile
from typing import Tuple

import librosa
import numpy as np
from scipy import ndimage
import soundfile as sf

TARGET_SR = 16000


def _ffmpeg_to_wav(raw_bytes: bytes) -> Tuple[np.ndarray, int]:
    """FFmpeg kullanarak gelen sesi gecici wav'e cevir ve oku.

    Bu fonksiyon mp4/aac gibi formatlar icin kullanilir. FFmpeg'in sistemde
    kurulu olmasi gerekir (macOS icin: `brew install ffmpeg`).
    """
    # Gecici dizinde giris ve cikis dosyalari olustur
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.bin")
        out_path = os.path.join(tmpdir, "output.wav")

        with open(in_path, "wb") as f:
            f.write(raw_bytes)

        cmd = [
            "ffmpeg",
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
            raise ValueError("FFmpeg ile ses donusumu basarisiz. FFmpeg kurulu mu?") from e

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
