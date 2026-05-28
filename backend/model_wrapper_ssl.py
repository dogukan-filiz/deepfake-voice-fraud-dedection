"""SSL+AASIST (XLSR-300M + AASIST) inference wrapper."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch

_AASIST_DIR = Path(__file__).resolve().parent / "aasist"
if str(_AASIST_DIR) not in sys.path:
    sys.path.insert(0, str(_AASIST_DIR))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_ssl_model_dir(model_dir: str | Path | None = None) -> Path:
    env_dir = os.getenv("LOCAL_SSL_MODEL_DIR")
    if model_dir is not None:
        chosen = Path(model_dir)
    elif env_dir:
        chosen = Path(env_dir)
    else:
        chosen = Path("models/ssl_aasist")
    if not chosen.is_absolute():
        chosen = (_repo_root() / chosen).resolve()
    return chosen


class XLSRAASISTModel:
    """SSL+AASIST inference wrapper.

    Expected directory contents:
      weights.pth   - state dict from TakHemlata pretrained checkpoint
      meta.json     - metadata (optional)
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = _resolve_ssl_model_dir(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"SSL model directory not found: {self.model_dir}")

        meta_path = self.model_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.sampling_rate: int = 16000
        self.window_sec: float = 64600 / self.sampling_rate
        self.stride_sec: float = self.window_sec

        self.id2label: Dict[int, str] = {0: "spoof", 1: "bonafide"}
        self.real_label_id: int = 1
        self.fake_label_id: int = 0

        from models.SSLAASIST import Model as SSLAASISTNet  # type: ignore

        self._model = SSLAASISTNet(device=self.device).to(self.device)

        weights_path = self.model_dir / "weights.pth"
        self.weights_path = weights_path
        if not weights_path.exists():
            raise FileNotFoundError(
                f"SSL+AASIST weights file not found: {weights_path}. "
                f"See {self.model_dir / 'README.md'} for download instructions."
            )

        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Some checkpoints prefix keys with 'module.' (DataParallel)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        # TakHemlata's checkpoint uses fairseq Wav2Vec2 naming for the XLSR backbone
        # (under the 'ssl_model.model.' prefix). Remap to HuggingFace Wav2Vec2Model naming.
        try:
            from ._fairseq_to_hf_xlsr import convert_xlsr_subkeys
        except ImportError:
            from _fairseq_to_hf_xlsr import convert_xlsr_subkeys
        state = convert_xlsr_subkeys(state, prefix="ssl_model.model.")

        load_report = self._model.load_state_dict(state, strict=False)
        if load_report.missing_keys:
            n_missing = len(load_report.missing_keys)
            print(f"[XLSRAASIST] WARNING: {n_missing} keys missing after remap; first 5:",
                  load_report.missing_keys[:5])
        if load_report.unexpected_keys:
            n_unexp = len(load_report.unexpected_keys)
            print(f"[XLSRAASIST] WARNING: {n_unexp} unexpected keys; first 5:",
                  load_report.unexpected_keys[:5])
        self._model.eval()

    @staticmethod
    def _chunk_audio(audio: np.ndarray, window_samples: int, stride_samples: int) -> List[np.ndarray]:
        if len(audio) <= window_samples:
            pad = window_samples - len(audio)
            if pad > 0:
                reps = int(np.ceil(window_samples / max(len(audio), 1)))
                audio = np.tile(audio, reps)[:window_samples]
            return [audio]
        starts = list(range(0, len(audio) - window_samples + 1, stride_samples))
        last_start = len(audio) - window_samples
        if starts[-1] != last_start:
            starts.append(last_start)
        return [audio[s : s + window_samples] for s in starts]

    @torch.inference_mode()
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        waveform = np.asarray(features["waveform"], dtype=np.float32)
        sr = int(features["sr"])
        spectral_resid = float(features.get("spectral_residual_score", 0.0))

        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        if waveform.size == 0:
            raise ValueError("Empty waveform - no audio data to process.")
        if not np.isfinite(waveform).all():
            raise ValueError("Audio data contains NaN or Inf values.")

        if sr != self.sampling_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sampling_rate)

        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1.5:
            waveform = waveform / peak

        window_samples = int(self.window_sec * self.sampling_rate)
        stride_samples = int(self.stride_sec * self.sampling_rate)
        chunks = self._chunk_audio(waveform, window_samples, stride_samples)

        p_real_chunks: List[float] = []
        p_fake_chunks: List[float] = []

        for chunk in chunks:
            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(self.device)
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            p_real_chunks.append(float(probs[0, self.real_label_id]))
            p_fake_chunks.append(float(probs[0, self.fake_label_id]))

        mean_p_fake = float(np.mean(p_fake_chunks))
        max_p_fake = float(np.max(p_fake_chunks))
        p_fake = 0.6 * mean_p_fake + 0.4 * max_p_fake
        p_real = 1.0 - p_fake

        if spectral_resid > 0.6:
            adjustment = 0.05 * (spectral_resid - 0.6)
            p_fake = min(p_fake + adjustment, 1.0)
            p_real = 1.0 - p_fake

        return {
            "p_real": round(p_real, 6),
            "p_fake": round(p_fake, 6),
            "spectral_residual": spectral_resid,
            "num_chunks": len(chunks),
            "max_chunk_p_fake": round(max_p_fake, 6),
        }
