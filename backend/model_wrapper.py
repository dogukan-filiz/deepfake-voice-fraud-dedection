from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_AASIST_DIR = Path(__file__).resolve().parent / "aasist"
if str(_AASIST_DIR) not in sys.path:
    sys.path.insert(0, str(_AASIST_DIR))

import librosa
import numpy as np
import torch
try:
    from .audio_processing import prepare_for_aasist
except ImportError:
    from audio_processing import prepare_for_aasist


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    env_dir = os.getenv("LOCAL_MODEL_DIR")
    if model_dir is not None:
        chosen = Path(model_dir)
    elif env_dir:
        chosen = Path(env_dir)
    else:
        repo = _repo_root()
        has_best_finetuned = (repo / "best_finetuned.pth").is_file()
        candidates = (
            [Path("models/aasist_finetuned"), Path("models/aasist_baseline")]
            if has_best_finetuned
            else [Path("models/aasist_baseline"), Path("models/aasist_finetuned")]
        )

        def _looks_like_aasist_model_dir(p: Path) -> bool:
            abs_p = _repo_root() / p
            return abs_p.is_dir() and (abs_p / "model_config.json").is_file() and (abs_p / "weights.pth").is_file()

        chosen = next((p for p in candidates if _looks_like_aasist_model_dir(p)), candidates[-1])

    if not chosen.is_absolute():
        chosen = (_repo_root() / chosen).resolve()
    return chosen


def _resolve_weights_path(model_dir: Path) -> Path:
    """Resolve which weights file to load.

    Priority:
    1) Explicit override via LOCAL_WEIGHTS_PATH env var
    2) If using finetuned dir and best_finetuned.pth exists at repo root, use it
    3) Default to <model_dir>/weights.pth
    """
    explicit = (os.getenv("LOCAL_WEIGHTS_PATH") or "").strip().strip('"')
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (_repo_root() / p).resolve()
        if p.is_file():
            return p

    repo = _repo_root()
    best = repo / "best_finetuned.pth"
    try:
        if model_dir.name.lower().endswith("finetuned") and best.is_file():
            return best
    except Exception:
        pass

    return model_dir / "weights.pth"


class DeepfakeVoiceModel:
    """AASIST-based deepfake voice detection model.

    AASIST architecture: raw-waveform graph attention network (~297k params).
    Expected directory contents: model_config.json, weights.pth, meta.json (optional).
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

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

        model_config_path = self.model_dir / "model_config.json"
        if not model_config_path.exists():
            raise FileNotFoundError(f"model_config.json not found: {model_config_path}")

        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)

        from models.AASIST import Model
        self._model = Model(model_config).to(self.device)

        weights_path = _resolve_weights_path(self.model_dir)
        self.weights_path = weights_path
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        state = torch.load(weights_path, map_location=self.device)
        self._model.load_state_dict(state)
        self._model.eval()

    @staticmethod
    def _chunk_audio(audio: np.ndarray, window_samples: int, stride_samples: int) -> List[np.ndarray]:
        """Split audio into fixed-size chunks."""
        if len(audio) <= window_samples:
            return [audio]

        starts = list(range(0, len(audio) - window_samples + 1, stride_samples))
        last_start = len(audio) - window_samples
        if starts[-1] != last_start:
            starts.append(last_start)

        return [audio[s : s + window_samples] for s in starts]

    @torch.inference_mode()
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Run inference on extracted audio features."""
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

        try:
            trimmed, _ = librosa.effects.trim(waveform, top_db=30)
            if trimmed.size >= int(0.25 * self.sampling_rate):
                waveform = trimmed
        except Exception:
            pass

        window_samples = int(self.window_sec * self.sampling_rate)
        stride_samples = int(self.stride_sec * self.sampling_rate)
        chunks = self._chunk_audio(waveform, window_samples, stride_samples)

        rms_list = [float(np.sqrt(np.mean(np.square(c)))) for c in chunks]
        max_rms = max(rms_list) if rms_list else 0.0
        if max_rms > 0:
            min_rms = max_rms * 0.032
            kept = [c for c, r in zip(chunks, rms_list) if r >= min_rms]
            if kept:
                chunks = kept

        p_real_chunks = []
        p_fake_chunks = []

        for chunk in chunks:
            chunk_np = prepare_for_aasist(chunk, self.sampling_rate)
            x = torch.from_numpy(chunk_np).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, logits = self._model(x)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                p_real_chunks.append(probs[:, 1])
                p_fake_chunks.append(probs[:, 0])

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


class HeuristicFallbackModel:
    """Fallback when AASIST model fails to load. Uses spectral residual heuristic."""

    @staticmethod
    def predict(features: Dict) -> Dict:
        spectral_resid = float(features.get("spectral_residual_score", 0.0))

        x0 = 0.45
        scale = 0.08
        z = (spectral_resid - x0) / max(scale, 1e-6)
        p_fake = float(1.0 / (1.0 + np.exp(-z)))
        p_real = float(1.0 - p_fake)
        return {
            "p_real": round(p_real, 6),
            "p_fake": round(p_fake, 6),
            "spectral_residual": spectral_resid,
            "num_chunks": 0,
            "max_chunk_p_fake": round(p_fake, 6),
        }


_model_instance: Optional[Any] = None  # XLSRAASISTModel | DeepfakeVoiceModel | HeuristicFallbackModel
_model_load_error: Optional[str] = None


def get_model_status() -> Dict:
    """Return model status without triggering load."""
    model_info: Dict[str, Any] = {
        "model_type": "ssl_aasist",
        "backbone": "facebook/wav2vec2-xls-r-300m",
    }

    if _model_instance is None:
        return {"loaded": False, "type": None, "error": _model_load_error, **model_info}

    t = type(_model_instance).__name__
    if t == "DeepfakeVoiceModel":
        model_info["model_type"] = "aasist_fallback"
    elif t == "HeuristicFallbackModel":
        model_info["model_type"] = "heuristic_fallback"
    elif t == "DFArenaModel":
        model_info["model_type"] = "df_arena"
        model_info["backbone"] = "Speech-Arena-2025/DF_Arena_1B_V_1"

    status: Dict[str, Any] = {
        "loaded": True,
        "type": t,
        "error": _model_load_error,
        **model_info,
    }

    for attr in ("model_dir", "weights_path", "id2label", "real_label_id", "fake_label_id"):
        if hasattr(_model_instance, attr):
            value = getattr(_model_instance, attr)
            status[attr] = str(value) if attr in ("model_dir", "weights_path") else value

    return status


def _load_ssl_aasist() -> Any:
    try:
        from .model_wrapper_ssl import XLSRAASISTModel  # type: ignore
    except ImportError:
        from model_wrapper_ssl import XLSRAASISTModel  # type: ignore
    return XLSRAASISTModel()


def _load_df_arena() -> Any:
    try:
        from .model_wrapper_df_arena import DFArenaModel  # type: ignore
    except ImportError:
        from model_wrapper_df_arena import DFArenaModel  # type: ignore
    return DFArenaModel()


_BACKENDS: Dict[str, Any] = {
    "ssl_aasist": ("SSL+AASIST (XLSR-300M + AASIST head)", _load_ssl_aasist),
    "df_arena": ("DF Arena 1B (Speech-Arena-2025)", _load_df_arena),
    "aasist": ("AASIST baseline", DeepfakeVoiceModel),
    "heuristic": ("heuristic spectral fallback", HeuristicFallbackModel),
}

_AUTO_CHAIN = ["ssl_aasist", "df_arena", "aasist", "heuristic"]


def get_model() -> Any:
    global _model_instance, _model_load_error
    if _model_instance is not None:
        return _model_instance

    try:
        from .config import settings  # type: ignore
    except ImportError:
        from config import settings  # type: ignore

    backend = (settings.MODEL_BACKEND or "auto").strip().lower()
    chain = _AUTO_CHAIN if backend == "auto" else [backend]
    if backend != "auto" and backend not in _BACKENDS:
        raise ValueError(
            f"Unknown MODEL_BACKEND '{backend}'. Valid: auto, {', '.join(_BACKENDS)}"
        )

    errors: list[str] = []
    for name in chain:
        desc, loader = _BACKENDS[name]
        try:
            _model_instance = loader()
            _model_load_error = "; ".join(errors) or None
            print(f"[MODEL] Loaded {desc} (backend={name})")
            return _model_instance
        except Exception as e:
            err = f"{name} load failed: {type(e).__name__}: {e}"
            errors.append(err)
            print("[MODEL]", err)

    # Heuristic never raises, but keep a hard floor anyway.
    _model_instance = HeuristicFallbackModel()
    _model_load_error = "; ".join(errors)
    print("[MODEL] All backends failed; using heuristic floor.", _model_load_error)
    return _model_instance
