"""DF Arena 1B (Speech-Arena-2025/DF_Arena_1B_V_1) inference wrapper.

Universal antispoofing model from the DF Arena leaderboard team. Trained on
ASVspoof 2019/2024, Codecfake, LibriSeVoc, DFADD, CTRSVDD, SpoofCeleb, MLAAD,
EnvSDD - far broader spoof coverage than the LA-only SSL+AASIST checkpoint.

Reported: avg EER 5.92% / F1 0.886 across eight public corpora.
License: non-commercial (research / academic use - fine for this project).

Weights auto-download from HF Hub on first load (~4 GB) and cache in
%USERPROFILE%/.cache/huggingface/hub. Requires trust_remote_code=True because
the model ships a custom `antispoofing` pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

MODEL_ID = "Speech-Arena-2025/DF_Arena_1B_V_1"
TARGET_SR = 16000
# Keep memory bounded on long clips: score in 30s windows and aggregate
# with the same 60/40 mean/worst-case blend the other wrappers use.
MAX_WINDOW_SEC = 30.0


class DFArenaModel:
    """transformers-pipeline wrapper with the project predict() schema."""

    def __init__(self, device: str | None = None) -> None:
        import torch
        from transformers import pipeline

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        self.model_id = MODEL_ID
        self.id2label: Dict[int, str] = {0: "spoof", 1: "bonafide"}
        self.real_label_id: int = 1
        self.fake_label_id: int = 0

        self._pipe = pipeline(
            "antispoofing",
            model=MODEL_ID,
            trust_remote_code=True,
            device=resolved_device,
        )

    def _score_window(self, window: np.ndarray) -> float:
        """Return p_fake (spoof probability) for one window."""
        out = self._pipe(window)
        scores = out.get("all_scores") or {}
        if "spoof" in scores:
            return float(scores["spoof"])
        # Fallback: derive from top label + score.
        label = str(out.get("label", "")).lower()
        score = float(out.get("score", 0.5))
        return score if label == "spoof" else 1.0 - score

    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        waveform = np.asarray(features["waveform"], dtype=np.float32).reshape(-1)
        sr = int(features["sr"])
        spectral_resid = float(features.get("spectral_residual_score", 0.0))

        if waveform.size == 0:
            raise ValueError("Empty waveform - no audio data to process.")
        if not np.isfinite(waveform).all():
            raise ValueError("Audio data contains NaN or Inf values.")
        if sr != TARGET_SR:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        window_samples = int(MAX_WINDOW_SEC * sr)
        if waveform.size <= window_samples:
            windows: List[np.ndarray] = [waveform]
        else:
            windows = [
                waveform[s : s + window_samples]
                for s in range(0, waveform.size, window_samples)
                if waveform[s : s + window_samples].size >= sr  # skip sub-second tail
            ] or [waveform[:window_samples]]

        p_fake_windows = [self._score_window(w) for w in windows]
        mean_p_fake = float(np.mean(p_fake_windows))
        max_p_fake = float(np.max(p_fake_windows))
        p_fake = 0.6 * mean_p_fake + 0.4 * max_p_fake
        p_real = 1.0 - p_fake

        return {
            "p_real": round(p_real, 6),
            "p_fake": round(p_fake, 6),
            "spectral_residual": spectral_resid,
            "num_chunks": len(windows),
            "max_chunk_p_fake": round(max_p_fake, 6),
        }
