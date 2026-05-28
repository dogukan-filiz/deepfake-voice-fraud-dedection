"""Unit tests for XLSRAASISTModel.

These tests use a MockSSLModel monkeypatch to avoid the 1.2 GB
facebook/wav2vec2-xls-r-300m download in CI. The full model is exercised
in tests/comprehensive_test.py (integration suite).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "aasist"))


class MockSSLModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.out_dim = 1024
        self.proj = nn.Linear(1, 1024)

    def extract_feat(self, x):
        if x.ndim == 3:
            x = x[:, :, 0]
        t = x.shape[1] // 320
        return torch.randn(x.shape[0], t, self.out_dim, device=x.device)


def test_model_forward_shape(monkeypatch):
    from models import SSLAASIST as ssl_mod
    monkeypatch.setattr(ssl_mod, "SSLModel", MockSSLModel)

    model = ssl_mod.Model(device=torch.device("cpu"))
    model.eval()
    waveform = torch.randn(1, 64600)
    with torch.no_grad():
        out = model(waveform)
    assert out.shape == (1, 2), f"expected [1, 2], got {tuple(out.shape)}"


def test_wrapper_predict_returns_schema(monkeypatch, tmp_path):
    from models import SSLAASIST as ssl_mod
    monkeypatch.setattr(ssl_mod, "SSLModel", MockSSLModel)

    weights = ssl_mod.Model(device=torch.device("cpu")).state_dict()
    weights_path = tmp_path / "weights.pth"
    torch.save(weights, weights_path)
    (tmp_path / "meta.json").write_text('{"name": "test", "id2label": {"0": "spoof", "1": "bonafide"}}', encoding="utf-8")

    from backend.model_wrapper_ssl import XLSRAASISTModel
    m = XLSRAASISTModel(model_dir=tmp_path, device="cpu")

    features = {
        "waveform": np.random.randn(16000 * 3).astype(np.float32),
        "sr": 16000,
        "spectral_residual_score": 0.2,
    }
    result = m.predict(features)
    for key in ("p_real", "p_fake", "spectral_residual", "num_chunks", "max_chunk_p_fake"):
        assert key in result, f"missing key {key}"
    assert 0.0 <= result["p_real"] <= 1.0
    assert 0.0 <= result["p_fake"] <= 1.0
    assert abs((result["p_real"] + result["p_fake"]) - 1.0) < 1e-5


def test_get_model_status_reports_ssl_primary(monkeypatch):
    import importlib
    import backend.model_wrapper as mw
    importlib.reload(mw)
    monkeypatch.setattr(mw, "_model_instance", None)
    monkeypatch.setattr(mw, "_model_load_error", None)
    status = mw.get_model_status()
    assert status["model_type"] == "ssl_aasist", (
        f"loader must advertise ssl_aasist as primary, got {status['model_type']}"
    )
