import os
import importlib
import sys


def test_ssl_model_dir_defaults_none(monkeypatch):
    monkeypatch.delenv("LOCAL_SSL_MODEL_DIR", raising=False)
    if "backend.config" in sys.modules:
        del sys.modules["backend.config"]
    cfg = importlib.import_module("backend.config")
    assert cfg.settings.LOCAL_SSL_MODEL_DIR is None


def test_ssl_model_dir_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_SSL_MODEL_DIR", str(tmp_path))
    if "backend.config" in sys.modules:
        del sys.modules["backend.config"]
    cfg = importlib.import_module("backend.config")
    assert cfg.settings.LOCAL_SSL_MODEL_DIR == str(tmp_path)
