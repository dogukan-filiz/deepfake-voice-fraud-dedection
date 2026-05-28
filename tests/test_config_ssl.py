def test_ssl_model_dir_defaults_none(monkeypatch):
    monkeypatch.delenv("LOCAL_SSL_MODEL_DIR", raising=False)
    from backend.config import Settings
    cfg = Settings()
    assert cfg.LOCAL_SSL_MODEL_DIR is None


def test_ssl_model_dir_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_SSL_MODEL_DIR", str(tmp_path))
    from backend.config import Settings
    cfg = Settings()
    assert cfg.LOCAL_SSL_MODEL_DIR == str(tmp_path)
