from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Fraud threshold on authenticity_score (p_real). Below = suspected fraud.
    # 0.01 calibrated for SSL+AASIST primary on 50+50 sample set (acc=82.5%).
    # AASIST baseline fallback uses ~0.35; if you swap primary, retune via
    # scripts/evaluation/enhanced_evaluate.py.
    AUTH_THRESHOLD: float = 0.01

    AUDIO_MIN_DURATION_SEC: float = 2.0
    AUDIO_MIN_PEAK_ABS: float = 5e-4
    AUDIO_MIN_RMS: float = 1e-4

    LOCAL_SSL_MODEL_DIR: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()
