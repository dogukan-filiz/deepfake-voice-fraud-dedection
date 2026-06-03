from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Fraud threshold on authenticity_score (p_real). Below = suspected fraud.
    # 0.49 calibrated for full_run_002 (SSL+AASIST fine-tuned, epoch 18).
    # Eval: macro_f1=1.0, real_recall=1.0, fake_recall=1.0 on test_audio 50+50.
    AUTH_THRESHOLD: float = 0.49

    AUDIO_MIN_DURATION_SEC: float = 2.0
    AUDIO_MIN_PEAK_ABS: float = 5e-4
    AUDIO_MIN_RMS: float = 1e-4

    LOCAL_SSL_MODEL_DIR: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()
