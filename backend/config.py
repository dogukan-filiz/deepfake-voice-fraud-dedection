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
    # Point directly at a weights file (overrides model dir / weights.pth lookup).
    LOCAL_SSL_WEIGHTS_PATH: Optional[str] = None

    # Model backend selector:
    #   auto      - try chain: ssl_aasist -> df_arena -> aasist -> heuristic
    #   ssl_aasist - XLSR-300M + AASIST head (local weights.pth required)
    #   df_arena  - Speech-Arena-2025/DF_Arena_1B_V_1 (auto-downloads from HF)
    #   aasist    - vanilla AASIST baseline (local weights)
    #   heuristic - spectral-anomaly sigmoid (no ML)
    MODEL_BACKEND: str = "auto"

    # Call-center normalization pipeline.
    # Set CALL_CHANNEL_MODE=true to apply telephony preprocessing before inference.
    CALL_CHANNEL_MODE: bool = False
    CALL_PROFILE: str = "narrowband_g711"   # narrowband_g711 | wideband_opus | bypass

    class Config:
        env_file = ".env"


settings = Settings()
