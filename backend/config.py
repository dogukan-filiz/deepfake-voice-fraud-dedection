from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Fraud threshold on authenticity_score (p_real). Below = suspected fraud.
    # 0.32 calibrated for SSL+AASIST full_run_002 + preprocess pipeline over
    # original + degraded OOD pool (opus/gain/headset/8k variants, n=700):
    # macro_f1=0.984, real_recall=0.991, fake_recall=0.977.
    # See training_runs/ood_experiment/ and scripts/calibrate_threshold.py.
    AUTH_THRESHOLD: float = 0.32

    AUDIO_MIN_DURATION_SEC: float = 2.0
    AUDIO_MIN_PEAK_ABS: float = 5e-4
    AUDIO_MIN_RMS: float = 1e-4

    LOCAL_SSL_MODEL_DIR: Optional[str] = None
    # Fine-tuned AASIST head loaded on top of the base checkpoint.
    # full_run_002: trained on balanced_500, perfect digital eval (F1=1.0).
    LOCAL_SSL_WEIGHTS_PATH: Optional[str] = "training_runs/full_run_002/best_head.pth"

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
