from pydantic_settings import BaseSettings


class Ayarlar(BaseSettings):
    # Authentication threshold (lower = more sensitive)
    AUTH_THRESHOLD: float = 0.5
    
    # Audio pre-checks
    AUDIO_MIN_DURATION_SEC: float = 0.5  # Reduced from 2.0 to 0.5
    AUDIO_MIN_PEAK_ABS: float = 5e-4
    AUDIO_MIN_RMS: float = 1e-4
    
    # Model selection (True for TensorFlow, False for PyTorch)
    USE_TENSORFLOW_MODEL: bool = True
    
    class Config:
        env_file = ".env"


ayarlar = Ayarlar()