from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AUTH_THRESHOLD: float = 0.5

    AUDIO_MIN_DURATION_SEC: float = 2.0
    AUDIO_MIN_PEAK_ABS: float = 5e-4
    AUDIO_MIN_RMS: float = 1e-4

    class Config:
        env_file = ".env"


settings = Settings()
