from pydantic_settings import BaseSettings


class Ayarlar(BaseSettings):
    # Gercek insan sesi olasiligi esigi (altinda ise riskli kabul edilir)
    AUTH_THRESHOLD: float = 0.5

    # Audio pre-checks (analysis starts only if these pass)
    # - If duration is below this threshold, reject.
    AUDIO_MIN_DURATION_SEC: float = 2.0
    # - If both peak and RMS are below these thresholds, treat as "no sound" and reject.
    AUDIO_MIN_PEAK_ABS: float = 5e-4
    AUDIO_MIN_RMS: float = 1e-4

    class Config:
        env_file = ".env"


ayarlar = Ayarlar()
