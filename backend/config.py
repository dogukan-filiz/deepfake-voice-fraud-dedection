from pydantic_settings import BaseSettings


class Ayarlar(BaseSettings):
    # Gercek insan sesi olasiligi esigi (altinda ise riskli kabul edilir)
    AUTH_THRESHOLD: float = 0.5

    class Config:
        env_file = ".env"


ayarlar = Ayarlar()
