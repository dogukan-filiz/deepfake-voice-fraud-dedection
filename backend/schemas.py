from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TahminSonucu(BaseModel):
    cagri_id: str
    authenticity_score: float  # 0-1 arasi: 1 -> gercek insan sesi
    is_suspected_fraud: bool
    p_real: float
    p_fake: float
    spectral_residual: float
    timestamp: datetime


class FeedbackIstegi(BaseModel):
    cagri_id: str
    is_false_positive: Optional[bool] = None
    is_confirmed_fraud: Optional[bool] = None
    aciklama: Optional[str] = None


class CagriKaydi(BaseModel):
    cagri_id: str
    authenticity_score: float
    is_suspected_fraud: bool
    timestamp: datetime
    notlar: Optional[str] = None
