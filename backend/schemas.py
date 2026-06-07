from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def calculate_risk_level(authenticity_score: float) -> RiskLevel:
    if authenticity_score >= 0.75:
        return RiskLevel.LOW
    elif authenticity_score >= 0.50:
        return RiskLevel.MEDIUM
    elif authenticity_score >= 0.25:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


class PredictionResult(BaseModel):
    call_id: str
    authenticity_score: float
    is_suspected_fraud: bool
    risk_level: RiskLevel
    p_real: float
    p_fake: float
    spectral_residual: float
    num_chunks: int = 0
    max_chunk_p_fake: float = 0.0
    timestamp: datetime


class FeedbackRequest(BaseModel):
    call_id: str
    is_false_positive: Optional[bool] = None
    is_confirmed_fraud: Optional[bool] = None
    description: Optional[str] = None


class CallRecord(BaseModel):
    call_id: str
    authenticity_score: float
    is_suspected_fraud: bool
    risk_level: RiskLevel = RiskLevel.MEDIUM
    timestamp: datetime
    notes: Optional[str] = None
    # Full score set persisted alongside the verdict. Optional/default None so
    # legacy JSON records and existing tests stay valid.
    p_real: Optional[float] = None
    p_fake: Optional[float] = None
    spectral_residual: Optional[float] = None
    model_backend: Optional[str] = None
    threshold: Optional[float] = None
    # Stored audio for playback. audio_path is relative to the repo root;
    # audio_filename is the display name (uploaded/test file name).
    audio_path: Optional[str] = None
    audio_filename: Optional[str] = None
