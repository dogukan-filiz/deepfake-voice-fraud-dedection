from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from audio_processing import ozellik_cikar
from model_wrapper import get_model
from schemas import TahminSonucu, CagriKaydi, FeedbackIstegi
from config import ayarlar


app = FastAPI(title="Banka Cagri Merkezi Deepfake Ses Tespit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Basit bir bellek ici cagri listesi (gercek sistemde DB kullan)
_CALL_LOG: list[CagriKaydi] = []

# Esik deger (gereksinime gore: skor dusukse riskli kabul edilecek)
THRESHOLD = ayarlar.AUTH_THRESHOLD


@app.post("/analyze", response_model=TahminSonucu)
async def analyze_call(file: UploadFile = File(...)):
    """Ses dosyasini analiz et, guvenilirlik skoru ve dolandiricilik riski uret.

    FR1, FR2, FR3, FR4, FR5 karsilanir.
    """

    raw_bytes = await file.read()

    # DEBUG: istemciden gelen dosya turunu logla
    print("[ANALYZE] content_type=", file.content_type, "size=", len(raw_bytes))

    # 1) Ozellik cikarimi (mel-spektrogram + spectral residual)
    # Safari canli kayitlari genelde audio/mp4;codecs=mp4a.40.2 olarak gonderir.
    # Bu format icin FFmpeg ile wav'e donusturmek daha guvenlidir.
    ffmpeg_gerek = file.content_type.startswith("audio/mp4")

    try:
        features = ozellik_cikar(raw_bytes, kullan_ffmpeg=ffmpeg_gerek)
    except ValueError as e:
        # Ornegin: desteklenmeyen veya bozuk ses formati durumunda
        raise HTTPException(status_code=400, detail=str(e))

    # 2) Model tahmini
    model = get_model()
    model_out = model.tahmin_et(features)

    p_real = model_out["p_real"]
    p_fake = model_out["p_fake"]
    spectral_resid = model_out["spectral_residual"]

    authenticity_score = float(p_real)
    is_suspected_fraud = authenticity_score < THRESHOLD

    cagri_id = f"call-{datetime.utcnow().timestamp()}"
    now = datetime.utcnow()

    kayit = CagriKaydi(
        cagri_id=cagri_id,
        authenticity_score=authenticity_score,
        is_suspected_fraud=is_suspected_fraud,
        timestamp=now,
    )
    _CALL_LOG.append(kayit)

    return TahminSonucu(
        cagri_id=cagri_id,
        authenticity_score=authenticity_score,
        is_suspected_fraud=is_suspected_fraud,
        p_real=p_real,
        p_fake=p_fake,
        spectral_residual=spectral_resid,
        timestamp=now,
    )


@app.get("/calls", response_model=List[CagriKaydi])
async def list_calls(limit: int = 50):
    """Son cagri kayitlarini listele (dashboard icin)."""
    return list(reversed(_CALL_LOG))[:limit]


@app.post("/feedback")
async def add_feedback(feedback: FeedbackIstegi):
    """Analist geri bildirimi al (FR7).

    Simdilik sadece log'a not dusuyoruz; gercek projede DB'ye yazilip
    yeniden egitim pipeline'ina girdi olarak kullanilir.
    """
    # Burada sadece basit bir ornek: ilgili cagri kaydini bul ve not ekle.
    for idx, kayit in enumerate(_CALL_LOG):
        if kayit.cagri_id == feedback.cagri_id:
            not_parcalari = []
            if feedback.is_false_positive:
                not_parcalari.append("false_positive")
            if feedback.is_confirmed_fraud:
                not_parcalari.append("confirmed_fraud")
            if feedback.aciklama:
                not_parcalari.append(feedback.aciklama)

            _CALL_LOG[idx] = CagriKaydi(
                cagri_id=kayit.cagri_id,
                authenticity_score=kayit.authenticity_score,
                is_suspected_fraud=kayit.is_suspected_fraud,
                timestamp=kayit.timestamp,
                notlar="; ".join(not_parcalari) if not_parcalari else kayit.notlar,
            )
            break

    return {"status": "ok"}


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Gercek zamanli izleme icin basit WebSocket (FR1, FR5, FR6).

    Frontend bu kanaldan periyodik olarak son cagriyi cekebilir veya
    ileride buraya direkt streaming baglanabilir.
    """
    await websocket.accept()
    try:
        last_len = 0
        while True:
            # Bu ornek, her yeni cagri eklenince butun log'u gonderiyor.
            # Gercekte sadece son kaydi gondermek daha mantikli.
            if len(_CALL_LOG) != last_len:
                last_len = len(_CALL_LOG)
                if _CALL_LOG:
                    son = _CALL_LOG[-1]
                    await websocket.send_json(son.dict())
            await websocket.receive_text()  # ping/pong icin basit bekleme
    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
