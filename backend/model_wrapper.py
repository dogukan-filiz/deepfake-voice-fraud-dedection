from typing import Dict

import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

MODEL_NAME = "facebook/wav2vec2-base-960h"  # Ornek; gercek projede kendi checkpoint'in
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepfakeSesModel:
    """Wav2Vec2 tabanli (veya benzeri) derin ogrenme model sarmalayicisi.

    Not: Burada hazir bir siniflandirma modeli varsayiyoruz. Kendi modelini
    fine-tune ettiginde sadece MODEL_NAME veya yolunu degistirmen yeterli.
    """

    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()

    @torch.inference_mode()
    def tahmin_et(self, features: Dict) -> Dict:
        """Ses ozelliklerinden gercek / yapay olasiliklarini hesapla.

        Dondurulen skor: 0-1 araliginda "gercek insan sesi" olasiligi.
        """
        waveform = features["waveform"]  # numpy 1D
        sr = features["sr"]
        spectral_resid = float(features["spectral_residual_score"])

        # Wav2Vec2 genelde 16k mono waveform bekler.
        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        if probs.shape[-1] == 2:
            # [gercek, yapay] gibi bir etiket duzeni varsayalim
            p_real = float(probs[0])
            p_fake = float(probs[1])
        else:
            # Eger farkli sayida class varsa kabaca ilk class'i "real" sayalim
            p_real = float(probs[0])
            p_fake = 1.0 - p_real

        # Spectral residual skorunu hesaba katan basit bir birlestirme (heuristik)
        # Daha duzgunu: bu ozelligi dogrudan modele girdin, yeniden egitin.
        # Burada sadece ornek olarak hafif bir ceza / odul uygulariz.
        alpha = 0.2  # spectral residual agirligi
        # Yüksek residual, "yapay" olasiligini artirsin.
        p_fake_adj = np.clip(p_fake + alpha * spectral_resid, 0.0, 1.0)
        p_real_adj = 1.0 - p_fake_adj

        return {
            "p_real": float(p_real_adj),
            "p_fake": float(p_fake_adj),
            "spectral_residual": spectral_resid,
        }


# Global, lazy yukleme icin yardimci
_model_instance: DeepfakeSesModel | None = None


def get_model() -> DeepfakeSesModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = DeepfakeSesModel()
    return _model_instance
