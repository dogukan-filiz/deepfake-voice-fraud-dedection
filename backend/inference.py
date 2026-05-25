import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


class DeepfakeAudioDetector:
    """
    Kaggle'da eğitilmiş Hugging Face audio classification modelini
    yerel projede inference için yükler.

    Beklenen model klasörü içeriği:
    - config.json
    - model.safetensors (veya pytorch_model.bin)
    - preprocessor_config.json
    - model_meta.json
    """

    def __init__(
        self,
        model_dir: str | Path = "models/orig_rerec_full",
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model klasörü bulunamadı: {self.model_dir}")

        meta_path = self.model_dir / "model_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"model_meta.json bulunamadı: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.device = self._resolve_device(device)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_dir)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.sampling_rate: int = int(self.meta["sampling_rate"])
        self.max_duration_seconds: float = float(self.meta["max_duration_seconds"])
        self.window_sec: float = float(self.meta.get("window_sec", self.max_duration_seconds))
        self.stride_sec: float = float(self.meta.get("stride_sec", self.window_sec / 2))

        raw_id2label = self.meta["id2label"]
        self.id2label: Dict[int, str] = {int(k): v for k, v in raw_id2label.items()}

        self.fake_label_id = self._find_label_id("fake")
        self.real_label_id = self._find_label_id("real")

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _find_label_id(self, target_label: str) -> int:
        for label_id, label_name in self.id2label.items():
            if label_name.lower() == target_label.lower():
                return label_id
        raise ValueError(f"'{target_label}' etiketi model_meta.json içinde bulunamadı.")

    @staticmethod
    def _chunk_audio(audio: np.ndarray, window_samples: int, stride_samples: int) -> List[np.ndarray]:
        if len(audio) <= window_samples:
            return [audio]

        starts = list(range(0, len(audio) - window_samples + 1, stride_samples))
        last_start = len(audio) - window_samples

        if starts[-1] != last_start:
            starts.append(last_start)

        return [audio[s:s + window_samples] for s in starts]

    def _load_audio(self, audio_path: str | Path) -> np.ndarray:
        audio_path = str(audio_path)
        audio, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        audio = audio.astype(np.float32)

        if audio is None or len(audio) == 0:
            raise ValueError(f"Boş veya okunamayan ses dosyası: {audio_path}")

        if not np.isfinite(audio).all():
            raise ValueError(f"Ses verisinde NaN/Inf bulundu: {audio_path}")

        return audio

    def predict_file(
        self,
        audio_path: str | Path,
        window_sec: Optional[float] = None,
        stride_sec: Optional[float] = None,
        chunk_batch_size: int = 8,
    ) -> Dict[str, Any]:
        """
        Tek ses dosyası için tahmin yapar.
        Sliding-window yaklaşımı kullanır.
        """
        if chunk_batch_size <= 0:
            raise ValueError("chunk_batch_size pozitif olmalıdır.")

        use_window_sec = float(window_sec) if window_sec is not None else self.window_sec
        use_stride_sec = float(stride_sec) if stride_sec is not None else self.stride_sec

        if use_window_sec <= 0 or use_stride_sec <= 0:
            raise ValueError("window_sec ve stride_sec pozitif olmalıdır.")

        audio = self._load_audio(audio_path)

        window_samples = int(self.sampling_rate * use_window_sec)
        stride_samples = int(self.sampling_rate * use_stride_sec)

        chunks = self._chunk_audio(audio, window_samples, stride_samples)
        logits_all: List[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(chunks), chunk_batch_size):
                batch_chunks = chunks[i:i + chunk_batch_size]

                inputs = self.feature_extractor(
                    batch_chunks,
                    sampling_rate=self.sampling_rate,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                logits_all.append(outputs.logits.detach().cpu())

        if not logits_all:
            raise RuntimeError("Hiç chunk üretilemedi; tahmin yapılamadı.")

        logits_tensor = torch.cat(logits_all, dim=0)            # [num_chunks, num_labels]
        mean_logits = logits_tensor.mean(dim=0, keepdim=True)   # [1, num_labels]
        probs = torch.softmax(mean_logits, dim=-1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_label = self.id2label[pred_id]

        fake_prob = float(probs[self.fake_label_id])
        real_prob = float(probs[self.real_label_id])

        return {
            "file_path": str(audio_path),
            "predicted_label": pred_label,
            "predicted_label_id": pred_id,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "num_chunks": len(chunks),
            "window_sec": use_window_sec,
            "stride_sec": use_stride_sec,
        }


if __name__ == "__main__":
    # Örnek kullanım:
    # python inference.py
    #
    # Dosya yolunu kendi bilgisayarınızdaki örnek bir sesle değiştirin.
    example_audio_path = "sample.wav"

    detector = DeepfakeAudioDetector(model_dir="models/orig_rerec_full")
    result = detector.predict_file(example_audio_path)

    print(json.dumps(result, indent=2, ensure_ascii=False))