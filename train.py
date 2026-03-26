"""Wav2Vec2 tabanli deepfake ses tespit modeli egitimi icin basit script.

Kullanim (venv icinde):

    python train.py \
        --metadata data/metadata.csv \
        --output_dir models/deepfake_wav2vec2 \
        --epochs 3 \
        --batch_size 4

Once `data/real` ve `data/fake` klasorlerine .wav dosyalarini koyup
`data/metadata.csv` dosyasini kendi verinle doldurmalisin.
"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, AdamW


SAMPLE_RATE = 16000


@dataclass
class AudioExample:
    path: str
    label: int


class AudioDataset(Dataset):
    def __init__(self, examples: List[AudioExample], processor: Wav2Vec2Processor):
        self.examples = examples
        self.processor = processor

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        ex = self.examples[idx]
        audio, sr = sf.read(ex.path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Beklenen sample rate {SAMPLE_RATE}, ama {sr} geldi: {ex.path}")

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=SAMPLE_RATE * 5,  # 5 saniyeye kadar ses
            truncation=True,
        )
        # processor cikti boyutlari: batch=1, bu nedenle squeeze yapiyoruz
        input_values = inputs["input_values"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        label = torch.tensor(ex.label, dtype=torch.long)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": label,
        }


def load_metadata(path: str) -> List[AudioExample]:
    examples: List[AudioExample] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for row in reader:
            audio_path = row["path"].strip()
            label = int(row["label"].strip())
            if not os.path.isfile(audio_path):
                print(f"[UYARI] Dosya bulunamadi, atliyorum: {audio_path}")
                continue
            examples.append(AudioExample(path=audio_path, label=label))
    if not examples:
        raise RuntimeError("metadata.csv bos veya hic gecerli satir yok. Once veriyi hazirla.")
    return examples


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    print(f"Metadata yukleniyor: {args.metadata}")
    examples = load_metadata(args.metadata)
    print(f"Toplam ornek sayisi: {len(examples)}")

    print("Processor ve model yukleniyor (facebook/wav2vec2-base-960h)...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base-960h",
        num_labels=2,
        problem_type="single_label_classification",
    )
    model.to(device)

    dataset = AudioDataset(examples, processor)

    # Basit train/val bolumu (80/20)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        # Basit validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                ).logits
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        val_acc = correct / max(1, total)

        print(f"Epoch {epoch + 1}/{args.epochs} - train_loss={avg_train_loss:.4f} val_acc={val_acc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Model kaydediliyor: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake ses tespiti icin Wav2Vec2 egitimi")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv", help="CSV metadata dosyasi yolu")
    parser.add_argument("--output_dir", type=str, default="models/deepfake_wav2vec2", help="Egitilen model cikis klasoru")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch sayisi")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Ogrenme orani")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
