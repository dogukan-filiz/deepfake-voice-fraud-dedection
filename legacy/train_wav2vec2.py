"""Wav2Vec2 tabanli deepfake ses tespit modeli icin egitim scripti.

Temel prensipler:

- Metadata, yeni klasor yapisindan gelir ve `path,label,split,source` kolonlarini icerebilir.
- `split` kolonu varsa `training` / `validation` / `testing` ayirimlari dogrudan kullanilir.
- `split` kolonu yoksa eski davranis korunur ve stratified train/val split uygulanir.
- Model: `facebook/wav2vec2-base-960h` uzerinde 2 sinifli classification head ile fine-tune.
- Metrikler: accuracy, precision, recall, F1.
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from torch.optim import AdamW

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


SAMPLE_RATE = 16000


@dataclass
class AudioExample:
    path: str
    label: int
    split: str = ""
    source: str = ""


class AudioDataset(Dataset):
    def __init__(self, examples: List[AudioExample], processor: Wav2Vec2Processor):
        self.examples = examples
        self.processor = processor

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        ex = self.examples[idx]

        try:
            audio, sr = sf.read(ex.path)
        except Exception as e1:
            # soundfile başarısız; librosa deneyelim
            try:
                audio, sr = librosa.load(ex.path, sr=None, mono=True)
            except Exception as e2:
                # librosa da başarısız; MP3 ise pydub deneyelim
                if str(ex.path).lower().endswith('.mp3') and PYDUB_AVAILABLE:
                    try:
                        sound = AudioSegment.from_mp3(ex.path)
                        samples = np.array(sound.get_array_of_samples(), dtype=np.float32)
                        if sound.channels == 2:
                            samples = samples.reshape((-1, 2)).mean(axis=1)
                        sr = sound.frame_rate
                        audio = samples / 32768.0
                    except Exception as e3:
                        print(f"[WARN] {ex.path} MP3 load failed: {e3}. Skipping.")
                        raise
                else:
                    print(f"[WARN] {ex.path} load failed (soundfile: {e1}, librosa: {e2}). Skipping.")
                    raise

        audio = np.asarray(audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Tip/olcek normalize
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
        else:
            audio = audio.astype(np.float32)

        # Bos/NaN/Inf guard
        if audio.size == 0:
            audio = np.zeros(1, dtype=np.float32)
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Sample rate tutarliligi
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Resample bazi edge-case'lerde NaN uretebilir
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Peak normalize + clip (wav2vec2 icin stabil)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0)

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=SAMPLE_RATE * 5,  # 5 saniyeye kadar ses
            truncation=True,
            return_attention_mask=True,
        )
        # processor cikti boyutlari: batch=1, bu nedenle squeeze yapiyoruz
        input_values = torch.nan_to_num(
            inputs["input_values"].squeeze(0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        attention_mask = inputs.get("attention_mask")
        attention_mask = attention_mask.squeeze(0) if attention_mask is not None else None

        label = torch.tensor(ex.label, dtype=torch.long)

        out = {"input_values": input_values, "labels": label, "path": ex.path}
        if attention_mask is not None:
            out["attention_mask"] = attention_mask
        return out


def load_metadata(path: str) -> List[AudioExample]:
    """Metadata.csv'den AudioExample listesi olustur.

    Beklenen kolonlar:
        path,label[,split,source]

    Etiketler:
        0 -> real
        1 -> fake
    """

    metadata_path = Path(path).expanduser().resolve()
    metadata_dir = metadata_path.parent
    examples: List[AudioExample] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for row in reader:
            audio_path_raw = row["path"].strip()
            # Label'i kesin olarak 0/1'e normalize et (0=real,1=fake)
            raw_label = row["label"].strip()
            try:
                label = int(raw_label)
            except ValueError:
                print(f"[UYARI] Label sayi degil ({raw_label}), atliyorum: {audio_path_raw}")
                continue
            if label not in (0, 1):
                print(f"[UYARI] Gecersiz label ({label}), atliyorum: {audio_path_raw}")
                continue
            audio_path = Path(audio_path_raw)
            if not audio_path.is_absolute():
                audio_path = (metadata_dir / audio_path).resolve()
            split = row.get("split", "").strip().lower()
            source = row.get("source", "").strip()
            if not os.path.isfile(audio_path):
                print(f"[UYARI] Dosya bulunamadi, atliyorum: {audio_path}")
                continue
            examples.append(AudioExample(path=str(audio_path), label=label, split=split, source=source))
    if not examples:
        raise RuntimeError("metadata.csv bos veya hic gecerli satir yok. Once veriyi hazirla.")
    return examples


def _normalize_split_name(split: str) -> str:
    split = split.strip().lower()
    if split in {"train", "training"}:
        return "training"
    if split in {"val", "valid", "validation"}:
        return "validation"
    if split in {"test", "testing"}:
        return "testing"
    return ""


def split_from_metadata(examples: List[AudioExample]) -> Tuple[List[AudioExample], List[AudioExample], List[AudioExample], bool]:
    """Metadata icindeki split kolonunu kullanarak train/val/test ayir.

    Dondurulen son deger explicit split kullanilip kullanilmadigini belirtir.
    """

    grouped: Dict[str, List[AudioExample]] = {"training": [], "validation": [], "testing": [], "": []}
    for ex in examples:
        grouped.setdefault(_normalize_split_name(ex.split), []).append(ex)

    if grouped["training"] and grouped["validation"]:
        unknown = grouped.get("", [])
        if unknown:
            print(f"[UYARI] {len(unknown)} ornek tanimsiz split ile geldi; training'e eklenecek.")
            grouped["training"].extend(unknown)
        return grouped["training"], grouped["validation"], grouped["testing"], True

    return [], [], [], False


def stratified_split(examples: List[AudioExample], test_size: float, seed: int) -> Tuple[List[AudioExample], List[AudioExample]]:
    """AudioExample listesi uzerinde stratified train/val bolumu uygula."""

    labels = [ex.label for ex in examples]
    train_idx, val_idx = train_test_split(
        np.arange(len(examples)),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_examples = [examples[i] for i in train_idx]
    val_examples = [examples[i] for i in val_idx]
    return train_examples, val_examples


def _checkpoint_dir(output_dir: str, checkpoint_dir: str) -> Path:
    if checkpoint_dir:
        return Path(checkpoint_dir)
    return Path(output_dir) / "checkpoints"


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: AdamW, epoch: int, args: argparse.Namespace) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def _torch_load_checkpoint(path: Path, device: torch.device) -> Dict:
    """Load a checkpoint saved by this script.

    Notes:
        PyTorch 2.6 changed torch.load default: weights_only=True.
        Our checkpoints include non-tensor state (rng, args), so we must load with weights_only=False.
    """

    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch: no weights_only arg
        return torch.load(path, map_location=device)


def _resolve_resume_path(resume_from: str) -> Path:
    p = Path(resume_from).expanduser()
    if p.is_dir():
        candidates = sorted(p.glob("epoch_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No epoch_*.pt found under resume_from directory: {p}")
        return candidates[-1].resolve()
    return p.resolve()


def _load_checkpoint_auto(
    path: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> Tuple[int, str, int]:
    """Load checkpoint and decide between full resume and warm-start.

    Returns:
        (start_epoch_index, mode, completed_epoch_human)
        - start_epoch_index: next epoch index to run (0-based)
        - mode: "full-resume" or "warm-start"
        - completed_epoch_human: epoch number that checkpoint corresponds to (1-based), or 0 if unknown
    """

    ckpt = _torch_load_checkpoint(Path(path), device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")

    if "model_state" not in ckpt:
        raise RuntimeError("Checkpoint does not contain 'model_state'.")

    model.load_state_dict(ckpt["model_state"])

    last_epoch_0_based = int(ckpt.get("epoch", -1))
    completed_epoch_human = max(0, last_epoch_0_based + 1)
    start_epoch = last_epoch_0_based + 1

    # Decide resume mode
    optimizer_state = ckpt.get("optimizer_state")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)

        rng = ckpt.get("rng") or {}
        try:
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception as e:
            print(f"[UYARI] RNG state restore basarisiz: {e}")

        return start_epoch, "full-resume", completed_epoch_human

    # Warm-start: model weights only; optimizer resets
    return max(0, start_epoch), "warm-start", completed_epoch_human


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Metadata yukleniyor: {args.metadata}")
    examples = load_metadata(args.metadata)
    print(f"Toplam ornek sayisi: {len(examples)}")

    explicit_train, explicit_val, explicit_test, explicit_splits_used = split_from_metadata(examples)
    if explicit_splits_used:
        train_examples = explicit_train
        val_examples = explicit_val
        test_examples = explicit_test
        print(
            f"Metadata split'i kullaniliyor -> train: {len(train_examples)}, val: {len(val_examples)}, test: {len(test_examples)}"
        )
    else:
        train_examples, val_examples = stratified_split(examples, test_size=0.2, seed=args.seed)
        test_examples = []
        print(f"Stratified split kullaniliyor -> train: {len(train_examples)}, val: {len(val_examples)}")

    train_labels = [ex.label for ex in train_examples]
    num_real = sum(1 for l in train_labels if l == 0)
    num_fake = sum(1 for l in train_labels if l == 1)
    print(f"Train sinif dagilimi -> real (0): {num_real}, fake (1): {num_fake}")

    for split_name, split_examples in (("train", train_examples), ("val", val_examples), ("test", test_examples)):
        if split_examples:
            split_real = sum(1 for ex in split_examples if ex.label == 0)
            split_fake = sum(1 for ex in split_examples if ex.label == 1)
            print(f"[DEBUG] {split_name} dagilimi -> real={split_real}, fake={split_fake}")

    # Ilk 20 ornek dosya yolunu yazdir
    print("[DEBUG] Ilk 20 ornek (metadata'dan):")
    for ex in examples[:20]:
        print("  ", ex.path, "label=", ex.label, "split=", ex.split, "source=", ex.source)

    print("Processor ve model yukleniyor (facebook/wav2vec2-base-960h)...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base-960h",
        num_labels=2,
        problem_type="single_label_classification",
    )
    model.to(device)

    train_dataset = AudioDataset(train_examples, processor)
    val_dataset = AudioDataset(val_examples, processor)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Class imbalance icin agirliklar (eğer >55/45 ise)
    total = num_real + num_fake
    p_real = num_real / total if total > 0 else 0.0
    p_fake = num_fake / total if total > 0 else 0.0
    use_class_weights = max(p_real, p_fake) > 0.55
    if use_class_weights:
        # Nadir sınıfa daha yüksek ağırlık ver
        w_real = 1.0 / max(p_real, 1e-6)
        w_fake = 1.0 / max(p_fake, 1e-6)
        class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32, device=device)
        print(f"[INFO] Class weights kullaniliyor: real={w_real:.4f}, fake={w_fake:.4f}")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("[INFO] Class weights kullanilmiyor (dagilim ~dengeli).")
        loss_fct = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = _checkpoint_dir(args.output_dir, args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if args.resume_from:
        resume_path = _resolve_resume_path(args.resume_from)
        print(f"[INFO] Checkpoint yukleniyor: {resume_path}")
        start_epoch, mode, completed_epoch_human = _load_checkpoint_auto(resume_path, model, optimizer, device)

        if mode == "full-resume":
            if completed_epoch_human > 0:
                print(f"[INFO] Resuming from epoch {completed_epoch_human} checkpoint (full resume)")
            else:
                print("[INFO] Resuming from checkpoint (full resume)")
        else:
            if completed_epoch_human > 0:
                print(
                    f"[INFO] Warm-start from epoch {completed_epoch_human} model weights; optimizer state not found"
                )
            else:
                print("[INFO] Warm-start from model weights; optimizer state not found")

        print(f"[INFO] Next epoch: {start_epoch + 1}")

    test_dataset = AudioDataset(test_examples, processor) if test_examples else None
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if test_dataset is not None
        else None
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        all_train_labels: List[int] = []
        all_train_preds: List[int] = []

        nan_skips = 0
        steps_this_epoch = 0

        for step, batch in enumerate(train_loader, start=1):
            if args.max_train_steps and step > args.max_train_steps:
                break

            optimizer.zero_grad(set_to_none=True)
            input_values = batch["input_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)

            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fct(logits, labels)
            if not torch.isfinite(loss):
                nan_skips += 1
                if nan_skips <= 10 or nan_skips % 50 == 0:
                    paths = batch.get("path")
                    if isinstance(paths, (list, tuple)):
                        paths_preview = list(paths)[:5]
                    else:
                        paths_preview = paths

                    in_finite = bool(torch.isfinite(input_values).all().item())
                    lg_finite = bool(torch.isfinite(logits).all().item())
                    print(
                        f"  [WARN] Non-finite loss (step={step}) -> skip. nan_skips={nan_skips} "
                        f"input_finite={in_finite} logits_finite={lg_finite} "
                        f"in_min={float(input_values.min()):.4g} in_max={float(input_values.max()):.4g} "
                        f"logits_min={float(logits.min()):.4g} logits_max={float(logits.max()):.4g} "
                        f"paths={paths_preview}"
                    )
                continue

            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item())
            steps_this_epoch += 1

            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                all_train_labels.extend(labels.detach().cpu().tolist())
                all_train_preds.extend(preds.detach().cpu().tolist())

            if args.log_every and step % args.log_every == 0:
                running_acc = accuracy_score(all_train_labels, all_train_preds) if all_train_labels else 0.0
                running_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0) if all_train_labels else 0.0
                print(
                    f"  [Train] Epoch {epoch + 1}, Step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} acc={running_acc:.4f} f1={running_f1:.4f} nan_skips={nan_skips}"
                )

        avg_train_loss = total_loss / max(1, steps_this_epoch)

        if not all_train_labels:
            print(
                "[ERROR] Bu epoch'ta hic gecerli batch islenemedi (loss non-finite?). "
                "Audio/processor/padding kaynakli sayisal problem olabilir."
            )
            return

        # Train metrikleri
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_prec = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_rec = recall_score(all_train_labels, all_train_preds, zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)

        model.eval()
        all_val_labels: List[int] = []
        all_val_preds: List[int] = []
        with torch.no_grad():
            for vstep, batch in enumerate(val_loader, start=1):
                if args.max_val_steps and vstep > args.max_val_steps:
                    break

                input_values = batch["input_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device, non_blocking=True)

                logits = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                ).logits
                preds = torch.argmax(logits, dim=-1)
                all_val_labels.extend(labels.cpu().tolist())
                all_val_preds.extend(preds.cpu().tolist())

        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_prec = precision_score(all_val_labels, all_val_preds, zero_division=0)
        val_rec = recall_score(all_val_labels, all_val_preds, zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"- train_loss={avg_train_loss:.4f} "
            f"train_acc={train_acc:.4f} train_prec={train_prec:.4f} "
            f"train_rec={train_rec:.4f} train_f1={train_f1:.4f} "
            f"val_acc={val_acc:.4f} val_prec={val_prec:.4f} "
            f"val_rec={val_rec:.4f} val_f1={val_f1:.4f}"
        )

        # Debug: ilk 20 prediction vs label yazdir
        debug_n = min(20, len(all_val_labels))
        if debug_n > 0:
            print("  [Debug] Ilk 20 val tahmini (label -> pred):")
            for i in range(debug_n):
                print(f"    {all_val_labels[i]} -> {all_val_preds[i]}")

        # Debug: model tek sinifa mi yapisiyor?
        if all_val_preds:
            uniq, counts = np.unique(all_val_preds, return_counts=True)
            dist = {int(k): int(v) for k, v in zip(uniq, counts)}
            print(f"  [Debug] Val prediction distribution: {dist}")

        ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
        _save_checkpoint(ckpt_path, model, optimizer, epoch, args)
        print(f"[INFO] Checkpoint kaydedildi: {ckpt_path}")

    if test_loader is not None:
        model.eval()
        all_test_labels: List[int] = []
        all_test_preds: List[int] = []
        with torch.no_grad():
            for tstep, batch in enumerate(test_loader, start=1):
                if args.max_test_steps and tstep > args.max_test_steps:
                    break

                input_values = batch["input_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device, non_blocking=True)

                logits = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                ).logits
                preds = torch.argmax(logits, dim=-1)
                all_test_labels.extend(labels.cpu().tolist())
                all_test_preds.extend(preds.cpu().tolist())

        test_acc = accuracy_score(all_test_labels, all_test_preds)
        test_prec = precision_score(all_test_labels, all_test_preds, zero_division=0)
        test_rec = recall_score(all_test_labels, all_test_preds, zero_division=0)
        test_f1 = f1_score(all_test_labels, all_test_preds, zero_division=0)
        print(
            f"Test sonuc = acc={test_acc:.4f} prec={test_prec:.4f} rec={test_rec:.4f} f1={test_f1:.4f}"
        )

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
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik icin seed")
    parser.add_argument("--log_every", type=int, default=50, help="Kac adimda bir train metrikleri yazilsin")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping (0 ise kapali)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker sayisi (Windows icin 0 onerilir)")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="0 ise tum epoch; >0 ise her epoch icin maksimum train step (hizli deneme icin)",
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=0,
        help="0 ise tum validation; >0 ise maksimum validation batch (hizli deneme icin)",
    )
    parser.add_argument(
        "--max_test_steps",
        type=int,
        default=0,
        help="0 ise tum test; >0 ise maksimum test batch (hizli deneme icin)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Checkpoint klasoru (bos ise output_dir\\checkpoints)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Checkpoint .pt dosyasi (ornegin checkpoints\\epoch_001.pt)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
