"""Offline evaluation for the exported epoch4 HuggingFace model.

Goals:
- Do NOT train.
- Use ONLY the exported model folder: models/deepfake_wav2vec2_epoch4
- Use existing backend inference pipeline (feature extraction + model wrapper).
- Evaluate on metadata rows with split == testing and files that exist on disk.

PowerShell example:
  python evaluate_epoch4.py --max_per_class 50
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from backend.audio_processing import ozellik_cikar
from backend.model_wrapper import DeepfakeSesModel


@dataclass(frozen=True)
class Row:
    path: Path
    label: int  # 0 real, 1 fake
    split: str
    source: str


def _normalize_split(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"test", "testing"}:
        return "testing"
    if v in {"train", "training"}:
        return "training"
    if v in {"val", "valid", "validation"}:
        return "validation"
    return v


def _read_metadata(metadata_csv: Path) -> List[Row]:
    metadata_csv = metadata_csv.expanduser().resolve()
    base_dir = metadata_csv.parent

    rows: List[Row] = []
    with metadata_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for r in reader:
            raw_path = (r.get("path") or "").strip()
            raw_label = (r.get("label") or "").strip()
            split = _normalize_split(r.get("split") or "")
            source = (r.get("source") or "").strip() or "<unknown>"

            if not raw_path:
                continue
            try:
                label = int(raw_label)
            except ValueError:
                continue
            if label not in (0, 1):
                continue

            p = Path(raw_path)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            rows.append(Row(path=p, label=label, split=split, source=source))

    if not rows:
        raise RuntimeError(f"No valid rows found in metadata: {metadata_csv}")
    return rows


def _filter_existing_testing(rows: Iterable[Row]) -> List[Row]:
    out: List[Row] = []
    for r in rows:
        if r.split != "testing":
            continue
        if not r.path.is_file():
            continue
        out.append(r)
    return out


def _balanced_sample(rows: List[Row], max_per_class: int, seed: int) -> List[Row]:
    rng = random.Random(seed)
    by_label: Dict[int, List[Row]] = {0: [], 1: []}
    for r in rows:
        by_label[r.label].append(r)

    for lbl in (0, 1):
        rng.shuffle(by_label[lbl])

    n0 = min(max_per_class, len(by_label[0]))
    n1 = min(max_per_class, len(by_label[1]))
    sample = by_label[0][:n0] + by_label[1][:n1]
    rng.shuffle(sample)
    return sample


def _predict_one(model: DeepfakeSesModel, wav_path: Path) -> Tuple[int, float, float]:
    raw = wav_path.read_bytes()
    # Prefer direct decode; retry with FFmpeg only if available/needed.
    try:
        feats = ozellik_cikar(raw, kullan_ffmpeg=False)
    except Exception:
        feats = ozellik_cikar(raw, kullan_ffmpeg=True)

    out = model.tahmin_et(feats)
    p_real = float(out.get("p_real", 0.0))
    p_fake = float(out.get("p_fake", 0.0))
    pred = 1 if p_fake > p_real else 0
    return pred, p_real, p_fake


def _fmt_cm(cm: np.ndarray) -> str:
    # cm is 2x2 for labels [0,1]
    return (
        "Confusion matrix (rows=true, cols=pred; label 0=real,1=fake)\n"
        f"[[{cm[0,0]:5d} {cm[0,1]:5d}]\n"
        f" [{cm[1,0]:5d} {cm[1,1]:5d}]]"
    )


def evaluate(
    model_dir: Path,
    metadata: Path,
    max_per_class: int,
    seed: int,
    device: Optional[str],
) -> int:
    model_dir = model_dir.expanduser().resolve()
    metadata = metadata.expanduser().resolve()

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    rows = _read_metadata(metadata)
    testing = _filter_existing_testing(rows)
    if not testing:
        raise RuntimeError("No existing testing samples found after filtering.")

    counts = Counter([r.label for r in testing])
    print(f"[DATA] testing existing samples: total={len(testing)} real(0)={counts.get(0,0)} fake(1)={counts.get(1,0)}")

    sample = _balanced_sample(testing, max_per_class=max_per_class, seed=seed)
    sample_counts = Counter([r.label for r in sample])
    print(f"[DATA] sampled: total={len(sample)} real(0)={sample_counts.get(0,0)} fake(1)={sample_counts.get(1,0)}")

    # Pick one real/fake path for manual API tests.
    real_pick = next((r for r in sample if r.label == 0), None)
    fake_pick = next((r for r in sample if r.label == 1), None)
    if real_pick:
        print(f"[PICK] real_sample={real_pick.path}")
    if fake_pick:
        print(f"[PICK] fake_sample={fake_pick.path}")

    print(f"[MODEL] loading: {model_dir}")
    model = DeepfakeSesModel(model_dir=model_dir, device=device)

    y_true: List[int] = []
    y_pred: List[int] = []
    per_item: List[dict] = []
    skipped = 0

    for i, r in enumerate(sample, start=1):
        try:
            pred, p_real, p_fake = _predict_one(model, r.path)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"[SKIP] {r.path} -> {type(e).__name__}: {e}")
            continue

        y_true.append(r.label)
        y_pred.append(pred)
        per_item.append(
            {
                "path": str(r.path),
                "label": int(r.label),
                "pred": int(pred),
                "p_real": float(p_real),
                "p_fake": float(p_fake),
                "source": r.source,
            }
        )

        if i % 10 == 0:
            print(f"[PROGRESS] {i}/{len(sample)} processed (kept={len(y_true)} skipped={skipped})")

    if not y_true:
        raise RuntimeError("All samples were skipped; cannot compute metrics.")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n[RESULT] metrics on kept samples")
    print(f"kept_samples={len(y_true)} skipped={skipped}")
    print(f"accuracy={acc:.4f}")
    print(f"precision_macro={prec:.4f}")
    print(f"recall_macro={rec:.4f}")
    print(f"f1_macro={f1:.4f}")
    print(_fmt_cm(cm))

    # Source breakdown (accuracy + counts)
    by_source: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for item in per_item:
        by_source[str(item.get("source", "<unknown>"))].append((int(item["label"]), int(item["pred"])))

    print("\n[RESULT] source breakdown (accuracy; count)")
    for src, pairs in sorted(by_source.items(), key=lambda kv: len(kv[1]), reverse=True):
        yt = [a for a, _ in pairs]
        yp = [b for _, b in pairs]
        a = accuracy_score(yt, yp)
        print(f"{src}: acc={a:.4f} n={len(pairs)}")

    # 3 correct + 3 wrong examples
    correct = [x for x in per_item if int(x["label"]) == int(x["pred"])][:3]
    wrong = [x for x in per_item if int(x["label"]) != int(x["pred"])][:3]

    def _show(tag: str, items: List[dict]) -> None:
        print(f"\n[EXAMPLES] {tag} (up to {len(items)})")
        for it in items:
            print(
                f"label={it['label']} pred={it['pred']} p_real={it['p_real']:.3f} p_fake={it['p_fake']:.3f} "
                f"source={it['source']} path={it['path']}"
            )

    _show("correct", correct)
    _show("wrong", wrong)

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline evaluation for epoch4 exported model")
    p.add_argument(
        "--model_dir",
        type=str,
        default=str(Path("models/deepfake_wav2vec2_epoch4")),
        help="HuggingFace model folder",
    )
    p.add_argument(
        "--metadata",
        type=str,
        default=str(Path("data/metadata.csv")),
        help="CSV with columns path,label,split,source",
    )
    p.add_argument(
        "--max_per_class",
        type=int,
        default=30,
        help="Max samples per class (0=use all, may be slow)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="Optional torch device string (e.g. cpu, cuda). Empty=auto",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_per_class = int(args.max_per_class)
    if max_per_class <= 0:
        # Keep behavior safe: require explicit bigger run.
        raise SystemExit("--max_per_class must be > 0 for safety (avoid huge offline run by accident).")

    device = args.device.strip() or None
    raise SystemExit(
        evaluate(
            model_dir=Path(args.model_dir),
            metadata=Path(args.metadata),
            max_per_class=max_per_class,
            seed=int(args.seed),
            device=device,
        )
    )
