"""Export epoch 4 training checkpoint (.pt) into a deployable HuggingFace model folder.

This project saves training checkpoints as a torch-saved dict containing:
- model_state
- optimizer_state
- rng
- epoch (0-based)
- args

Those checkpoints are NOT directly loadable by the backend.
Backend expects a HuggingFace model directory with (at least):
- config.json
- model.safetensors (or pytorch_model.bin)
- preprocessor_config.json

Usage (from repo root):
  python export_epoch4.py \
    --checkpoint "training_runs\\run_main\\checkpoints\\epoch_004.pt" \
    --output_dir "models\\deepfake_wav2vec2_epoch4"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor


DEFAULT_BASE_MODEL = "facebook/wav2vec2-base-960h"


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict:
    # PyTorch 2.6+: weights_only defaults to True; our checkpoint includes non-tensor objects.
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(obj)}")
    return obj


def _load_processor(base_model: str) -> Wav2Vec2Processor:
    try:
        return Wav2Vec2Processor.from_pretrained(base_model, local_files_only=True)
    except Exception:
        # Fallback: allow HF Hub download if not cached.
        return Wav2Vec2Processor.from_pretrained(base_model)


def _load_model(base_model: str, num_labels: int) -> Wav2Vec2ForSequenceClassification:
    try:
        return Wav2Vec2ForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels,
            problem_type="single_label_classification",
            local_files_only=True,
        )
    except Exception:
        return Wav2Vec2ForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )


def export_checkpoint(checkpoint_path: Path, output_dir: Path, base_model: str) -> None:
    device = torch.device("cpu")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ckpt = _torch_load_checkpoint(checkpoint_path, device=device)

    if "model_state" not in ckpt:
        raise RuntimeError("Checkpoint does not contain 'model_state'.")

    epoch_0 = int(ckpt.get("epoch", -1))
    epoch_human = epoch_0 + 1 if epoch_0 >= 0 else None

    print(f"[EXPORT] checkpoint = {checkpoint_path}")
    if epoch_human is not None:
        print(f"[EXPORT] checkpoint epoch = {epoch_human} (stored={epoch_0}, 0-based)")

    processor = _load_processor(base_model)
    model = _load_model(base_model, num_labels=2)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        # strict=False allows buffers/new keys across versions; but we still surface issues.
        print(f"[WARN] Missing keys: {len(missing)}")
        if len(missing) <= 20:
            print("        ", missing)
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 20:
            print("        ", unexpected)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer safe serialization if available.
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(output_dir)

    processor.save_pretrained(output_dir)

    # Optional metadata file used by backend for labels/sampling settings.
    meta = {
        "source": "export_epoch4.py",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": epoch_human,
        "base_model": base_model,
        "sampling_rate": 16000,
        "max_duration_seconds": 5.0,
        "window_sec": 5.0,
        "stride_sec": 2.5,
        "id2label": {"0": "real", "1": "fake"},
        "label2id": {"real": 0, "fake": 1},
    }
    (output_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[EXPORT] done -> {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export epoch_004.pt checkpoint to HuggingFace model folder")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path("training_runs/run_main/checkpoints/epoch_004.pt")),
        help="Path to epoch_004.pt (or another epoch_*.pt)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("models/deepfake_wav2vec2_epoch4")),
        help="Target HuggingFace model directory",
    )
    p.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model name used during training (default matches train.py)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_checkpoint(Path(args.checkpoint), Path(args.output_dir), base_model=args.base_model)
