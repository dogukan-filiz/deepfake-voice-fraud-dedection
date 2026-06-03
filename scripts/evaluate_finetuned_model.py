"""Evaluate a fine-tuned AASIST head checkpoint and compare against baseline."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "aasist"))

WINDOW = 64600
SR = 16000


def load_and_predict(wrapper, df, split, device):
    import librosa
    import math

    rows = df[df["split"] == split]
    scores = []
    labels = []

    model = wrapper._model
    model.eval()
    model.ssl_model.eval()

    for _, row in rows.iterrows():
        label = 1 if row["label"] == "real" else 0
        try:
            audio, _ = librosa.load(row["path"], sr=SR, mono=True)
            if len(audio) == 0:
                raise ValueError("empty audio")
            if len(audio) < WINDOW:
                reps = math.ceil(WINDOW / len(audio))
                audio = np.tile(audio, reps)[:WINDOW]
            else:
                audio = audio[:WINDOW]
            audio = audio.astype(np.float32)
            x = torch.from_numpy(audio).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                p_real = float(probs[0, 1].cpu())
            scores.append(p_real)
            labels.append(label)
        except Exception as e:
            print(f"  [WARN] {row['path']}: {e}")

    return np.array(scores), np.array(labels)


def compute_metrics(scores, labels, threshold=0.5):
    from sklearn.metrics import (
        f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
    )
    preds = (scores >= threshold).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    real_recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    fake_recall = recall_score(labels, preds, pos_label=0, zero_division=0)
    prec_real = precision_score(labels, preds, pos_label=1, zero_division=0)
    prec_fake = precision_score(labels, preds, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()

    # Find best threshold by F1
    best_f1, best_t = 0.0, threshold
    for t in np.arange(0.0, 1.01, 0.01):
        p = (scores >= t).astype(int)
        f = f1_score(labels, p, average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)

    return {
        "threshold": round(threshold, 2),
        "macro_f1": round(macro_f1, 4),
        "real_recall": round(real_recall, 4),
        "fake_recall": round(fake_recall, 4),
        "precision_real": round(prec_real, 4),
        "precision_fake": round(prec_fake, 4),
        "auc_roc": round(float(auc), 4),
        "best_threshold_by_f1": round(best_t, 2),
        "best_macro_f1": round(best_f1, 4),
        "confusion_matrix_fake0_real1": cm,
    }


def delta(new_val, old_val):
    if old_val is None:
        return "N/A"
    d = new_val - old_val
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned AASIST head")
    parser.add_argument("--checkpoint", required=True, help="Path to best_head.pth")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--baseline_metrics", default=None,
                        help="Path to baseline_metrics.json for delta comparison")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[eval] Loading XLSRAASISTModel + checkpoint {args.checkpoint}...")
    from backend.model_wrapper_ssl import XLSRAASISTModel
    wrapper = XLSRAASISTModel(device=str(device))

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = wrapper._model.load_state_dict(state, strict=False)
    print(f"[eval] Checkpoint loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    df = pd.read_csv(args.metadata)
    n_eval = len(df[df["split"] == args.split])
    print(f"[eval] Evaluating on split='{args.split}': {n_eval} samples")

    scores, labels = load_and_predict(wrapper, df, args.split, device)
    metrics = compute_metrics(scores, labels, threshold=args.threshold)

    # Load baseline for delta
    baseline = None
    if args.baseline_metrics and Path(args.baseline_metrics).exists():
        baseline = json.loads(Path(args.baseline_metrics).read_text())

    print(f"\n=== FINETUNED EVAL ({args.split.upper()} SPLIT) ===")
    print(f"{'Metric':<30} {'Finetuned':>10}  {'Baseline':>10}  {'Delta':>10}")
    print("-" * 65)
    for key in ["macro_f1", "real_recall", "fake_recall", "auc_roc"]:
        new_v = metrics.get(key, None)
        base_v = baseline.get(key if key != "macro_f1" else "best_macro_f1", None) if baseline else None
        d = delta(new_v, base_v) if new_v is not None else "N/A"
        base_str = f"{base_v:.4f}" if base_v is not None else "N/A"
        print(f"  {key:<28} {new_v:>10.4f}  {base_str:>10}  {d:>10}")

    result = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "finetuned_metrics": metrics,
        "baseline_metrics": baseline,
    }
    out_path = output_dir / "final_eval.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[eval] Saved to {out_path}")


if __name__ == "__main__":
    main()
