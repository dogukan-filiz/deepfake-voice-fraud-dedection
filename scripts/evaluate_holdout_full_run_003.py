"""
scripts/evaluate_holdout_full_run_003.py

Holdout evaluation for full_run_003 best checkpoint.
Reads audio directly from disk, runs XLSRAASISTModel inference,
saves JSON + CSV + LOG outputs.

Usage:
  python3 scripts/evaluate_holdout_full_run_003.py \
      --checkpoint training_runs/full_run_003/best_head.pth \
      --benchmark test_audio \
      --output_dir training_runs/full_run_003

  python3 scripts/evaluate_holdout_full_run_003.py \
      --checkpoint training_runs/full_run_003/best_head.pth \
      --benchmark microphone \
      --output_dir training_runs/full_run_003
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "aasist"))

SR = 16000
WINDOW = 64600  # ~4 sec at 16kHz


def load_audio(path):
    import librosa
    try:
        audio, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception as e:
        print(f"  WARN: could not load {path}: {e}")
        return np.zeros(WINDOW, dtype=np.float32)
    if len(audio) == 0:
        return np.zeros(WINDOW, dtype=np.float32)
    if len(audio) < WINDOW:
        reps = math.ceil(WINDOW / len(audio))
        audio = np.tile(audio, reps)[:WINDOW]
    else:
        audio = audio[:WINDOW]
    audio = audio.astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 1.5:
        audio /= peak
    return audio


def load_model(checkpoint_path, device):
    from backend.model_wrapper_ssl import XLSRAASISTModel
    wrapper = XLSRAASISTModel(device=str(device))
    nn_model = wrapper._model          # SSLModel + AASIST head nn.Module
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    # Checkpoint contains head-only keys (no "ssl_model." prefix)
    missing, unexpected = nn_model.load_state_dict(state, strict=False)
    print(f"  Head loaded — missing={len(missing)} unexpected={len(unexpected)}")
    nn_model.eval()
    nn_model.to(device)
    return nn_model


def predict_file(model, audio_np, device):
    x = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)           # shape [1, 2]; label 1=real, 0=fake
        probs = torch.softmax(logits, dim=-1)
    p_real = float(probs[0, 1])    # class 1 = real
    p_fake = float(probs[0, 0])    # class 0 = fake
    return p_real, p_fake


def collect_files(benchmark):
    """Return list of (path, true_label) for benchmark."""
    samples = []
    if benchmark == "test_audio":
        for p in sorted((REPO_ROOT / "test_audio" / "real").glob("*")):
            if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                samples.append((p, "real"))
        for p in sorted((REPO_ROOT / "test_audio" / "fake").glob("*")):
            if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                samples.append((p, "fake"))
    elif benchmark == "microphone":
        for p in sorted((REPO_ROOT / "microphone_benchmark" / "real").glob("*")):
            if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                samples.append((p, "real"))
        for p in sorted((REPO_ROOT / "microphone_benchmark" / "fake_replayed").glob("*")):
            if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                samples.append((p, "fake"))
    return samples


def compute_metrics(results, threshold):
    """Compute classification metrics at given threshold."""
    tp = fp = tn = fn = 0
    for r in results:
        pred = "real" if r["p_real"] >= threshold else "fake"
        true = r["true_label"]
        if true == "real" and pred == "real":
            tp += 1
        elif true == "fake" and pred == "fake":
            tn += 1
        elif true == "real" and pred == "fake":
            fn += 1
        elif true == "fake" and pred == "real":
            fp += 1

    real_total = tp + fn
    fake_total = tn + fp
    real_recall = tp / real_total if real_total else 0.0
    fake_recall = tn / fake_total if fake_total else 0.0
    real_prec = tp / (tp + fp) if (tp + fp) else 0.0
    fake_prec = tn / (tn + fn) if (tn + fn) else 0.0
    real_f1 = (2 * real_prec * real_recall / (real_prec + real_recall)
               if (real_prec + real_recall) else 0.0)
    fake_f1 = (2 * fake_prec * fake_recall / (fake_prec + fake_recall)
               if (fake_prec + fake_recall) else 0.0)
    macro_f1 = (real_f1 + fake_f1) / 2
    accuracy = (tp + tn) / len(results) if results else 0.0
    balanced_acc = (real_recall + fake_recall) / 2

    return {
        "threshold": threshold,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "real_recall": round(real_recall, 4),
        "fake_recall": round(fake_recall, 4),
        "real_precision": round(real_prec, 4),
        "fake_precision": round(fake_prec, 4),
        "real_f1": round(real_f1, 4),
        "fake_f1": round(fake_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def compute_auc(results):
    """Simple AUC-ROC via trapezoidal rule."""
    try:
        from sklearn.metrics import roc_auc_score
        y_true = [1 if r["true_label"] == "real" else 0 for r in results]
        y_score = [r["p_real"] for r in results]
        if len(set(y_true)) < 2:
            return None
        return round(roc_auc_score(y_true, y_score), 4)
    except ImportError:
        # Manual trapezoidal AUC
        y_true = [1 if r["true_label"] == "real" else 0 for r in results]
        y_score = [r["p_real"] for r in results]
        pairs = sorted(zip(y_score, y_true), reverse=True)
        tp = fp = 0
        prev_tp = prev_fp = 0
        auc = 0.0
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return None
        for _, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += (tp + prev_tp) / 2
                prev_tp = tp
            prev_fp = fp
        return round(auc / (n_pos * n_neg), 4)


def threshold_sweep(results):
    thresholds = [round(t, 2) for t in np.arange(0.10, 0.91, 0.01)]
    sweep = [compute_metrics(results, t) for t in thresholds]

    best_f1 = max(sweep, key=lambda x: x["macro_f1"])
    best_bacc = max(sweep, key=lambda x: x["balanced_accuracy"])
    best_real = max(sweep, key=lambda x: x["real_recall"])
    best_fake = max(sweep, key=lambda x: x["fake_recall"])

    return sweep, {
        "best_macro_f1":      best_f1,
        "best_balanced_acc":  best_bacc,
        "best_real_recall":   best_real,
        "best_fake_recall":   best_fake,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", required=True, choices=["test_audio", "microphone"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--default_threshold", type=float, default=0.49)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "test_audio_holdout_eval" if args.benchmark == "test_audio" else "microphone_holdout_eval"
    log_path = out / f"{prefix}.log"

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log(f"=== full_run_003 Holdout Evaluation: {args.benchmark} ===")
    log(f"Checkpoint : {args.checkpoint}")
    log(f"Device     : {args.device}")
    log(f"Threshold  : {args.default_threshold} (default; sweep follows)")
    log()

    # Load model
    log("Loading model...")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    log("Model loaded.")
    log()

    # Collect files
    samples = collect_files(args.benchmark)
    log(f"Files found: {len(samples)}")
    real_count = sum(1 for _, l in samples if l == "real")
    fake_count = sum(1 for _, l in samples if l == "fake")
    log(f"  real: {real_count}  fake: {fake_count}")
    log()

    # Inference
    log("Running inference...")
    results = []
    for path, true_label in samples:
        audio = load_audio(path)
        p_real, p_fake = predict_file(model, audio, device)
        results.append({
            "file": path.name,
            "true_label": true_label,
            "p_real": round(p_real, 6),
            "p_fake": round(p_fake, 6),
            "pred_label": "real" if p_real >= args.default_threshold else "fake",
            "correct": ("real" if p_real >= args.default_threshold else "fake") == true_label,
        })
        log(f"  {path.name:50s}  true={true_label}  p_real={p_real:.4f}  pred={'real' if p_real>=args.default_threshold else 'fake'}  {'OK' if results[-1]['correct'] else 'WRONG'}")

    log()

    # Metrics at default threshold
    metrics = compute_metrics(results, args.default_threshold)
    auc = compute_auc(results)

    log(f"=== Metrics @ threshold={args.default_threshold} ===")
    log(f"  Accuracy         : {metrics['accuracy']:.4f}")
    log(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    log(f"  Real recall      : {metrics['real_recall']:.4f}")
    log(f"  Fake recall      : {metrics['fake_recall']:.4f}")
    log(f"  Real precision   : {metrics['real_precision']:.4f}")
    log(f"  Fake precision   : {metrics['fake_precision']:.4f}")
    log(f"  AUC-ROC          : {auc if auc is not None else 'N/A'}")
    log(f"  Confusion matrix :")
    log(f"    TP (real→real) : {metrics['tp']}")
    log(f"    TN (fake→fake) : {metrics['tn']}")
    log(f"    FP (fake→real) : {metrics['fp']}")
    log(f"    FN (real→fake) : {metrics['fn']}")
    log()

    # p_real stats by class
    real_scores = [r["p_real"] for r in results if r["true_label"] == "real"]
    fake_scores = [r["p_real"] for r in results if r["true_label"] == "fake"]
    log("=== p_real distribution by true class ===")
    log(f"  real  — mean={np.mean(real_scores):.4f}  min={np.min(real_scores):.4f}  max={np.max(real_scores):.4f}")
    log(f"  fake  — mean={np.mean(fake_scores):.4f}  min={np.min(fake_scores):.4f}  max={np.max(fake_scores):.4f}")
    log()

    # FP / FN lists
    fps = [r for r in results if r["true_label"] == "fake" and r["pred_label"] == "real"]
    fns = [r for r in results if r["true_label"] == "real" and r["pred_label"] == "fake"]
    log(f"=== False Positives (fake predicted as real): {len(fps)} ===")
    for r in fps:
        log(f"  {r['file']}  p_real={r['p_real']:.4f}")
    log()
    log(f"=== False Negatives (real predicted as fake): {len(fns)} ===")
    for r in fns:
        log(f"  {r['file']}  p_real={r['p_real']:.4f}")
    log()

    # Threshold sweep
    log("=== Threshold Sweep (0.10 → 0.90) ===")
    sweep, best = threshold_sweep(results)
    log(f"  Best macro F1        : threshold={best['best_macro_f1']['threshold']}  F1={best['best_macro_f1']['macro_f1']}  real_recall={best['best_macro_f1']['real_recall']}  fake_recall={best['best_macro_f1']['fake_recall']}")
    log(f"  Best balanced acc    : threshold={best['best_balanced_acc']['threshold']}  bacc={best['best_balanced_acc']['balanced_accuracy']}  real_recall={best['best_balanced_acc']['real_recall']}  fake_recall={best['best_balanced_acc']['fake_recall']}")
    log(f"  Best real recall     : threshold={best['best_real_recall']['threshold']}  real_recall={best['best_real_recall']['real_recall']}  fake_recall={best['best_real_recall']['fake_recall']}")
    log(f"  Best fake recall     : threshold={best['best_fake_recall']['threshold']}  fake_recall={best['best_fake_recall']['fake_recall']}  real_recall={best['best_fake_recall']['real_recall']}")
    log()

    # Save log
    log_path.write_text("\n".join(lines), encoding="utf-8")

    # Save CSV
    import csv as csv_mod
    csv_path = out / f"{prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Save JSON
    json_path = out / f"{prefix}.json"
    payload = {
        "benchmark": args.benchmark,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "default_threshold": args.default_threshold,
        "sample_count": len(results),
        "real_count": real_count,
        "fake_count": fake_count,
        "metrics_at_default_threshold": {**metrics, "auc_roc": auc},
        "p_real_stats": {
            "real_class": {"mean": round(float(np.mean(real_scores)), 4),
                           "min":  round(float(np.min(real_scores)), 4),
                           "max":  round(float(np.max(real_scores)), 4)},
            "fake_class": {"mean": round(float(np.mean(fake_scores)), 4),
                           "min":  round(float(np.min(fake_scores)), 4),
                           "max":  round(float(np.max(fake_scores)), 4)},
        },
        "false_positives": [{"file": r["file"], "p_real": r["p_real"]} for r in fps],
        "false_negatives": [{"file": r["file"], "p_real": r["p_real"]} for r in fns],
        "threshold_sweep_best": {k: v for k, v in best.items()},
        "threshold_sweep_full": sweep,
        "per_file_results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nOutputs saved:")
    print(f"  {log_path}")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
