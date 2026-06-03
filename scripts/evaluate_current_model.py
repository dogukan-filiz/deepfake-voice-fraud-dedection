"""Baseline evaluation of the current SSL+AASIST model on test_audio/."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "aasist"))


def load_audio_files(audio_dir: Path):
    exts = {".wav", ".flac", ".mp3", ".ogg"}
    return sorted([p for p in audio_dir.iterdir() if p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation on test_audio/")
    parser.add_argument("--real_dir", default=str(REPO_ROOT / "test_audio" / "real"))
    parser.add_argument("--fake_dir", default=str(REPO_ROOT / "test_audio" / "fake"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "training_runs" / "baseline"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[baseline] Loading XLSRAASISTModel...")
    t0 = time.time()
    from backend.model_wrapper_ssl import XLSRAASISTModel
    from backend.audio_processing import extract_features
    wrapper = XLSRAASISTModel(device=args.device)
    print(f"[baseline] Model loaded in {time.time()-t0:.1f}s  device={wrapper.device}")

    real_files = load_audio_files(Path(args.real_dir))
    fake_files = load_audio_files(Path(args.fake_dir))
    print(f"[baseline] Files: {len(real_files)} real, {len(fake_files)} fake")

    scores = []
    labels = []
    errors = []

    for path, label in [(p, "real") for p in real_files] + [(p, "fake") for p in fake_files]:
        try:
            feats = extract_features(path.read_bytes())
            result = wrapper.predict(feats)
            scores.append(result["p_real"])
            labels.append(1 if label == "real" else 0)
        except Exception as e:
            errors.append({"file": str(path), "error": str(e)})
            print(f"  [WARN] {path.name}: {e}")

    scores = np.array(scores)
    labels = np.array(labels)

    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
    )

    auc = roc_auc_score(labels, scores)

    # Threshold sweep
    best_f1, best_thresh = 0.0, 0.5
    sweep_rows = []
    for t in np.arange(0.0, 1.01, 0.01):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        rr = recall_score(labels, preds, pos_label=1, zero_division=0)
        fr = recall_score(labels, preds, pos_label=0, zero_division=0)
        sweep_rows.append({"threshold": round(float(t), 2), "macro_f1": round(f1, 4),
                           "real_recall": round(rr, 4), "fake_recall": round(fr, 4)})
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)

    preds_best = (scores >= best_thresh).astype(int)
    cm = confusion_matrix(labels, preds_best, labels=[0, 1]).tolist()

    metrics = {
        "n_real": int((labels == 1).sum()),
        "n_fake": int((labels == 0).sum()),
        "n_errors": len(errors),
        "auc_roc": round(float(auc), 4),
        "best_threshold": round(best_thresh, 2),
        "best_macro_f1": round(best_f1, 4),
        "real_recall_at_best": round(float(recall_score(labels, preds_best, pos_label=1, zero_division=0)), 4),
        "fake_recall_at_best": round(float(recall_score(labels, preds_best, pos_label=0, zero_division=0)), 4),
        "precision_real_at_best": round(float(precision_score(labels, preds_best, pos_label=1, zero_division=0)), 4),
        "precision_fake_at_best": round(float(precision_score(labels, preds_best, pos_label=0, zero_division=0)), 4),
        "confusion_matrix_fake0_real1": cm,
        "current_auth_threshold": 0.01,
    }

    import pandas as pd
    pd.DataFrame(sweep_rows).to_csv(output_dir / "threshold_sweep.csv", index=False)
    (output_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== BASELINE RESULTS ===")
    for k, v in metrics.items():
        if k != "confusion_matrix_fake0_real1":
            print(f"  {k}: {v}")
    print(f"  confusion_matrix [[TN,FP],[FN,TP]]: {cm}")
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
