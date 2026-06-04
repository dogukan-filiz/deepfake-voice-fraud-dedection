"""
scripts/calibrate_threshold.py

Pick the AUTH_THRESHOLD that maximizes macro-F1 over one or more per-file CSVs
produced by scripts/ood_experiment.py (combine original + degraded conditions
so the threshold holds up on out-of-domain input, not just the clean set).

Usage:
    python scripts/calibrate_threshold.py training_runs/ood_experiment_df_arena/per_file.csv
    python scripts/calibrate_threshold.py a.csv b.csv --conditions original opus_24k headset_sim
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """y: 1 = fake/fraud, 0 = real/clean."""
    f1s = []
    for cls in (0, 1):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="restrict to these conditions (default: all)")
    args = ap.parse_args()

    rows: list[dict] = []
    for path in args.csvs:
        with open(Path(path), newline="", encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f)]

    if args.conditions:
        rows = [r for r in rows if r["condition"] in set(args.conditions)]
    rows = [r for r in rows if float(r["p_real"]) >= 0]

    y_true = [1 if r["label"] == "fake" else 0 for r in rows]
    scores = [float(r["p_real"]) for r in rows]

    best = (0.0, 0.5)
    for thr in np.arange(0.01, 1.0, 0.01):
        y_pred = [1 if s < thr else 0 for s in scores]
        f1 = macro_f1(y_true, y_pred)
        if f1 > best[0]:
            best = (f1, float(thr))

    f1, thr = best
    y_pred = [1 if s < thr else 0 for s in scores]
    real_idx = [i for i, t in enumerate(y_true) if t == 0]
    fake_idx = [i for i, t in enumerate(y_true) if t == 1]
    real_rec = sum(1 for i in real_idx if y_pred[i] == 0) / max(len(real_idx), 1)
    fake_rec = sum(1 for i in fake_idx if y_pred[i] == 1) / max(len(fake_idx), 1)

    print(f"n={len(rows)}  conditions={args.conditions or 'all'}")
    print(f"best threshold = {thr:.2f}  macro_f1 = {f1:.4f}")
    print(f"real recall = {real_rec:.4f}   fake recall = {fake_rec:.4f}")


if __name__ == "__main__":
    main()
