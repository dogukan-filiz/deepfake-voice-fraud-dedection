"""
Evaluate full_run_002 (running backend at :8010) on FoR (Fake-or-Real) dataset.
Two-domain sample: for-original (digital) + for-rerecorded (mic/replay).
Posts each file to /analyze; uses live backend config (threshold, call_channel).

Usage:
  python scripts/evaluate_for_dataset.py --per_class 200 --out training_runs/full_run_002/for_eval
"""
import argparse
import json
import random
import time
from pathlib import Path

import requests

DATASET_ROOT = Path("D:/dataset")
BACKEND = "http://127.0.0.1:8010/analyze"

DOMAINS = {
    "for_original": DATASET_ROOT / "for-original" / "validation",     # digital, mp3
    "for_rerecorded": DATASET_ROOT / "for-rerecorded" / "testing",    # mic/replay, wav
}


def sample_files(folder: Path, n: int, seed: int = 42):
    files = [p for p in folder.iterdir() if p.is_file()]
    random.Random(seed).shuffle(files)
    return files[:n]


def analyze(path: Path, timeout=60):
    with open(path, "rb") as f:
        r = requests.post(BACKEND, files={"file": (path.name, f, "application/octet-stream")}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def eval_domain(name: str, root: Path, per_class: int):
    rows = []
    tp = tn = fp = fn = err = 0
    for label in ("real", "fake"):
        folder = root / label
        files = sample_files(folder, per_class)
        print(f"  [{name}/{label}] {len(files)} files")
        for i, p in enumerate(files):
            try:
                out = analyze(p)
                p_real = out["p_real"]
                fraud = out["is_suspected_fraud"]
                pred = "fake" if fraud else "real"
                correct = (pred == label)
                if label == "fake":
                    if fraud: tp += 1
                    else: fn += 1
                else:
                    if fraud: fp += 1
                    else: tn += 1
                rows.append({"domain": name, "file": p.name, "true": label,
                             "p_real": round(p_real, 4), "p_fake": round(out["p_fake"], 4),
                             "pred": pred, "correct": correct})
            except Exception as e:
                err += 1
                rows.append({"domain": name, "file": p.name, "true": label,
                             "p_real": -1, "p_fake": -1, "pred": "ERROR", "correct": False})
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(files)} done")
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0
    rec_real = tn / (tn + fp) if (tn + fp) else 0
    rec_fake = tp / (tp + fn) if (tp + fn) else 0
    prec_fake = tp / (tp + fp) if (tp + fp) else 0
    prec_real = tn / (tn + fn) if (tn + fn) else 0
    f1_fake = 2 * prec_fake * rec_fake / (prec_fake + rec_fake) if (prec_fake + rec_fake) else 0
    f1_real = 2 * prec_real * rec_real / (prec_real + rec_real) if (prec_real + rec_real) else 0
    macro_f1 = (f1_fake + f1_real) / 2

    real_scores = [r["p_real"] for r in rows if r["true"] == "real" and r["p_real"] >= 0]
    fake_scores = [r["p_real"] for r in rows if r["true"] == "fake" and r["p_real"] >= 0]

    def stats(s):
        return {"mean": round(sum(s)/len(s), 4), "min": round(min(s), 4), "max": round(max(s), 4)} if s else None

    summary = {
        "domain": name, "n": n, "errors": err,
        "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "real_recall": round(rec_real, 4), "fake_recall": round(rec_fake, 4),
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "real_p_real": stats(real_scores), "fake_p_real": stats(fake_scores),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=200)
    ap.add_argument("--out", default="training_runs/full_run_002/for_eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    health = requests.get("http://127.0.0.1:8010/health", timeout=10).json()
    print(f"Backend: threshold={health['threshold']} call_channel={health['call_channel_mode']} "
          f"weights={health['active_weights_path']}")

    all_summaries = []
    all_rows = []
    t0 = time.time()
    for name, root in DOMAINS.items():
        print(f"\n=== {name} ===")
        s, rows = eval_domain(name, root, args.per_class)
        all_summaries.append(s)
        all_rows.extend(rows)
        print(f"  acc={s['accuracy']} macro_f1={s['macro_f1']} "
              f"real_recall={s['real_recall']} fake_recall={s['fake_recall']}")

    elapsed = round(time.time() - t0, 1)
    result = {
        "model": "full_run_002",
        "threshold": health["threshold"],
        "call_channel_mode": health["call_channel_mode"],
        "active_weights": health["active_weights_path"],
        "per_class": args.per_class,
        "elapsed_sec": elapsed,
        "domains": all_summaries,
    }

    (out / "for_eval_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # CSV
    import csv
    with open(out / "for_eval_per_file.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "file", "true", "p_real", "p_fake", "pred", "correct"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nDone in {elapsed}s. Saved to {out}")
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
