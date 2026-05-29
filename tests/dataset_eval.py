#!/usr/bin/env python3
"""
Dataset evaluation runner for TD test sets T2 (in-domain) and T3 (cross-domain).

Usage:
  python tests/dataset_eval.py --set T2          # 200+200 random from for-original/validation
  python tests/dataset_eval.py --set T3          # full 408+408 for-rerecorded/testing
  python tests/dataset_eval.py --set T2 --n 50   # override per-class sample count

Backend must be running on http://127.0.0.1:8010.
Output JSON: tests/results/<set>_results.json
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

BACKEND = "http://127.0.0.1:8010"
DATASET_ROOT = Path(r"D:\dataset")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SET_DEFINITIONS = {
    "T2": {
        "src": DATASET_ROOT / "for-original" / "validation",
        "label": "in-domain",
        "n_per_class": 200,
    },
    "T3": {
        "src": DATASET_ROOT / "for-rerecorded" / "testing",
        "label": "cross-domain",
        "n_per_class": 408,
    },
}


def sample_files(src: Path, n_per_class: int, seed: int = 42) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    out: List[Tuple[str, str]] = []
    for cls in ("real", "fake"):
        cdir = src / cls
        files = sorted(p for p in cdir.iterdir() if p.suffix.lower() == ".wav")
        if len(files) > n_per_class:
            picked = rng.sample(files, n_per_class)
        else:
            picked = files
        out.extend((str(p), cls) for p in picked)
    rng.shuffle(out)
    return out


def analyze(path: str, timeout: float = 300.0) -> Dict:
    t0 = time.perf_counter()
    try:
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, "audio/wav")}
            r = requests.post(f"{BACKEND}/analyze", files=files, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return {"success": False, "status": -1, "error": f"{type(e).__name__}: {str(e)[:200]}", "latency_s": time.perf_counter() - t0}
    latency = time.perf_counter() - t0
    if r.status_code != 200:
        return {"success": False, "status": r.status_code, "error": r.text[:200], "latency_s": latency}
    try:
        j = r.json()
    except ValueError:
        return {"success": False, "status": r.status_code, "error": "invalid_json", "latency_s": latency}
    return {
        "success": True,
        "p_real": j["p_real"],
        "p_fake": j["p_fake"],
        "authenticity_score": j["authenticity_score"],
        "is_suspected_fraud": j["is_suspected_fraud"],
        "risk_level": j["risk_level"],
        "predicted": "fake" if j["is_suspected_fraud"] else "real",
        "latency_s": latency,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=list(SET_DEFINITIONS.keys()))
    p.add_argument("--n", type=int, default=None, help="override n_per_class")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = SET_DEFINITIONS[args.set]
    n = args.n if args.n else cfg["n_per_class"]
    src: Path = cfg["src"]

    print(f"[{args.set}] {cfg['label']} | source={src} | n_per_class={n}")
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    samples = sample_files(src, n, seed=args.seed)
    print(f"[{args.set}] sampled {len(samples)} files")

    health = requests.get(f"{BACKEND}/health", timeout=5).json()
    print(f"[{args.set}] backend ok, model_type={health.get('model', {}).get('model_type')}")

    rows = []
    checkpoint_path = RESULTS_DIR / f"{args.set}_checkpoint.json"
    t_start = time.perf_counter()
    for i, (path, actual) in enumerate(samples, 1):
        res = analyze(path)
        res.update({"file": os.path.basename(path), "actual": actual})
        rows.append(res)
        if i % 20 == 0 or i == len(samples):
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / i) * (len(samples) - i)
            ok_so_far = sum(1 for r in rows if r["success"])
            print(f"  [{i}/{len(samples)}] elapsed={elapsed:6.1f}s eta={eta:6.1f}s ok={ok_so_far}", flush=True)
            with open(checkpoint_path, "w", encoding="utf-8") as cp:
                json.dump({"completed": i, "rows": rows}, cp)

    ok = [r for r in rows if r["success"]]
    failed = [r for r in rows if not r["success"]]

    y_true = [r["actual"] for r in ok]
    y_pred = [r["predicted"] for r in ok]
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0) if y_true else 0.0
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0) if y_true else {}
    cm = confusion_matrix(y_true, y_pred, labels=["real", "fake"]).tolist() if y_true else []

    latencies = [r["latency_s"] for r in ok]
    latencies.sort()
    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        k = int(p / 100 * (len(latencies) - 1))
        return latencies[k]

    out = {
        "set": args.set,
        "label": cfg["label"],
        "source": str(src),
        "n_per_class_requested": n,
        "n_total_attempted": len(samples),
        "n_success": len(ok),
        "n_failed": len(failed),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "accuracy": acc,
            "f1_weighted": f1w,
            "classification_report": report,
            "confusion_matrix_labels": ["real", "fake"],
            "confusion_matrix": cm,
        },
        "latency_s": {
            "n": len(latencies),
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50": pct(50),
            "p95": pct(95),
            "min": latencies[0] if latencies else 0.0,
            "max": latencies[-1] if latencies else 0.0,
        },
        "failures": failed[:20],
        "details": rows,
    }

    out_path = RESULTS_DIR / f"{args.set}_results.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)
    print(f"[{args.set}] DONE | acc={acc:.4f} f1w={f1w:.4f} | results -> {out_path}")


if __name__ == "__main__":
    main()
