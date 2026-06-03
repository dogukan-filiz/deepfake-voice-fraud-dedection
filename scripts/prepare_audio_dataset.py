"""Build metadata.csv from an audio directory with real/ and fake/ subdirs."""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LABEL_SUBDIRS = {"real": "real", "fake": "fake"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def collect_files(input_dir: Path):
    rows = []
    for label_name, label in LABEL_SUBDIRS.items():
        subdir = input_dir / label_name
        if not subdir.exists():
            print(f"[prepare] WARNING: subdir not found: {subdir}")
            continue
        files = sorted([p for p in subdir.iterdir() if p.suffix.lower() in AUDIO_EXTS])
        for p in files:
            rows.append({"path": str(p), "label": label})
        print(f"[prepare] {label_name}: {len(files)} files")
    return rows


def split_stratified(rows, per_class_counts, seed):
    rng = random.Random(seed)
    real_rows = [r for r in rows if r["label"] == "real"]
    fake_rows = [r for r in rows if r["label"] == "fake"]

    rng.shuffle(real_rows)
    rng.shuffle(fake_rows)

    def take(lst, n):
        n = min(n, len(lst))
        return lst[:n], lst[n:]

    n_train, n_val, n_test = per_class_counts
    real_train, real_rest = take(real_rows, n_train)
    real_val, real_rest = take(real_rest, n_val)
    real_test, _ = take(real_rest, n_test)

    fake_train, fake_rest = take(fake_rows, n_train)
    fake_val, fake_rest = take(fake_rest, n_val)
    fake_test, _ = take(fake_rest, n_test)

    result = []
    for r in real_train + fake_train:
        r["split"] = "train"
        result.append(r)
    for r in real_val + fake_val:
        r["split"] = "val"
        result.append(r)
    for r in real_test + fake_test:
        r["split"] = "test"
        result.append(r)
    return result


def split_full(rows, seed):
    rng = random.Random(seed)
    real_rows = [r for r in rows if r["label"] == "real"]
    fake_rows = [r for r in rows if r["label"] == "fake"]
    rng.shuffle(real_rows)
    rng.shuffle(fake_rows)

    def assign_splits(lst, ratios=(0.70, 0.15, 0.15)):
        n = len(lst)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, r in enumerate(lst):
            if i < n_train:
                r["split"] = "train"
            elif i < n_train + n_val:
                r["split"] = "val"
            else:
                r["split"] = "test"
        return lst

    return assign_splits(real_rows) + assign_splits(fake_rows)


def main():
    parser = argparse.ArgumentParser(description="Build metadata.csv for training pipeline")
    parser.add_argument("--input_dir", required=True,
                        help="Root audio directory with real/ and fake/ subdirs")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write metadata.csv")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: small fixed-size splits for pipeline validation only")
    parser.add_argument("--train_per_class", type=int, default=20)
    parser.add_argument("--val_per_class", type=int, default=10)
    parser.add_argument("--test_per_class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_files(input_dir)
    if not rows:
        print("[prepare] ERROR: No audio files found.")
        sys.exit(1)

    if args.smoke:
        print(f"[prepare] SMOKE mode: {args.train_per_class} train / {args.val_per_class} val / "
              f"{args.test_per_class} test per class (PIPELINE VALIDATION ONLY — not for metrics)")
        rows = split_stratified(rows, (args.train_per_class, args.val_per_class, args.test_per_class),
                                args.seed)
    else:
        print("[prepare] Full mode: 70/15/15 stratified split")
        rows = split_full(rows, args.seed)

    df = pd.DataFrame(rows, columns=["path", "label", "split"])
    out_path = output_dir / "metadata.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[prepare] metadata.csv saved to {out_path}")
    print(df.groupby(["split", "label"]).size().to_string())
    print(f"  Total rows: {len(df)}")

    if args.smoke:
        print("\n  NOTE: Smoke metadata is for pipeline smoke testing only.")
        print("  Do NOT use smoke evaluation results as final model metrics.")


if __name__ == "__main__":
    main()
