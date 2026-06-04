"""
scripts/prepare_balanced_dataset_v4.py

Generate call-channel-normalized dataset for full_run_004.

Strategy: normalize-then-train
  - Read metadata_v3.csv (split assignments already fixed)
  - Apply narrowband_g711 and/or wideband_opus to every source file
  - Save normalized WAVs to data/call_normalized_v4/
  - Write training_runs/full_run_004/metadata_v4.csv

Output layout:
  data/call_normalized_v4/
    narrowband_g711/
      train/real/   train/fake/
      val/real/     val/fake/
      test/real/    test/fake/
      holdout/test_audio/real/         holdout/test_audio/fake/
      holdout/microphone_benchmark/real/  holdout/microphone_benchmark/fake_replayed/
    wideband_opus/
      (same structure)

Usage:
  # Dry-run (no file writes, just statistics)
  python3 scripts/prepare_balanced_dataset_v4.py --dry_run

  # Real run (writes all normalized WAVs + metadata_v4.csv)
  python3 scripts/prepare_balanced_dataset_v4.py

  # One profile only
  python3 scripts/prepare_balanced_dataset_v4.py --profiles g711_only

  # Limit files for quick test
  python3 scripts/prepare_balanced_dataset_v4.py --dry_run --max_files 20
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from normalize_call_channel import process_file, compute_stats, MODES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

META_V3   = REPO / "training_runs/full_run_003/metadata_v3.csv"
OUT_ROOT  = REPO / "data/call_normalized_v4"
META_V4_DIR = REPO / "training_runs/full_run_004"
META_V4   = META_V4_DIR / "metadata_v4.csv"

PROFILE_MAP = {
    "g711_only":  ["narrowband_g711"],
    "opus_only":  ["wideband_opus"],
    "both":       ["narrowband_g711", "wideband_opus"],
}

PROFILE_SHORT = {
    "narrowband_g711": "g711",
    "wideband_opus":   "opus",
}

QC_SAMPLE_N = 10   # files to include in QC table
TARGET_SR   = 16000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bytes_to_human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def wav_bytes(n_samples):
    """Estimated PCM_16 WAV size in bytes."""
    return n_samples * 2 + 44   # 2 bytes/sample + WAV header


def output_path(profile, row, src_path):
    """Compute output WAV path for a given row + profile."""
    split = row["split"]
    label = row["label"]

    if split == "holdout":
        bname = row.get("benchmark_name", "")
        if bname == "test_audio":
            sub = OUT_ROOT / profile / "holdout/test_audio" / label
        elif bname == "microphone_benchmark":
            # for fake_replayed we keep label as-is (label=fake, but subdir is fake_replayed)
            ch = row.get("channel_type", "")
            if ch == "replayed":
                sub = OUT_ROOT / profile / "holdout/microphone_benchmark/fake_replayed"
            else:
                sub = OUT_ROOT / profile / "holdout/microphone_benchmark" / label
        else:
            sub = OUT_ROOT / profile / "holdout/other" / label
    else:
        sub = OUT_ROOT / profile / split / label

    stem = Path(src_path).stem
    return sub / f"{stem}_{PROFILE_SHORT[profile]}.wav"


def load_duration_sec(path):
    """Get audio duration without full decode."""
    try:
        info = sf.info(str(path))
        return info.duration
    except Exception:
        try:
            import librosa
            return librosa.get_duration(path=str(path))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Count files and estimate disk; do not write")
    parser.add_argument("--profiles", choices=list(PROFILE_MAP.keys()),
                        default="both")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Process at most N source files (useful for quick tests)")
    parser.add_argument("--qc_samples", type=int, default=QC_SAMPLE_N,
                        help="Files to include in QC table (dry_run only uses fast path)")
    parser.add_argument("--meta_v3", default=str(META_V3))
    parser.add_argument("--output_root", default=str(OUT_ROOT))
    parser.add_argument("--meta_v4", default=str(META_V4))
    args = parser.parse_args()

    profiles = PROFILE_MAP[args.profiles]
    out_root = Path(args.output_root)
    meta_v4_path = Path(args.meta_v4)

    lines = []
    def log(s=""):
        print(s)
        lines.append(str(s))

    log("=" * 70)
    log("prepare_balanced_dataset_v4.py")
    log(f"  Source metadata : {args.meta_v3}")
    log(f"  Output root     : {out_root}")
    log(f"  metadata_v4.csv : {meta_v4_path}")
    log(f"  Profiles        : {profiles}")
    log(f"  Dry-run         : {args.dry_run}")
    log(f"  Max files       : {args.max_files or 'unlimited'}")
    log("=" * 70)
    log()

    # -----------------------------------------------------------------------
    # Read metadata_v3
    # -----------------------------------------------------------------------
    with open(args.meta_v3, newline="", encoding="utf-8") as f:
        rows_v3 = list(csv.DictReader(f))

    log(f"Loaded {len(rows_v3)} rows from metadata_v3")

    # Limit if requested
    if args.max_files:
        rows_v3 = rows_v3[:args.max_files]
        log(f"  Truncated to {len(rows_v3)} rows (--max_files)")
    log()

    # -----------------------------------------------------------------------
    # Path existence check
    # -----------------------------------------------------------------------
    log("--- Path existence check ---")
    missing = []
    for row in rows_v3:
        p = Path(row["path"])
        if not p.exists():
            missing.append(row["path"])
    if missing:
        log(f"  WARN: {len(missing)} source files not found:")
        for m in missing[:10]:
            log(f"    {m}")
        if len(missing) > 10:
            log(f"    ... ({len(missing) - 10} more)")
    else:
        log(f"  All {len(rows_v3)} source files found. ✓")
    log()

    # -----------------------------------------------------------------------
    # Plan: for each row × profile, compute output path
    # -----------------------------------------------------------------------
    plan = []
    for row in rows_v3:
        src = Path(row["path"])
        for profile in profiles:
            dst = output_path(profile, row, src)
            plan.append({
                "row":     row,
                "src":     src,
                "dst":     dst,
                "profile": profile,
            })

    log(f"Total output files planned: {len(plan)}")
    log()

    # -----------------------------------------------------------------------
    # Split / domain / profile distribution report
    # -----------------------------------------------------------------------
    log("--- Distribution report (planned) ---")

    split_counts   = {}
    label_counts   = {}
    domain_counts  = {}
    profile_counts = {}
    source_counts  = {}

    for p in plan:
        row = p["row"]
        split   = row["split"]
        label   = row["label"]
        channel = row.get("channel_type", "?")
        profile = p["profile"]
        source  = row.get("source", "?")

        split_counts[split]           = split_counts.get(split, 0) + 1
        label_counts[label]           = label_counts.get(label, 0) + 1
        domain_counts[(channel,label)]= domain_counts.get((channel,label), 0) + 1
        profile_counts[profile]       = profile_counts.get(profile, 0) + 1
        source_counts[source]         = source_counts.get(source, 0) + 1

    log("  Split distribution:")
    for s, c in sorted(split_counts.items()):
        log(f"    {s:12s}: {c}")
    log()

    log("  Label distribution:")
    for l, c in sorted(label_counts.items()):
        log(f"    {l:10s}: {c}")
    log()

    log("  Domain (channel_type × label) distribution:")
    for (ch, lb), c in sorted(domain_counts.items()):
        log(f"    {ch:15s} × {lb:6s}: {c}")
    log()

    log("  Call profile distribution:")
    for pr, c in sorted(profile_counts.items()):
        log(f"    {pr:20s}: {c}")
    log()

    log("  Source distribution:")
    for src, c in sorted(source_counts.items()):
        log(f"    {src:50s}: {c}")
    log()

    # -----------------------------------------------------------------------
    # Train balance check (first profile only to match v3 structure)
    # -----------------------------------------------------------------------
    log("--- Train balance check (per profile) ---")
    for profile in profiles:
        train_rows = [p for p in plan if p["row"]["split"] == "train" and p["profile"] == profile]
        real_n = sum(1 for p in train_rows if p["row"]["label"] == "real")
        fake_n = sum(1 for p in train_rows if p["row"]["label"] == "fake")
        ratio  = real_n / fake_n if fake_n else float("inf")
        log(f"  [{profile}]  train: real={real_n}  fake={fake_n}  ratio={ratio:.2f}")
    log()

    # -----------------------------------------------------------------------
    # Leakage check: holdout files must not appear in train/val/test
    # -----------------------------------------------------------------------
    log("--- Leakage check ---")
    holdout_stems = {
        Path(p["src"]).stem
        for p in plan
        if p["row"]["split"] == "holdout"
    }
    train_stems = {
        Path(p["src"]).stem
        for p in plan
        if p["row"]["split"] in ("train", "val", "test")
    }
    overlap = holdout_stems & train_stems
    if overlap:
        log(f"  FAIL: {len(overlap)} holdout files appear in train/val/test splits!")
        for s in list(overlap)[:5]:
            log(f"    {s}")
    else:
        log(f"  OK — no holdout stems in train/val/test ({len(holdout_stems)} holdout, {len(train_stems)} trainable)")
    log()

    # -----------------------------------------------------------------------
    # Disk size estimate
    # -----------------------------------------------------------------------
    log("--- Estimated disk usage ---")
    total_bytes = 0
    dur_samples = []
    existing_bytes = 0
    already_done = 0
    for p in plan:
        dur = load_duration_sec(p["src"])
        if dur is None:
            dur = 5.0   # fallback
        n_samples = int(dur * TARGET_SR)
        b = wav_bytes(n_samples)
        total_bytes += b
        dur_samples.append(dur)
        if p["dst"].exists():
            already_done += 1
            existing_bytes += p["dst"].stat().st_size

    avg_dur = float(np.mean(dur_samples)) if dur_samples else 0
    log(f"  Files planned     : {len(plan)}")
    log(f"  Already on disk   : {already_done}")
    log(f"  Avg source dur    : {avg_dur:.1f} s")
    log(f"  Estimated total   : {bytes_to_human(total_bytes)}")
    if already_done:
        log(f"  Already written   : {bytes_to_human(existing_bytes)} ({already_done} files)")
    log()

    # -----------------------------------------------------------------------
    # QC sample table (actual decode + normalize of a small sample)
    # -----------------------------------------------------------------------
    log(f"--- QC sample table (n={args.qc_samples} files, profile={profiles[0]}) ---")
    qc_rows = [p for p in plan if p["profile"] == profiles[0]]
    # Sample across splits
    import random; random.seed(42)
    qc_sample = random.sample(qc_rows, min(args.qc_samples, len(qc_rows)))

    qc_results = []
    for p in qc_sample:
        src = p["src"]
        profile = p["profile"]
        if not src.exists():
            continue
        try:
            audio_out = process_file(src, mode=profile)
            stats = compute_stats(audio_out)
            clipping = int((np.abs(audio_out) > 0.99).mean() * len(audio_out))
            clipping_flag = 1 if clipping > 5 else 0
            qc_results.append({
                "file":          src.name[:40],
                "split":         p["row"]["split"],
                "label":         p["row"]["label"],
                "profile":       profile,
                "duration_s":    stats["duration_sec"],
                "sample_rate":   TARGET_SR,
                "rms_db":        stats["rms_db"],
                "silence_ratio": stats["silence_ratio"],
                "clipping":      clipping_flag,
                "ok":            stats["duration_sec"] > 0.1,
            })
        except Exception as e:
            qc_results.append({
                "file": src.name[:40], "split": p["row"]["split"],
                "label": p["row"]["label"], "profile": profile,
                "error": str(e), "ok": False,
            })

    # Print QC table
    hdr = f"  {'File':42s} {'Split':8s} {'Label':6s} {'Dur':6s} {'RMS':7s} {'Sil':6s} {'Clip':5s} {'OK':4s}"
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))
    for q in qc_results:
        if "error" in q:
            log(f"  {'ERROR':42s} {q['split']:8s} {q['label']:6s} {'—':6s} {'—':7s} {'—':6s} {'—':5s} FAIL")
            continue
        log(
            f"  {q['file']:42s} {q['split']:8s} {q['label']:6s} "
            f"{q['duration_s']:6.2f} {q['rms_db']:7.1f} {q['silence_ratio']:6.3f} "
            f"{'YES' if q['clipping'] else 'no':5s} {'✓' if q['ok'] else '✗':4s}"
        )
    qc_fails = sum(1 for q in qc_results if not q.get("ok", False))
    log(f"\n  QC: {len(qc_results)} tested, {len(qc_results)-qc_fails} OK, {qc_fails} FAIL")
    log()

    # -----------------------------------------------------------------------
    # Ready-for-training check
    # -----------------------------------------------------------------------
    log("--- full_run_004 readiness ---")
    issues = []
    if missing:
        issues.append(f"{len(missing)} source files missing")
    if overlap:
        issues.append(f"{len(overlap)} leakage conflicts")
    if qc_fails > 0:
        issues.append(f"{qc_fails} QC failures")
    train_plan = [p for p in plan if p["row"]["split"] == "train"]
    if len(train_plan) < 100:
        issues.append("fewer than 100 training rows — check metadata")

    if issues:
        log("  NOT READY. Issues found:")
        for iss in issues:
            log(f"    - {iss}")
        ready = False
    else:
        log("  READY for training. ✓")
        ready = True
    log()

    # -----------------------------------------------------------------------
    # Dry-run exit
    # -----------------------------------------------------------------------
    if args.dry_run:
        log("=" * 70)
        log("DRY-RUN complete — no files written.")
        log("Run without --dry_run to generate normalized audio + metadata_v4.csv")
        log("=" * 70)
        return

    # -----------------------------------------------------------------------
    # Write normalized files
    # -----------------------------------------------------------------------
    log("=" * 70)
    log("Writing normalized audio files...")
    log()

    t0 = time.time()
    n_written  = 0
    n_skipped  = 0
    n_error    = 0
    v4_rows    = []

    for i, p in enumerate(plan):
        src     = p["src"]
        dst     = p["dst"]
        row     = p["row"]
        profile = p["profile"]

        # Skip if already written (idempotent)
        if dst.exists():
            n_skipped += 1
            # Still add to metadata
        else:
            if not src.exists():
                n_error += 1
                print(f"  SKIP (not found): {src}")
                continue
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                audio_out = process_file(src, mode=profile)
                sf.write(str(dst), audio_out, TARGET_SR, subtype="PCM_16")
                n_written += 1
            except Exception as e:
                n_error += 1
                print(f"  ERROR [{src.name}]: {e}")
                continue

        # Compute stats for metadata
        try:
            info = sf.info(str(dst))
            dur  = info.duration
        except Exception:
            dur = 0.0

        # Build v4 row
        v4_row = dict(row)
        v4_row["path"]         = str(dst)
        v4_row["codec"]        = "wav_pcm_norm"
        v4_row["call_profile"] = profile
        v4_row["norm_source"]  = row["path"]
        v4_row["duration_sec"] = round(dur, 3)
        v4_rows.append(v4_row)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(plan) - i - 1) / rate
            print(f"  [{i+1}/{len(plan)}]  written={n_written}  skipped={n_skipped}  "
                  f"err={n_error}  {remaining:.0f}s remaining")

    elapsed = time.time() - t0
    log(f"Done: {n_written} written, {n_skipped} skipped, {n_error} errors in {elapsed:.1f}s")
    log()

    # -----------------------------------------------------------------------
    # Write metadata_v4.csv
    # -----------------------------------------------------------------------
    meta_v4_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows_v3[0].keys()) + ["call_profile", "norm_source"]
    # Ensure all v4 rows have all fields
    for v4_row in v4_rows:
        for f in fieldnames:
            if f not in v4_row:
                v4_row[f] = ""

    with open(meta_v4_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(v4_rows)

    log(f"metadata_v4.csv written: {meta_v4_path} ({len(v4_rows)} rows)")
    log()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    log("=" * 70)
    log("SUMMARY")
    log(f"  Output root      : {out_root}")
    log(f"  metadata_v4.csv  : {meta_v4_path}")
    log(f"  Files written    : {n_written}")
    log(f"  Files skipped    : {n_skipped}")
    log(f"  Errors           : {n_error}")
    log(f"  Total rows       : {len(v4_rows)}")
    log(f"  Ready            : {'YES ✓' if ready else 'NO — see issues above'}")
    log("=" * 70)


if __name__ == "__main__":
    main()
