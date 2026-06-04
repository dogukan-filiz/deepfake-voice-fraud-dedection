"""
scripts/ood_experiment.py

Out-of-distribution degradation experiment for the deployed deepfake detector.

Generates degraded variants of the 50+50 test library with FFmpeg (WhatsApp/Opus,
loudness shifts, headset bandlimit, narrowband resample), scores every variant with
the same model the backend loads, and reports per-variant p_real shifts and verdict
flips. This quantifies WHICH input transformations push real voices into fraud
territory before we design the inference preprocessing pipeline.

Usage:
    .venv/Scripts/python.exe scripts/ood_experiment.py
    .venv/Scripts/python.exe scripts/ood_experiment.py --variants opus_24k gain_minus12
    .venv/Scripts/python.exe scripts/ood_experiment.py --preprocess   # Phase 2 re-run

Outputs to training_runs/ood_experiment/:
    results.json   - summary metrics per variant
    results.md     - human-readable report
    per_file.csv   - every (file, variant) score row
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGET_SR = 16000

# variant name -> list of ffmpeg arg lists applied in sequence (round-trips)
# {tmp} placeholders resolved at runtime.
VARIANTS: dict[str, dict] = {
    "opus_24k": {
        "desc": "WhatsApp voice note sim (libopus 24kbps round-trip)",
        "steps": [
            ["-c:a", "libopus", "-b:a", "24k", "-ar", "16000", "{tmp}.opus"],
            ["-ar", "16000", "-ac", "1", "{out}"],
        ],
    },
    "opus_12k": {
        "desc": "Poor-network voice note (libopus 12kbps round-trip)",
        "steps": [
            ["-c:a", "libopus", "-b:a", "12k", "-ar", "16000", "{tmp}.opus"],
            ["-ar", "16000", "-ac", "1", "{out}"],
        ],
    },
    "gain_minus12": {
        "desc": "Quiet headset mic (-12 dB)",
        "steps": [["-af", "volume=-12dB", "-ar", "16000", "-ac", "1", "{out}"]],
    },
    "gain_plus6": {
        "desc": "Hot mic (+6 dB, clipping allowed)",
        "steps": [["-af", "volume=6dB", "-ar", "16000", "-ac", "1", "{out}"]],
    },
    "headset_sim": {
        "desc": "Consumer headset response (HP 100Hz, LP 7kHz) + light noise",
        "steps": [["-af", "highpass=f=100,lowpass=f=7000", "-ar", "16000", "-ac", "1", "{out}"]],
        "add_noise_snr_db": 30.0,
    },
    "resample_8k": {
        "desc": "Narrowband 8kHz round-trip",
        "steps": [
            ["-ar", "8000", "-ac", "1", "{tmp}_8k.wav"],
            ["-ar", "16000", "-ac", "1", "{out}"],
        ],
    },
}


def ffmpeg_exe() -> str:
    exe = os.environ.get("FFMPEG_EXE") or os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            raise RuntimeError("FFmpeg not found (PATH, FFMPEG_EXE, or imageio-ffmpeg).")
    return exe


def run_ffmpeg(exe: str, args: list[str]) -> None:
    proc = subprocess.run(
        [exe, "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{proc.stderr.strip()}")


def add_noise(wav_path: Path, snr_db: float) -> None:
    import soundfile as sf

    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    rms = float(np.sqrt(np.mean(data**2))) or 1e-9
    noise_rms = rms / (10 ** (snr_db / 20))
    rng = np.random.default_rng(1337)  # deterministic across runs
    noisy = data + rng.normal(0.0, noise_rms, size=data.shape)
    sf.write(wav_path, np.clip(noisy, -1.0, 1.0).astype(np.float32), sr)


def generate_variant(exe: str, src: Path, out: Path, spec: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp_base = str(Path(td) / "step")
        current_input = src
        for step in spec["steps"]:
            resolved = [s.format(tmp=tmp_base, out=str(out)) for s in step]
            run_ffmpeg(exe, ["-i", str(current_input), *resolved])
            produced = resolved[-1]
            current_input = Path(produced)
    snr = spec.get("add_noise_snr_db")
    if snr is not None:
        add_noise(out, snr)


def load_model():
    os.environ.setdefault("LOCAL_SSL_MODEL_DIR", "models/ssl_aasist")
    os.environ.setdefault(
        "LOCAL_SSL_WEIGHTS_PATH", "training_runs/full_run_002/best_head.pth"
    )
    os.chdir(ROOT)
    from backend.model_wrapper import get_model, get_model_status

    model = get_model()
    status = get_model_status()
    return model, status


def score_file(model, wav_path: Path, preprocess: bool) -> dict:
    import soundfile as sf

    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    data = data.astype(np.float32)
    if preprocess:
        from backend.preprocess import preprocess_waveform

        data = preprocess_waveform(data, sr)
    return model.predict({"waveform": data, "sr": sr})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()))
    parser.add_argument("--threshold", type=float, default=0.49,
                        help="full_run_002 calibrated threshold")
    parser.add_argument("--degraded_dir", default="test_audio_degraded")
    parser.add_argument("--output_dir", default="training_runs/ood_experiment")
    parser.add_argument("--preprocess", action="store_true",
                        help="apply backend.preprocess pipeline before scoring")
    parser.add_argument("--regen", action="store_true",
                        help="regenerate degraded files even if present")
    parser.add_argument("--limit", type=int, default=None,
                        help="max files per label (quick pass)")
    args = parser.parse_args()

    exe = ffmpeg_exe()
    degraded_root = ROOT / args.degraded_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, Path]] = []
    for label in ("real", "fake"):
        d = ROOT / "test_audio" / label
        files = sorted(d.glob("*.wav"))
        if args.limit:
            files = files[: args.limit]
        sources += [(label, p) for p in files]
    if not sources:
        raise SystemExit("test_audio/real|fake empty - nothing to do")

    # --- 1. Generate degraded variants ---------------------------------
    for vname in args.variants:
        spec = VARIANTS[vname]
        vdir = degraded_root / vname
        existing = len(list(vdir.glob("*/*.wav"))) if vdir.exists() else 0
        if existing >= len(sources) and not args.regen:
            print(f"[gen] {vname}: {existing} files present, skipping")
            continue
        print(f"[gen] {vname}: {spec['desc']}")
        for label, src in sources:
            out = vdir / label / src.name
            generate_variant(exe, src, out, spec)

    # --- 2. Score original + variants -----------------------------------
    model, status = load_model()
    print(f"[model] {status}")
    suffix = "_preprocessed" if args.preprocess else ""

    rows: list[dict] = []
    conditions = ["original"] + list(args.variants)
    for cond in conditions:
        print(f"[score] {cond}", flush=True)
        for n, (label, src) in enumerate(sources):
            if n % 10 == 0:
                print(f"  {cond}: {n}/{len(sources)}", flush=True)
            path = src if cond == "original" else degraded_root / cond / label / src.name
            try:
                out = score_file(model, path, args.preprocess)
                rows.append({
                    "condition": cond,
                    "label": label,
                    "file": src.name,
                    "p_real": out["p_real"],
                    "p_fake": out["p_fake"],
                })
            except Exception as e:  # keep going, record failure
                print(f"  ERROR {cond}/{label}/{src.name}: {e}")
                rows.append({
                    "condition": cond, "label": label, "file": src.name,
                    "p_real": -1.0, "p_fake": -1.0,
                })

    # --- 3. Aggregate ----------------------------------------------------
    thr = args.threshold
    base = {(r["label"], r["file"]): r["p_real"] for r in rows if r["condition"] == "original"}

    summary: dict[str, dict] = {}
    for cond in conditions:
        cond_rows = [r for r in rows if r["condition"] == cond and r["p_real"] >= 0]
        real = [r for r in cond_rows if r["label"] == "real"]
        fake = [r for r in cond_rows if r["label"] == "fake"]
        real_flagged = sum(1 for r in real if r["p_real"] < thr)       # false alarms
        fake_missed = sum(1 for r in fake if r["p_real"] >= thr)       # misses
        shifts = [r["p_real"] - base[(r["label"], r["file"])]
                  for r in cond_rows if (r["label"], r["file"]) in base]
        summary[cond] = {
            "desc": VARIANTS.get(cond, {}).get("desc", "unmodified"),
            "n": len(cond_rows),
            "real_p_real_mean": round(float(np.mean([r["p_real"] for r in real])), 4) if real else None,
            "fake_p_real_mean": round(float(np.mean([r["p_real"] for r in fake])), 4) if fake else None,
            "mean_p_real_shift": round(float(np.mean(shifts)), 4) if shifts else 0.0,
            "real_flagged_as_fraud": real_flagged,
            "real_false_alarm_rate": round(real_flagged / max(len(real), 1), 4),
            "fake_missed_as_clean": fake_missed,
            "fake_miss_rate": round(fake_missed / max(len(fake), 1), 4),
        }

    result = {
        "threshold": thr,
        "preprocess": args.preprocess,
        "model_status": {k: str(v) for k, v in status.items()},
        "summary": summary,
    }

    json_path = out_dir / f"results{suffix}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = out_dir / f"per_file{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "label", "file", "p_real", "p_fake"])
        w.writeheader()
        w.writerows(rows)

    md = [f"# OOD Degradation Experiment{' (with preprocessing)' if args.preprocess else ''}",
          "", f"Threshold: {thr}", "",
          "| Variant | Real FA | FA rate | Fake miss | Miss rate | mean dp_real | real mean p_real | fake mean p_real |",
          "|---|---|---|---|---|---|---|---|"]
    for cond, s in summary.items():
        md.append(
            f"| {cond} | {s['real_flagged_as_fraud']} | {s['real_false_alarm_rate']:.0%} "
            f"| {s['fake_missed_as_clean']} | {s['fake_miss_rate']:.0%} "
            f"| {s['mean_p_real_shift']:+.3f} | {s['real_p_real_mean']} | {s['fake_p_real_mean']} |"
        )
    md_path = out_dir / f"results{suffix}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nOutputs: {json_path}, {csv_path}, {md_path}")


if __name__ == "__main__":
    main()
