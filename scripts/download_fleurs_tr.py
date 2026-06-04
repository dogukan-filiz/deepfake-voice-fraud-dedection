"""
scripts/download_fleurs_tr.py
Download FLEURS Turkish (google/fleurs, tr_tr config) to data/fleurs_tr/.
No token required. CC-BY-4.0 license.
Outputs: 16kHz mono WAV files + fleurs_tr_metadata.csv

Usage:
  python3 scripts/download_fleurs_tr.py [--max_samples 2000] [--output_dir data/fleurs_tr]
"""
import os, sys, csv, argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def normalize_audio(data, target_rms=0.05):
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    rms = float(np.sqrt(np.mean(data ** 2)))
    if rms > 1e-6:
        data = data * (target_rms / rms)
    return np.clip(data, -1.0, 1.0)


def qc_audio(data, sr, min_dur=2.0, max_dur=20.0, max_silence=0.8):
    dur = len(data) / sr
    if dur < min_dur or dur > max_dur:
        return False, "duration={:.1f}s".format(dur)
    silence_ratio = float((np.abs(data) < 0.001).mean())
    if silence_ratio > max_silence:
        return False, "silence={:.0%}".format(silence_ratio)
    peak = float(np.abs(data).max())
    if peak < 1e-4:
        return False, "too_quiet peak={:.6f}".format(peak)
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/fleurs_tr")
    parser.add_argument("--max_samples", type=int, default=3000,
                        help="Max samples to download (0=all ~3607)")
    parser.add_argument("--splits", default="train+validation+test",
                        help="HF splits to include")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FLEURS TR from HuggingFace (google/fleurs, tr_tr)...")
    print("No token required. License: CC-BY-4.0")
    print("Output dir:", out_dir)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    import soundfile as sf

    from datasets import Audio
    ds = load_dataset("google/fleurs", "tr_tr", split=args.splits, trust_remote_code=False)
    # Disable auto-decoding to avoid torchcodec/FFmpeg dependency issues.
    # We decode manually with soundfile/librosa instead.
    ds = ds.cast_column("audio", Audio(decode=False))
    total = len(ds)
    if args.max_samples > 0:
        total = min(total, args.max_samples)
    print("Total samples to process: {}".format(total))

    COLS = ["filename", "split_orig", "language", "speaker_id",
            "duration_sec", "sample_rate", "clipping_flag", "silence_ratio",
            "qc_pass", "qc_reason", "raw_transcription"]

    csv_path = out_dir / "fleurs_tr_metadata.csv"
    written = 0
    skipped_qc = 0
    skipped_err = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLS)
        writer.writeheader()

        for idx, item in enumerate(ds):
            if args.max_samples > 0 and idx >= args.max_samples:
                break

            if idx % 200 == 0:
                print("  {}/{} written={} skipped_qc={} skipped_err={}".format(
                    idx, total, written, skipped_qc, skipped_err))

            try:
                audio = item["audio"]
                # decode=False → audio is {"path": str, "bytes": bytes}
                # Decode manually with soundfile (avoids torchcodec requirement)
                import io
                audio_bytes = audio.get("bytes")
                audio_path = audio.get("path", "")
                if audio_bytes:
                    raw, sr = sf.read(io.BytesIO(audio_bytes))
                elif audio_path and os.path.isfile(audio_path):
                    raw, sr = sf.read(audio_path)
                else:
                    skipped_err += 1
                    continue
                raw = np.array(raw, dtype=np.float32)

                # Resample to 16kHz if needed
                if sr != 16000:
                    import librosa
                    raw = librosa.resample(raw, orig_sr=sr, target_sr=16000)
                    sr = 16000

                dur = len(raw) / sr
                peak = float(np.abs(raw).max())
                silence_ratio = float((np.abs(raw) < 0.001).mean())
                clip_flag = int(peak > 0.99)

                qc_ok, qc_reason = qc_audio(raw, sr)

                fname = "fleurs_tr_{:05d}.wav".format(idx)
                fpath = out_dir / fname

                if qc_ok:
                    normed = normalize_audio(raw)
                    sf.write(str(fpath), normed, sr, subtype="PCM_16")
                    written += 1
                else:
                    skipped_qc += 1

                writer.writerow({
                    "filename": fname,
                    "split_orig": item.get("split", ""),
                    "language": "tr",
                    "speaker_id": str(item.get("speaker_id", "")),
                    "duration_sec": round(dur, 2),
                    "sample_rate": sr,
                    "clipping_flag": clip_flag,
                    "silence_ratio": round(silence_ratio, 4),
                    "qc_pass": int(qc_ok),
                    "qc_reason": qc_reason,
                    "raw_transcription": item.get("raw_transcription", "")[:100],
                })

            except Exception as e:
                skipped_err += 1
                print("  ERROR idx={}: {}".format(idx, str(e)[:80]))

    print("\nDone.")
    print("  Written WAV files: {}".format(written))
    print("  Skipped (QC fail): {}".format(skipped_qc))
    print("  Skipped (error):   {}".format(skipped_err))
    print("  Metadata CSV:      {}".format(csv_path))

    # Summary stats
    if written > 0:
        import soundfile as _sf
        durations = []
        for fp in out_dir.glob("*.wav"):
            try:
                info = _sf.info(str(fp))
                durations.append(info.frames / info.samplerate)
            except Exception:
                pass
        if durations:
            print("\nQuality summary ({} files):".format(len(durations)))
            print("  Duration: mean={:.1f}s  min={:.1f}s  max={:.1f}s  total={:.0f}min".format(
                np.mean(durations), min(durations), max(durations),
                sum(durations) / 60))


if __name__ == "__main__":
    main()
