"""
Prepare balanced anti-spoofing dataset v2 (500 real + 500 fake).
Token security: reads from HF_TOKEN env var or cached CLI session.
Never prints or logs token values.

Sources:
  Real:  LJSpeech   (ajaykarthick/wavefake-audio, label='R')
       + Turkish     (shunyalabs/turkish-speech-dataset)
       + GS real     (garystafford/deepfake-audio-detection, label=0)
  Fake:  WaveFake    (ajaykarthick/wavefake-audio, WF1-WF7)
       + GS fake     (garystafford/deepfake-audio-detection, label=1, excluded list applied)

Targets:
  Real train: 150 LJSpeech + 150 TR + 100 GS-real = 400
  Real val:    20 LJSpeech +  20 TR +  10 GS-real =  50
  Fake train: 150 WaveFake (WF1-5, 30 each) + 250 GS-fake = 400
  Fake val:    25 WaveFake (WF6)             +  25 GS-fake =  50
  Test: test_audio/real + test_audio/fake (external benchmark, never touched here)
"""

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import librosa

SR = 16000


def _get_token():
    """Read HF token from env or cached session. Never logs the value."""
    import huggingface_hub
    from huggingface_hub import get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    if token:
        huggingface_hub.login(token=token, add_to_git_credential=False)
        print("[auth] HuggingFace token found — logged in")
    else:
        print("[auth] No HF token. Using cached files only.")
        print("       If download fails: huggingface-cli login")
    return token


def _hf_download(repo_id, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, filename, repo_type="dataset")


def save_wav(raw: bytes, out_path: Path, src_sr: int = None):
    buf = io.BytesIO(raw)
    try:
        arr, sr = sf.read(buf, dtype="float32")
        if len(arr.shape) > 1:
            arr = arr[:, 0]
        if sr != SR:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
    except Exception:
        buf.seek(0)
        arr, sr = librosa.load(buf, sr=SR, mono=True)
    sf.write(str(out_path), arr.astype(np.float32), SR, subtype="PCM_16")


# ── WaveFake + LJSpeech ────────────────────────────────────────────────────

WFSYS = {
    "WF1": "melgan", "WF2": "full_band_melgan", "WF3": "multi_band_melgan",
    "WF4": "hifigan", "WF5": "waveglow", "WF6": "parallel_wavegan",
    "WF7": "conformer_fastspeech2",
}
TRAIN_WF_SYSTEMS = ["WF1", "WF2", "WF3", "WF4", "WF5"]
VAL_WF_SYSTEM    = "WF6"


def collect_wavefake_lj(out: Path,
                        lj_train=150, lj_val=20,
                        wf_train_per_sys=30, wf_val=25):
    real_dir = out / "real"; real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir = out / "fake"; fake_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    lj_cnt   = {"train": 0, "val": 0}
    wf_cnt   = defaultdict(lambda: {"train": 0, "val": 0})

    lj_tgt   = {"train": lj_train, "val": lj_val}
    wf_tgt   = {s: {"train": wf_train_per_sys} for s in TRAIN_WF_SYSTEMS}
    wf_tgt[VAL_WF_SYSTEM] = {"val": wf_val}

    # partition 0-1 → train utterances, partition 2 → val utterances
    for part_idx, split in [(0, "train"), (1, "train"), (2, "val")]:
        fname = f"data/partition{part_idx}-00000-of-00001.parquet"
        print(f"[wavefake] partition {part_idx} ({split})...")
        path = _hf_download("ajaykarthick/wavefake-audio", fname)
        df   = pq.read_table(path).to_pandas()
        df   = df.sort_values(["audio_id", "real_or_fake"]).reset_index(drop=True)

        for _, row in df.iterrows():
            lbl  = row["real_or_fake"]
            aid  = row["audio_id"]
            raw  = row["audio"]["bytes"]

            if lbl == "R":
                if lj_cnt[split] >= lj_tgt[split]:
                    continue
                idx = lj_cnt[split]
                fout = real_dir / f"lj_{split}_{part_idx}_{idx:04d}.wav"
                if not fout.exists():
                    save_wav(raw, fout)
                rows.append({"path": str(fout), "label": "real", "split": split,
                             "language": "en", "source": "ljspeech",
                             "generation_type": "natural", "system": "ljspeech_original",
                             "utterance_id": aid})
                lj_cnt[split] += 1

            elif lbl in TRAIN_WF_SYSTEMS and split == "train":
                if wf_cnt[lbl]["train"] >= wf_tgt[lbl]["train"]:
                    continue
                sys_name = WFSYS[lbl]
                idx = wf_cnt[lbl]["train"]
                fout = fake_dir / f"wf_{sys_name}_{part_idx}_{idx:04d}.wav"
                if not fout.exists():
                    save_wav(raw, fout)
                rows.append({"path": str(fout), "label": "fake", "split": "train",
                             "language": "en", "source": "wavefake",
                             "generation_type": "synthetic_vocoder", "system": sys_name,
                             "utterance_id": aid})
                wf_cnt[lbl]["train"] += 1

            elif lbl == VAL_WF_SYSTEM and split == "val":
                if wf_cnt[VAL_WF_SYSTEM]["val"] >= wf_tgt[VAL_WF_SYSTEM]["val"]:
                    continue
                sys_name = WFSYS[lbl]
                idx = wf_cnt[VAL_WF_SYSTEM]["val"]
                fout = fake_dir / f"wf_{sys_name}_{part_idx}_{idx:04d}.wav"
                if not fout.exists():
                    save_wav(raw, fout)
                rows.append({"path": str(fout), "label": "fake", "split": "val",
                             "language": "en", "source": "wavefake",
                             "generation_type": "synthetic_vocoder", "system": sys_name,
                             "utterance_id": aid})
                wf_cnt[VAL_WF_SYSTEM]["val"] += 1

    print(f"  LJSpeech: {dict(lj_cnt)}")
    print(f"  WaveFake train: {[wf_cnt[s]['train'] for s in TRAIN_WF_SYSTEMS]}")
    print(f"  WaveFake val (WF6): {wf_cnt[VAL_WF_SYSTEM]['val']}")
    return rows


# ── Shunyalabs Turkish ─────────────────────────────────────────────────────

def collect_turkish(out: Path, n_train=150, n_val=20):
    real_dir = out / "real"; real_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    splits = [("train", "data/train-00000-of-00004.parquet", n_train),
              ("val",   "data/validation-00000-of-00001.parquet", n_val)]
    for split, parquet_file, target in splits:
        print(f"[turkish] {parquet_file} ({split}, n={target})...")
        path = _hf_download("shunyalabs/turkish-speech-dataset", parquet_file)
        df   = pq.read_table(path).to_pandas()
        n = 0
        for idx, row in df.iterrows():
            if n >= target:
                break
            audio = row["audio"]
            raw   = audio["bytes"] if isinstance(audio, dict) else audio
            fout  = real_dir / f"tr_{split}_{n:04d}.wav"
            if not fout.exists():
                save_wav(raw, fout)
            rows.append({"path": str(fout), "label": "real", "split": split,
                         "language": "tr", "source": "shunyalabs_tr",
                         "generation_type": "natural", "system": "human_speech_tr",
                         "utterance_id": f"tr_{split}_{idx}"})
            n += 1
        print(f"  {split}: {n}")
    return rows


# ── Garystafford ──────────────────────────────────────────────────────────

def collect_garystafford(out: Path, excluded_fake_set: set,
                         real_train=100, real_val=10,
                         fake_train=250, fake_val=25):
    real_dir = out / "real"; real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir = out / "fake"; fake_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    real_cnt = {"train": 0, "val": 0}
    fake_cnt = {"train": 0, "val": 0}
    real_tgt = {"train": real_train, "val": real_val}
    fake_tgt = {"train": fake_train, "val": fake_val}

    for parquet_file in [
        "data/train-00000-of-00002.parquet",   # label=0 real
        "data/train-00001-of-00002.parquet",   # label=1 fake
    ]:
        print(f"[garystafford] {parquet_file}...")
        path = _hf_download("garystafford/deepfake-audio-detection", parquet_file)
        df   = pq.read_table(path).to_pandas()
        lval = int(df["label"].iloc[0])
        is_fake = lval == 1

        for _, row in df.iterrows():
            audio  = row["audio"]
            raw    = audio["bytes"] if isinstance(audio, dict) else audio
            gs_path = audio.get("path", "") if isinstance(audio, dict) else ""

            if is_fake:
                # Skip excluded (test_audio overlap)
                if gs_path in excluded_fake_set:
                    continue
                # train first, then val
                if fake_cnt["train"] < fake_tgt["train"]:
                    split = "train"
                elif fake_cnt["val"] < fake_tgt["val"]:
                    split = "val"
                else:
                    continue
                idx  = fake_cnt[split]
                fout = fake_dir / f"gs_fake_{split}_{idx:04d}.wav"
                if not fout.exists():
                    save_wav(raw, fout)
                rows.append({"path": str(fout), "label": "fake", "split": split,
                             "language": "en", "source": "garystafford",
                             "generation_type": "synthetic_tts",
                             "system": "elevenlabs_tts",
                             "utterance_id": f"gs_{gs_path}"})
                fake_cnt[split] += 1

            else:
                # real
                if real_cnt["train"] < real_tgt["train"]:
                    split = "train"
                elif real_cnt["val"] < real_tgt["val"]:
                    split = "val"
                else:
                    continue
                idx  = real_cnt[split]
                fout = real_dir / f"gs_real_{split}_{idx:04d}.wav"
                if not fout.exists():
                    save_wav(raw, fout)
                rows.append({"path": str(fout), "label": "real", "split": split,
                             "language": "en", "source": "garystafford",
                             "generation_type": "natural",
                             "system": "youtube_speech",
                             "utterance_id": f"gs_{gs_path}"})
                real_cnt[split] += 1

        print(f"  real: {dict(real_cnt)}  fake: {dict(fake_cnt)}")
    return rows


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, out: Path):
    print("\n" + "=" * 65)
    print("DATASET SUMMARY — v2")
    print("=" * 65)
    print("\n-- real/fake --")
    print(df["label"].value_counts().to_string())
    print("\n-- split x label --")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print("\n-- source x split x label --")
    print(df.groupby(["source", "split", "label"]).size().to_string())
    print("\n-- language x label --")
    print(df.groupby(["language", "label"]).size().to_string())
    print("\n-- fake system x split --")
    fake = df[df["label"] == "fake"]
    print(fake.groupby(["system", "split"]).size().to_string())

    print("\n-- Utterance leakage check --")
    tu = set(df[df["split"] == "train"]["utterance_id"])
    vu = set(df[df["split"] == "val"]["utterance_id"])
    tv = tu & vu
    print(f"  Train∩Val utterances: {len(tv)} {'LEAKAGE' if tv else 'clean'}")

    print("\n-- Sample paths (3 real + 3 fake) --")
    for lbl in ["real", "fake"]:
        for _, r in df[df["label"] == lbl].head(3).iterrows():
            print(f"  [{lbl:<4}][{r['split']:<5}][{r['source']:<16}][{r['language']}] {Path(r['path']).name}")

    total_bytes = sum(Path(p).stat().st_size for p in df["path"] if Path(p).exists())
    print(f"\nDisk: {total_bytes / 1e6:.1f} MB")

    # Benchmark safety
    print("\n-- Benchmark safety --")
    gs_excl_in_train = [p for p in df["utterance_id"] if "el_0001_c_part_002" in str(p)]
    print(f"  test_audio files in train/val: {len(gs_excl_in_train)} (should be 0)")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare balanced dataset v2")
    parser.add_argument("--output_dir",           required=True)
    parser.add_argument("--garystafford_excluded", required=True,
                        help="Path to overlap_pcm.json from overlap check")
    args = parser.parse_args()

    _get_token()  # reads HF_TOKEN env or cached session — never logged

    with open(args.garystafford_excluded) as f:
        overlap_data = json.load(f)
    excluded_fake_set = set(overlap_data.get("excluded_fake_files", []))
    print(f"[overlap] Excluded fake files (test_audio): {len(excluded_fake_set)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []

    print("\n[1/3] WaveFake + LJSpeech...")
    all_rows.extend(collect_wavefake_lj(out))

    print("\n[2/3] Turkish real (shunyalabs)...")
    all_rows.extend(collect_turkish(out))

    print("\n[3/3] Garystafford real + fake...")
    all_rows.extend(collect_garystafford(out, excluded_fake_set))

    df = pd.DataFrame(all_rows)
    meta = out / "metadata.csv"
    df.to_csv(meta, index=False)
    print(f"\nSaved metadata.csv → {meta} ({len(df)} rows)")
    print_summary(df, out)


if __name__ == "__main__":
    main()
