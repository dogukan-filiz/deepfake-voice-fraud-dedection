"""
Prepare balanced anti-spoofing dataset (500 real + 500 fake).
Uses direct parquet access — no datasets streaming, no torchcodec required.

Sources:
  Real:  LJSpeech original     (ajaykarthick/wavefake-audio label='R', lang=en)
       + Turkish speech         (shunyalabs/turkish-speech-dataset, lang=tr)
  Fake:  WaveFake vocoders      (ajaykarthick/wavefake-audio WF1-WF7, lang=en)

Split (system + utterance leakage prevention):
  Train: utterances partitions 0-1, fake systems WF1-WF5 (seen)
  Val:   utterances partition   2,  fake system  WF6     (unseen)
  Test:  utterances partition   3,  fake system  WF7     (unseen)

Targets:
  Real:  200 LJSpeech-EN + 200 Turkish = 400 train
         25  LJSpeech-EN +  25 Turkish =  50 val
         25  LJSpeech-EN +  25 Turkish =  50 test
  Fake:  WF1..WF5: 80 each = 400 train
         WF6:       50 val
         WF7:       50 test
  Total: 500 real + 500 fake
"""

import argparse
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import librosa
from huggingface_hub import hf_hub_download, login

HF_TOKEN = None  # Set via --hf_token arg

SYSTEM_LABEL_TO_NAME = {
    "WF1": "melgan",
    "WF2": "full_band_melgan",
    "WF3": "multi_band_melgan",
    "WF4": "hifigan",
    "WF5": "waveglow",
    "WF6": "parallel_wavegan",
    "WF7": "conformer_fastspeech2",
}

TRAIN_FAKE_SYSTEMS = {"WF1", "WF2", "WF3", "WF4", "WF5"}
VAL_FAKE_SYSTEM    = "WF6"
TEST_FAKE_SYSTEM   = "WF7"


def save_wav_bytes(raw_bytes: bytes, out_path: Path, target_sr: int = 16000):
    buf = io.BytesIO(raw_bytes)
    audio, sr = librosa.load(buf, sr=target_sr, mono=True)
    sf.write(str(out_path), audio.astype(np.float32), target_sr, subtype="PCM_16")


def get_parquet(repo_id: str, filename: str) -> str:
    return hf_hub_download(repo_id, filename, repo_type="dataset", token=HF_TOKEN)


def collect_wavefake_lj(output_root: Path,
                        train_per_system: int = 80,
                        lj_real_train: int = 200,
                        val_fake: int = 50, lj_real_val: int = 25,
                        test_fake: int = 50, lj_real_test: int = 25):
    """
    Read ajaykarthick/wavefake-audio parquets.
    partition 0-1 → train, partition 2 → val, partition 3 → test.
    Returns list of metadata rows.
    """
    real_dir = output_root / "real"
    fake_dir = output_root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    configs = [
        # (partition_idx, split, lj_real_target, fake_system(s), fake_target_per_sys_or_total)
        (0, "train"),
        (1, "train"),
        (2, "val"),
        (3, "test"),
    ]

    lj_counts  = {"train": 0, "val": 0, "test": 0}
    fake_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    lj_real_targets = {"train": lj_real_train, "val": lj_real_val, "test": lj_real_test}
    fake_targets = {
        **{s: {"train": train_per_system} for s in TRAIN_FAKE_SYSTEMS},
        VAL_FAKE_SYSTEM:  {"val":  val_fake},
        TEST_FAKE_SYSTEM: {"test": test_fake},
    }

    for part_idx, split in configs:
        fname = f"data/partition{part_idx}-00000-of-00001.parquet"
        print(f"[wavefake] Downloading partition {part_idx} ({split})...")
        path = get_parquet("ajaykarthick/wavefake-audio", fname)
        tbl  = pq.read_table(path)
        df   = tbl.to_pandas()

        # Sort by audio_id to get deterministic utterance ordering
        df = df.sort_values(["audio_id", "real_or_fake"]).reset_index(drop=True)

        for _, row in df.iterrows():
            label    = row["real_or_fake"]
            audio_id = row["audio_id"]
            audio    = row["audio"]

            if label == "R":
                target = lj_real_targets[split]
                if lj_counts[split] >= target:
                    continue
                idx   = lj_counts[split]
                fname_out = f"lj_{split}_{part_idx}_{idx:04d}.wav"
                out   = real_dir / fname_out
                if not out.exists():
                    save_wav_bytes(audio["bytes"], out)
                rows.append({
                    "path": str(out), "label": "real", "split": split,
                    "language": "en", "source": "ljspeech",
                    "generation_type": "natural", "system": "ljspeech_original",
                    "utterance_id": audio_id,
                })
                lj_counts[split] += 1

            elif label in TRAIN_FAKE_SYSTEMS and split == "train":
                target = fake_targets[label]["train"]
                if fake_counts[label]["train"] >= target:
                    continue
                sys_name  = SYSTEM_LABEL_TO_NAME[label]
                idx       = fake_counts[label]["train"]
                fname_out = f"wf_{sys_name}_{part_idx}_{idx:04d}.wav"
                out       = fake_dir / fname_out
                if not out.exists():
                    save_wav_bytes(audio["bytes"], out)
                rows.append({
                    "path": str(out), "label": "fake", "split": "train",
                    "language": "en", "source": "wavefake",
                    "generation_type": "synthetic_vocoder", "system": sys_name,
                    "utterance_id": audio_id,
                })
                fake_counts[label]["train"] += 1

            elif label == VAL_FAKE_SYSTEM and split == "val":
                target = fake_targets[VAL_FAKE_SYSTEM]["val"]
                if fake_counts[VAL_FAKE_SYSTEM]["val"] >= target:
                    continue
                sys_name  = SYSTEM_LABEL_TO_NAME[label]
                idx       = fake_counts[VAL_FAKE_SYSTEM]["val"]
                fname_out = f"wf_{sys_name}_{part_idx}_{idx:04d}.wav"
                out       = fake_dir / fname_out
                if not out.exists():
                    save_wav_bytes(audio["bytes"], out)
                rows.append({
                    "path": str(out), "label": "fake", "split": "val",
                    "language": "en", "source": "wavefake",
                    "generation_type": "synthetic_vocoder", "system": sys_name,
                    "utterance_id": audio_id,
                })
                fake_counts[VAL_FAKE_SYSTEM]["val"] += 1

            elif label == TEST_FAKE_SYSTEM and split == "test":
                target = fake_targets[TEST_FAKE_SYSTEM]["test"]
                if fake_counts[TEST_FAKE_SYSTEM]["test"] >= target:
                    continue
                sys_name  = SYSTEM_LABEL_TO_NAME[label]
                idx       = fake_counts[TEST_FAKE_SYSTEM]["test"]
                fname_out = f"wf_{sys_name}_{part_idx}_{idx:04d}.wav"
                out       = fake_dir / fname_out
                if not out.exists():
                    save_wav_bytes(audio["bytes"], out)
                rows.append({
                    "path": str(out), "label": "fake", "split": "test",
                    "language": "en", "source": "wavefake",
                    "generation_type": "synthetic_vocoder", "system": sys_name,
                    "utterance_id": audio_id,
                })
                fake_counts[TEST_FAKE_SYSTEM]["test"] += 1

        print(f"  partition {part_idx} done. LJ real: {dict(lj_counts)}")
        for sys_label in sorted(TRAIN_FAKE_SYSTEMS):
            c = fake_counts[sys_label]["train"]
            if c > 0:
                print(f"    WF train {sys_label}: {c}")
        if fake_counts[VAL_FAKE_SYSTEM]["val"]:
            print(f"    WF6 val: {fake_counts[VAL_FAKE_SYSTEM]['val']}")
        if fake_counts[TEST_FAKE_SYSTEM]["test"]:
            print(f"    WF7 test: {fake_counts[TEST_FAKE_SYSTEM]['test']}")

    return rows


def collect_turkish_real(output_root: Path,
                         n_train: int = 200, n_val: int = 25, n_test: int = 25):
    """
    Read shunyalabs/turkish-speech-dataset parquets.
    Returns list of metadata rows.
    """
    real_dir = output_root / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    split_files = {
        "train": ("data/train-00000-of-00004.parquet", n_train),
        "val":   ("data/validation-00000-of-00001.parquet", n_val),
        "test":  ("data/test-00000-of-00002.parquet", n_test),
    }

    for split, (parquet_file, target) in split_files.items():
        print(f"[turkish] Downloading {parquet_file} ({split}, target={target})...")
        path = get_parquet("shunyalabs/turkish-speech-dataset", parquet_file)
        tbl  = pq.read_table(path)
        df   = tbl.to_pandas()

        collected = 0
        for idx, row in df.iterrows():
            if collected >= target:
                break
            audio = row["audio"]
            raw   = audio["bytes"] if isinstance(audio, dict) else audio

            fname_out = f"tr_{split}_{collected:04d}.wav"
            out       = real_dir / fname_out
            if not out.exists():
                # Bytes are already WAV at 16 kHz per schema metadata
                buf = io.BytesIO(raw)
                try:
                    audio_arr, sr = sf.read(buf, dtype="float32")
                    if len(audio_arr.shape) > 1:
                        audio_arr = audio_arr[:, 0]
                    if sr != 16000:
                        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
                    sf.write(str(out), audio_arr, 16000, subtype="PCM_16")
                except Exception as e:
                    # Fallback: write raw bytes directly if soundfile can parse
                    try:
                        buf.seek(0)
                        audio_arr, sr = librosa.load(buf, sr=16000, mono=True)
                        sf.write(str(out), audio_arr.astype(np.float32), 16000, subtype="PCM_16")
                    except Exception as e2:
                        print(f"  [WARN] {fname_out} skip: {e2}")
                        continue

            rows.append({
                "path": str(out), "label": "real", "split": split,
                "language": "tr", "source": "shunyalabs_tr",
                "generation_type": "natural", "system": "human_speech_tr",
                "utterance_id": f"tr_{split}_{idx}",
            })
            collected += 1

        print(f"  {split}: {collected} Turkish samples collected")

    return rows


def print_summary(df: pd.DataFrame, output_root: Path):
    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)

    print("\n-- Total real/fake --")
    print(df["label"].value_counts().to_string())

    print("\n-- Train/val/test split --")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())

    print("\n-- Source distribution --")
    print(df.groupby(["source", "split", "label"]).size().to_string())

    print("\n-- Language distribution --")
    print(df.groupby(["language", "label", "split"]).size().to_string())

    print("\n-- Generation type --")
    print(df.groupby(["generation_type", "label"]).size().to_string())

    print("\n-- System distribution (fake only) --")
    fake = df[df["label"] == "fake"]
    print(fake.groupby(["system", "split"]).size().to_string())

    print("\n-- Sample paths (5 real + 5 fake) --")
    for label in ["real", "fake"]:
        for _, r in df[df["label"] == label].head(5).iterrows():
            print(f"  [{label:<4}][{r['split']:<5}][{r['source']:<18}][lang={r['language']}] {Path(r['path']).name}")

    # Leakage checks
    print("\n-- Utterance leakage check --")
    train_u = set(df[df["split"] == "train"]["utterance_id"])
    val_u   = set(df[df["split"] == "val"]["utterance_id"])
    test_u  = set(df[df["split"] == "test"]["utterance_id"])
    tv = train_u & val_u
    tt = train_u & test_u
    vt = val_u   & test_u
    print(f"  Train∩Val:  {len(tv)} {'⚠ LEAKAGE' if tv else '✓ clean'}")
    print(f"  Train∩Test: {len(tt)} {'⚠ LEAKAGE' if tt else '✓ clean'}")
    print(f"  Val∩Test:   {len(vt)} {'⚠ LEAKAGE' if vt else '✓ clean'}")

    print("\n-- Fake system leakage check --")
    fake_df = df[df["label"] == "fake"]
    tr_sys  = set(fake_df[fake_df["split"] == "train"]["system"])
    va_sys  = set(fake_df[fake_df["split"] == "val"]["system"])
    te_sys  = set(fake_df[fake_df["split"] == "test"]["system"])
    print(f"  Train systems: {sorted(tr_sys)}")
    print(f"  Val   systems: {sorted(va_sys)} {'⚠ SEEN IN TRAIN' if va_sys & tr_sys else '✓ unseen'}")
    print(f"  Test  systems: {sorted(te_sys)} {'⚠ SEEN IN TRAIN' if te_sys & tr_sys else '✓ unseen'}")

    # Turkish warning if absent
    tr_real = df[(df["language"] == "tr") & (df["label"] == "real")]
    if len(tr_real) == 0:
        print("\n⚠ WARNING: No Turkish real data collected.")
        print("  Real recall improvement for Turkish audio NOT guaranteed.")
        print("  Reason: shunyalabs download failed or skipped.")
    else:
        print(f"\n✓ Turkish real samples: {len(tr_real)} ({tr_real['split'].value_counts().to_dict()})")

    # Disk usage
    total_bytes = sum(Path(p).stat().st_size for p in df["path"] if Path(p).exists())
    print(f"\nTotal audio on disk: {total_bytes / 1e6:.1f} MB")
    print(f"Output: {output_root}/")
    print(f"metadata.csv: {len(df)} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--hf_token",         default=None)
    parser.add_argument("--train_per_system", type=int, default=80)
    parser.add_argument("--val_fake",         type=int, default=50)
    parser.add_argument("--test_fake",        type=int, default=50)
    parser.add_argument("--skip_turkish",     action="store_true")
    args = parser.parse_args()

    global HF_TOKEN
    HF_TOKEN = args.hf_token

    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("[auth] HuggingFace login OK")
    else:
        print("[auth] No HF token provided — using public datasets only")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    print("\n[1/2] WaveFake + LJSpeech (ajaykarthick/wavefake-audio)...")
    wf_rows = collect_wavefake_lj(
        output_root,
        train_per_system=args.train_per_system,
        val_fake=args.val_fake,
        test_fake=args.test_fake,
    )
    all_rows.extend(wf_rows)
    print(f"  → {len(wf_rows)} rows (LJSpeech real + WaveFake fake)")

    if not args.skip_turkish:
        print("\n[2/2] Turkish real speech (shunyalabs/turkish-speech-dataset)...")
        try:
            tr_rows = collect_turkish_real(output_root)
            all_rows.extend(tr_rows)
            print(f"  → {len(tr_rows)} rows (Turkish real)")
        except Exception as e:
            print(f"  ⚠ Turkish collection failed: {e}")
            print("  Proceeding without Turkish data.")
    else:
        print("\n[2/2] Skipping Turkish (--skip_turkish).")
        print("  ⚠ No Turkish real data. Turkish recall improvement NOT guaranteed.")

    df = pd.DataFrame(all_rows)
    meta_path = output_root / "metadata.csv"
    df.to_csv(meta_path, index=False)

    print_summary(df, output_root)


if __name__ == "__main__":
    main()
