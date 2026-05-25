"""Pick 50 real + 50 fake samples from ASVspoof 2019 LA dev set.

Usage: python scripts/pick_test_samples.py

Reads the protocol file to find bonafide/spoof labels,
copies 50 of each into test_audio/real/ and test_audio/fake/.
Converts FLAC to WAV (16kHz mono) for direct use with the API.
"""
import os
import shutil
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "test_audio" / "raw" / "LA" / "LA"
PROTO = RAW / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
FLAC_DIR = RAW / "ASVspoof2019_LA_dev" / "flac"
OUT_REAL = ROOT / "test_audio" / "real"
OUT_FAKE = ROOT / "test_audio" / "fake"

N = 50


def main():
    if not PROTO.exists():
        print(f"Protocol file not found: {PROTO}")
        print("Make sure dataset is downloaded to test_audio/raw/")
        return

    bonafide = []
    spoof = []
    with open(PROTO, "r") as f:
        for line in f:
            parts = line.strip().split()
            # format: speaker_id file_id - attack_type label
            file_id = parts[1]
            label = parts[-1]  # bonafide or spoof
            flac_path = FLAC_DIR / f"{file_id}.flac"
            if not flac_path.exists():
                continue
            if label == "bonafide":
                bonafide.append(flac_path)
            else:
                spoof.append(flac_path)

    print(f"Found {len(bonafide)} bonafide, {len(spoof)} spoof")

    random.seed(42)
    pick_real = random.sample(bonafide, min(N, len(bonafide)))
    pick_fake = random.sample(spoof, min(N, len(spoof)))

    OUT_REAL.mkdir(parents=True, exist_ok=True)
    OUT_FAKE.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        import numpy as np

        def convert(src, dst):
            data, sr = sf.read(str(src))
            if data.ndim == 2:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            sf.write(str(dst), data, sr, subtype="PCM_16")

        for p in pick_real:
            convert(p, OUT_REAL / f"{p.stem}.wav")
        for p in pick_fake:
            convert(p, OUT_FAKE / f"{p.stem}.wav")
        print(f"Converted {len(pick_real)} real + {len(pick_fake)} fake WAVs")

    except ImportError:
        for p in pick_real:
            shutil.copy2(p, OUT_REAL / p.name)
        for p in pick_fake:
            shutil.copy2(p, OUT_FAKE / p.name)
        print(f"Copied {len(pick_real)} real + {len(pick_fake)} fake FLACs (no soundfile for WAV conversion)")

    print(f"Output: {OUT_REAL} and {OUT_FAKE}")


if __name__ == "__main__":
    main()
