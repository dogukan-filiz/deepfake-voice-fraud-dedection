"""
scripts/normalize_call_channel.py

Deterministic call-channel normalization pipeline.
Transforms any input audio into a consistent telephony representation
suitable for bank call-center deepfake detection.

Pipeline (in order):
  1. Decode → mono, 16 kHz
  2. RMS normalization (target: -20 dBFS)
  3. Energy-based leading/trailing silence trim
  4. Telephony bandpass filter (300–3400 Hz narrow or 300–7600 Hz wide)
  5. Codec channel simulation (G.711 µ-law or Opus 12 kbps)
  6. Upsample back to 16 kHz mono WAV

Fully deterministic — no random operations.
Safe for both training data generation and production inference.

Usage:
  # Single file
  python3 scripts/normalize_call_channel.py \
      --input audio.wav \
      --output normalized.wav \
      --mode narrowband_g711

  # Batch directory
  python3 scripts/normalize_call_channel.py \
      --input_dir test_audio/real \
      --output_dir /tmp/call_norm/real \
      --mode narrowband_g711 \
      --report

Modes:
  narrowband_g711   : 300-3400 Hz BP + G.711 µ-law @ 8kHz → upsample 16kHz
  wideband_opus     : 300-7600 Hz BP + Opus 12kbps @ 16kHz encode-decode
  narrowband_clean  : 300-3400 Hz BP only, no codec sim
  bypass            : mono + RMS norm + silence trim only (no bandpass/codec)
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SR  = 16000
TARGET_RMS_DB  = -20.0   # dBFS RMS target
SILENCE_THRESH_DB = -40.0  # energy below this = silence
SILENCE_MIN_LEN    = 0.1   # seconds of silence at edges to trim
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


# ---------------------------------------------------------------------------
# Step 1: Decode → mono 16 kHz numpy array
# ---------------------------------------------------------------------------

def load_mono_16k(path: Path) -> np.ndarray:
    """Load audio file → mono float32 @ 16 kHz."""
    import librosa
    audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2: RMS normalization
# ---------------------------------------------------------------------------

def rms_normalize(audio: np.ndarray, target_db: float = TARGET_RMS_DB) -> np.ndarray:
    """Normalize RMS to target_db dBFS. Clip-safe."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms
    audio = audio * gain
    # Soft peak clip to ±1.0 without hard distortion
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak * 0.99
    return audio


# ---------------------------------------------------------------------------
# Step 3: Energy-based silence trim
# ---------------------------------------------------------------------------

def trim_silence(audio: np.ndarray, sr: int = TARGET_SR,
                 thresh_db: float = SILENCE_THRESH_DB,
                 min_len: float = SILENCE_MIN_LEN) -> np.ndarray:
    """Remove leading/trailing silence below thresh_db energy."""
    frame_len = int(sr * 0.02)   # 20 ms frames
    hop_len   = frame_len // 2
    thresh_rms = 10 ** (thresh_db / 20.0)

    # Compute per-frame RMS
    n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)
    frame_rms = np.array([
        np.sqrt(np.mean(audio[i*hop_len : i*hop_len+frame_len]**2))
        for i in range(n_frames)
    ])

    voiced = frame_rms > thresh_rms
    if not voiced.any():
        return audio   # fully silent → return as-is

    first = int(np.argmax(voiced))
    last  = int(len(voiced) - np.argmax(voiced[::-1]) - 1)

    min_frames = int(min_len * sr / hop_len)
    first = max(0, first - min_frames)
    last  = min(len(voiced) - 1, last + min_frames)

    start_sample = first * hop_len
    end_sample   = min(len(audio), (last + 1) * hop_len + frame_len)
    return audio[start_sample:end_sample]


# ---------------------------------------------------------------------------
# Step 4: Bandpass filter
# ---------------------------------------------------------------------------

def bandpass_filter(audio: np.ndarray, sr: int = TARGET_SR,
                    lo_hz: float = 300.0, hi_hz: float = 3400.0,
                    order: int = 5) -> np.ndarray:
    """Butterworth bandpass filter."""
    nyq = sr / 2.0
    lo  = max(lo_hz / nyq, 0.001)
    hi  = min(hi_hz / nyq, 0.999)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5a: G.711 µ-law codec simulation via ffmpeg
# ---------------------------------------------------------------------------

def g711_codec_sim(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Encode to G.711 µ-law @ 8 kHz, decode back to 16 kHz.
    Uses ffmpeg subprocess — deterministic.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_wav  = tmp / "in.wav"
        g711_wav = tmp / "g711.wav"
        out_wav  = tmp / "out.wav"

        # Write input
        sf.write(str(in_wav), audio, sr, subtype="PCM_16")

        # Encode: → 8 kHz µ-law
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(in_wav),
            "-ar", "8000", "-ac", "1",
            "-acodec", "pcm_mulaw", str(g711_wav)
        ], capture_output=True)
        if r.returncode != 0:
            return audio   # fallback: return input unchanged

        # Decode: → 16 kHz PCM
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(g711_wav),
            "-ar", str(TARGET_SR), "-ac", "1",
            "-acodec", "pcm_s16le", str(out_wav)
        ], capture_output=True)
        if r.returncode != 0:
            return audio

        result, _ = sf.read(str(out_wav), dtype="float32")
        return result


# ---------------------------------------------------------------------------
# Step 5b: Opus codec simulation via ffmpeg (12 kbps)
# ---------------------------------------------------------------------------

def opus_codec_sim(audio: np.ndarray, sr: int = TARGET_SR,
                   bitrate: str = "12k") -> np.ndarray:
    """
    Encode to Opus @ bitrate, decode back to 16 kHz PCM.
    Uses ffmpeg — deterministic (fixed bitrate, no VBR variance).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_wav  = tmp / "in.wav"
        opus_f  = tmp / "out.opus"
        out_wav = tmp / "out.wav"

        sf.write(str(in_wav), audio, sr, subtype="PCM_16")

        # Encode to Opus OGG
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(in_wav),
            "-c:a", "libopus",
            "-b:a", bitrate,
            "-ar", "16000", "-ac", "1",
            "-vbr", "off",        # constant bitrate → deterministic
            str(opus_f)
        ], capture_output=True)
        if r.returncode != 0:
            return audio

        # Decode to 16 kHz PCM
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(opus_f),
            "-ar", str(TARGET_SR), "-ac", "1",
            "-acodec", "pcm_s16le", str(out_wav)
        ], capture_output=True)
        if r.returncode != 0:
            return audio

        result, _ = sf.read(str(out_wav), dtype="float32")
        return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

MODES = {
    "narrowband_g711": {
        "bandpass": (300.0, 3400.0),
        "codec": "g711",
        "desc": "300-3400 Hz BP + G.711 µ-law @ 8kHz→16kHz (classic PSTN)",
    },
    "wideband_opus": {
        "bandpass": (300.0, 7600.0),
        "codec": "opus_12k",
        "desc": "300-7600 Hz BP + Opus 12 kbps @ 16kHz (WhatsApp-like VoIP)",
    },
    "narrowband_clean": {
        "bandpass": (300.0, 3400.0),
        "codec": None,
        "desc": "300-3400 Hz BP only, no codec simulation",
    },
    "bypass": {
        "bandpass": None,
        "codec": None,
        "desc": "Mono + RMS norm + silence trim only",
    },
}


def process_file(input_path: Path, mode: str = "narrowband_g711") -> np.ndarray:
    """Apply full call-channel normalization pipeline. Returns float32 array @ 16kHz."""
    cfg = MODES[mode]

    # 1. Decode
    audio = load_mono_16k(input_path)

    # 2. RMS normalize
    audio = rms_normalize(audio)

    # 3. Silence trim
    audio = trim_silence(audio)

    # 4. Bandpass
    if cfg["bandpass"]:
        lo, hi = cfg["bandpass"]
        audio = bandpass_filter(audio, lo_hz=lo, hi_hz=hi)

    # 5. Codec sim
    if cfg["codec"] == "g711":
        audio = g711_codec_sim(audio)
    elif cfg["codec"] == "opus_12k":
        audio = opus_codec_sim(audio, bitrate="12k")

    # Re-normalize after codec (codec can shift levels)
    audio = rms_normalize(audio)

    return audio


def compute_stats(audio: np.ndarray, sr: int = TARGET_SR) -> dict:
    """Compute audio quality stats."""
    if len(audio) == 0:
        return {"duration_sec": 0.0, "rms_db": None, "silence_ratio": 1.0, "peak_db": None}
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(rms + 1e-9)
    peak = np.abs(audio).max()
    peak_db = 20 * np.log10(peak + 1e-9)
    silence_ratio = float((np.abs(audio) < 1e-4).mean())
    return {
        "duration_sec": round(len(audio) / sr, 3),
        "rms_db":       round(float(rms_db), 2),
        "peak_db":      round(float(peak_db), 2),
        "silence_ratio": round(silence_ratio, 4),
        "n_samples":    len(audio),
    }


def spectral_stats(audio: np.ndarray, sr: int = TARGET_SR) -> dict:
    """Estimate energy in telephony bands."""
    from scipy.fft import rfft, rfftfreq
    N = min(len(audio), 8192)
    chunk = audio[:N]
    mag = np.abs(rfft(chunk * np.hanning(N)))
    freqs = rfftfreq(N, 1.0 / sr)

    def band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(mag[mask].mean()) if mask.any() else 0.0

    total = band_energy(0, sr / 2) + 1e-9
    return {
        "e_sub300":    round(band_energy(0,    300)   / total, 4),
        "e_300_3400":  round(band_energy(300,  3400)  / total, 4),
        "e_3400_7600": round(band_energy(3400, 7600)  / total, 4),
        "e_7600_plus": round(band_energy(7600, sr/2)  / total, 4),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single input audio file")
    group.add_argument("--input_dir", type=Path, help="Input directory (batch)")

    parser.add_argument("--output", type=Path, help="Single output WAV path")
    parser.add_argument("--output_dir", type=Path, help="Output directory (batch)")
    parser.add_argument("--mode", choices=list(MODES.keys()), default="narrowband_g711")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Max files to process per directory")
    parser.add_argument("--report", action="store_true",
                        help="Print detailed before/after stats")
    parser.add_argument("--report_json", type=Path,
                        help="Save report to JSON file")
    args = parser.parse_args()

    cfg = MODES[args.mode]
    print(f"Mode: {args.mode} — {cfg['desc']}")

    # Collect input files
    if args.input:
        files = [(args.input, args.output)]
    else:
        in_dir = args.input_dir
        out_dir = args.output_dir or Path("/tmp/call_norm_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_files = sorted([
            f for f in in_dir.rglob("*") if f.suffix.lower() in AUDIO_EXTS
        ])
        if args.max_files:
            audio_files = audio_files[:args.max_files]
        files = [(f, out_dir / (f.stem + "_normalized.wav")) for f in audio_files]

    report_rows = []

    for in_path, out_path in files:
        if not in_path.is_file():
            print(f"SKIP (not found): {in_path}")
            continue

        print(f"\n{'─'*60}")
        print(f"Input : {in_path}")

        # Before stats
        raw, raw_sr = sf.read(str(in_path), dtype="float32", always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        before = compute_stats(raw, raw_sr)
        before_spec = spectral_stats(raw, raw_sr)
        before["sample_rate"] = raw_sr

        # Process
        try:
            audio_out = process_file(in_path, mode=args.mode)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Write output
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio_out, TARGET_SR, subtype="PCM_16")

        # After stats
        after = compute_stats(audio_out, TARGET_SR)
        after_spec = spectral_stats(audio_out, TARGET_SR)
        after["sample_rate"] = TARGET_SR

        if args.report:
            print(f"Output: {out_path}")
            print(f"\n  BEFORE:")
            print(f"    SR={before['sample_rate']} Hz  dur={before['duration_sec']}s  rms={before['rms_db']} dBFS  peak={before['peak_db']} dBFS  silence={before['silence_ratio']:.3f}")
            print(f"    Spectrum — sub300={before_spec['e_sub300']:.3f}  300-3400={before_spec['e_300_3400']:.3f}  3400-7600={before_spec['e_3400_7600']:.3f}  7600+={before_spec['e_7600_plus']:.3f}")
            print(f"  AFTER ({args.mode}):")
            print(f"    SR={after['sample_rate']} Hz  dur={after['duration_sec']}s  rms={after['rms_db']} dBFS  peak={after['peak_db']} dBFS  silence={after['silence_ratio']:.3f}")
            print(f"    Spectrum — sub300={after_spec['e_sub300']:.3f}  300-3400={after_spec['e_300_3400']:.3f}  3400-7600={after_spec['e_3400_7600']:.3f}  7600+={after_spec['e_7600_plus']:.3f}")
            # Bandpass confirmation
            if cfg["bandpass"]:
                lo, hi = cfg["bandpass"]
                sub_energy = after_spec["e_sub300"]
                above_energy = after_spec.get("e_7600_plus", 0) if hi > 7000 else after_spec["e_3400_7600"] + after_spec["e_7600_plus"]
                bp_ok = sub_energy < 0.05 and above_energy < 0.05
                print(f"    Bandpass check ({lo:.0f}-{hi:.0f} Hz): {'PASS ✓' if bp_ok else 'WARN (energy outside band)'}")

        row = {
            "file": str(in_path),
            "output": str(out_path) if out_path else None,
            "mode": args.mode,
            "before": {**before, "spectrum": before_spec},
            "after":  {**after,  "spectrum": after_spec},
        }
        report_rows.append(row)

    print(f"\n{'='*60}")
    print(f"Processed: {len(report_rows)} files")
    print(f"Mode: {args.mode}")

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report_rows, indent=2))
        print(f"Report: {args.report_json}")

    return report_rows


if __name__ == "__main__":
    main()
