#!/usr/bin/env python3
"""
TD §3 functional tests for non-WAV formats (FFmpeg fallback path) and audio
validation rejections (silence / too-short).

Backend must be running on http://127.0.0.1:8010 with FFmpeg available.

Run:
  python tests/test_format_and_validation.py
"""
import io
import json
import math
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

import requests

BACKEND = "http://127.0.0.1:8010"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / "format_and_validation_results.json"


def write_wav(path: Path, samples: bytes, sr: int = 16000, sw: int = 2, ch: int = 1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(samples)


def sine_pcm16(seconds: float, freq: float = 440.0, sr: int = 16000, amp: float = 0.3) -> bytes:
    n = int(seconds * sr)
    buf = bytearray()
    for i in range(n):
        v = int(amp * 32767 * math.sin(2 * math.pi * freq * i / sr))
        buf.extend(struct.pack("<h", v))
    return bytes(buf)


def silent_pcm16(seconds: float, sr: int = 16000) -> bytes:
    return b"\x00\x00" * int(seconds * sr)


def ffmpeg_convert(src_wav: Path, dst: Path) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src_wav), str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[:300]}")


def post(path: Path, mime: str):
    with open(path, "rb") as f:
        files = {"file": (path.name, f, mime)}
        r = requests.post(f"{BACKEND}/analyze", files=files, timeout=60)
    return r


def case_ok(resp, name: str) -> dict:
    return {
        "case": name,
        "status_code": resp.status_code,
        "passed": resp.status_code == 200,
        "body": resp.json() if resp.status_code == 200 else resp.text[:200],
    }


def case_reject(resp, name: str) -> dict:
    return {
        "case": name,
        "status_code": resp.status_code,
        "passed": 400 <= resp.status_code < 500,
        "body": resp.text[:200],
    }


def main():
    results = []

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        wav_3s = td / "tone_3s.wav"
        write_wav(wav_3s, sine_pcm16(3.0))

        silent_3s = td / "silent_3s.wav"
        write_wav(silent_3s, silent_pcm16(3.0))

        short_wav = td / "short_1s.wav"
        write_wav(short_wav, sine_pcm16(1.0))

        for ext, mime in [("ogg", "audio/ogg"), ("webm", "audio/webm"), ("m4a", "audio/mp4")]:
            converted = td / f"tone_3s.{ext}"
            try:
                ffmpeg_convert(wav_3s, converted)
            except Exception as e:
                results.append({"case": f"format_{ext}", "passed": False, "error": str(e)})
                continue
            r = post(converted, mime)
            results.append(case_ok(r, f"format_{ext}"))

        r = post(silent_3s, "audio/wav")
        results.append(case_reject(r, "silence_reject"))

        r = post(short_wav, "audio/wav")
        results.append(case_reject(r, "too_short_reject"))

    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    summary = {
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "cases": results,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
