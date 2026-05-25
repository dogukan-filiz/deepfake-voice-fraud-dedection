import io
import json
import uuid
import urllib.request

import numpy as np
import soundfile as sf

API = "http://127.0.0.1:8020/analyze"


def wav_bytes(duration_sec: float, *, leading_silence_sec: float = 0.0, trailing_silence_sec: float = 0.0) -> bytes:
    sr = 16000
    n = int(duration_sec * sr)
    t = np.arange(n, dtype=np.float32) / sr
    tone = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    ls = np.zeros(int(leading_silence_sec * sr), dtype=np.float32)
    ts = np.zeros(int(trailing_silence_sec * sr), dtype=np.float32)
    x = np.concatenate([ls, tone, ts])

    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def post_analyze(wav: bytes) -> dict:
    boundary = "----Boundary" + uuid.uuid4().hex
    crlf = "\r\n"

    parts: list[bytes] = []
    parts.append(f"--{boundary}{crlf}".encode())
    parts.append(
        (
            f'Content-Disposition: form-data; name="file"; filename="test.wav"{crlf}'
            f"Content-Type: audio/wav{crlf}{crlf}"
        ).encode()
    )
    parts.append(wav)
    parts.append(crlf.encode())
    parts.append(f"--{boundary}--{crlf}".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_case(name: str, wav: bytes) -> None:
    out = post_analyze(wav)
    print(
        f"{name:28s} | score={out['authenticity_score']:.4f} p_real={out['p_real']:.4f} p_fake={out['p_fake']:.4f}"
    )


if __name__ == "__main__":
    print("POST", API)
    run_case("short_2s", wav_bytes(2.0))
    run_case("long_20s", wav_bytes(20.0))
    run_case("long_20s_lead5s_sil", wav_bytes(20.0, leading_silence_sec=5.0))
    run_case("long_20s_trail5s_sil", wav_bytes(20.0, trailing_silence_sec=5.0))
