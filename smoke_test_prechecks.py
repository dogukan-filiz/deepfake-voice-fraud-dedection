import io
import json
import uuid
import urllib.error
import urllib.request

import numpy as np
import soundfile as sf

API = "http://127.0.0.1:8020/analyze"


def make_wav(duration_sec: float, *, silence: bool = False) -> bytes:
    sr = 16000
    n = int(sr * duration_sec)
    if silence:
        x = np.zeros(n, dtype=np.float32)
    else:
        t = np.arange(n, dtype=np.float32) / sr
        x = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def post_analyze(wav: bytes) -> tuple[int, str]:
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

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def run_case(name: str, wav: bytes) -> None:
    status, body = post_analyze(wav)
    detail = None
    try:
        j = json.loads(body)
        detail = j.get("detail")
    except Exception:
        detail = body[:200]
    print(f"{name:22s} -> HTTP {status} | detail={detail}")


if __name__ == "__main__":
    print("POST", API)
    run_case("too_short_1s", make_wav(1.0))
    run_case("silent_3s", make_wav(3.0, silence=True))
    run_case("ok_3s", make_wav(3.0, silence=False))
