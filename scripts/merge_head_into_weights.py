"""
Merge a fine-tuned AASIST head checkpoint into a full SSL+AASIST weights file.

Base weights.pth layout (DataParallel-wrapped):
  module.ssl_model.*   → 429 backbone keys — KEPT unchanged
  module.<head>.*      → 245 head keys    — REPLACED from checkpoint

Head checkpoint layout (no module. prefix):
  pos_S, master1, LL.weight, ...  → prepend "module." before inserting
"""

import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",   required=True, help="Full weights.pth (module.ssl_model.* + module.<head>.*)")
    parser.add_argument("--head",   required=True, help="best_head.pth (head keys, no module. prefix)")
    parser.add_argument("--output", required=True, help="Output merged .pth path")
    args = parser.parse_args()

    base_path = Path(args.base)
    head_path = Path(args.head)
    out_path  = Path(args.output)

    print(f"[merge] Base:   {base_path}  ({base_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[merge] Head:   {head_path}  ({head_path.stat().st_size / 1e6:.1f} MB)")

    print("[merge] Loading base weights...")
    base_state = torch.load(base_path, map_location="cpu")

    print("[merge] Loading head checkpoint...")
    head_ckpt  = torch.load(head_path, map_location="cpu")
    head_state = head_ckpt.get("model_state_dict", head_ckpt)

    base_keys      = list(base_state.keys())
    backbone_keys  = {k for k in base_keys if k.startswith("module.ssl_model.")}
    base_head_keys = {k for k in base_keys if not k.startswith("module.ssl_model.")}

    print(f"\n[merge] Base: {len(base_keys)} total")
    print(f"         backbone (module.ssl_model.*): {len(backbone_keys)}")
    print(f"         head    (module.<other>.*):    {len(base_head_keys)}")
    print(f"[merge] Head ckpt: {len(head_state)} keys (no module. prefix)")

    # Validate mapping: strip "module." from base head keys, compare with head_state keys
    base_head_stripped = {k.replace("module.", "", 1) for k in base_head_keys}
    missing = set(head_state.keys()) - base_head_stripped
    extra   = base_head_stripped - set(head_state.keys())
    if missing:
        print(f"[WARNING] Head ckpt keys with no base counterpart: {missing}")
    if extra:
        print(f"[WARNING] Base head keys not covered by checkpoint: {extra}")
    if not missing and not extra:
        print(f"[merge] Key mapping: perfect 1:1 match ({len(head_state)} keys)")

    # Build merged state dict
    merged = {}
    backbone_count = 0
    head_count = 0

    for k, v in base_state.items():
        if k.startswith("module.ssl_model."):
            merged[k] = v
            backbone_count += 1
        # skip old head keys

    for k, v in head_state.items():
        merged_key = "module." + k
        merged[merged_key] = v
        head_count += 1

    print(f"\n[merge] Merged: {backbone_count} backbone + {head_count} head = {len(merged)} total")
    assert len(merged) == len(base_keys), \
        f"Key count mismatch: merged={len(merged)} != base={len(base_keys)}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)
    out_size = out_path.stat().st_size / 1e6
    print(f"[merge] Saved  → {out_path}  ({out_size:.1f} MB)")
    print(f"[merge] SHA256:  {sha256(out_path)}")

    epoch = head_ckpt.get("epoch", "unknown")
    val   = head_ckpt.get("val_metrics", {})
    print(f"\n[merge] Head source: epoch={epoch}")
    print(f"[merge] Val metrics: {val}")


if __name__ == "__main__":
    main()
