"""Convert fairseq Wav2Vec2 state-dict keys to HuggingFace Wav2Vec2Model keys.

Reference: transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py
"""
from __future__ import annotations

import re
from typing import Dict

import torch

# Keys present in fairseq pretraining checkpoint but unused by Wav2Vec2Model (inference).
# These are quantizer / pretraining heads — drop them.
_DROP_PREFIXES = (
    "quantizer.",
    "project_q.",
    "project_hid.",
    "final_proj.",
    "label_embs_concat",
)


def _convert_key(key: str) -> str | None:
    """Map one fairseq Wav2Vec2 key to its HF equivalent. Return None to drop."""
    # Drop pretraining-only weights
    for prefix in _DROP_PREFIXES:
        if key.startswith(prefix):
            return None

    # mask_emb -> masked_spec_embed
    if key == "mask_emb":
        return "masked_spec_embed"

    # feature_extractor.conv_layers.N.0.{weight,bias} -> feature_extractor.conv_layers.N.conv.{w,b}
    m = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", key)
    if m:
        return f"feature_extractor.conv_layers.{m.group(1)}.conv.{m.group(2)}"

    # feature_extractor.conv_layers.N.2.1.{weight,bias} -> feature_extractor.conv_layers.N.layer_norm.{w,b}
    # (layer-norm variant; group-norm variant is N.0 only on layer 0 but XLS-R uses layer-norm everywhere)
    m = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)", key)
    if m:
        return f"feature_extractor.conv_layers.{m.group(1)}.layer_norm.{m.group(2)}"

    # post_extract_proj -> feature_projection.projection
    if key == "post_extract_proj.weight":
        return "feature_projection.projection.weight"
    if key == "post_extract_proj.bias":
        return "feature_projection.projection.bias"

    # Top-level layer_norm (between extractor and encoder) -> feature_projection.layer_norm
    if key == "layer_norm.weight":
        return "feature_projection.layer_norm.weight"
    if key == "layer_norm.bias":
        return "feature_projection.layer_norm.bias"

    # encoder.pos_conv.0.* -> encoder.pos_conv_embed.conv.*
    m = re.fullmatch(r"encoder\.pos_conv\.0\.(weight_g|weight_v|weight|bias)", key)
    if m:
        return f"encoder.pos_conv_embed.conv.{m.group(1)}"

    # encoder.layer_norm.* stays as encoder.layer_norm.*
    if key in ("encoder.layer_norm.weight", "encoder.layer_norm.bias"):
        return key

    # encoder.layers.N.* — multiple sub-mappings
    m = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", key)
    if m:
        layer_idx, rest = m.group(1), m.group(2)
        new_rest = _convert_encoder_layer_subkey(rest)
        if new_rest is None:
            return None
        return f"encoder.layers.{layer_idx}.{new_rest}"

    # Unknown key — return as-is (will be flagged by strict load if it matters)
    return key


def _convert_encoder_layer_subkey(sub: str) -> str | None:
    # self_attn.* -> attention.*
    if sub.startswith("self_attn."):
        attr = sub[len("self_attn."):]
        if attr in (
            "k_proj.weight", "k_proj.bias",
            "q_proj.weight", "q_proj.bias",
            "v_proj.weight", "v_proj.bias",
            "out_proj.weight", "out_proj.bias",
        ):
            return f"attention.{attr}"
        return None
    # self_attn_layer_norm.* -> layer_norm.*
    if sub.startswith("self_attn_layer_norm."):
        return "layer_norm." + sub[len("self_attn_layer_norm."):]
    # fc1.* -> feed_forward.intermediate_dense.*
    if sub.startswith("fc1."):
        return "feed_forward.intermediate_dense." + sub[len("fc1."):]
    # fc2.* -> feed_forward.output_dense.*
    if sub.startswith("fc2."):
        return "feed_forward.output_dense." + sub[len("fc2."):]
    # final_layer_norm.* stays
    if sub.startswith("final_layer_norm."):
        return sub
    return None


def convert_xlsr_subkeys(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Remap any keys starting with ``prefix`` (e.g. 'ssl_model.model.') from fairseq to HF naming.

    Non-matching keys are passed through unchanged. Pretraining-head keys are dropped.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not k.startswith(prefix):
            out[k] = v
            continue
        sub = k[len(prefix):]
        new_sub = _convert_key(sub)
        if new_sub is None:
            continue  # drop
        out[prefix + new_sub] = v
    return out
