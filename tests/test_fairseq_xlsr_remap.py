"""Tests for fairseq -> HF Wav2Vec2 state-dict key remapper."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from backend._fairseq_to_hf_xlsr import _convert_key, convert_xlsr_subkeys


def test_conv_layer_weight():
    assert _convert_key("feature_extractor.conv_layers.0.0.weight") == "feature_extractor.conv_layers.0.conv.weight"
    assert _convert_key("feature_extractor.conv_layers.3.0.bias") == "feature_extractor.conv_layers.3.conv.bias"


def test_conv_layer_norm():
    assert _convert_key("feature_extractor.conv_layers.0.2.1.weight") == "feature_extractor.conv_layers.0.layer_norm.weight"
    assert _convert_key("feature_extractor.conv_layers.6.2.1.bias") == "feature_extractor.conv_layers.6.layer_norm.bias"


def test_post_extract_proj():
    assert _convert_key("post_extract_proj.weight") == "feature_projection.projection.weight"


def test_top_level_layer_norm():
    assert _convert_key("layer_norm.weight") == "feature_projection.layer_norm.weight"


def test_mask_emb():
    assert _convert_key("mask_emb") == "masked_spec_embed"


def test_pos_conv():
    assert _convert_key("encoder.pos_conv.0.weight_g") == "encoder.pos_conv_embed.conv.weight_g"
    assert _convert_key("encoder.pos_conv.0.bias") == "encoder.pos_conv_embed.conv.bias"


def test_encoder_layer_attention():
    assert _convert_key("encoder.layers.0.self_attn.k_proj.weight") == "encoder.layers.0.attention.k_proj.weight"
    assert _convert_key("encoder.layers.5.self_attn.out_proj.bias") == "encoder.layers.5.attention.out_proj.bias"


def test_encoder_layer_norms():
    assert _convert_key("encoder.layers.0.self_attn_layer_norm.weight") == "encoder.layers.0.layer_norm.weight"
    assert _convert_key("encoder.layers.0.final_layer_norm.bias") == "encoder.layers.0.final_layer_norm.bias"


def test_encoder_layer_ffn():
    assert _convert_key("encoder.layers.0.fc1.weight") == "encoder.layers.0.feed_forward.intermediate_dense.weight"
    assert _convert_key("encoder.layers.0.fc2.bias") == "encoder.layers.0.feed_forward.output_dense.bias"


def test_drops_quantizer():
    assert _convert_key("quantizer.codevectors") is None
    assert _convert_key("project_q.weight") is None
    assert _convert_key("project_hid.bias") is None


def test_convert_xlsr_subkeys_preserves_non_prefixed():
    t = torch.zeros(1)
    state = {
        "ssl_model.model.feature_extractor.conv_layers.0.0.weight": t,
        "ssl_model.model.quantizer.codevectors": t,
        "pos_S": t,
        "out_layer.weight": t,
    }
    out = convert_xlsr_subkeys(state, prefix="ssl_model.model.")
    assert "ssl_model.model.feature_extractor.conv_layers.0.conv.weight" in out
    assert "ssl_model.model.feature_extractor.conv_layers.0.0.weight" not in out
    assert "ssl_model.model.quantizer.codevectors" not in out  # dropped
    assert "pos_S" in out  # unchanged
    assert "out_layer.weight" in out
