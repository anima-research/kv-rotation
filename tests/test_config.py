"""Config parsing: rope_type handling + per-layer RoPE gating (afmoe vs Llama)."""

from __future__ import annotations

from types import SimpleNamespace

from kvrot.config import ArchSpec, RoPESpec


def test_default_rope_type_maps_to_none():
    # transformers 5.x standardizes null rope_scaling to {"rope_type": "default"}
    cfg = SimpleNamespace(
        head_dim=128, hidden_size=3072, num_attention_heads=24,
        rope_theta=10000.0, rope_scaling={"rope_type": "default"},
    )
    spec = RoPESpec.from_hf_config(cfg)
    assert spec.scaling_type == "none"
    assert spec.build_inv_freq().shape[0] == 64


def test_llama3_rope_type_parsed():
    cfg = SimpleNamespace(
        head_dim=128, hidden_size=3072, num_attention_heads=24, rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
        },
    )
    spec = RoPESpec.from_hf_config(cfg)
    assert spec.scaling_type == "llama3"
    assert spec.build_inv_freq().shape[0] == 64


def test_unknown_rope_type_does_not_crash():
    cfg = SimpleNamespace(
        head_dim=128, hidden_size=3072, num_attention_heads=24,
        rope_theta=10000.0, rope_scaling={"rope_type": "yarn", "factor": 4.0},
    )
    spec = RoPESpec.from_hf_config(cfg)  # must not raise; harness uses live-model inv_freq
    assert spec.scaling_type == "none"


def test_afmoe_applies_rope_sliding_only():
    layer_types = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
    cfg = SimpleNamespace(
        model_type="afmoe", num_hidden_layers=4, head_dim=128, hidden_size=3072,
        num_attention_heads=48, num_key_value_heads=8, sliding_window=4096,
        rope_theta=10000.0, rope_scaling={"rope_type": "default"}, layer_types=layer_types,
    )
    arch = ArchSpec.from_hf_config(cfg)
    assert arch.applies_rope == [True, True, True, False]
    assert sum(arch.applies_rope) == len(arch.sliding_layers)
    assert len(arch.full_attention_layers) == 1


def test_llama_applies_rope_all_layers():
    cfg = SimpleNamespace(
        model_type="llama", num_hidden_layers=4, head_dim=128, hidden_size=3072,
        num_attention_heads=24, num_key_value_heads=8, sliding_window=None,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
        },
        layer_types=None,
    )
    arch = ArchSpec.from_hf_config(cfg)
    assert arch.applies_rope == [True] * 4
    assert arch.rope.scaling_type == "llama3"
