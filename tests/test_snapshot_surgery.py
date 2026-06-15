"""Proof that evict + reindex on a KVSnapshot is exact for the positional component.

We build a synthetic cache by rotating known unrotated keys at their original positions,
run the full surgery (sink-aware evict, then recompact + re-rotate), and check the
result is bit-identical to rotating the *original* keys directly at the new positions.
No model required — this is pure tensor math, runnable on CPU.
"""

from __future__ import annotations

import torch

from kvrot.config import EvictionSpec
from kvrot.eviction import compute_keep_indices, new_positions_for
from kvrot.rope import default_inv_freq, reindex_keys
from kvrot.snapshot import KVSnapshot, evict, reindex

HEAD_DIM = 64
THETA = 10000.0
B, H, S = 1, 4, 40


def _make_snapshot(inv_freq, n_layers=3):
    k0 = [torch.randn(B, H, S, HEAD_DIM) for _ in range(n_layers)]  # unrotated truth
    pos = torch.arange(S, dtype=torch.float32)
    keys = [reindex_keys(k, pos, inv_freq) for k in k0]  # store rotated at 0..S-1
    values = [torch.randn(B, H, S, HEAD_DIM) for _ in range(n_layers)]
    positions = [torch.arange(S, dtype=torch.long) for _ in range(n_layers)]
    layer_types = ["full_attention"] * n_layers
    snap = KVSnapshot(keys=keys, values=values, positions=positions, layer_types=layer_types)
    return snap, k0, values


def test_evict_then_reindex_is_exact():
    inv = default_inv_freq(HEAD_DIM, THETA)
    snap, k0, values = _make_snapshot(inv)

    spec = EvictionSpec(policy="sink_window", num_sink_tokens=4, evict_count=10, recompact=True)
    keep = compute_keep_indices(S, spec)
    new_pos = new_positions_for(keep, snap.positions[0], recompact=True)

    out = reindex(evict(snap, keep), new_pos, inv)

    # Expected: take the *unrotated* truth at kept indices, rotate at the new positions.
    for li in range(snap.num_layers):
        expected = reindex_keys(
            k0[li].index_select(-2, keep), new_pos.float(), inv
        )
        assert torch.allclose(out.keys[li], expected, atol=1e-5), f"layer {li} key mismatch"
        # values are position-free: just the kept slice, untouched
        assert torch.equal(out.values[li], values[li].index_select(-2, keep))
        # positions renumbered contiguously
        assert torch.equal(out.positions[li], torch.arange(keep.shape[0], dtype=torch.long))


def test_next_position_after_surgery():
    inv = default_inv_freq(HEAD_DIM, THETA)
    snap, _, _ = _make_snapshot(inv)
    spec = EvictionSpec(policy="sink_window", num_sink_tokens=4, evict_count=10)
    keep = compute_keep_indices(S, spec)
    new_pos = new_positions_for(keep, snap.positions[0], recompact=True)
    out = reindex(evict(snap, keep), new_pos, inv)
    # kept S-10 tokens, compacted to 0..(S-11) → next position is S-10
    assert out.next_position() == S - 10


def test_reindex_skips_nope_layers():
    """Hybrid model (trinity): re-rotate RoPE layers, leave NoPE layers' keys untouched."""
    inv = default_inv_freq(HEAD_DIM, THETA)
    pos = torch.arange(S, dtype=torch.float32)
    rope_keys = reindex_keys(torch.randn(B, H, S, HEAD_DIM), pos, inv)  # layer 0: rotated
    nope_keys = torch.randn(B, H, S, HEAD_DIM)                          # layer 1: never rotated
    snap = KVSnapshot(
        keys=[rope_keys, nope_keys],
        values=[torch.randn(B, H, S, HEAD_DIM) for _ in range(2)],
        positions=[torch.arange(S, dtype=torch.long) for _ in range(2)],
        layer_types=["sliding_attention", "full_attention"],
        applies_rope=[True, False],
    )
    keep = compute_keep_indices(S, EvictionSpec(policy="sink_window", num_sink_tokens=4, evict_count=10))
    new_pos = new_positions_for(keep, snap.positions[0], recompact=True)
    out = reindex(evict(snap, keep), new_pos, inv)

    # RoPE layer: re-rotated to the new positions
    expected0 = reindex_keys(
        rope_keys.index_select(-2, keep),
        (new_pos - snap.positions[0].index_select(0, keep)).float(),
        inv,
    )
    assert torch.allclose(out.keys[0], expected0, atol=1e-5)
    # NoPE layer: keys are byte-for-byte the evicted slice (no rotation applied)
    assert torch.equal(out.keys[1], nope_keys.index_select(-2, keep))


def test_sliding_layer_can_evict_more_than_full_layer():
    """Per-layer keep sets: a hybrid model drops more on sliding layers (smoke test)."""
    inv = default_inv_freq(HEAD_DIM, THETA)
    snap, _, _ = _make_snapshot(inv, n_layers=2)
    keep_full = compute_keep_indices(S, EvictionSpec(policy="sink_window", num_sink_tokens=4, evict_count=8))
    keep_slide = compute_keep_indices(S, EvictionSpec(policy="oldest", evict_count=20))
    out = evict(snap, [keep_full, keep_slide])
    assert out.seq_len(0) == keep_full.shape[0]
    assert out.seq_len(1) == keep_slide.shape[0]
    assert out.seq_len(1) < out.seq_len(0)
