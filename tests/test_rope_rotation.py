"""Tier 0 proof: re-rotating stored keys is bit-exact for the positional component."""

from __future__ import annotations

import torch

from kvrot.rope import (
    assert_rotation_homomorphism,
    default_inv_freq,
    llama3_inv_freq,
    reindex_keys,
    rotate_half,
)

HEAD_DIM = 128
THETA = 500000.0


def _inv_freq():
    return default_inv_freq(HEAD_DIM, THETA)


def test_rotate_half_preserves_norm():
    x = torch.randn(2, 3, 5, HEAD_DIM)
    # [x, rotate_half(x)] is an orthogonal transform of each 2-D pair → norm preserved
    inv = _inv_freq()
    delta = torch.full((5,), 7.0)
    y = reindex_keys(x, delta, inv)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_compose_shift_equals_single_rotation():
    """R(-k)·R(p) == R(p-k): re-rotating a stored key matches rotating from scratch."""
    inv = _inv_freq()
    k0 = torch.randn(1, 8, 16, HEAD_DIM)  # unrotated
    pos = torch.arange(16, dtype=torch.float32)
    stored = reindex_keys(k0, pos, inv)  # rotate at original positions

    k = 5
    new_pos = pos - k
    reindexed = reindex_keys(stored, torch.full((16,), float(-k)), inv)
    from_scratch = reindex_keys(k0, new_pos, inv)
    assert torch.allclose(reindexed, from_scratch, atol=1e-5)


def test_inverse_recovers_original():
    inv = _inv_freq()
    k0 = torch.randn(1, 4, 10, HEAD_DIM)
    pos = torch.arange(10, dtype=torch.float32) + 3
    stored = reindex_keys(k0, pos, inv)
    recovered = reindex_keys(stored, -pos, inv)
    assert torch.allclose(recovered, k0, atol=1e-5)


def test_homomorphism_holds_for_default_and_llama3():
    assert_rotation_homomorphism(default_inv_freq(HEAD_DIM, THETA))
    assert_rotation_homomorphism(
        llama3_inv_freq(
            HEAD_DIM,
            THETA,
            factor=32.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        )
    )
