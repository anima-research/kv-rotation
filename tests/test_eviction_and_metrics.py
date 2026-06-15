"""Eviction keep-index logic + drift-metric sanity."""

from __future__ import annotations

import torch

from kvrot.config import EvictionSpec
from kvrot.eviction import compute_keep_indices, new_positions_for
from kvrot.metrics import DriftReport, stepwise_kl, top1_agreement


def test_policy_none_keeps_all():
    keep = compute_keep_indices(20, EvictionSpec(policy="none"))
    assert torch.equal(keep, torch.arange(20))


def test_oldest_drops_leading_block_including_sinks():
    keep = compute_keep_indices(20, EvictionSpec(policy="oldest", evict_count=5))
    assert torch.equal(keep, torch.arange(5, 20))
    assert 0 not in keep.tolist()  # sinks are gone — the catastrophic baseline


def test_sink_window_keeps_sinks_and_recent():
    keep = compute_keep_indices(20, EvictionSpec(policy="sink_window", num_sink_tokens=4, evict_count=6))
    # keep 0,1,2,3 then 10..19
    assert keep.tolist() == [0, 1, 2, 3] + list(range(10, 20))


def test_sink_window_degenerate_keeps_only_sinks():
    keep = compute_keep_indices(8, EvictionSpec(policy="sink_window", num_sink_tokens=4, evict_count=100))
    assert keep.tolist() == [0, 1, 2, 3]


def test_recompact_positions_are_contiguous():
    keep = compute_keep_indices(20, EvictionSpec(policy="sink_window", num_sink_tokens=4, evict_count=6))
    old = torch.arange(20)
    new = new_positions_for(keep, old, recompact=True)
    assert torch.equal(new, torch.arange(keep.shape[0]))
    # without recompact, original positions are preserved (gap remains)
    new_gap = new_positions_for(keep, old, recompact=False)
    assert new_gap.tolist() == [0, 1, 2, 3] + list(range(10, 20))


def test_kl_zero_for_identical_logits():
    logits = torch.randn(8, 100)
    kl = stepwise_kl(logits, logits)
    assert torch.allclose(kl, torch.zeros(8), atol=1e-6)
    assert torch.equal(top1_agreement(logits, logits), torch.ones(8))


def test_drift_report_fields():
    ref = torch.randn(16, 256)
    other = ref + 0.01 * torch.randn(16, 256)
    rep = DriftReport.from_logits(ref, other, label="noisy")
    assert rep.n_steps == 16
    assert rep.mean_kl > 0
    assert 0.0 <= rep.top1_agreement <= 1.0
    assert len(rep.per_step_kl) == 16
