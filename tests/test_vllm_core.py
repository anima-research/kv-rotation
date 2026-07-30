"""CPU tests for kvrot_vllm.core — no vLLM, no GPU, no model.

The load-bearing test is surgery parity: SessionKVStore.apply_plan must
reproduce the proven KVSnapshot evict+reindex path bit-exactly, per layer,
including the NoPE gate. The rest covers the paged-layout adapters (both
backend layouts), slot math, claim arithmetic, and store bookkeeping.
"""

from __future__ import annotations

import pytest
import torch

from kvrot.rope import default_inv_freq, reindex_keys
from kvrot.snapshot import KVSnapshot, evict, reindex
from kvrot_vllm.core import (
    EvictionPlanMsg,
    KvrotParams,
    SessionKVStore,
    align_down,
    compute_claim,
    extract_tokens,
    infer_kv_axis,
    inject_tokens,
    parse_kvrot_params,
    slot_mapping_from_blocks,
)

H, D, BS = 2, 8, 4  # kv_heads, head_dim, block_size
INV = default_inv_freq(D, 10000.0)


def make_store(seq: int, applies_rope: list[bool], seed: int = 0) -> SessionKVStore:
    g = torch.Generator().manual_seed(seed)
    store = SessionKVStore(applies_rope=applies_rope, inv_freq=INV, device=torch.device("cpu"))
    layers = [torch.randn(2, seq, H, D, generator=g) for _ in applies_rope]
    store.append(0, layers)
    return store


# ---------------------------------------------------------------------------
# claim / slot arithmetic
# ---------------------------------------------------------------------------


def test_align_down():
    assert align_down(0, 16) == 0
    assert align_down(15, 16) == 0
    assert align_down(16, 16) == 16
    assert align_down(33, 16) == 32
    with pytest.raises(ValueError):
        align_down(5, 0)


def test_compute_claim_never_claims_full_prompt():
    # stored covers the whole prompt: claim must still leave >=1 token to compute
    assert compute_claim(stored_tokens=32, prompt_len=32, block_size=16) == 16
    # normal case: survivors 37, new turn brings prompt to 50
    assert compute_claim(37, 50, 16) == 32
    # store shorter than a block -> nothing claimable
    assert compute_claim(7, 50, 16) == 0
    # empty session
    assert compute_claim(0, 10, 16) == 0


def test_slot_mapping_from_blocks():
    # blocks [7, 3, 9], bs=4 -> token 0..3 in block 7, 4..7 in block 3, ...
    slots = slot_mapping_from_blocks([7, 3, 9], 4, start=0, count=12)
    assert slots.tolist() == [28, 29, 30, 31, 12, 13, 14, 15, 36, 37, 38, 39]
    # mid-range extraction (save of a chunk)
    slots = slot_mapping_from_blocks([7, 3, 9], 4, start=5, count=4)
    assert slots.tolist() == [13, 14, 15, 36]
    with pytest.raises(ValueError):
        slot_mapping_from_blocks([7], 4, start=2, count=4)  # exceeds allocation
    with pytest.raises(ValueError):
        slot_mapping_from_blocks([7], 4, start=0, count=0)


# ---------------------------------------------------------------------------
# paged layout adapters
# ---------------------------------------------------------------------------


def test_infer_kv_axis():
    assert infer_kv_axis((2, 10, BS, H, D)) == 0  # FlashAttention
    assert infer_kv_axis((10, 2, BS, H, D)) == 1  # FlashInfer
    with pytest.raises(ValueError):
        infer_kv_axis((2, 2, BS, H, D))  # ambiguous
    with pytest.raises(ValueError):
        infer_kv_axis((10, 3, BS, H, D))  # no size-2 axis
    with pytest.raises(ValueError):
        infer_kv_axis((10, 2, BS, H))  # not 5-D (MLA-ish)


@pytest.mark.parametrize("kv_axis", [0, 1])
def test_extract_inject_round_trip(kv_axis: int):
    nb = 6
    shape = (2, nb, BS, H, D) if kv_axis == 0 else (nb, 2, BS, H, D)
    paged = torch.randn(*shape)
    baseline = paged.clone()
    slots = slot_mapping_from_blocks([4, 1, 5], BS, 0, 10)

    src = torch.randn(2, 10, H, D)
    inject_tokens(paged, slots, src, kv_axis)
    got = extract_tokens(paged, slots, kv_axis)
    assert torch.equal(got, src)

    # inject wrote ONLY the mapped slots (and in-place on the original tensor)
    mask = torch.zeros(nb * BS, dtype=torch.bool)
    mask[slots] = True
    flat_now = paged.movedim(kv_axis, 0).reshape(2, nb * BS, H, D)
    flat_before = baseline.movedim(kv_axis, 0).reshape(2, nb * BS, H, D)
    assert torch.equal(flat_now[:, ~mask], flat_before[:, ~mask])
    assert not torch.equal(flat_now[:, mask], flat_before[:, mask])


def test_inject_rejects_bad_src_shape():
    paged = torch.randn(2, 4, BS, H, D)
    slots = slot_mapping_from_blocks([0, 1], BS, 0, 5)
    with pytest.raises(ValueError):
        inject_tokens(paged, slots, torch.randn(2, 4, H, D), 0)  # wrong T
    with pytest.raises(ValueError):
        inject_tokens(paged, slots, torch.randn(2, 5, H, D + 1), 0)  # wrong D


# ---------------------------------------------------------------------------
# 3-D shard regression for reindex_keys (per-rank connector shape)
# ---------------------------------------------------------------------------


def test_reindex_keys_3d_shard_matches_4d():
    k4 = torch.randn(1, H, 9, D)  # HF-style [B, H, S, D]
    delta = torch.arange(9) - torch.arange(9) * 2  # arbitrary shifts
    ref = reindex_keys(k4, delta.float(), INV)
    got = reindex_keys(k4[0], delta.float(), INV)  # [H, S, D] shard
    assert torch.equal(got, ref[0])


# ---------------------------------------------------------------------------
# session store bookkeeping
# ---------------------------------------------------------------------------


def test_store_append_contiguity_and_truncate():
    store = make_store(6, [True, False])
    assert store.seq_len() == 6
    with pytest.raises(ValueError):  # gap
        store.append(8, [torch.randn(2, 2, H, D) for _ in range(2)])
    with pytest.raises(ValueError):  # wrong layer count
        store.append(6, [torch.randn(2, 2, H, D)])
    store.append(6, [torch.randn(2, 3, H, D) for _ in range(2)])
    assert store.seq_len() == 9
    store.truncate(4)
    assert store.seq_len() == 4
    with pytest.raises(ValueError):
        store.truncate(10)


def test_apply_plan_validates():
    store = make_store(10, [True])
    with pytest.raises(ValueError):  # desynced ledger
        store.apply_plan(EvictionPlanMsg(keep=[0, 1], src_len=9))
    with pytest.raises(ValueError):  # not strictly increasing
        store.apply_plan(EvictionPlanMsg(keep=[3, 3, 5], src_len=10))
    with pytest.raises(ValueError):  # out of range
        store.apply_plan(EvictionPlanMsg(keep=[0, 10], src_len=10))


# ---------------------------------------------------------------------------
# THE test: surgery parity with the proven KVSnapshot path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "applies_rope",
    [
        [True, True, True],          # llama-like: all RoPE
        [True, False, True, False],  # afmoe-like: NoPE gating interleaved
    ],
)
def test_apply_plan_matches_snapshot_surgery(applies_rope: list[bool]):
    seq = 12
    keep = [0, 1, 4, 5, 6, 9, 11]  # sink + scattered survivors
    store = make_store(seq, applies_rope, seed=42)
    # keep an independent copy in HF snapshot layout [B, H, S, D]
    keys = [t[0].permute(1, 0, 2).unsqueeze(0).clone() for t in store.layers]
    values = [t[1].permute(1, 0, 2).unsqueeze(0).clone() for t in store.layers]
    snap = KVSnapshot(
        keys=keys,
        values=values,
        positions=[torch.arange(seq) for _ in applies_rope],
        layer_types=["full_attention"] * len(applies_rope),
        applies_rope=list(applies_rope),
    )

    keep_t = torch.tensor(keep)
    new_pos = torch.arange(len(keep))  # recompact
    ref = reindex(evict(snap, keep_t), new_pos, INV)

    store.apply_plan(EvictionPlanMsg(keep=keep, src_len=seq))

    assert store.seq_len() == len(keep)
    for i in range(len(applies_rope)):
        got_k = store.layers[i][0].permute(1, 0, 2)  # [H, K, D]
        got_v = store.layers[i][1].permute(1, 0, 2)
        assert torch.equal(got_k, ref.keys[i][0]), f"layer {i} keys diverge"
        assert torch.equal(got_v, ref.values[i][0]), f"layer {i} values diverge"


def test_apply_plan_nope_layers_byte_identical():
    store = make_store(8, [False, True], seed=7)
    orig_nope = store.layers[0].clone()
    keep = [0, 2, 3, 7]
    store.apply_plan(EvictionPlanMsg(keep=keep, src_len=8))
    # NoPE layer: pure index_select, bit-identical rows, no rotation applied
    assert torch.equal(store.layers[0], orig_nope[:, torch.tensor(keep)])


# ---------------------------------------------------------------------------
# driver message schema
# ---------------------------------------------------------------------------


def test_parse_kvrot_params():
    assert parse_kvrot_params(None) is None
    assert parse_kvrot_params({}) is None
    assert parse_kvrot_params({"other_connector": {"x": 1}}) is None
    p = parse_kvrot_params(
        {"kvrot": {"session_id": "s1", "plan": {"keep": [0, 1, 5], "src_len": 9}}}
    )
    assert isinstance(p, KvrotParams)
    assert p.session_id == "s1"
    assert p.plan is not None and p.plan.keep == [0, 1, 5]
    p2 = parse_kvrot_params({"kvrot": {"session_id": "s2"}})
    assert p2 is not None and p2.plan is None
    with pytest.raises(Exception):  # empty keep rejected by schema
        parse_kvrot_params({"kvrot": {"session_id": "s", "plan": {"keep": [], "src_len": 1}}})


# ---------------------------------------------------------------------------
# end-to-end mini rotation cycle (store -> plan -> claim -> inject -> compare)
# ---------------------------------------------------------------------------


def test_end_to_end_rotation_cycle():
    """Simulate one eviction turn at connector level and check the paged cache
    ends up holding exactly the surgered KV at the claimed slots."""
    applies_rope = [True, False]
    seq, keep = 11, [0, 1, 2, 5, 6, 7, 8, 10]  # 8 survivors
    store = make_store(seq, applies_rope, seed=3)
    ref_store = make_store(seq, applies_rope, seed=3)  # same seed -> same content

    plan = EvictionPlanMsg(keep=keep, src_len=seq)
    store.apply_plan(plan)

    claim = compute_claim(len(keep), prompt_len=len(keep) + 5, block_size=BS)
    assert claim == 8  # 8 survivors, aligned to 4, prompt leaves room
    store.truncate(claim)

    nb = 4
    paged = [torch.zeros(2, nb, BS, H, D) for _ in applies_rope]
    block_ids = [2, 0]  # claim/BS = 2 blocks
    slots = slot_mapping_from_blocks(block_ids, BS, 0, claim)
    for i, p in enumerate(paged):
        inject_tokens(p, slots, store.layers[i], 0)

    # reference: independent surgery, then read back through extract
    ref_store.apply_plan(plan)
    for i, p in enumerate(paged):
        got = extract_tokens(p, slots, 0)
        assert torch.equal(got, ref_store.layers[i][:, :claim])


# ---------------------------------------------------------------------------
# regression: transformers 5.x relocates rope_theta into the rope dict
# (exp12 gate-2 failure 2026-07-29: theta=10000 table built for Llama-3.2's
# theta=500000 -> rotated cells corrupt while everything else passes)
# ---------------------------------------------------------------------------


class _Cfg:
    """Minimal 5.x-style config: unset attrs raise AttributeError."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_rope_theta_read_from_rope_scaling_dict():
    from kvrot.config import RoPESpec

    cfg = _Cfg(
        head_dim=128,
        hidden_size=3072,
        num_attention_heads=24,
        rope_scaling={
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )
    spec = RoPESpec.from_hf_config(cfg)
    assert spec.theta == 500000.0
    assert spec.scaling_type == "llama3"


def test_rope_theta_read_from_rope_parameters_dict():
    from kvrot.config import RoPESpec

    cfg = _Cfg(
        head_dim=128,
        hidden_size=4096,
        num_attention_heads=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    spec = RoPESpec.from_hf_config(cfg)
    assert spec.theta == 10000.0
    assert spec.scaling_type == "none"


def test_rope_theta_top_level_fallback():
    from kvrot.config import RoPESpec

    cfg = _Cfg(head_dim=64, hidden_size=2048, num_attention_heads=32, rope_theta=1e6)
    spec = RoPESpec.from_hf_config(cfg)
    assert spec.theta == 1e6
