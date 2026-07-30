"""vLLM-free core logic for the kvrot KV connector.

Everything here is exercised by the CPU test suite (``tests/test_vllm_core.py``)
without vLLM installed. The conventions match vLLM v1's paged KV cache:

- Per layer, per TP rank, the paged cache is ONE fused tensor holding K and V.
  Two layouts exist depending on the attention backend (design §5.1):

  * FlashAttention: ``(2, num_blocks, block_size, kv_heads, head_dim)``
  * FlashInfer:     ``(num_blocks, 2, block_size, kv_heads, head_dim)``

  ``kv_axis`` (0 or 1) names the K/V axis of size 2.

- A request's token *i* lives at slot ``block_ids[i // bs] * bs + i % bs``.

- Stored keys are post-RoPE (rotated at their in-request position), exactly as
  in HF — so ``kvrot.rope.reindex_keys`` applies unchanged. vLLM positions are
  the in-request indices ``0..N-1``, i.e. always recompacted (design §5.3);
  an eviction plan therefore only needs keep-indices, and the per-token
  rotation delta is ``arange(K) - keep``.

- NoPE layers (afmoe's 15 global layers) must pass through surgery
  byte-identical — ``applies_rope`` gates the rotation per layer (design §5.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from kvrot.rope import reindex_keys

# ---------------------------------------------------------------------------
# Driver <-> connector message schema (rides in kv_transfer_params["kvrot"])
# ---------------------------------------------------------------------------


class EvictionPlanMsg(BaseModel):
    """Turn-aligned eviction plan, computed driver-side (chat.turn_keep_indices).

    ``keep`` is strictly-increasing indices into the session's stored tokens;
    ``src_len`` is the driver's belief of the store length, asserted worker-side
    so a desynced ledger fails loudly instead of rotating garbage.
    """

    model_config = ConfigDict(frozen=True)

    keep: list[int] = Field(min_length=1)
    src_len: int = Field(ge=1)


class KvrotParams(BaseModel):
    """Per-request connector params: ``{"kvrot": {...}}`` in kv_transfer_params."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(min_length=1)
    plan: EvictionPlanMsg | None = None


def parse_kvrot_params(kv_transfer_params: dict | None) -> KvrotParams | None:
    """Extract our namespaced params; None when the request isn't ours."""
    if not kv_transfer_params:
        return None
    raw = kv_transfer_params.get("kvrot")
    if raw is None:
        return None
    return KvrotParams.model_validate(raw)


# ---------------------------------------------------------------------------
# Claim / slot arithmetic
# ---------------------------------------------------------------------------


def align_down(num_tokens: int, block_size: int) -> int:
    """Largest multiple of ``block_size`` that is <= num_tokens."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return max(0, num_tokens) // block_size * block_size


def compute_claim(stored_tokens: int, prompt_len: int, block_size: int) -> int:
    """How many prompt tokens to claim as externally matched.

    Block-aligned (scheduler expectation, design §5.6) and never the full
    prompt (``assert num_new_tokens > 0`` in the scheduler has no backstop):
    the ragged tail plus at least the final token are recomputed by vLLM from
    the injected prefix — rolling_replay semantics.
    """
    return min(align_down(stored_tokens, block_size), align_down(prompt_len - 1, block_size))


def slot_mapping_from_blocks(
    block_ids: list[int], block_size: int, start: int, count: int
) -> Tensor:
    """Slots (paged-pool token indices) for request tokens [start, start+count).

    Mirrors ExampleConnector: ``slot = block_ids[i // bs] * bs + i % bs``.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    end = start + count
    if end > len(block_ids) * block_size:
        raise ValueError(
            f"token range [{start}, {end}) exceeds allocated blocks "
            f"({len(block_ids)} x {block_size})"
        )
    pos = torch.arange(start, end, dtype=torch.int64)
    blocks = torch.tensor(block_ids, dtype=torch.int64)
    return blocks[pos // block_size] * block_size + pos % block_size


# ---------------------------------------------------------------------------
# Paged-tensor layout adapters
# ---------------------------------------------------------------------------


def infer_kv_axis(paged_shape: tuple[int, ...]) -> int:
    """Which axis of the fused paged tensor is the K/V axis (size 2).

    FlashAttention puts it at 0, FlashInfer at 1. Refuses to guess when
    ambiguous (both axes size 2, or neither) — pass an explicit override via
    connector extra config in that case.
    """
    if len(paged_shape) != 5:
        raise ValueError(
            f"expected a 5-D fused KV paged tensor (no MLA support), got shape {paged_shape}"
        )
    ax0, ax1 = paged_shape[0] == 2, paged_shape[1] == 2
    if ax0 and not ax1:
        return 0
    if ax1 and not ax0:
        return 1
    raise ValueError(
        f"cannot infer K/V axis from paged shape {paged_shape} (ambiguous); "
        "set kvrot_kv_axis explicitly in kv_connector_extra_config"
    )


def _blk_off(paged: Tensor, slots: Tensor, kv_axis: int) -> tuple[Tensor, Tensor]:
    """Slot indices -> (block, in-block offset) index tensors on paged's device.

    NEVER flatten the paged tensor with reshape here: for the FlashInfer layout
    (kv_axis=1) a permute+reshape silently COPIES, so an in-place scatter would
    write to a throwaway tensor — the classic silent-no-op injection bug.
    Advanced indexing on the original tensor is in-place in both layouts.
    """
    if kv_axis not in (0, 1):
        raise ValueError(f"kv_axis must be 0 or 1, got {kv_axis}")
    if paged.dim() != 5:
        raise ValueError(f"expected 5-D fused KV paged tensor, got shape {tuple(paged.shape)}")
    if paged.shape[kv_axis] != 2:
        raise ValueError(
            f"K/V axis {kv_axis} has size {paged.shape[kv_axis]}, expected 2 "
            f"(shape {tuple(paged.shape)})"
        )
    block_size = paged.shape[2]
    slots = slots.to(paged.device)
    return slots // block_size, slots % block_size


def extract_tokens(paged: Tensor, slots: Tensor, kv_axis: int) -> Tensor:
    """Gather per-token KV from the paged tensor -> ``[2, T, kv_heads, head_dim]``."""
    blk, off = _blk_off(paged, slots, kv_axis)
    if kv_axis == 0:  # (2, nb, bs, H, D)
        return paged[:, blk, off]  # [2, T, H, D]
    # (nb, 2, bs, H, D): advanced indices on dims 0 and 2 sandwich the slice ->
    # result comes back token-major [T, 2, H, D]
    return paged[blk, :, off].permute(1, 0, 2, 3)


def inject_tokens(paged: Tensor, slots: Tensor, src: Tensor, kv_axis: int) -> None:
    """Scatter ``src [2, T, kv_heads, head_dim]`` into the paged tensor at ``slots``.

    In-place on the original paged tensor (see ``_blk_off`` on why not reshape).
    """
    blk, off = _blk_off(paged, slots, kv_axis)
    expected = (2, slots.shape[0], paged.shape[3], paged.shape[4])
    if tuple(src.shape) != expected:
        raise ValueError(f"src shape {tuple(src.shape)} != expected {expected}")
    src = src.to(device=paged.device, dtype=paged.dtype)
    if kv_axis == 0:
        paged[:, blk, off] = src
    else:
        paged[blk, :, off] = src.permute(1, 0, 2, 3)


# ---------------------------------------------------------------------------
# Per-session KV store + surgery
# ---------------------------------------------------------------------------


@dataclass
class SessionKVStore:
    """One session's KV, this TP rank's shard: per layer ``[2, T, kv_heads, head_dim]``.

    ``applies_rope``/``inv_freq`` are per-model constants shared by all
    sessions; they live here so surgery is self-contained.
    """

    applies_rope: list[bool]
    inv_freq: Tensor
    device: torch.device
    layers: list[Tensor] = field(default_factory=list)

    def seq_len(self) -> int:
        return 0 if not self.layers else int(self.layers[0].shape[1])

    def num_layers(self) -> int:
        return len(self.applies_rope)

    def append(self, start: int, layer_tensors: list[Tensor]) -> None:
        """Append extracted KV for request tokens [start, start+T) — contiguous only."""
        if len(layer_tensors) != self.num_layers():
            raise ValueError(
                f"got {len(layer_tensors)} layers, store expects {self.num_layers()}"
            )
        cur = self.seq_len()
        if start != cur:
            raise ValueError(
                f"non-contiguous append: store has {cur} tokens, save starts at {start}"
            )
        moved = [t.to(self.device) for t in layer_tensors]
        if not self.layers:
            self.layers = moved
        else:
            self.layers = [torch.cat([old, new], dim=1) for old, new in zip(self.layers, moved)]

    def truncate(self, num_tokens: int) -> None:
        """Drop stored tokens beyond ``num_tokens`` (the un-claimed ragged tail —
        vLLM recomputes those from the injected prefix and they get re-saved)."""
        if num_tokens > self.seq_len():
            raise ValueError(f"cannot truncate to {num_tokens}: store has {self.seq_len()}")
        if num_tokens < self.seq_len():
            self.layers = [t[:, :num_tokens].contiguous() for t in self.layers]

    def apply_plan(self, plan: EvictionPlanMsg) -> None:
        """Evict + re-rotate in place: keep ``plan.keep``, recompact to 0..K-1.

        Keys on RoPE layers get the exact ``R(new - old)`` re-rotation
        (kvrot Tier 0); NoPE layers pass through byte-identical; V is
        position-free and only evicted, never touched.
        """
        cur = self.seq_len()
        if plan.src_len != cur:
            raise ValueError(
                f"eviction plan built for src_len={plan.src_len} but store has "
                f"{cur} tokens — driver ledger and connector store are desynced"
            )
        keep = torch.tensor(plan.keep, dtype=torch.int64, device=self.device)
        if keep.numel() > 1 and not bool((keep[1:] > keep[:-1]).all()):
            raise ValueError("plan.keep must be strictly increasing")
        if int(keep[-1]) >= cur or int(keep[0]) < 0:
            raise ValueError(f"plan.keep out of range [0, {cur})")

        # recompacted new positions are 0..K-1, so delta = new - old = arange - keep
        delta = torch.arange(keep.numel(), dtype=torch.int64, device=self.device) - keep

        new_layers: list[Tensor] = []
        for i, t in enumerate(self.layers):
            kept = t.index_select(1, keep)  # [2, K, H, D]
            if self.applies_rope[i]:
                k = kept[0].permute(1, 0, 2)  # [H, K, D] — seq at dim -2
                k = reindex_keys(k, delta, self.inv_freq)
                kept = torch.stack([k.permute(1, 0, 2), kept[1]])
            new_layers.append(kept.contiguous())
        self.layers = new_layers
