"""KvrotConnector — vLLM v1 KV connector doing save -> surgery -> inject between
turns, so a rolling chat session keeps its (re-rotated) KV instead of recomputing.

Loaded out-of-tree (no vLLM source edits, no installs)::

    --kv-transfer-config '{"kv_connector": "KvrotConnector",
                           "kv_connector_module_path": "kvrot_vllm.connector",
                           "kv_role": "kv_both",
                           "kv_connector_extra_config": {"kvrot_store_device": "cpu"}}'

Written against the pinned vLLM 0.16.0 on node1 (source evidence in
notes/design-vllm-playground.md §12). Protocol per turn (design §4):

1. Driver sends prompt = survivors + new turn, with
   ``kv_transfer_params={"kvrot": {"session_id": ..., "plan": {...}|None}}``.
2. Scheduler-side: claim ``compute_claim(...)`` tokens as externally matched
   (block-aligned, never all — the tail + reply are recomputed, design §5.7).
3. Worker-side ``start_load_kv``: apply the eviction plan to the session store
   (evict + RoPE-gated re-rotation, recompact), truncate to the claim, scatter
   into the freshly allocated paged blocks.
4. Worker-side ``wait_for_save``: extract this step's newly computed *prompt*
   KV from the paged cache and append to the session store (prefill-only save;
   decode steps save nothing).

Requires: APC off (design §5.5), HMA off (automatic once kv_transfer_config is
set — asserted at init), bf16 KV (``--kv-cache-dtype auto``).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from kvrot.config import ArchSpec
from kvrot.rope import assert_rotation_homomorphism
from kvrot_vllm.core import (
    EvictionPlanMsg,
    KvrotParams,
    SessionKVStore,
    compute_claim,
    extract_tokens,
    infer_kv_axis,
    inject_tokens,
    parse_kvrot_params,
    slot_mapping_from_blocks,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")


def _build_inv_freq_verified(hf_config: Any, arch: ArchSpec) -> torch.Tensor:
    """The rotation table, cross-checked against transformers' own builder.

    A wrong-but-static table passes the homomorphism assert and corrupts
    rotated cells only moderately (exp12 gate 2 caught exactly this for a
    mis-read llama3 theta), so build the reference table with transformers'
    ROPE_INIT_FUNCTIONS whenever possible and fail loudly on mismatch.
    Re-rotation is exact only when attention_scaling == 1.0 (a scaled table
    makes the stored transform non-orthonormal), so that is a hard error.
    """
    spec_table = arch.rope.build_inv_freq()
    scaling = (
        getattr(hf_config, "rope_scaling", None)
        or getattr(hf_config, "rope_parameters", None)
        or {}
    )
    rtype = scaling.get("rope_type", scaling.get("type", "default")) if isinstance(
        scaling, dict
    ) else "default"
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        fn = ROPE_INIT_FUNCTIONS.get(rtype or "default")
        if fn is None:
            logger.warning(
                "kvrot: no transformers rope init for type %r; using spec table unverified",
                rtype,
            )
            return spec_table
        ref, attention_scaling = fn(hf_config, device="cpu")
        if abs(float(attention_scaling) - 1.0) > 1e-6:
            raise RuntimeError(
                f"rope attention_scaling={attention_scaling} != 1.0 — cached-key "
                "re-rotation is inexact for this scaling scheme (yarn?); refusing"
            )
        ref = ref.float().cpu()
        err = float((ref - spec_table).abs().max())
        if err > 1e-5:
            logger.warning(
                "kvrot: RoPESpec table deviates from transformers (%s, max err "
                "%.3e) — using the transformers table",
                rtype,
                err,
            )
        return ref
    except ImportError:
        logger.warning("kvrot: transformers unavailable; using spec table unverified")
        return spec_table


@dataclass
class _LoadOp:
    req_id: str
    session_id: str
    num_tokens: int  # the claim; store is truncated to this after plan application
    block_ids: list[int]
    plan: EvictionPlanMsg | None


@dataclass
class _SaveOp:
    req_id: str
    session_id: str
    start: int  # first token index (request coords == store coords)
    count: int
    block_ids: list[int]  # accumulated blocks covering [0, start+count)


@dataclass
class KvrotConnectorMetadata(KVConnectorMetadata):
    loads: list[_LoadOp] = field(default_factory=list)
    saves: list[_SaveOp] = field(default_factory=list)


class KvrotConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._block_size: int = vllm_config.cache_config.block_size
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._strict: bool = bool(extra.get("kvrot_strict", True))
        self._max_sessions: int = int(extra.get("kvrot_max_sessions", 4))
        self._kv_axis_override: int | None = (
            int(extra["kvrot_kv_axis"]) if "kvrot_kv_axis" in extra else None
        )

        # HMA must be off so SWA layers retain full-length KV (design §5.4).
        # vLLM auto-disables it when kv_transfer_config is set; fail loudly if
        # a future version changes that.
        hma_disabled = getattr(
            vllm_config.scheduler_config, "disable_hybrid_kv_cache_manager", None
        )
        if hma_disabled is False:
            raise RuntimeError(
                "KvrotConnector requires the hybrid KV cache manager to be "
                "disabled (sliding-window layers must retain full-length KV); "
                "pass --disable-hybrid-kv-cache-manager"
            )

        if vllm_config.cache_config.enable_prefix_caching:
            raise RuntimeError(
                "KvrotConnector requires --no-enable-prefix-caching: APC "
                "shadows connector claims and re-hashes injected rotated "
                "blocks (design §5.5)"
            )

        # Per-layer RoPE gating + frequencies, config-only (design §5.2).
        hf_config = vllm_config.model_config.hf_config
        self._arch = ArchSpec.from_hf_config(hf_config)
        self._inv_freq = _build_inv_freq_verified(hf_config, self._arch)
        assert_rotation_homomorphism(self._inv_freq)
        n_rope = sum(self._arch.applies_rope)
        logger.info(
            "KvrotConnector(%s): %d layers (%d RoPE / %d NoPE), block_size=%d, "
            "strict=%s",
            role.name,
            len(self._arch.applies_rope),
            n_rope,
            len(self._arch.applies_rope) - n_rope,
            self._block_size,
            self._strict,
        )

        if role == KVConnectorRole.SCHEDULER:
            # session_id -> tokens currently held in the (worker) stores.
            # Scheduler-side mirror, updated when ops are emitted.
            self._session_tokens: dict[str, int] = {}
            # req_id -> parsed params, recorded in update_state_after_alloc
            self._req_params: dict[str, KvrotParams] = {}
            # req_id -> (accumulated block ids, prompt_len, claimed)
            self._req_blocks: dict[str, list[int]] = {}
            self._req_prompt_len: dict[str, int] = {}
            self._req_claimed: dict[str, int] = {}
            self._pending_load: set[str] = set()
        else:
            # WORKER state: per-rank session stores + registered paged tensors.
            self._stores: dict[str, SessionKVStore] = {}
            self._session_last_used: dict[str, float] = {}
            self._layer_tensors: list[torch.Tensor] | None = None
            self._kv_axis: int | None = self._kv_axis_override
            self._store_device = torch.device(str(extra.get("kvrot_store_device", "cpu")))
            # instrumentation (design §7 Gate 0): cumulative op/token counters
            self.stats: dict[str, int] = {
                "load_ops": 0,
                "loaded_tokens": 0,
                "save_ops": 0,
                "saved_tokens": 0,
                "plans_applied": 0,
            }

    # ==================================================================
    # Scheduler side
    # ==================================================================

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        params = parse_kvrot_params(request.kv_transfer_params)
        if params is None:
            return 0, False
        stored = (
            len(params.plan.keep) if params.plan is not None
            else self._session_tokens.get(params.session_id, 0)
        )
        prompt_len = len(request.prompt_token_ids or [])
        claim = compute_claim(stored, prompt_len, self._block_size)
        return max(0, claim - num_computed_tokens), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = parse_kvrot_params(request.kv_transfer_params)
        if params is None:
            return
        rid = request.request_id
        if rid not in self._req_params:
            self._req_params[rid] = params
        if num_external_tokens > 0:
            self._pending_load.add(rid)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = KvrotConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            rid = new_req.req_id
            params = self._req_params.get(rid)
            if params is None:
                continue
            if len(new_req.block_ids) != 1:
                raise RuntimeError(
                    f"expected a single KV cache group (HMA off), got "
                    f"{len(new_req.block_ids)} — hybrid allocation is active?"
                )
            prompt_len = len(new_req.prompt_token_ids or [])
            block_ids = list(new_req.block_ids[0])
            self._req_blocks[rid] = block_ids
            self._req_prompt_len[rid] = prompt_len
            claimed = new_req.num_computed_tokens  # == external claim (APC off)
            self._req_claimed[rid] = claimed

            if rid in self._pending_load:
                self._pending_load.discard(rid)
                meta.loads.append(
                    _LoadOp(
                        req_id=rid,
                        session_id=params.session_id,
                        num_tokens=claimed,
                        block_ids=block_ids,
                        plan=params.plan,
                    )
                )
                # store gets truncated to the claim at load time
                self._session_tokens[params.session_id] = claimed

            self._emit_save(meta, scheduler_output, rid, claimed, params)

        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            params = self._req_params.get(rid)
            if params is None:
                continue
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is not None:
                if rid in cached.resumed_req_ids:
                    # resumed-from-preemption: new_block_ids REPLACES (v1 does
                    # not restore our injected KV — unsupported single-user
                    # edge; fail loudly in strict mode)
                    if self._strict:
                        raise RuntimeError(
                            f"request {rid} resumed from preemption — "
                            "KvrotConnector v1 does not support preemption "
                            "recovery; rerun the turn"
                        )
                    self._req_blocks[rid] = list(new_block_ids[0])
                else:
                    self._req_blocks[rid].extend(new_block_ids[0])
            start = cached.num_computed_tokens[i]
            self._emit_save(meta, scheduler_output, rid, start, params)

        if self._pending_load and self._strict:
            raise RuntimeError(
                f"loads pending for {sorted(self._pending_load)} but their "
                "requests were not scheduled this step"
            )
        return meta

    def _emit_save(
        self,
        meta: KvrotConnectorMetadata,
        scheduler_output: SchedulerOutput,
        rid: str,
        start: int,
        params: KvrotParams,
    ) -> None:
        """Save only prompt-token KV (prefill-only policy, design §5.7)."""
        prompt_len = self._req_prompt_len.get(rid, 0)
        n_sched = scheduler_output.num_scheduled_tokens.get(rid, 0)
        count = min(n_sched, prompt_len - start)
        if count <= 0:
            return
        meta.saves.append(
            _SaveOp(
                req_id=rid,
                session_id=params.session_id,
                start=start,
                count=count,
                block_ids=list(self._req_blocks[rid]),
            )
        )
        s = params.session_id
        self._session_tokens[s] = max(self._session_tokens.get(s, 0), start + count)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        rid = request.request_id
        params = self._req_params.pop(rid, None)
        self._req_blocks.pop(rid, None)
        self._req_prompt_len.pop(rid, None)
        claimed = self._req_claimed.pop(rid, None)
        self._pending_load.discard(rid)
        if params is None:
            return False, None
        return False, {
            "kvrot": {
                "session_id": params.session_id,
                "stored_tokens": self._session_tokens.get(params.session_id, 0),
                "claimed_tokens": claimed,
            }
        }

    # ==================================================================
    # Worker side
    # ==================================================================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        n_layers = len(self._arch.applies_rope)
        by_idx: dict[int, torch.Tensor] = {}
        for name, tensor in kv_caches.items():
            m = _LAYER_IDX_RE.search(name)
            if m is None:
                raise RuntimeError(f"cannot parse layer index from KV cache name {name!r}")
            by_idx[int(m.group(1))] = tensor
        if sorted(by_idx) != list(range(n_layers)):
            raise RuntimeError(
                f"registered KV caches cover layers {sorted(by_idx)}, expected "
                f"0..{n_layers - 1} (pipeline parallelism is not supported)"
            )
        self._layer_tensors = [by_idx[i] for i in range(n_layers)]
        if self._kv_axis is None:
            self._kv_axis = infer_kv_axis(tuple(self._layer_tensors[0].shape))
        logger.info(
            "KvrotConnector registered %d paged KV tensors, shape=%s, kv_axis=%d, "
            "dtype=%s, store_device=%s",
            n_layers,
            tuple(self._layer_tensors[0].shape),
            self._kv_axis,
            self._layer_tensors[0].dtype,
            self._store_device,
        )

    def _get_store(self, session_id: str, *, create: bool) -> SessionKVStore:
        store = self._stores.get(session_id)
        if store is None:
            if not create:
                raise RuntimeError(
                    f"no KV store for session {session_id!r} on this rank — a "
                    "claim was made for KV that was never saved"
                )
            self._evict_lru_sessions()
            store = SessionKVStore(
                applies_rope=list(self._arch.applies_rope),
                inv_freq=self._inv_freq.to(self._store_device),
                device=self._store_device,
            )
            self._stores[session_id] = store
        self._session_last_used[session_id] = time.monotonic()
        return store

    def _evict_lru_sessions(self) -> None:
        while len(self._stores) >= self._max_sessions:
            victim = min(self._session_last_used, key=self._session_last_used.get)
            logger.warning("KvrotConnector evicting LRU session %r", victim)
            self._stores.pop(victim, None)
            self._session_last_used.pop(victim, None)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, KvrotConnectorMetadata)
        if not metadata.loads:
            return
        if self._layer_tensors is None:
            raise RuntimeError("start_load_kv before register_kv_caches")

        for op in metadata.loads:
            store = self._get_store(op.session_id, create=False)
            if op.plan is not None:
                store.apply_plan(op.plan)
                self.stats["plans_applied"] += 1
            if store.seq_len() < op.num_tokens:
                raise RuntimeError(
                    f"claim of {op.num_tokens} tokens exceeds store "
                    f"({store.seq_len()}) for session {op.session_id!r}"
                )
            store.truncate(op.num_tokens)
            slots = slot_mapping_from_blocks(
                op.block_ids, self._block_size, 0, op.num_tokens
            )
            for i, paged in enumerate(self._layer_tensors):
                inject_tokens(paged, slots, store.layers[i], self._kv_axis)
            self.stats["load_ops"] += 1
            self.stats["loaded_tokens"] += op.num_tokens
            logger.debug(
                "kvrot load: session=%s tokens=%d plan=%s",
                op.session_id,
                op.num_tokens,
                op.plan is not None,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return  # loads are synchronous in start_load_kv

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return  # extraction happens once per step in wait_for_save

    def wait_for_save(self):
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, KvrotConnectorMetadata)
        if not metadata.saves:
            return
        if self._layer_tensors is None:
            raise RuntimeError("wait_for_save before register_kv_caches")

        for op in metadata.saves:
            store = self._get_store(op.session_id, create=True)
            slots = slot_mapping_from_blocks(
                op.block_ids, self._block_size, op.start, op.count
            )
            extracted = [
                extract_tokens(paged, slots, self._kv_axis).to(self._store_device)
                for paged in self._layer_tensors
            ]
            store.append(op.start, extracted)
            self.stats["save_ops"] += 1
            self.stats["saved_tokens"] += op.count
            logger.debug(
                "kvrot save: session=%s range=[%d,%d) store_len=%d",
                op.session_id,
                op.start,
                op.start + op.count,
                store.seq_len(),
            )
