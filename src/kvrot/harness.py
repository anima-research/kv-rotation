"""Rolling-context measurement harness (run on node1, GPU).

Measures the north-star metric: per-step KL of the next-token distribution between the
**full-context** cache (reference) and a **rotated** cache, on a matched (teacher-forced)
continuation. This isolates the cache-surgery effect from divergent sampling.

Pipeline per trial:
  1. Prefill ``context[:-1]`` once → snapshot ``S0`` (the full cache).
  2. Reference: greedy-decode N tokens from ``S0`` → continuation ``C_ref`` + logits.
  3. For each eviction variant: surgery on ``S0`` (evict + reindex), then teacher-force
     ``C_ref`` through the rotated cache → logits. Compare to reference.

Requires the ``hf`` extra (transformers/accelerate). Lazy-imported so the CPU tests
never need it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import ArchSpec, EvictionSpec
from .eviction import compute_keep_indices, new_positions_for
from .metrics import DriftReport
from .rope import assert_rotation_homomorphism
from .snapshot import KVSnapshot, evict, from_hf_cache, reindex, to_hf_dynamic_cache


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    arch: ArchSpec
    inv_freq: Tensor
    device: torch.device


def _ensure_default_rope_init() -> None:
    """Remote-code models written for transformers 4.x (afmoe) look up
    ``ROPE_INIT_FUNCTIONS['default']``, which 5.x removed (standard RoPE moved to a
    class-based system). Register a standard default matching 4.x semantics. Correctness is
    independently verified: the harness pulls inv_freq off the live model and asserts the
    rotation homomorphism at load."""
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:
        return
    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _default(config, device=None, seq_len=None, **kwargs):
        base = float(getattr(config, "rope_theta", 10000.0))
        dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device).float() / dim)
        )
        return inv_freq, 1.0  # (inv_freq, attention_scaling)

    ROPE_INIT_FUNCTIONS["default"] = _default


def load_model(
    path: str,
    *,
    dtype=torch.bfloat16,
    device: str | None = None,
    device_map: str | None = None,
    attn_implementation: str | None = None,
    max_memory_frac: float | None = None,
) -> LoadedModel:
    """Load a model + tokenizer. Single-GPU by default (device_map=None → .to(device));
    pass device_map='auto' for sharded models like trinity. Pass
    attn_implementation='eager' when you need output_attentions (Tier-2 importance).
    ``max_memory_frac`` (device_map='auto' only) is the fraction of each GPU's *free* memory
    to budget for weights (default 0.92); the rest is left for activations + KV."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    # Remote-code models written for older transformers (afmoe targets 4.57) may read config
    # attributes that transformers 5.x no longer defaults — and 5.x now *raises* AttributeError
    # on unset attrs instead of returning None. Backfill the ones modeling code reads directly.
    for _attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        try:
            getattr(cfg, _attr)
        except AttributeError:
            setattr(cfg, _attr, None)
    # RoPE compat: transformers 5.x routes rope_type "default" through a module method
    # (compute_default_rope_parameters) that 4.x remote-code rotary classes (afmoe) don't define,
    # breaking both __init__ and _init_weights. Re-express plain RoPE as linear scaling with
    # factor 1.0 — identical math (inv_freq unchanged, scaling 1.0) but a path 5.x supports
    # everywhere. Only applied when the model is genuinely plain RoPE (not llama3/yarn/etc.).
    _rs = getattr(cfg, "rope_scaling", None)
    _rtype = _rs.get("rope_type", _rs.get("type")) if isinstance(_rs, dict) else None
    if _rs is None or _rtype in (None, "default"):
        cfg.rope_scaling = {"rope_type": "linear", "factor": 1.0}
    arch = ArchSpec.from_hf_config(cfg)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    _ensure_default_rope_init()  # backfill ROPE_INIT_FUNCTIONS['default'] for 4.x remote code
    extra = {"attn_implementation": attn_implementation} if attn_implementation else {}
    if isinstance(device_map, str) and device_map not in ("cpu", "") and torch.cuda.is_available():
        # Cap per-GPU weight budget from *free* memory (not total). On a shared box other users
        # may already hold VRAM; a fixed fraction of *total* over-promises on a contended GPU,
        # so accelerate spills the unplaceable shards to disk (the exp07 slowdown) or OOMs the
        # forward. Sizing from free adapts to contention and leaves (1-frac) of free for
        # activations + KV. A *lower* frac can also fix accelerate imbalance: a high ceiling lets
        # it overload the last GPU and then offload to disk while early GPUs sit half-empty;
        # capping nearer the even-split-per-GPU forces a balanced fit. Run nvidia-smi first.
        frac = 0.92 if max_memory_frac is None else max_memory_frac
        mm = {}
        for i in range(torch.cuda.device_count()):
            free_i, _total_i = torch.cuda.mem_get_info(i)
            mm[i] = int(free_i * frac)
        extra["max_memory"] = mm
        print("[load] max_memory/GPU (GiB): " + ", ".join(f"{i}:{mm[i] / 2**30:.0f}" for i in mm))
    # Pass our patched cfg so the backfilled attrs take effect. transformers 5.x renamed
    # torch_dtype -> dtype; fall back for older versions.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, config=cfg, dtype=dtype, trust_remote_code=True, device_map=device_map, **extra
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            path, config=cfg, torch_dtype=dtype, trust_remote_code=True, device_map=device_map, **extra
        )
    model = model.eval()
    if device_map is None:
        model = model.to(dev)
    inv_freq = _find_inv_freq(model, arch).to(dev)
    assert_rotation_homomorphism(inv_freq)  # fail loudly if re-rotation would be inexact
    return LoadedModel(model=model, tokenizer=tok, arch=arch, inv_freq=inv_freq, device=dev)


def _find_inv_freq(model, arch: ArchSpec) -> Tensor:
    """Prefer the live model's own ``inv_freq`` buffer; fall back to rebuilding from config."""
    for name, buf in model.named_buffers():
        if name.endswith("inv_freq"):
            return buf.detach().float().cpu()
    try:
        return arch.rope.build_inv_freq()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not locate inv_freq on the model nor rebuild it from config; "
            "re-rotation needs the exact frequencies the model rotates with."
        ) from e


@torch.no_grad()
def _greedy_decode(lm: LoadedModel, prefill_ids: Tensor, last_tok: Tensor, n_steps: int):
    """Decode ``n_steps`` greedily from a fresh full cache. Returns (cont_ids[1,N], logits[N,V])."""
    from transformers import DynamicCache

    cache = DynamicCache()
    L = prefill_ids.shape[1]
    out = lm.model(input_ids=prefill_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values

    cur = last_tok
    pos = L  # last_tok sits at absolute position L (prefill held 0..L-1)
    cont, logits = [], []
    for _ in range(n_steps):
        out = lm.model(
            input_ids=cur,
            past_key_values=cache,
            use_cache=True,
            position_ids=torch.tensor([[pos]], device=lm.device),
            cache_position=torch.tensor([pos], device=lm.device),
        )
        cache = out.past_key_values
        step_logit = out.logits[:, -1, :].float()
        nxt = step_logit.argmax(-1, keepdim=True).to(lm.device)  # logits may be on lm_head's shard
        logits.append(step_logit)
        cont.append(nxt)
        cur = nxt
        pos += 1
    return torch.cat(cont, dim=1), torch.cat(logits, dim=0)


@torch.no_grad()
def _teacher_forced_logits(lm: LoadedModel, snapshot: KVSnapshot, probe_ids: Tensor) -> Tensor:
    """Logits predicting each probe token, conditioned on the rotated ``snapshot``."""
    cache = to_hf_dynamic_cache(snapshot)
    start_pos = snapshot.next_position()
    start_cache = snapshot.seq_len()
    n = probe_ids.shape[1]
    out = lm.model(
        input_ids=probe_ids,
        past_key_values=cache,
        use_cache=True,
        position_ids=torch.arange(start_pos, start_pos + n, device=lm.device).unsqueeze(0),
        cache_position=torch.arange(start_cache, start_cache + n, device=lm.device),
    )
    return out.logits[0].float()  # [N, V]


@torch.no_grad()
def prefill_snapshot(lm: LoadedModel, context_ids: Tensor) -> KVSnapshot:
    """Prefill a context and return its full-cache snapshot (positions 0..C-1)."""
    from transformers import DynamicCache

    context_ids = context_ids.to(lm.device)
    out = lm.model(input_ids=context_ids, past_key_values=DynamicCache(), use_cache=True)
    c = context_ids.shape[1]
    return from_hf_cache(
        out.past_key_values,
        positions=torch.arange(c, device=lm.device),
        layer_types=lm.arch.layer_types,
        applies_rope=lm.arch.applies_rope,
    )


@torch.no_grad()
def prefill_with_importance(lm: LoadedModel, context_ids: Tensor) -> tuple[KVSnapshot, Tensor]:
    """Prefill with attention capture; return (snapshot, per-token importance).

    Importance = total attention *received* by each token, summed over all queries, heads
    and layers (the H2O accumulated-attention signal). Needs the model loaded with
    ``attn_implementation='eager'``. Note the known recency/causal bias (early tokens have
    more potential receivers) and that this is *reactive* — it reflects what was attended
    during prefill, not what a future query will need.
    """
    from transformers import DynamicCache

    context_ids = context_ids.to(lm.device)
    out = lm.model(
        input_ids=context_ids,
        past_key_values=DynamicCache(),
        use_cache=True,
        output_attentions=True,
    )
    if not getattr(out, "attentions", None):
        raise RuntimeError(
            "model returned no attentions; load with attn_implementation='eager'"
        )
    c = context_ids.shape[1]
    snap = from_hf_cache(
        out.past_key_values,
        positions=torch.arange(c, device=lm.device),
        layer_types=lm.arch.layer_types,
        applies_rope=lm.arch.applies_rope,
    )
    imp = torch.zeros(c, device=lm.device, dtype=torch.float32)
    for a in out.attentions:  # each [B, heads, q, k]
        imp += a[0].sum(dim=(0, 1)).float()  # attention received per key token -> [c]
    return snap, imp


@torch.no_grad()
def answer_logprob(
    lm: LoadedModel, snap: KVSnapshot, suffix_ids: Tensor, target_ids: Tensor
) -> tuple[float, bool]:
    """Total log-prob the (rotated) cache assigns to ``target_ids`` after ``suffix_ids``,
    and whether greedy decoding would reproduce the target exactly. Recall metric."""
    suffix_ids = suffix_ids.to(lm.device)
    target_ids = target_ids.to(lm.device)
    k = target_ids.shape[1]
    feed = suffix_ids if k == 1 else torch.cat([suffix_ids, target_ids[:, :-1]], dim=1)
    logits = _teacher_forced_logits(lm, snap, feed)  # [feed_len, V]
    m = suffix_ids.shape[1]
    tgt_logits = logits[m - 1 : m - 1 + k]  # [k, V]
    logp = torch.log_softmax(tgt_logits, dim=-1)
    tgt = target_ids[0]
    total = float(logp.gather(1, tgt.unsqueeze(1)).sum().item())
    greedy_match = bool((tgt_logits.argmax(-1) == tgt).all().item())
    return total, greedy_match


@torch.no_grad()
def rolling_replay(
    lm: LoadedModel,
    context_ids: Tensor,
    feed_tokens: Tensor,
    *,
    budget: int,
    evict_block: int,
    sinks: int,
    recompact: bool,
) -> tuple[Tensor, int, int]:
    """Replay ``feed_tokens`` through a bounded cache that evicts when it exceeds ``budget``.

    Simulates a real rolling session: every time the cache hits ``budget`` we pop
    ``evict_block`` post-sink tokens and re-rotate. Returns (per-step logits [G, V],
    max absolute position reached, number of evictions). Teacher-forced, so it can be
    compared step-for-step against a full-context reference fed the same tokens.
    """
    from transformers import DynamicCache

    context_ids = context_ids.to(lm.device)
    feed_tokens = feed_tokens.to(lm.device)
    prefill_ids = context_ids[:, :-1]
    cm1 = prefill_ids.shape[1]
    out = lm.model(input_ids=prefill_ids, past_key_values=DynamicCache(), use_cache=True)
    cache = out.past_key_values
    positions = torch.arange(cm1, device=lm.device)
    cache_len, pos_next, max_pos, n_evict = cm1, cm1, cm1, 0

    logits_out = []
    for t in range(feed_tokens.shape[1]):
        if cache_len >= budget:
            snap = from_hf_cache(
                cache,
                positions=positions,
                layer_types=lm.arch.layer_types,
                applies_rope=lm.arch.applies_rope,
            )
            keep = compute_keep_indices(
                cache_len,
                EvictionSpec(policy="sink_window", num_sink_tokens=sinks, evict_count=evict_block),
            ).to(lm.device)
            new_pos = new_positions_for(keep, positions, recompact=recompact)
            snap = reindex(evict(snap, keep), new_pos, lm.inv_freq)
            cache = to_hf_dynamic_cache(snap)
            positions = new_pos
            cache_len = int(keep.shape[0])
            pos_next = int(positions.max().item()) + 1
            n_evict += 1

        out = lm.model(
            input_ids=feed_tokens[:, t : t + 1],
            past_key_values=cache,
            use_cache=True,
            position_ids=torch.tensor([[pos_next]], device=lm.device),
            cache_position=torch.tensor([cache_len], device=lm.device),
        )
        cache = out.past_key_values
        logits_out.append(out.logits[:, -1, :].float())
        positions = torch.cat([positions, torch.tensor([pos_next], device=lm.device)])
        cache_len += 1
        pos_next += 1
        max_pos = max(max_pos, pos_next)

    return torch.cat(logits_out, dim=0), max_pos, n_evict


@torch.no_grad()
def run_trial(
    lm: LoadedModel,
    context_ids: Tensor,
    *,
    n_steps: int = 32,
    variants: dict[str, EvictionSpec] | None = None,
) -> dict[str, DriftReport]:
    """One rolling-context trial. ``context_ids`` is ``[1, L]`` on ``lm.device``."""
    if variants is None:
        variants = {
            "oldest (drops sinks)": EvictionSpec(policy="oldest", evict_count=64),
            "sink+rerotate": EvictionSpec(policy="sink_window", num_sink_tokens=4, evict_count=64),
        }

    context_ids = context_ids.to(lm.device)
    prefill_ids, last_tok = context_ids[:, :-1], context_ids[:, -1:]
    L = prefill_ids.shape[1]

    # 1) reference continuation (full cache)
    c_ref, ref_logits = _greedy_decode(lm, prefill_ids, last_tok, n_steps)

    # snapshot of the full prefill cache for surgery (re-prefill to get a clean copy)
    from transformers import DynamicCache

    base = lm.model(input_ids=prefill_ids, past_key_values=DynamicCache(), use_cache=True)
    s0 = from_hf_cache(
        base.past_key_values,
        positions=torch.arange(L, device=lm.device),
        layer_types=lm.arch.layer_types,
        applies_rope=lm.arch.applies_rope,
    )

    # probe = inputs that should predict C_ref (teacher forcing)
    probe = torch.cat([last_tok, c_ref[:, :-1]], dim=1)

    reports: dict[str, DriftReport] = {}
    for name, spec in variants.items():
        keep = compute_keep_indices(L, spec).to(lm.device)
        new_pos = new_positions_for(keep, s0.positions[0], recompact=spec.recompact)
        snap = reindex(evict(s0.clone(), keep), new_pos, lm.inv_freq)
        rot_logits = _teacher_forced_logits(lm, snap, probe)
        reports[name] = DriftReport.from_logits(ref_logits, rot_logits, label=name)
    return reports
