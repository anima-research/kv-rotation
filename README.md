# kvrot — accurate KV-cache rotation under prefix eviction

**Problem.** In long-running or agentic LLM serving you want a *rolling
context*: pop the oldest turns to make room for new ones. But a transformer's
KV cache is positionally and content-bound to the original prefix, so today
any prefix change invalidates it and forces a recompute. That kills prompt
caching for exactly the workload that needs it most.

**Goal.** *Exact-vs-full-prompt* behaviour while reclaiming KV space, at
~stable latency / throughput / memory. After popping a prefix block, the model
should behave as if the full context were still present — not as if the early
tokens had been deleted. (This is deliberately *not* the same target as
"behave as if the early tokens never existed," which just needs a recompute
and discards information we'd rather keep.)

**The feasibility law that organises the project:**

> When you pop a prefix block `B` and keep the (re-rotated) survivor KV, you
> preserve `B`'s **indirect** influence exactly — it's already baked into the
> survivors' keys/values. You only lose the **direct** future→`B` attention
> path. So:
>
> **behavioural drift vs. full-prompt ≈ the future attention mass that would
> have landed on `B`.**

Consequences: a stale prefix (low future attention) is near-exact to drop and
cheap; a still-relevant prefix must be kept (compressed) or selectively
recomputed; and on sliding-window-attention layers, a block evicted beyond the
window is provably exact to drop, for free.

See [`notes/feasibility.md`](notes/feasibility.md) for the literature synthesis
and [`notes/references.md`](notes/references.md) for citations.

## The mechanism stack (cheap → expensive)

| Tier | Mechanism | Cost | Status |
|---|---|---|---|
| 0 | RoPE re-rotation of survivor keys | ~free | ✅ proven bit-exact |
| 1 | sink-aware eviction (keep first N tokens) | ~free | ✅ |
| 2 | importance-aware eviction (H2O accumulated attention) | ~free | ✅ on Llama; not yet ported to the MoE target |
| 3 | selective recompute (CacheBlend-style, high-deviation survivors) | compute | not yet built |
| 4 | learned consolidation (Gist/Cartridges-style) | training | not yet built |

## North-star metric

`exact-vs-full` is operationalised as the per-step **KL(p_full ‖ p_rotated)**
of the next-token distribution over a teacher-forced continuation, plus top-1
agreement and a factual-recall probe — measured against the *full-context*
cache, not a shortened-prompt recompute. We separate **information loss**
(`KL(full ‖ shortened-recompute)`, the cost of forgetting) from **mechanism
error** (`KL(shortened-recompute ‖ rotation)`, the cost of the cheap-reuse
trick itself) — the latter is the honest verdict on whether rotation is doing
its job.

## Status

Validated on **Llama-3.2-3B** (full-attention, the harder case) and at scale
on a real ~389B hybrid sliding-window MoE target: the mechanism is faithful
(mechanism error at or below irreducible information loss in every raw-content
setting measured), continuity-preserving (closer to the full-context
distribution than a clean recompute is, both behaviourally and — per an
independent computational-signature instrument — internally), and orders of
magnitude cheaper than the recompute it replaces.

Turn-aligned eviction on templated multi-turn chat exposed a real failure mode
(low-entropy chat scaffolding reads near-seam contamination the mechanism
otherwise hides) that recovers on naturalized dialogue — see the ledger below
for the full story, including the open threads it motivates (selective
recompute, importance-aware eviction on the MoE target).

**The serving-stack port landed (exp12):** an out-of-tree vLLM v1 KV connector
(`src/kvrot_vllm/`, zero vLLM source changes) runs the rotation inside vLLM at
tensor-parallel scale — rotated-vs-HF-oracle parity **below cross-stack kernel
noise** on the 389B target (TP=8, 8k ctx), a rotated turn costing ~1 s wall on
a stack decoding at ~100 tok/s. Remaining production work is in-place paged
rotation (no extract/re-inject round trip).

**→ [`RESULTS.md`](RESULTS.md) is the full results ledger**, one entry per
experiment (exp01–exp12): goal, setup, numbers, verdict.
**→ [`docs/API.md`](docs/API.md)** — the cache-control API contract (raw vLLM
`kv_transfer_params` protocol + the playground REST API).

## Layout

```
src/kvrot/
  rope.py        # Tier 0: exact RoPE re-rotation of stored (pre-rotated) keys
  snapshot.py     # KVSnapshot + cache surgery (evict / reindex), HF adapters
  eviction.py     # eviction policies -> keep-indices + new positions
  metrics.py      # KL-vs-full, top-1 agreement, drift reports
  config.py       # strongly-typed (pydantic) arch / rope / eviction specs
  harness.py      # real-model rolling-context experiment runner (GPU)
  chat.py         # turn-aligned eviction: chat-template turn spans, synthesis
  natural.py      # naturalized-dialogue eval support, device-placement helpers
  sigbridge.py    # cached-replay bridge into an external signature-extraction
                   # instrument (exp11) — computational-character comparison
src/kvrot_vllm/
  core.py         # vLLM-free connector logic: session KV store, plan
                   # application, slot math, paged-layout adapters (CPU-tested)
  connector.py    # out-of-tree vLLM v1 KV connector (KvrotConnector):
                   # save -> evict+re-rotate (per TP rank) -> inject
tests/            # CPU correctness tests (rotation math, surgery, eviction,
                   # metrics, config, data, chat, sigbridge, vllm-core) — no
                   # model needed
experiments/      # runnable measurement scripts (exp01–exp12, GPU-backed)
notes/            # feasibility synthesis, design docs, chronological journals
```

## Quickstart (CPU — proves the rotation math)

```bash
uv sync --extra dev
uv run pytest      # correctness tests: rotation math, cache surgery, eviction,
                    # chat turn-spans, config, metrics — no GPU/model required
```

## Real-model runs (GPU)

`experiments/exp01`–`exp11` are the runnable measurement scripts; each nominal
`expNN_*.py` file corresponds to a ledger entry in `RESULTS.md`. They expect a
CUDA-capable host with the `hf` (and, for exp11, `sig`) extras installed:

```bash
uv pip install -e ".[hf,dev]"
python experiments/exp01_rotation_drift.py \
    --model <path-or-hf-id> --context-len 512 --gen 32 --evict 64 128 256
```

GPU-host-specific setup (shared-machine constraints, exact model paths, sync
workflow) is kept in local, untracked notes rather than this repo.
