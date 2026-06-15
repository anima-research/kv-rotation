# kvrot — accurate KV-cache rotation under prefix eviction

> **New here — including a fresh agent picking this up?** Start with **[HANDOFF.md](HANDOFF.md)**:
> the idea + reframing, node1 setup, code map, results so far, gotchas, and next steps.

**Problem.** In long-running / agentic serving you want a *rolling context*: pop the
oldest turns to make room for new ones. But the KV cache is positionally and
content-bound to the original prefix, so today any prefix change invalidates it and
forces a recompute. We own the serving infra, so we can do better.

**Goal (pinned).** *Exact-vs-full-prompt* behaviour while reclaiming KV space, at
**~stable latency / throughput / memory**. I.e. after popping a prefix block, the
model should behave as if the full context were still present — not as if the early
tokens had been deleted.

**The feasibility law that organises everything:**

> When you pop a prefix block `B` and keep the (re-rotated) survivor KV, you preserve
> `B`'s **indirect** influence *exactly* (it is already baked into the survivors' KV).
> You only lose the **direct** future→`B` attention path. So
>
>   **behavioural drift vs. full-prompt ≈ the future attention mass that would have landed on `B`.**

- Stale prefix (low future attention) → near-exact, cheap. Feasible.
- Still-relevant prefix → must keep it (compressed) or selectively recompute. Costs.
- Sliding-window layers, `B` ≥ W before the oldest survivor → **provably exact, free.**

See [`notes/feasibility.md`](notes/feasibility.md) for the full research synthesis and
[`notes/references.md`](notes/references.md) for verified citations.

## The mechanism stack (cheap → expensive)

| Tier | Mechanism | Cost | Exactness |
|---|---|---|---|
| 0 | RoPE re-rotation (position fix) | ~free | exact for the positional component |
| 1 | sink-aware eviction | ~free | near-exact for stale content |
| 2 | importance-aware eviction | ~free | widens what's safe to drop |
| 3 | selective recompute (CacheBlend-style) | compute | drives drift → 0, empirical |
| 4 | learned consolidation (Gist/Cartridges) | training | best for still-relevant content |

## North-star metric

`exact-vs-full` is operationalised as the per-step **KL(p_full ‖ p_rotated)** of the
next-token distribution over a continuation, plus top-1 agreement, plus a continuity
task — measured against the *full-context* cache (not a shortened-prompt recompute).

## Layout

```
src/kvrot/
  rope.py        # Tier 0: exact RoPE re-rotation of stored (pre-rotated) keys
  snapshot.py    # KVSnapshot + cache surgery (evict / reindex), HF adapters
  eviction.py    # eviction policies -> keep-indices + new positions
  metrics.py     # KL-vs-full, top-1 agreement, drift reports
  config.py      # strongly-typed (pydantic) arch / rope / eviction specs
  harness.py     # real-model rolling-context experiment (run on node1)
tests/           # CPU correctness tests (rotation math + surgery), no model needed
experiments/     # runnable measurement scripts (node1)
notes/           # feasibility synthesis + references
```

## Quickstart (dev box, CPU — proves the rotation math)

```bash
uv sync --extra dev
uv run pytest
```

## Real-model runs (node1, GPU)

See [`NODE1.md`](NODE1.md). Targets: `llama-3.2-3b-instruct` (hard case, full attn)
then `Trinity-Large-Preview` (hybrid SWA, the eventual goal).
