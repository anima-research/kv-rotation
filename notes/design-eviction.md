# Design: eviction mechanism, reference frames, and the comparison matrix

*Companion to `feasibility.md`. Pins down what we reuse, what "success" is measured against,
the policy choices, and the experiment matrix. Written for alignment (manager-reviewable).*

## 1. What the rotation mechanism actually does

Block `[1..50]` cached. To drop message `[1]`:

1. Keep `[2..50]`'s KV **as-is** (no recompute).
2. Free `[1]`'s KV slots.
3. **Re-rotate** `[2..50]`'s keys so their positions become contiguous again (RoPE `R(−k)`).
4. Prefill only the genuinely new tokens going forward.

vs. the two naive options:
- **toss + recompute-all**: loses continuity *and* re-prefills everything.
- **re-cache `[2..50]`**: a full prefill of the survivors.

Rotation turns an O(K) *prefill* of the survivors into an O(K) *elementwise rotation* of
their keys (no network forward) — that's the whole efficiency win.

## 2. "Dropping [1] is fine" → measure against the right reference

Losing `[1]`'s information is acceptable (often intended). So success is **not** "behave like
the full context"; it's two separable questions, each with its own reference:

| quantity | meaning | want |
|---|---|---|
| **KL(full ‖ shortened-recompute)** | *information loss* from forgetting `[1]` | small **if** `[1]` is stale (policy's job) |
| **KL(shortened-recompute ‖ rotation)** | *mechanism error*: reuse-contamination vs a clean recompute | **≈ 0** (mechanism's job) |
| KL(full ‖ rotation) | total drift (≈ sum of the two) | — |

- **shortened-recompute** = freshly prefilling `[2..50]` (the survivors) — the "correct answer"
  once we've decided `[1]` is gone. Its KV is *clean*.
- **rotation** = the survivors' KV *reused* from the full pass (computed while `[1]` was
  present) and re-rotated — so it carries residual `[1]` *contamination*. KL(shortened ‖
  rotation) is exactly that contamination, and it's the verdict on the mechanism.

The mechanism is a win iff KL(shortened ‖ rotation) ≈ 0 at ≪ recompute cost. Information loss
is a *separate*, policy-controlled budget.

## 3. Eviction policy axis (what to drop)

- **Oldest-first (contiguous, block/message-level)** — drop the oldest block. Retained
  context is a clean shorter transcript `[k..50]`. Simple, interpretable, the default.
- **Importance (token-level, sub-message)** — drop the least-attended tokens anywhere. Keeps
  salient old info (exp04 kept a fact oldest-first lost). Retained context is a *compressed,
  non-contiguous* view.

These are choices on one axis; both use the same evict+re-rotate mechanism.

## 4. Cache ↔ prompt sync (practical consequence)

- **Contiguous oldest-first**: cache stays a clean prefix-ish of the conversation → app sends
  `[k..50]+new`, sync is trivial.
- **Importance**: retained subset is non-contiguous → the model's working memory diverges
  from "the transcript". Fine if the serving layer owns the cache, but it breaks naive prefix
  matching (e.g. vLLM automatic prefix caching) and needs an explicit prompt↔cache map.

## 5. The comparison matrix (exp05+)

| variant | KL vs full | KL vs shortened | recall (fact in dropped span) | tokens recomputed | per-evict latency |
|---|---|---|---|---|---|
| full (no evict) | 0 | — | ✓ | — | — |
| shortened-recompute / no-KV | = info loss | 0 | ✗ (correctly) | K (full prefill) | high |
| **rotation** | ~info loss | **~0 (target)** | matches shortened | **0** | **low** |
| evict, no re-rotate (gap) | ? | ? | matches shortened | 0 | low |
| naive (drop sinks) | catastrophic | catastrophic | ✗ | 0 | low |

Reading: rotation should land on top of shortened-recompute for *quality* while sitting with
the free/low-latency variants on *cost* — i.e. the quality of a recompute at the price of a
rotation. exp05 fills these cells on Llama-3.2-3B.
