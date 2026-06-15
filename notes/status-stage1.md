# KV-Cache Rotation — Stage 1 Status (2026-06-14)

**TL;DR.** Goal: let a long-running / agentic session drop its oldest context to reclaim
KV-cache space *without* the usual full-cache invalidation, while the model keeps behaving
as if the full context were still present. Stage-1 result: on the hardest model class, the
cheap mechanism (retain a few attention-"sink" tokens + re-rotate positions) reproduces
full-context behavior near-exactly — **100% greedy next-token agreement even after dropping
7/8 of the context** — whereas naive dropping is catastrophic. The approach is
**conditionally feasible and measurable**. The main thing to confirm is the precise
definition of "exact" we're targeting.

## Purpose — please confirm alignment

Today, any change to the prompt prefix invalidates the KV cache from that point on,
forcing a recompute. That breaks prompt caching for exactly the workloads we care about:
rolling context, where we pop early turns to make room for new ones.

The one definition that shapes the whole project — **"exact relative to *what*?"**:

- **Exact vs. the *full* prompt** (behavior unchanged, just reclaim the space) — **this is
  our target.** It matches the "cache = the model's continuative self" framing: keep the
  *influence* of the early context, shrink its footprint.
- **Exact vs. the *shortened* prompt** (as if the early tokens never existed) — provably
  requires a near-full recompute, and it *discards* the early information. Not our goal.

➡ **Confirm:** we are optimizing "no behavioral drift from full context," not "faithfully
forget the dropped tokens."

## What we've found

- **Feasibility law.** Behavioral drift ≈ the future attention the model *would* have paid
  to the dropped tokens. Feasibility is therefore content-dependent: dropping *stale*
  context is near-free; dropping still-referenced context costs. Corollaries: evict by
  *importance*, not age — and never drop the first few tokens (they are disproportionately
  load-bearing "attention sinks"; dropping them is what makes naive eviction blow up).
- **Honest limit.** Literal bit-exactness *while* reclaiming space is provably impossible
  for a standard full-attention model. The tractable, well-defined target is
  **measurably-bounded drift**, and we have the metric for it: per-step divergence of the
  model's next-token distribution vs. the full-context run.
- **Early evidence (Llama-3.2-3B — deliberately the *hard*, all-full-attention case).**
  Sink-aware drop + position re-rotation gives **3–4 orders of magnitude less drift** than
  naive dropping, with **100% greedy next-token agreement** vs. full context after popping
  up to **7/8 of an 8k-token context**. Naive dropping collapses agreement to ~35–60%.
- **Our real target is *easier*.** Trinity is already a hybrid sliding-window model:
  ~**75% of its layers** can drop old context with *zero* error once it falls outside the
  window. The hard part is confined to the remaining ~25% (the global layers).
- **Cost.** The cheap mechanism *reduces* memory and adds only a small position-fix —
  latency/throughput/memory stay flat. Only the "still-relevant content" fallback
  (selective recompute) adds compute, and only when we choose to spend it.

*Caveat on the numbers:* these are early, single-model results on a generic continuation.
We have **not yet probed recall** of facts from the dropped span, so "100% agreement"
should be read as "the continuation matches full context," not yet "remembers everything
that was dropped." That probe is the next step.

## Next steps

1. **Deepen on the 3B (cheap, fast).** (a) recall/continuity probe — can it still answer
   about dropped content?; (b) long-horizon test of *repeated* pops; (c) importance-based
   eviction (drop what's unattended, not merely what's old).
2. **Trinity.** Validate the "~75% of layers free" structure on the real target and handle
   its sliding-window cache.
3. **Serving.** Port the winning mechanism into vLLM (the production stack).

## Decisions where her input would help

- The **definition of "exact"** above (full vs. shortened).
- The **success bar**: what drift / behavioral threshold, on which workloads (e.g. agentic
  rollouts), counts as "good enough"?
- **Eviction policy**: is it acceptable to evict by *importance* (the system decides what's
  stale) rather than strict oldest-first?
- **Cost envelope** for the still-relevant case: how much selective recompute — or one-time
  training to compress old context — is on the table?

---

## The core experiment: removing old context, and which method wins

**The question.** When a long session has to drop its oldest message(s) to free up room, what
is the best way to do it? Today, changing the prefix invalidates the cache, so the only options
are bad. We put the candidate methods head-to-head on the same setup and measured both quality
and cost.

**Methods compared (drop the oldest block, then continue):**
- **Full context** — keep everything. The reference for "ideal behaviour"; not viable (runs
  out of room).
- **Naive drop** — just delete the oldest cache entries and keep going.
- **Recompute / no-cache** — drop the old message and re-process the remaining context from
  scratch. Correct, but a full recompute every turn.
- **Rotation (our method)** — keep the surviving cache as-is, surgically remove the dropped
  message's entries, and cheaply re-index positions. No recompute.
- **Importance-based** — instead of always dropping the oldest, drop the least-used content.

**How we scored it.** We measure how far each method's next-token behaviour is from the
references, split into two separable questions:
- *Information loss* — how much does forgetting the old message change behaviour at all? (A
  policy choice; fine when the message is stale.)
- *Mechanism error* — how far is our cheap rotation from an actual clean recompute? (The
  verdict on whether the shortcut is sound.)
Plus whether the model still recalls planted facts, and the compute cost.

**Results (Llama-3.2-3B, deliberately the hardest case — all-global attention):**
- **The shortcut is sound:** rotation reproduces a clean recompute almost exactly (mechanism
  error ≈ 0.014 nats; identical fact-recall behaviour).
- **It's better than recompute for continuity:** for the "behave as if the full context were
  present" goal, rotation lands *closer to full* (≈0.027) than a recompute does (≈0.091) —
  because the surviving cache still carries the dropped message's influence, which a recompute
  throws away. (Measured on a continuation produced by the full model; a neutral-source rerun
  is queued to size this precisely — the direction is robust.)
- **It's ~64× cheaper** than recomputing, with zero tokens reprocessed.
- **Naive drop is catastrophic** (≈100–1000× worse) — you must keep a few "anchor" tokens and
  be deliberate about what you drop.
- **Importance-based dropping** preserves a specific buried fact that age-based dropping loses,
  at the same budget — i.e. dropping by relevance beats dropping by age.
- On recall: the model loses the dropped message's *verbatim* details (exactly as a recompute
  would) but retains its *influence* on the ongoing conversation.

**Why this is the important bit.** This is the head-to-head that validates the whole premise:
you can roll the context forward by removing old messages and *reuse* the existing cache —
getting recompute-quality (better, for continuity) at a tiny fraction of the cost — instead of
either losing continuity (naive) or paying a full recompute every turn. It is the concrete
evidence that "accurate KV rotation" is real and worth building into the serving stack.

---

## Addendum (2026-06-14): pre-flight review of the trinity architecture

**What we did.** Before spending GPU time running the rotation method on Trinity-Large-Preview,
we read trinity's actual model implementation (the `afmoe` code) to ground the mechanism in how
the model really works, rather than carrying over assumptions from the Llama prototype.

**What we found.** Trinity applies rotary position encoding (RoPE) **only on its sliding-window
(local) layers**. Its 15 global (full-attention) layers use **no positional encoding at all**
(NoPE). This is a deliberate, increasingly common design for long-context models — but it
differs from Llama, where *every* layer is position-encoded.

**Why it mattered.** Our mechanism re-rotates the positions of the surviving cache entries when
we evict. That is correct for position-encoded layers — but applying it to the NoPE global
layers would have rotated vectors that were never rotated to begin with, silently corrupting
them. On Llama this never surfaces (all layers are RoPE); on trinity it would have produced
unexplained quality degradation that is painful to diagnose after the fact. Catching it by
reading the source cost ~an hour and avoided a likely multi-day debugging detour on expensive
hardware.

**The fix (built and unit-tested, no GPU needed).** Re-rotation is now per-layer-aware: it
rotates only the layers that were actually position-encoded and leaves the NoPE layers' cache
untouched. Covered by a dedicated test; the full local suite is green.

**Why this is good news for trinity — it makes the target easier, not harder:**
- The 15 global (NoPE) layers need **no** positional correction on eviction — we simply drop
  the stale entries.
- The 45 sliding layers re-rotate cheaply, and anything evicted beyond their 4096-token window
  is provably exact (the model cannot attend to it anyway).
- So the only approximation lives in the ~25% of layers that are global, and even there it is
  pure, well-understood content drift — not a positional artifact.

We also confirmed trinity uses a standard KV cache with mask-based windowing, so no bespoke
cache infrastructure is required.

**Methodology note.** This is the pattern we'll keep: read the model's source of truth before
committing compute, so each experiment tests the *idea* rather than an incorrect assumption
about the model.
