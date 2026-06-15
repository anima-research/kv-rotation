# KV-cache rotation under prefix eviction — feasibility synthesis

*Status: Phase 1 underway. Tier-0 rotation proven exact on CPU (`tests/`). exp01 on
Llama-3.2-3B (node1): sink-aware evict + re-rotation is near-exact vs full prompt
(mean KL ~1e-4–1e-3 nats, 100% greedy top-1) while naive popping is catastrophic
(KL 2.6–4.5). See `journal.md` for the running log and numbers.*

## 1. The goal, precisely

Pop the oldest prefix from a running context to reclaim KV space, while keeping the
model's behaviour **as if the full context were still present** (exact-vs-*full*, not
exact-vs-shortened), at ~stable latency / throughput / memory.

The "exact relative to *what*" distinction is load-bearing:

- **Exact vs. the shortened prompt** (as if early tokens never existed): provably needs
  (near-)full recompute, and it *discards* the early information — the opposite of the
  intent.
- **Exact vs. the full prompt** (behaviour unchanged, just reclaim space): the actual
  goal. Not deletion — *lossless-as-possible consolidation*.

## 2. The feasibility law

> When you pop a prefix block `B` and keep the (re-rotated) survivor KV, you preserve
> `B`'s **indirect** influence *exactly* — it is already baked into the survivors' KV at
> every layer. You lose only the **direct** future→`B` attention path. Therefore
>
>   **behavioural drift vs. full-prompt ≈ the future attention mass that would have landed on `B`.**

Consequences:

| Case | Drift | Cost to stay near-exact |
|---|---|---|
| Stale prefix (low future attention) | small | ~free (Tier 0+1) |
| Still-relevant prefix | large | keep it (compress, Tier 4) or recompute (Tier 3) |
| SWA layer, `B` ≥ W before oldest survivor | **zero** | ~free, **provably exact** |

The corollary is the design principle: **evict by importance, not by age** — and never
drop the leading tokens (attention sinks).

## 3. Why bit-exact-vs-full *while reclaiming space* is impossible in general

For a standard RoPE full-attention model, the survivors' K/V at layers ≥1 were computed
by attending over `B`; we keep those (so indirect influence survives), but once `B`'s
keys are freed, **future** queries can no longer attend to `B` directly. Softmax
renormalisation + MLP nonlinearities give no cheap algebraic inverse. So exact-vs-full
holds only where future attention to `B` is exactly zero — which is precisely the SWA
regime. Everywhere else it is an approximation whose error we can *measure and bound*,
not eliminate for free. (No surveyed method — CacheBlend, EPIC, APE… — is exact; all are
empirically quality-preserving. See `references.md`.)

## 4. The mechanism stack

| Tier | Mechanism | Cost | Exactness | Status |
|---|---|---|---|---|
| 0 | RoPE re-rotation (`reindex_keys`) | ~free | **exact** for the positional component | ✅ proven (CPU tests) |
| 1 | sink-aware eviction | ~free | near-exact for stale content | ✅ implemented |
| 2 | importance-aware eviction (H2O/SnapKV) | ~free | widens what's safe to drop | ⏳ Phase 2 |
| 3 | selective recompute (CacheBlend) | compute | drives drift → 0 (empirical) | ⏳ Phase 2/3 |
| 4 | learned consolidation (Gist/Cartridges) | training | best for still-relevant content | ⏳ Phase 4 |

Key numbers from the literature (verified — see `references.md`):

- **Sinks are load-bearing.** Naive window (drop oldest, no sinks): perplexity
  5.40 → **5158** on Llama-2-13B. Keep ~4 sink tokens → stable to **4M tokens**
  (StreamingLLM). The first token can hold **>50%** of attention mass.
- **RoPE re-rotation** = re-index positions *within the cache* (not original text
  positions). Exact for position; in HF/vLLM/SGLang the stored key is **pre-rotated**,
  so we must read → `R(−k)` → write back; V is untouched.
- **Selective recompute** (only needed when drift exceeds budget): CacheBlend ~15% of
  tokens → ≤0.002 F1 loss, 2.2–3.3× TTFT; EPIC ~constant ~20 boundary tokens → 0–7%
  drop. Both built for RAG chunk-fusion, *not* prefix-pop — a literature gap we fill.
- **Consolidation**: Cartridges (per-context KV distillation) 38.6× memory / 26.4×
  throughput at full-ICL quality; Activation Beacon 8× (<9 GPU-h to train, drop-in).

Cost-stability: Tiers 0–2 *reduce* memory and add only a sub-forward-pass rotation →
latency/throughput/memory stable-or-better. Tier 3 is the only cost adder, gated behind
a drift budget.

## 5. Trinity is the favourable case

`Trinity-Large-Preview` (`afmoe`, ~389B MoE): 60 layers, `layer_types = [sliding×3, full]×15`
→ **45 sliding-window (W=4096) + 15 full-attention** layers; RoPE θ=10000, GQA 8 KV
heads, head_dim 128, muP, 256k context.

Implication via §2: on the **45 SWA layers** (75%), once the evicted block is ≥4096
tokens before the oldest survivor, those tokens are outside the window → **exact eviction
for free**. The content-contamination problem is confined to the **15 global layers**, so
Tier 3 recompute (if used) touches only 1/4 of layers. This is materially easier than a
pure full-attention model.

Caveats (verify on node1): SWA "exact" is rigorously *per-layer*; the multi-layer
receptive field (~W·L) means whole-model exactness needs consistent eviction across
layers. muP changes attention-logit scaling — matters when we compute attention scores
ourselves for Tier-2 importance, not for the harness (which uses the model's own forward).

**Llama-3.2-3B** is the hard control: 28 layers, *all* full attention, θ=500k + llama3
scaling, GQA 8 KV. We prove the approximate tiers here first.

## 6. North-star metric

`exact-vs-full` ≡ per-step **KL(p_full ‖ p_rotated)** of the next-token distribution over
a *teacher-forced* continuation (same tokens fed to both, isolating cache effect from
divergent sampling), + top-1 agreement, + a continuity task (early fact needed later).
Plus cost deltas. Target: KL ≈ 0 / agreement ≈ 100% within a drift budget, at ≤ original
cost. Note: the consolidation literature measures drift as perplexity/accuracy, *not*
behavioural/agentic continuity — this metric is a genuine contribution.

## 7. Plan

- **Phase 0** ✅ scaffold + this synthesis + verified refs.
- **Phase 1** (now) HF harness, Tier 0+1, KL-vs-full baselines on Llama-3.2-3B;
  reproduce the sink failure; confirm sink+rerotate keeps KL small.
- **Phase 2** Tier 2 (importance) + Tier 3 (selective recompute); map the drift/cost
  Pareto frontier; characterise "what is safe to evict".
- **Phase 3** port the winner to vLLM (in-place KV mutate API + Δ-rotation kernel + APC
  hash handling) and/or the close-to-metal testbed; validate cost ~stable in-engine.
- **Phase 4** learned consolidation for still-relevant content (Cartridges/Beacon),
  if the Pareto frontier shows it's needed.

## 8. Open questions / risks

- Importance-based "X% budget = full quality" numbers are from *static* prompts; we must
  re-measure under repeated prefix-pop in long rollouts.
- "Pitfalls of KV-cache compression" (arXiv 2510.00231): eviction can make models
  silently ignore instructions / leak system prompts — our metric must catch this, not
  just track perplexity.
- vLLM internals to verify in-repo (Phase 3): exact `attention/layer.py` path, FlashInfer
  read/write path, APC re-hashing after position shift.
- Whether trinity's per-layer SWA exactness holds end-to-end given cross-layer receptive
  field — verify empirically.
