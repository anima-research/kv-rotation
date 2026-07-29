# Results ledger

Distilled findings from `experiments/exp01`–`exp11`, in run order. Each entry is
goal → setup → key numbers → verdict. Full chronological detail (every number,
every caveat) lives in `notes/journal.md` and the per-experiment draft journals
in `notes/`; this file is the summary a new reader should start with.

Two metrics recur throughout:

- **KL(p_full ‖ p_x)** — per-step KL divergence between the next-token
  distribution under the full (unevicted) cache and under condition `x`,
  averaged over a teacher-forced continuation. The behavioural drift metric.
- **Mechanism error, `KL(shortened-recompute ‖ rotation)`** — isolates *how far
  rotation's cheap cache reuse is from a clean recompute of the same shortened
  context*, separate from the information loss of forgetting the evicted block.
  This is the verdict on the mechanism itself (see [`README.md`](README.md) for
  the full decomposition).

## Summary

| # | Experiment | Model | Headline |
|---|---|---|---|
| 01 | Drift vs. full prompt | Llama-3.2-3B | Sink-aware rotation near-exact (KL ~1e-4–1e-3, 100% top-1); dropping sinks is catastrophic (KL 2.6–4.5). |
| 02 | Recall/continuity probe | Llama-3.2-3B | Retained facts survive eviction almost perfectly (−0.025 ≈ full −0.021); evicted facts are faithfully lost. |
| 03 | Long-horizon multi-eviction | Llama-3.2-3B | Drift stays bounded over repeated pops; recompaction caps absolute position for arbitrarily long sessions. |
| 04 | Tier-2 importance eviction | Llama-3.2-3B | At 40% budget, importance-based eviction retains a mid-context fact that age-based eviction loses. |
| 05 | Comparison matrix + decomposition | Llama-3.2-3B | Mechanism faithful (KL 0.014 vs. clean recompute) and **64× cheaper**; rotation ends up *closer to full* than the recompute it replaces. |
| 06 | Trinity smoke test | Trinity-Large-Preview (afmoe, ~389B MoE) | Loads across 8 GPUs; per-layer RoPE gating (45 sliding / 15 NoPE-global) correct; rotation homomorphism holds on real weights. |
| 07 | Trinity eviction matrix at scale | Trinity-Large-Preview | Mechanism near-error-free at 8k/16k real context (mech-KL ≤3e-4); trinity tolerates naive sink-dropping eviction (SWA+NoPE favourable case). |
| 08 / 08b | Trinity recall + clean perf | Trinity-Large-Preview | Retained facts recall ≈ full (100% top-1) at 389B; rotation is 29–51 ms vs. 20–29 s clean GPU-resident recompute (~560–720×, HF naive model-parallelism). |
| 09 | Chat-shaped eval (turn-aligned) | Trinity-Large-Preview | Turn-aligned oldest-turn eviction on templated chat breaks the mechanism (mech-KL 0.41–0.68, ~1000× worse) — traced to *what the reply reads*, not eviction geometry. |
| 10 | Naturalized dialogue | Trinity-Large-Preview | On natural (non-templated) model-generated dialogue, the exp09 pathology collapses ~15× (mech-KL 0.04) — it was the low-entropy chat scaffold, not chat shape itself. |
| 11 | Signature preservation | Llama-3.2-3B | Independent instrument (anamnesis-pl v3, 2,713-dim computational signature) confirms the behavioural finding: recompute drifts *further* from full than rotation does, on both token-KL (12/12 boost cells) and signature distance (confirmatory p=0.026). |

---

## Phase 1 — mechanism validation on Llama-3.2-3B

### exp01 — drift vs. full prompt

**Goal.** Establish the base case: does sink-aware eviction + RoPE re-rotation
track the full-context model, and how catastrophic is naive (sink-dropping)
eviction by comparison?

**Setup.** Per-step KL(p_full ‖ p_rotated) over 32 teacher-forced tokens, top-1
agreement, at 512- and 8192-token contexts, evicting 64–7168 tokens.

**Results.**

| context | evict | oldest (drops sinks) | sink + rotation | top-1 (sink+rot) |
|---|---|---|---|---|
| 512  | 64  | KL 2.60 | **7.3e-4** | **100%** |
| 512  | 256 | KL 4.19 | **3.9e-3** | **100%** |
| 8192 | 1024 | KL 2.76 | **5.9e-5** | **100%** |
| 8192 | 7168 | KL 4.50 | **7.3e-4** | **100%** |

**Verdict.** Dropping attention sinks is catastrophic (KL 2.6–4.5, top-1 down to
34–62%) even on a modern instruction model — reproduces StreamingLLM's finding
via a behavioural KL metric. Sink-aware eviction + re-rotation is 3–4 orders of
magnitude tighter and behaviourally exact under greedy decoding, even after
evicting up to 7/8 of the context.

### exp02 — recall / continuity probe

**Goal.** exp01 shows the *continuation* matches; does *information* survive?

**Setup.** Plant a passcode fact at a known span; measure log-prob and greedy
recall of the answer under full vs. rotated cache, with the fact either
retained or evicted.

**Results.** Full-context reference logprob −0.021. Fact retained + unrelated
context evicted: −0.025 (recalled). Fact inside the evicted span: −15.9 (lost,
correctly). Naive eviction (drops sinks and the fact): −21.2 (lost).

**Verdict.** Retained information survives eviction almost perfectly; dropping
sinks or the fact's own span both fail *correctly* — the honest cost that makes
"evict by importance, not age" non-negotiable.

### exp03 — long-horizon multi-eviction rollout

**Goal.** Does drift blow up over many repeated evictions in a long session?

**Setup.** Context 256, generate 512, budget 320, evict 128/pop, 4 evictions.

**Results.** Mean KL 1.6–1.9e-3 across the whole 512-step / 4-eviction rollout
— same order as a single eviction, not compounding. Recompaction caps absolute
position at ~budget (320) instead of growing unbounded (767 and rising).

**Verdict.** Drift does not compound over repeated pops. Recompaction is what
makes arbitrarily long rolling sessions possible without ever exceeding a
model's RoPE range — recompact by default.

### exp04 — Tier-2 importance-aware eviction

**Goal.** Does an attention-based (H2O-style) importance signal beat purely
age-based eviction when a salient fact sits mid-context?

**Setup.** Context 438, fact planted at ~50%. Three strategies at equal
kept-budget (K=175, 40%): oldest, sink-window, importance.

**Results.**

| strategy (K=175) | mean KL | recall logprob | fact kept? |
|---|---|---|---|
| oldest | 5.96 | −21.4 | no |
| sink_window | 0.81 | −15.5 | no |
| **importance** | **0.75** | **−0.04** | **yes** |

(full-context reference recall logprob: −0.029)

**Verdict.** Importance-based eviction retains the salient mid-context fact
that both age-based strategies drop, at equal budget, while also matching or
beating sink-window on KL. Caveat: absolute KL is content-dependent (richer,
less-redundant text drifts more than exp01's repetitive filler) — relative
ordering is the robust result.

### exp05 — comparison matrix + drift decomposition

**Goal.** Separate *information loss* (is dropping the block OK?) from
*mechanism error* (is cheap reuse as good as a clean recompute?) by adding the
missing baseline: a fresh recompute of the shortened context.

**Setup.** Context 512, evict 128 after 4 sinks, fact planted in the evicted
span.

**Results.**

| comparison | mean KL | meaning |
|---|---|---|
| KL(full ‖ shortened-recompute) | 9.1e-2 | information loss of forgetting the block |
| **KL(shortened ‖ rotation)** | **1.4e-2** | mechanism error — the verdict |
| KL(full ‖ rotation) | 2.7e-2 | total drift |
| KL(full ‖ naive) | 3.41 | drops sinks — catastrophic |

Cost: rotation surgery 18 ms vs. recompute 1164 ms (383 tokens) — **64× faster**.

**Verdict.** The mechanism is faithful (mech-KL 0.014) — and the headline
surprise: **rotation ends up closer to full (0.027) than the clean recompute
does (0.091)**, because survivors still carry the evicted block's baked-in
influence that a fresh recompute discards. For the exact-vs-*full* goal,
rotation isn't just cheaper than recompute — it's better. (Caveat: continuation
was generated by the full model, biasing toward KV-sharing variants; direction
held up under later scrutiny — see exp08b, exp11.)

---

## Phase 2 — Trinity-Large-Preview at scale

Target architecture: **afmoe**, ~389B MoE, 60 layers = 45 sliding-window
(W=4096) + 15 full-attention, θ=10000, no RoPE scaling, GQA 8 KV heads. Source
review found RoPE is applied **only on the sliding layers** — the 15 global
layers are NoPE, so re-rotating them would corrupt keys that were never
rotated. `ArchSpec.applies_rope` is per-layer aware; NoPE layers are dropped
byte-for-byte on eviction, never rotated.

### exp06 — Trinity smoke test

**Goal.** Bring the mechanism up end-to-end on the real 389B target for the
first time.

**Results.** Loads across 8 GPUs; arch confirmed (60 layers, 45 sliding + 15
full, GQA 8, head_dim 128); `applies_rope = 45/60` matches the sliding-layer
count exactly; rotation homomorphism holds on real weights; eviction runs
end-to-end (sink+rot mean-KL 9.3e-4, top-1 100%). Required 5 transformers
4.57→5.3 compatibility shims (documented in `notes/journal.md`, all verified
not to touch model math).

**Verdict.** Mechanism correctness carries to the real target. Small-scale
(256-ctx) sanity only — the matrix at real context length is exp07.

### exp07 — Trinity eviction matrix at scale

**Goal.** First real-long-context, at-scale measurement: real pg19 book text,
8k/16k context spanning the 4096 SWA boundary, evict 60% after 4 sinks.

**Results.**

| length | mech-KL(short‖rot) | info-loss KL(full‖short) | rotation | recompute | speedup |
|---|---|---|---|---|---|
| 8192  | **1.31e-4** | 3.95e-4 | 34 ms | 24.4 s | 721× |
| 16384 | **3.04e-4** | 6.68e-5 | 48 ms | 42.3 s | 890× |

**Verdict.** Mechanism error is the same tiny order as (or below)
irreducible information loss — sub-millinat at 389B on real long context.
Trinity tolerates naive (sink-dropping) eviction at scale (KL 1.2–1.5e-4, same
order as sink-aware) — unlike Llama, confirming the hybrid SWA+NoPE design is
largely free of attention-sink fragility. The "rotation closer to full than
recompute" continuity thesis held at 8k but flipped at 16k — later shown
(exp08b) to be an artifact of disk-offload-inflated recompute timing, not a
real effect. Needle recall wasn't tested here (planted fact fell in the
evicted zone) — that's exp08.

### exp08 / exp08b — Trinity recall preservation + clean performance

**Goal.** exp07's needle was evicted, so it only proved faithful forgetting.
Sweep a needle fact across evicted (depth 0.40) and retained (depth 0.80)
zones; separately, eliminate a GPU-placement disk-offload artifact to get an
honest recompute timing baseline.

**Results (exp08, recall).**

| length | needle zone | recall (full) | recall (rotation) | top-1 |
|---|---|---|---|---|
| 8192  | evicted | −0.13 | −14.20 (correctly lost) | 100% |
| 8192  | **kept** | −0.09 | **−0.12** | 100% |
| 16384 | **kept** | −0.12 | **−0.15** | 100% |

**Results (exp08b, clean decomposition, no disk offload).** Mechanism error
1.4–4.7e-4 across both lengths/zones — same order as exp07. Rotation is
*closer to full than recompute* for evicted content at **both** lengths once
the offload confound is removed (8k: 5.15e-4 vs. 2.47e-3, 4.8×; 16k: 2.01e-5
vs. 3.66e-4, 18×) — exp07's 16k "flip" does not survive a clean measurement.
Warm rotation 29–51 ms vs. GPU-resident recompute 20–29 s → **~560–720×** (HF
naive model-parallelism; a tensor-parallel production engine will narrow this
ratio, though rotation itself stays tens-of-ms regardless).

**Verdict.** Recall is preserved at scale when the fact is retained (≈ full,
100% top-1), faithfully dropped when evicted. The continuity advantage
(rotation > recompute) is real and confirmed at both context lengths once
placement artifacts are controlled. Trinity feasibility is settled on raw
document content; the open question becomes realism of the workload shape.

---

## Phase 3 — chat-shaped realism

### exp09 — turn-aligned chat eviction

**Goal.** All prior evals used raw document continuations. Does the mechanism
hold when eviction is turn-aligned on multi-turn chat and the measured
continuation is the *next assistant reply* rather than doc continuation?

**Setup.** Trinity, 8k/16k context, 12 templated user/assistant turns, evict
oldest turns at various fractions.

**Results.**

| cell | KL(full‖rot) | top-1 | info-loss KL(full‖short) | mech-KL(short‖rot) | rotation | recompute |
|---|---|---|---|---|---|---|
| L≈8922, ef=0.5 | 0.581 | 91% | 0.047 | **0.684** | 44 ms | 31.7 s (718×) |
| L≈17529, ef=0.5 | 0.469 | 78% | 0.158 | **0.414** | 61 ms | 62.1 s (1020×) |

Sweeps across evict-fraction (0.3/0.5/0.7) and turn count (8/12/24) all showed
the same pattern: mechanism error 3–4 orders above raw-doc exp07/08b, scaling
with evicted fraction. Recall stayed clean throughout (16/16 across all cells).

**Diagnosis (same-day follow-up).** KV-geometry analysis found contamination
magnitude near the eviction seam is **nearly identical between raw-doc and
chat regimes** (cosine similarity to a clean recompute within 1–3%). The
divergence is not contamination magnitude — it's *readout*: an assistant reply
is a low-entropy, template-locked, globally-attending operation over the
conversational scaffold (turn headers, acknowledgment boilerplate), which is
exquisitely sensitive to the same near-seam contamination a raw-doc
continuation simply doesn't read. Swapping *content* (chat template ↔ raw
prose, same eviction mask) moved mech-KL ~4000×; swapping *eviction geometry*
moved it ~1.5×.

**Verdict.** Turn-aligned oldest-turn eviction on templated chat is not
deployment-ready as-is — the mechanism itself is unchanged (still bit-exact
CPU math), but reusing survivor KV after mid-context eviction diverges hard
from a clean recompute specifically when the read is the low-entropy chat
scaffold. Motivates Tier-2 importance eviction and Tier-3 selective recompute
of the near-seam survivor band.

### exp10 — naturalized dialogue

**Goal.** exp09 used templated, low-entropy chat turns. Does the mechanism
recover on *natural*, high-entropy model-generated dialogue through the same
chat template and eviction policy?

**Setup.** 8 multi-turn conversations (multiparty, majority model-generated,
non-Trinity content), same turn-aligned oldest-first eviction at ef=0.5, same
facts/probes as exp09, 8k/16k lengths (16 cells).

**Results (mean across 8 conversations).**

| length | KL(full‖rot) | top-1 | info-loss KL(full‖short) | mech-KL(short‖rot) |
|---|---|---|---|---|
| 8192  | 0.032 | 92% | 0.061 | **0.042** |
| 16384 | 0.027 | 91% | 0.047 | **0.039** |

Mechanism error collapsed **~15×** versus exp09's templated-chat numbers, back
to the same order as raw-doc results. In **16/16 cells**, rotation was closer
to full than the clean recompute — the continuity advantage (exp05/07/08b)
reappeared on chat-shaped content once the content was natural. Recall stayed
16/16 clean.

**Verdict.** The exp09 pathology was never about chat *shape* — it was the
low-entropy templated acknowledgment scaffold specifically. On natural
dialogue, turn-aligned oldest-turn eviction + rotation is back in the
deployable regime. (A planned two-party fresh-generation follow-up hit an
unrelated model-serving pathology — deep multi-turn chat-format generation
degenerating into repetition loops — and was deprioritized; the naturalization
finding does not depend on it, since it's already established through
identical-pipeline naturalized content.)

---

## Phase 4 — signature preservation (computational-character crossover)

### exp11 — does rotation preserve *how* the model computes, not just its next-token distribution?

**Goal.** All prior metrics are behavioural (KL, recall). exp11 asks a
different question with an independent instrument: does eviction+rotation
preserve the model's internal computational signature (residual-stream
geometry, attention-routing statistics, gate dynamics — a 2,713-dimensional
feature space from an external interpretability tool, "anamnesis-pl v3") as
well as, or better than, a clean recompute?

**Setup.** Llama-3.2-3B, four cache conditions per cell — FULL / ROT
(evict→rotate) / REC (fresh recompute of the shortened context) / NAIVE
(evict, no positional fix) — compared by signature distance to FULL, on both
freshly-generated 3B-native dialogue (Regime A) and raw documents (Regime B).
12 initial cells, later extended with an independently pre-registered 12-cell
confirmatory set (n=24 total) after the initial result narrowly missed
significance.

**Results — behavioural axis (unanimous).** token-KL(FULL‖REC) >
token-KL(FULL‖ROT) in **24/24 cells** across both grids.

**Results — signature axis.**

| test | initial (n=12) | confirmatory (n=12, independent) | pooled |
|---|---|---|---|
| primary: paired d_ROT < d_REC (Wilcoxon, one-sided) | p=0.125 (not significant; direction favourable 6/8 dialogue) | **p=0.0261** ✅ | p=0.0029 (18/24) |
| scale-free T3 (residual-PCA) distance, ROT<REC | 11/12, p=0.0005 | 12/12, p=0.00024 | 23/24, p≈3e-7 |

**Mechanistic finding.** Whole-signature deviation Δ_NAIVE is colinear with
Δ_ROT in every cell (cos 0.97–0.99) and near-orthogonal to Δ_REC in dialogue —
ROT and NAIVE share the *same cache content* (the evicted context's baked-in
influence, undisturbed by mere key rotation), while REC's recomputed cache has
that influence surgically removed. That is the axis the signature instrument
actually separates on. A follow-up per-feature decomposition found the effect
is not sparse: 531–915 of T3's 1,250 dimensions (Benjamini–Hochberg FDR
q<0.05, depending on n) individually discriminate ROT from REC, concentrated
at first-generated-token, mid-to-late residual-PCA layers — broadly and
redundantly encoded, not carried by a handful of features.

**Verdict.** Confirmed on the pre-registered independent set: eviction +
re-rotation preserves the model's computational character measurably better
than a clean recompute of the shortened context does, corroborating the
behavioural finding (exp05/08b/10) with an orthogonal instrument. The
project's continuity thesis — "rotation keeps what recompute throws away" —
now has evidence at the level of next-token distributions, factual recall,
*and* internal computational signature.

---

## Open threads

- **Tier 3 (selective recompute)** — not yet built. Motivated concretely by
  exp09's diagnosis: recompute the near-seam survivor band (worst-contaminated
  ~7% of keys) rather than the whole shortened prefix.
- **Tier 2 on Trinity** — needs `output_attentions` support in afmoe's forward
  pass; unconfirmed whether it threads through.
- **Production (vLLM/TP) port** — every timing ratio so far (~560–1000×) is HF
  naive model-parallelism (one GPU active at a time); the production-honest
  ratio under tensor parallelism is unmeasured. Rotation's own cost (tens of
  ms, mostly an unoptimized Python loop) is expected to shrink further under a
  fused kernel regardless of the recompute baseline.
- **exp10 stage 2** (clean two-party, Trinity-authored dialogue) — blocked on
  a Trinity chat-format generation pathology (repetition-loop degeneration in
  deep multi-turn chat completions); not required for the naturalization
  finding, which already holds on identical-pipeline naturalized content.
