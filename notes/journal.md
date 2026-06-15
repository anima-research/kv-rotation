# Research journal — KV-cache rotation

Running log of what we did, why, and what we found. Newest entries at the bottom.

---

## 2026-06-14 — Kickoff, scoping, research

**What.** Manager brought the idea: make prefix eviction in a rolling/agentic context
*not* invalidate the KV cache. Scoped it with the user:
- Goal = **exact-vs-full-prompt** (behave as if full context present), not
  exact-vs-shortened. ~stable latency/throughput/memory.
- RoPE models first (generalise later). Contiguous prefix pop. Map all 3 training
  regimes (inference-only / light FT / full).
- Substrate: HF transformers prototype → vLLM / close-to-metal testbed. Final target:
  `Trinity-Large-Preview`.

**Why it's non-trivial.** Re-rotation (the literal "rotation") is cheap and exact for
the *positional* component, but survivor KV at layers ≥1 is *content-contaminated* by the
evicted tokens. So bit-exact-vs-full while reclaiming space is provably impossible for
general full-attention RoPE; the tractable target is *measurably-bounded* drift.

**Research.** Ran a 5-lane literature + systems sweep (see `references.md`). Key takeaways:
- Sinks are load-bearing: naive window (drop oldest, no sinks) → ppl 5.4→5158; keep ~4
  sinks → stable to millions of tokens (StreamingLLM). First token can hold >50% mass.
- No published method is *exact*; CacheBlend/EPIC are quality-preserving via selective
  recompute (~15% / ~20 boundary tokens). All target RAG chunk-fusion, not prefix-pop.
- Consolidation (Cartridges 38.6× / Activation Beacon 8×) is the training-based path.
- Nobody addresses RoPE re-indexing of evicted prefixes, and everyone measures drift as
  perplexity, not behavioural KL-vs-full. **Both gaps are ours to fill.**

**Formalised the feasibility law:**
> drift vs full-prompt ≈ the future attention mass that would have landed on the evicted
> block. Stale prefix → near-exact & cheap; still-relevant → compress/recompute; SWA
> layer with block outside the window → provably exact & free.

---

## 2026-06-14 — Target architectures

Read configs on node1 `/models/`:
- **Llama-3.2-3B** (starter / *hard* case): 28 layers, **all full-attention**, θ=500k +
  llama3 scaling (eff. 131k), GQA 8 KV heads, head_dim 128.
- **Trinity-Large-Preview** (final goal, *favourable* case): `afmoe` ~389B MoE, 60 layers,
  `[sliding×3, full]×15` → **45 sliding-window(4096) + 15 full** layers, θ=10000,
  **no scaling**, GQA 8 KV, muP, 256k context.

**Implication:** on trinity's 45 SWA layers, once the evicted block is ≥4096 before the
oldest survivor it's outside the window → **exact eviction for free (75% of layers)**.
Contamination confined to 15 global layers. Trinity is *easier* than the 3B for exactness.

---

## 2026-06-14 — Phase 0/1 scaffold + Tier 0 proof (dev box, CPU)

Built the `kvrot` package: `rope` (re-rotation), `snapshot` (cache surgery),
`eviction` (policies), `metrics` (KL-vs-full), `config` (pydantic), `harness` (real-model).

**Proved Tier 0 exact on CPU** (`tests/`, 14/14): the full surgery
(sink-aware evict → recompact → re-rotate) yields keys **bit-identical (atol 1e-5)** to
rotating the original keys directly at the new positions. The rotation-homomorphism check
guards against dynamic-NTK frequencies that would make re-rotation inexact.

---

## 2026-06-14 — node1 setup

- Disk: home `/` 114 GB free (our upload <1 MB); `/models` is 100% full but we never
  write there.
- Shared venv `~/luxi-files/.venv-shared`: py3.12, **torch 2.11+cu128, transformers 5.3.0,
  accelerate 1.13**, 8× **idle B200** (183 GB each). pytest absent → we run via
  `PYTHONPATH=src`; the rotation math is already proven and is re-checked at model load.
- **transformers 5.x breaking change:** `cache.key_cache[i]` → `cache.layers[i].keys`
  (layers also carry `is_sliding` — useful for trinity). Patched `snapshot._extract_kv`
  and the loader (`dtype=` kwarg, single-GPU load). CPU tests still 14/14.
- Workflow: edit locally → `rsync` to `~/luxi-files/kv-rotation` → run with the shared
  venv on `CUDA_VISIBLE_DEVICES=1`. Never install into the shared venv.

---

## 2026-06-14 — exp01: drift vs full prompt (Llama-3.2-3B, the hard case)

Metric: per-step KL(p_full ‖ p_rotated) over 32 teacher-forced tokens; top-1 agreement.

**512-token context:**

| evict | oldest (drops sinks) | sink+rot | top1 (sink+rot) |
|---|---|---|---|
| 64  | KL 2.60, top1 56% | **7.3e-4** | **100%** |
| 128 | KL 3.50, top1 53% | **1.7e-3** | **100%** |
| 256 | KL 4.19, top1 41% | **3.9e-3** | **100%** |

**8192-token context:**

| evict | oldest | sink+rot | sink,no-recompact |
|---|---|---|---|
| 1024 | KL 2.76 | **5.9e-5** | 6.7e-5 |
| 4096 | KL 3.22 | **1.1e-4** | 1.0e-4 |
| 7168 | KL 4.50 | **7.3e-4** | 6.1e-4 |

**Findings.**
1. **Dropping sinks is catastrophic** even on a modern model: KL 2.6–4.5 nats, top-1
   collapses to 34–62%. Reproduces StreamingLLM via our KL-vs-full metric.
2. **Sink-aware + re-rotation is near-exact**: KL ~1e-4–1e-3 nats (3–4 orders smaller),
   **100% greedy top-1 agreement** even after evicting *half to 7/8* of the context. For
   greedy decoding this is behaviourally exact vs the full prompt.
3. **Drift shrinks with retained context** (8192-ctx drift < 512-ctx drift at equal
   evict). Matches the feasibility law: more retained context ⇒ the evicted block is a
   smaller fraction of total attention mass.
4. **Recompaction ≈ leaving a gap** (and at evict 7168 no-recompact is *slightly*
   better). Hypothesis "recompact matters at long context" **not confirmed at 8192**.

**Interpretation of (4).** For a model with long RoPE scaling (Llama eff. 131k),
positions ≤ ~8200 sit well inside the scaled range, so a positional gap costs nothing for
a *single* eviction. Re-rotation's real job is **long-horizon hygiene**: keeping absolute
positions bounded across *many* evictions so they never exceed the scaled range. Expect a
*different* answer on trinity (θ=10k, **no scaling**) — there the gap should bite sooner,
so recompaction likely matters per-eviction. To test.

**Caveat.** Teacher-forced KL on a generic continuation measures distributional drift; it
does not yet probe *recall* of evicted facts. The planted facts (HALCYON/4096/Okonkwo)
live in the sink region and are retained — a dedicated continuity probe is next.

---

## 2026-06-14 — exp02: recall/continuity probe (Llama-3.2-3B)

**Why.** exp01's KL/top-1 shows the *continuation* matches, not that *information* survives.
Planted "the vault passcode is 48127" at a known token span, measured the log-prob (and
greedy recall) of the answer under full vs. rotated cache.

| scenario | logprob(answer) | recall |
|---|---|---|
| full context (reference) | −0.021 | ✅ |
| **Probe 1: fact kept in sinks, evict 300 unrelated tokens** | **−0.025** | **✅** |
| Probe 1: naive oldest (drops sinks+fact) | −21.2 | ❌ |
| Probe 2: fact is *in* the evicted span | −15.9 | ❌ |

**Findings.**
- **Retained info survives eviction essentially perfectly** (−0.025 ≈ full −0.021): popping
  unrelated context does not damage what's kept. This is the result that separates *memory*
  from *fluency* — the manager-report caveat is now addressed.
- Both failures are *correct* behaviour: dropping sinks corrupts everything (−21); dropping
  the span that holds a live fact loses it (−16). The latter is the honest cost that makes
  "evict by importance, not age" non-negotiable.

---

## 2026-06-14 — exp03: long-horizon multi-eviction rollout (Llama-3.2-3B)

**Setup.** context 256, generate 512, cache budget 320, evict 128/pop, 4 sinks → 4
evictions. Full reference cache grows 255→767; bounded cache stays ~budget.

| variant | mean KL | early (0:256) | late (256:) | max KL | max position |
|---|---|---|---|---|---|
| recompact | 1.90e-3 | 1.30e-3 | 2.50e-3 | 2.37e-2 | **320** |
| no-recompact | 1.63e-3 | 1.12e-3 | 2.15e-3 | 1.92e-2 | **767** |

**Findings.**
- **Drift does NOT blow up over repeated pops**: mean KL ~1.6–1.9e-3 over 512 steps / 4
  evictions — same order as a single eviction. The bounded cache tracks the full-context
  model across the whole rollout. Early→late growth is gentle (~2×), not runaway.
- **Recompaction caps absolute position at ~budget (320) vs unbounded (767, growing with
  session length).** The concrete long-horizon argument: without it, positions grow without
  bound and eventually exceed any RoPE range; with it, arbitrarily long sessions are
  possible — at negligible fidelity cost (no-recompact is even marginally lower KL while
  in-range). Conclusion: **recompact by default.**

---

## 2026-06-14 — exp04: Tier-2 importance-aware eviction (Llama-3.2-3B)

**Setup.** context 438, planted fact at ~50% (span [218,228)). Three strategies at EQUAL
kept-budget, scored on mean KL vs full (16-tok continuation) and recall of the mid-context
fact. Importance = accumulated attention received (H2O), captured via eager attention.

| budget | strategy | mean KL | recall_lp | fact kept? |
|---|---|---|---|---|
| K=175 (40%) | oldest | 5.96 | −21.4 ❌ | no |
| | sink_window | 0.81 | −15.5 ❌ | no |
| | **importance** | **0.75** | **−0.04 ✅** | **yes** |
| K=262 (60%) | oldest | 3.53 | −0.22 ✅ | yes |
| | sink_window | 0.71 | −0.03 ✅ | yes |
| | importance | 0.71 | −0.03 ✅ | yes |

(full-context recall_lp = −0.029)

**Findings.**
- **Headline:** at 40% budget, importance-eviction *retains* the salient mid-context fact and
  preserves recall (−0.04 ≈ full −0.03), while both age-based strategies drop it and lose it
  (−15 to −21). The accumulated-attention signal *did* catch a one-off mid-context fact (the
  distinctive passcode token attracts attention) — better than the reactive-not-predictive
  worry. Importance also matches/beats sink_window on KL.
- Sinks dominate KL: `oldest` (drops sinks) is catastrophic (KL 3.5–6) even when it happens
  to keep the fact (K=262).

**Honest caveat — KL is content-dependent.** Mean KL here (~0.7) is ~100× exp01's ~1e-3. The
surgery path is verified correct (recall −0.04 when the fact is kept), so this is *real*
drift, not an artifact: exp01's context was one sentence repeated (very redundant → dropping
half barely changes the distribution); exp04's richer context drifts more (repetitive filler
likely makes the model sensitive to exact repetition structure that eviction disrupts).
⇒ **our early "1e-3" was an easy case; we need realistic agentic-transcript eval contexts.**
Relative ordering (importance ≥ sink_window ≫ oldest) and the recall win are robust.

---

## 2026-06-14 — exp05: comparison matrix + drift decomposition (Llama-3.2-3B)

**Why.** Separate "is dropping [1] OK?" (information loss) from "is the mechanism good?"
(reuse contamination) by adding the missing baseline: a fresh recompute of the shortened
context. Fact planted in the *evicted* span. C=512, evict 128 after 4 sinks (keep 383).

| pair | mean KL | meaning |
|---|---|---|
| KL(full ‖ shortened) | 9.1e-2 | information loss of forgetting [1] |
| KL(shortened ‖ rotation) | **1.4e-2** | mechanism error (contamination) — the verdict |
| KL(full ‖ rotation) | 2.7e-2 | total drift |
| KL(full ‖ naive) | 3.41 | drops sinks — catastrophic |

**Recall (fact in evicted span):** full −0.04 ✅ · shortened −16.2 ❌ · rotation −16.0 ❌
**Cost:** rotation surgery 18 ms / 0 tokens recomputed vs shortened recompute 1164 ms / 383
tokens → **64× faster**.

**Findings.**
1. **Mechanism is faithful:** KL(shortened ‖ rotation)=0.014 — reuse+re-rotate reproduces a
   clean recompute closely, and matches it exactly on recall (both lose the dropped verbatim
   passcode: −16.0 ≈ −16.2).
2. **…and continuity-preserving — the headline:** rotation is *closer to full* (0.027) than
   the clean recompute is (0.091). Expected from first principles — rotation's survivors still
   carry [1]'s baked-in influence that a fresh recompute discards. So for the exact-vs-*full*
   ("continuative self") goal, rotation retains continuity **better** than recompute, not just
   cheaper. ⚠ Caveat: continuation here was generated by *full* (biases toward KV-sharing
   variants); neutral-source rerun queued to size the pure effect. Direction is sound.
3. **Verbatim vs echo:** you lose the dropped fact's exact tokens (unrecoverable, = recompute)
   but keep its influence on style/continuity.
4. **64× cheaper** than re-prefilling survivors (grows with context length); surgery is
   unoptimized Python-loop — a fused kernel would be ≪1 ms.

**Net:** recompute-quality (better, for continuity) at rotation-cost.

---

## 2026-06-14 — Trinity (afmoe) source review + hybrid-cache handling (no GPU)

Read `modeling_afmoe.py` before writing any trinity code (source of truth). Findings:

1. **RoPE is applied ONLY on sliding layers; the 15 global (full-attention) layers are NoPE.**
   `apply_rotary_pos_emb` is gated by `is_local_attention` (modeling_afmoe.py:374). So
   re-rotating the global layers on eviction would *corrupt* keys that were never rotated.
   **Fix built:** `ArchSpec.applies_rope` (afmoe → sliding-only, else all-True); `KVSnapshot`
   carries `applies_rope`; `reindex()` rotates RoPE layers and leaves NoPE layers' keys
   byte-for-byte. New test `test_reindex_skips_nope_layers`. 15/15 CPU tests pass.
2. **afmoe uses a plain `DynamicCache`** (:537) and enforces the window purely via the
   attention MASK (`create_sliding_window_causal_mask`, :555-566) — it stores all keys for all
   layers. So our `to_hf_dynamic_cache` rebuild is the *correct* cache type; the model rebuilds
   the right per-layer masks from `cache_position`. No custom cache class needed.
3. QK-norm (`k_norm` before RoPE) composes fine with re-rotation; muP is just input scaling
   (×√hidden); attention scale is standard `head_dim^-0.5`. None affect the surgery.

**Revised trinity feasibility — better than expected:**
- 15 global NoPE layers: eviction needs **no** re-rotation (position-free) — just drop keys;
  only content contamination.
- 45 sliding RoPE layers: re-rotate; evicting beyond the 4096 window is exact.
- ⇒ trinity rotation = re-rotate sliding layers only; content drift confined to the 15 global
  layers (+ within-window sliding). Simpler and lower-risk than a uniform RoPE model.

Hybrid-cache code done + unit-tested locally. Trinity GPU run (loads ~778 GB across 8 B200s)
pending go-ahead.

---

## 2026-06-15 — Trinity smoke test PASSES (afmoe, ~389B MoE, 8×B200)

Brought trinity up end-to-end. afmoe targets transformers 4.57; the shared venv is 5.3, so
five compat fixes in `load_model` (all verified; none touch model math):
1. **rope_scaling parse** — 5.x standardizes null→`{"rope_type":"default"}`; RoPESpec now maps
   default/None→none.
2. **token-ids** — backfill pad/bos/eos (5.x raises on unset attrs; afmoe reads them directly).
3. **pass `config=cfg`** to `from_pretrained` so the backfills take effect.
4. **plain RoPE as `linear`/factor=1.0** — sidesteps 5.x routing `"default"` rope through a
   module method (`compute_default_rope_parameters`) that 4.x rotary classes lack (hit in both
   `__init__` and `_init_weights`). Identical math.
5. **`device_map="auto"` max_memory cap (0.80×VRAM)** — default packs weights to the brim and
   OOMs the forward by ~36 MiB; reserve per-GPU headroom.

**Smoke results (context 256, evict 64, gen 16):**
- Arch confirmed: 60 layers (45 sliding W=4096 + 15 full), GQA 8 KV, head_dim 128, θ=10000, no scaling.
- **applies_rope = 45/60 == #sliding ✓** — per-layer RoPE gating matches the afmoe source (NoPE
  on the 15 global layers); re-rotation touches only the 45 RoPE layers.
- **Rotation homomorphism held on the real weights ✓** — re-rotation is exact.
- Eviction ran end-to-end: sink+rot mean_KL=9.3e-4 (top1 100%); oldest mean_KL=8.1e-4 (top1 100%).

**Observation (needs at-scale confirmation):** unlike Llama (where `oldest` drops sinks and is
catastrophic), trinity tolerates `oldest` here too. Plausibly the hybrid SWA + NoPE-global design
is far less position-0-sink-dependent — consistent with "trinity is the favourable case." But
this is a tiny sanity run (256 ctx / 64 evict, both within the 4096 window); the proper matrix is
needed before concluding.

---

## 2026-06-15 — exp07: trinity real-data eviction matrix AT SCALE (afmoe, 8×B200)

**Setup.** First real-long-context, at-scale test on the deployment target. Real pg19 book text
(doc 0), synthetic needle (made-up passcode) at depth 0.40, evict 60% after 4 sinks; full
decomposition + needle recall + cost timing. Lengths 8192 / 16384 (both span the 4096 SWA
boundary). Arch reconfirmed on live weights: 60L (45 sliding W=4096 + 15 full), applies_rope=45/60.

| L | keep/P | evict | info-loss KL(full‖short) | **mech** KL(short‖rot) | total KL(full‖rot) | naive KL(full‖naive) | rot ms | recompute ms | speedup |
|---|---|---|---|---|---|---|---|---|---|
| 8192  | 3277/8191  | 4914 | 3.95e-4 | **1.31e-4** | 1.54e-4 | 1.49e-4 | 33.8 | 24383 | 721× |
| 16384 | 6554/16383 | 9829 | 6.68e-5 | **3.04e-4** | 2.66e-4 | 1.16e-4 | 47.6 | 42326 | 890× |

Recall (needle EVICTED in both): full −0.159/−0.108 ✅ · shortened −15.2/−16.0 ❌ · rotation
−14.3/−15.3 ❌.

**Findings.**
1. **Mechanism runs correctly at scale on the real target.** Per-layer RoPE gating (rotate the 45
   sliding layers, leave the 15 NoPE global layers byte-for-byte) executes end-to-end at 389B
   across 8 GPUs on real context spanning the window. All drift ≤4e-4 nats. This is the at-scale
   confirmation the exp06 256-ctx smoke lacked.
2. **Mechanism error is the same tiny order as information loss — the verdict.** At 8k the cheap
   reuse is *lower* error (1.3e-4) than the unavoidable info loss of forgetting (3.95e-4); at 16k
   it's 3.0e-4 vs a near-zero 6.7e-5 info loss. Either way sub-millinat: reuse+re-rotate ≈ a clean
   recompute on the real target.
3. **Continuity thesis (rotation closer to full than recompute): real but NOT universal.** At 8k
   rotation total (1.54e-4) < recompute info-loss (3.95e-4) — replicates exp05's headline on
   trinity real data. At 16k it flips (rotation 2.66e-4 > recompute 6.7e-5): the evicted block was
   so stale (info-loss ≈ 0) that recompute is nearly identical to full and the still-tiny mechanism
   error dominates. ⇒ the win is largest exactly when the evicted block still carried influence;
   when it's fully stale, recompute and rotation both ≈ full and ordering is noise.
4. **Trinity tolerates naive (drop-sinks) eviction — now confirmed at scale.** naive KL
   1.5e-4/1.2e-4, same order as sink-aware (vs Llama's catastrophic 2.6–4.5). Upgrades exp06's
   tentative observation: the hybrid SWA+NoPE design is largely free of the attention-sink
   fragility StreamingLLM identified (NoPE globals have no position-0 anchor; sliding layers attend
   only within 4096, so token 0 isn't special). Keep sinks as ~free insurance, but on trinity they
   are **not load-bearing** — a real architectural advantage for eviction.
5. **Needle was EVICTED (depth 0.40 < 0.60 evict zone), so this run tests faithful *forgetting*,
   not recall preservation.** Both shortened and rotation correctly fail recall (you can't recall
   what you dropped), and rotation matches recompute on it (−14.3 ≈ −15.2; −15.3 ≈ −16.0; both
   « full). Faint signal: rotation is consistently ~1 nat *less* wrong than recompute — it retains
   a trace of the evicted needle's baked-in influence a clean recompute discards (corroborates the
   indirect-influence thesis, but it's ~1 nat in a −15 nat hole; don't over-read it).

**Caveats.**
- **Recompute timing inflated by disk offload.** The 0.80×VRAM cap forced partial disk offload
  ("parameters … offloaded to the disk"), so the 24–42 s recompute baseline is pessimistic;
  rotation's 34–48 ms is clean GPU tensor work. The orders-of-magnitude win is robust, the exact
  721/890× is config-dependent. Rotation cost grew sub-linearly (34→48 ms as kept doubled).
- **Memorization:** pg19 is kotodama training data → absolute drift reads low; trust the relative
  decomposition + the synthetic needle.
- **Mechanism error grows mildly with evicted-block size** (1.3e-4→3.0e-4 as evict 4914→9829);
  sub-millinat at 16k, worth watching at 64k+.

**Net:** the mechanism is correct, cheap, and near-error-free at 389B scale on real long context,
and trinity is confirmed the *favourable* case (SWA+NoPE → naive-tolerant). The headline
continuity win is context-dependent (largest when the evicted block mattered). Recall-preservation
at scale (needle-kept run) and clean GPU-resident timing are the two open follow-ups.

---

## 2026-06-15 — exp08: trinity recall preservation + clean timing (rotation-only, real data)

**Why.** exp07's needle landed in the evicted zone, so it only proved faithful *forgetting*. This is
the exp02 analog on the real target: sweep the needle across evicted (depth 0.40) and retained
(depth 0.80) regions and check recall under rotation, with **NO recompute in the measured path**
(rotation is the object of study; the shortened recompute is behind an opt-in flag, unused here).
Also instruments per-GPU memory and times the surgery cleanly. `--mem-frac 0.95`, evict 60% after 4
sinks, real pg19 (doc 0).

| L | needle | zone | recall full | recall rotation | KL(full‖rot) | top1 | rot ms |
|---|---|---|---|---|---|---|---|
| 8192  | @0.40 | evicted | −0.13 ✅ | **−14.20 ❌** | 3.31e-4 | 100% | 34 |
| 8192  | @0.80 | **kept** | −0.09 ✅ | **−0.12 ✅** | 4.20e-5 | 100% | 29 |
| 16384 | @0.40 | evicted | −0.09 ✅ | **−15.37 ❌** | 1.43e-4 | 100% | 49 |
| 16384 | @0.80 | **kept** | −0.12 ✅ | **−0.15 ✅** | 6.21e-5 | 100% | 52 |

**Findings.**
1. **Recall is preserved at scale when the fact is retained — the result exp07 was missing.** Needle
   in the kept tail (depth 0.80): rotation recall −0.12/−0.15 ≈ full −0.09/−0.12, greedy YES. Popping
   60% of the prefix + re-rotating survivors does not damage a retained fact, at 389B on real 8k/16k.
   The trinity analog of exp02 (Llama) now holds.
2. **Faithful forgetting when evicted** (depth 0.40): rotation −14.2/−15.4, greedy no — correct, you
   can't recall a dropped fact (matches exp07).
3. **Distributional drift negligible everywhere:** KL(full‖rotation) 4e-5–3e-4, **top1 100%** in all
   four cells. KL is ~5× lower when the high-attention needle is retained (4e-5 vs 3e-4) — keeping a
   salient token makes the continuation more faithful.
4. **Rotation surgery 29–52 ms**, pure GPU tensor ops (never touches disk) — the clean cost, scaling
   sub-linearly with kept tokens.

**Memory (the diagnostic the user asked for).** Even at 0.95×free (169 GiB/GPU budget) a small amount
still offloaded to disk — but it's an accelerate **imbalance, not a capacity wall**: g7 ran hot at
164/178 GiB while g0–g6 sat at 71–82/178 (95–101 GiB free *each*; ~600 GiB aggregate unused).
Activations for 16k add only +2–4 GiB (g7 peak 168). ⇒ **context length is nowhere near the memory
limit** (room for 64k+), and to kill the residual offload we should rebalance placement
(`device_map="balanced"` or bump to 0.98) rather than raise the cap — g7 is the bottleneck, not total
VRAM. The earlier "stuck-looking" phase (VRAM held, GPU 0%, load 1.0) was just deserializing ~727 GiB
from disk single-threaded; not a hang.

**Observability fix.** Piping python through `tee` block-buffers stdout, so our progress prints didn't
appear until the run ended (the apparent "stuck"). Fixed: run with `PYTHONUNBUFFERED=1`, and exp08 now
flushes every print + stamps phases with elapsed wall-clock (`[t+…s] …`). The `Loading checkpoint
shards` tqdm bar already streams load progress to stderr.

**Net:** rotation preserves recall for retained facts (≈ full, 100% top1) and faithfully forgets
evicted ones, at 389B on real long context — exp07's open gap is closed. Memory headroom is large; the
disk offload is a placement quirk to tidy (before any recompute-timing comparison), not a wall.

---

## Next

- [x] Continuity probe — done (exp02): retained facts survive; dropping live facts loses them.
- [x] Long-horizon rollout — done (exp03): drift bounded; recompaction caps positions.
- [x] Tier-2 importance eviction — done (exp04): preserves recall age-based eviction loses.
- [x] Comparison matrix — done (exp05): mechanism faithful (KL 0.014) & 64× cheaper than recompute; rotation closer to full than recompute.
- [ ] Neutral-continuation rerun of exp05 to size the "rotation beats recompute for continuity" effect cleanly (current run used full's continuation).
- [x] Trinity hybrid-cache handling — done (per-layer RoPE gating; plain DynamicCache is the
      correct cache type for afmoe). Code + test landed, no GPU needed.
- [x] Trinity GPU validation (exp06 smoke) — done: loads across 8 B200s, per-layer RoPE gating
      correct (45/60), homomorphism holds, eviction runs near-exact. 5 compat fixes documented.
- [x] Trinity eviction matrix at scale — done (exp07): 8k/16k real pg19, 60% evict across the 4096
      boundary. Mechanism near-error-free (mech-KL ≤3e-4 ≈ info-loss); rotation 34–48 ms vs
      disk-offloaded recompute 24–42 s; **naive/oldest confirmed safe at scale** (no sink fragility
      — SWA+NoPE is the favourable case). Continuity-beats-recompute held at 8k, flipped at 16k.
- [x] Trinity recall-preservation at scale — done (exp08): needle at depth 0.80 (retained) recalls
      under rotation (−0.12/−0.15 ≈ full) with 100% top1; depth 0.40 (evicted) faithfully drops it.
- [~] Clean trinity timing: rotation surgery is clean (29–52 ms, GPU-resident, exp08). The
      recompute *comparison* still needs the disk offload fixed first (rebalance placement) so the
      recompute baseline isn't disk-bound — then one `--with-recompute` cell gives the honest ratio.
- [ ] Fix trinity placement imbalance (exp08): `auto` overloads g7 (164/178) while g0–g6 sit at
      ~80/178 → small disk offload. Try `device_map="balanced"` or `--mem-frac 0.98`; lots of headroom.
- [ ] Re-check recompact-vs-gap on trinity (θ=10k, no scaling) — exp07 didn't vary it; θ=10k should
      make the positional gap bite sooner than on Llama (exp01).
- [ ] Realistic eval contexts (agentic transcripts, not synthetic repetition) — exp04 shows
      absolute KL is content-dependent; calibrate on real workloads.
- [ ] Tier 3 selective recompute on the global layers only (cost-bounded).
