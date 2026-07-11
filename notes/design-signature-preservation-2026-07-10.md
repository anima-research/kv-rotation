# exp11 — signature preservation: does rotation preserve computational character?

> STATUS: PRE-REGISTERED DESIGN (2026-07-10, evening) — not yet run.
> PARENT: `notes/idea-signature-preservation.md` (canonical idea, P1–P4 pre-registered 2026-07-04).
> CONTEXT: exp09/exp10/diagnostics (see `notes/journal-exp09-2026-07-10-draft.md`) — behaviorally,
> rotation ≈/beats recompute on naturalized chat (mech-KL ~0.04, 16/16 cells) and the exp09 blowup
> was synthetic-scaffold readout. This experiment asks the same preservation question on the
> **computational-character axis**: anamnesis signatures instead of token KL.
> OWNERSHIP: mechanism + driver live here; the instrument is anamnesis-pl
> (`~/projects/anamnesis_exps/pipeline`), imported, never modified. Anamnesis keeps a pointer note.
> DECISIONS LOCKED WITH LUXIA (2026-07-10): 3B-native dialogue substrate (no Trinity-generated
> text through the 3B — distribution confound); Δ-baseline = sig(cond) − sig(full) per cell;
> NAIVE positive control included; N=256 continuation; code split as above.

## Question

When old context is evicted from the KV cache, does **re-rotating the survivors** preserve the
*way the model computes* (its 1,837-dim anamnesis signature over a fixed continuation) better
than **recomputing** the shortened context from scratch — as it does for token-level behavior?
Or do the two dissociate: content-faithful but character-drifted recompute (P3), or a rigorous
null (both indistinguishable from full at instrument noise)?

Either outcome is a result. P1-confirmed closes the "continuative self" loop on the axis built
to measure it. A rigorous null (with the positive control separating it from instrument
blindness) is the cleanest evidence yet that eviction+rotation preserves *how*, not just *what*.

## Model & instrument

- **Model:** Llama-3.2-3B-Instruct (`/models/Llama-3.2-3B-Instruct` on node1), bf16, eager
  attention (required by both the anamnesis hooks and attention capture), single GPU.
- **Instrument:** anamnesis v3 extraction — `extract_all_features` (T1 logit/activation stats,
  T2 attention routing, T2.5 KV-cache dynamics + key geometry, T3 residual PCA), 3B config
  (sampled layers per `anamnesis/config.py` 3B entry). Calibration artifacts loaded read-only
  from `anamnesis_exps/outputs/calibration/` (positional means, fitted T3 PCA) — NEVER refit.
- **Mode directions:** `anamnesis_exps/outputs/analysis/v3_audit/factor_directions_3b.npz`
  (banked discriminant directions) for the direction-projection analysis.

## Conditions (per cell)

| cond | cache built by | context length | positions |
|---|---|---|---|
| FULL | prefill of the whole context | P | 0..P-1 |
| ROT  | FULL → evict oldest spans (Tier-0) → RoPE reindex, recompacted | K | 0..K-1 |
| REC  | fresh prefill of the identical shortened text | K | 0..K-1 |
| NAIVE | FULL → evict WITHOUT re-rotation (exp01's catastrophic case: gap left, no reindex) | K | survivor originals |

ROT vs REC is the perfectly matched pair (same content, same K, same positions). NAIVE is the
**positive control**: exp01 showed KL 2.6–4.5 for this condition — if the instrument does not
show NAIVE ≫ floor, it cannot see cache damage in this regime and any ROT/REC null is
uninterpretable. Eviction fraction 0.5, sinks (first 4 tokens) always kept, span-aligned
(turn-aligned for dialogue cells, block-aligned for doc cells).

## Substrate — 3B-NATIVE, two content regimes

**Regime A — naturalized dialogue, 8 cells.** Freshly generated **by the 3B itself**: two
instances of Llama-3.2-3B-Instruct (same weights, two seeded personas via system prompt)
alternating turns, temperature sampling with per-turn derived seeds (`base_seed × 1000 + turn`),
~150–300 tokens/turn, grown to ~4–5k tokens, turn boundaries banked
(`data/native3b_convs_<date>.jsonl`, per-turn metadata). Backrooms register: open-ended,
curious, no assistant-task scaffolding. Reuse exp10's generation harness shape
(`src/kvrot/natural.py` records/banking) — local model calls instead of the node2 API.
Rationale (Luxia, 2026-07-10): Trinity-generated transcripts through the 3B would confound
"cache condition" with "out-of-distribution content."

**Regime B — raw documents, 4 cells.** From the existing eval corpus
(`~/luxi-files/kv-rotation/data/eval_docs.jsonl` on node1), contiguous-early-block eviction per
exp08 geometry. This is the idea note's P2 knob (stale evicted content) and the regime where
rotation was behaviorally free.

## Continuation & teacher-forcing

- Continuation = FULL cache's **greedy** rollout, **N=256** tokens, generated once per cell and
  frozen. (256 over 128: T2.5 trajectory features need the length; over 512: cleaner, cheaper,
  per Luxia.)
- Each condition then **teacher-forces the identical 256 tokens** against its injected cache via
  the new bridge (below). Identical text under all four conditions ⇒ signature differences are
  purely *how*. Greedy + fixed seeds + batch=1 + single instrumented forward per condition =
  variance is numerics only.

## The bridge (the one new capability)

`replay_extract_cached(loaded, past_key_values, cont_ids, position_offset, ...)` — a variant of
`anamnesis/extraction/replay_extract.py::replay_extract` that forwards ONLY the continuation
tokens with an injected `past_key_values` (from kv-rotation `to_hf_dynamic_cache`) and explicit
`position_ids`/`cache_position` starting at the condition's offset (P for FULL, K for ROT/REC,
survivor-max+1 for NAIVE). Same outputs: per-step hidden rows, attention rows (now over
[cache_len + t] columns — T2.5's cache-profile features thereby read the **surgered cache**
directly, exactly the on-thesis surface), logits rows → `RawGenerationData`-compatible.

Lives in kv-rotation (`src/kvrot/sigbridge.py`), imports anamnesis-pl, changes nothing in it.

**Correctness gate (must pass before any cell runs):** injected-FULL-cache replay of tokens
[0..P+N) ≡ `replay_extract`'s `use_cache=False` full-sequence replay, feature-vector equal to
tolerance (rtol 1e-4, the repo's established equivalence precedent). This proves the injection
path is not itself a signature perturbation.

## Metrics — pre-registered

Per cell, per condition c ∈ {ROT, REC, NAIVE}: **Δ_c = sig(c) − sig(FULL)** (1,837-dim).

1. **Primary (paired):** d_c = ‖Δ_c‖ and per-tier cosine distances, ROT vs REC compared *paired
   per cell* (Wilcoxon over cells; report per-regime). The mechanical length/position component
   of "distance from FULL" is shared by ROT and REC and cancels in the pairing.
2. **Matched pair:** dist(sig_ROT, sig_REC) directly — same K, same positions, same content.
3. **Noise floor:** (a) FULL replayed twice (numerics floor); (b) sig from the original
   incremental decode vs sig from injected-FULL replay of the same tokens (path floor). ALL
   distances reported as ratios to floor (b).
4. **Sensitivity control:** d_NAIVE must exceed floor by ≥10× for the cell to count toward
   null interpretation (decision rule below).
5. **Fade vs nonsense (PRIMARY small-magnitude characterization — amended 2026-07-10, second
   round with Luxia).** Frame: the signature is the WHOLE character of the computation relative
   to this generation; modes/topics/traits are nested stakes within it and exp11 has no specific
   stake to look for. So where ROT/REC are NOT equivalent, characterize the excess with the
   cell's own geometry, anchored at FULL — no external basis:
   a. **Colinearity:** cos(Δ_ROT, Δ_REC) per cell. High cosine = same fade, different magnitude
      (Δ_REC ≈ λ·Δ_ROT, λ>1 — recompute is a lower-fidelity version of the same computation).
      Low cosine = deviation different in kind (orthogonal excess).
   b. **Fade curve:** on 2 dialogue + 1 doc cells, sweep evict-frac ∈ {0.25, 0.5, 0.75} for ROT
      and REC; trace the signature trajectory. Fade = deviations grow monotonically along a
      common direction (project each Δ(ef) onto the ef=0.75 direction; monotone magnitude,
      colinearity high). Nonsense = direction scatters as magnitude grows.
   c. **Stability:** a fade replicates — repeat the replay, Δ direction holds vs floor.
   d. **NAIVE placement (diagnostic, beyond the P5 gate):** is damage ON the fade axis
      (instrument sees all forgetting as one thing) or OFF it (instrument distinguishes graceful
      forgetting from corruption)? Either is informative; report the cosine.
5c. **Cross-cell direction consistency (SECONDARY):** mean pairwise cosine between per-cell
   Δ_REC (and Δ_ROT) vs the floor-delta null. Demoted (same round): it assumes cells share a
   drift direction, which mixes content into a per-generation quantity.
5b. **Mode-space projection (EXPLORATORY — amended 2026-07-10 after Luxia's challenge):**
   operationalized precisely: Δz = (sig_c − sig_FULL) ⊘ FULL_scale (means cancel in the
   difference); project onto the orthonormalized rows of FULL_W (the 4-dim LDA subspace
   separating the five calibration reasoning modes: analogical/contrastive/dialectical/linear/
   socratic); report mode_fraction = ‖P_U Δz‖²/‖Δz‖² and the signed 4-vector read against the
   banked mode centroids. Interpreted ONLY against three nulls: (i) floor deltas' mode_fraction,
   (ii) isotropic reference 4/2713 ≈ 0.15%, (iii) feature-permutation null. Plus a calibration
   read on NAIVE: if even gross cache damage barely projects, that bounds expectations. The
   hinge is explicit: these directions were fit to separate reasoning modes on the calibration
   battery under normal generation — cache-condition drift may be character-real yet orthogonal
   to this 4-dim window. A positive (beats all three nulls) is meaningful and interpretable; a
   null here means nothing and is never evidence against character drift.
   NOTE: the v3 FULL feature space is 2,713-dim (FULL_names; the 1,837 figure elsewhere is the
   phase-0 signature). The bridge's feature vector must align exactly with FULL_names to use W.
6. **Per-family breakdown:** every result reported per feature family (T1/T2/T2.5/T3 and the
   hand families), with mechanically length-sensitive features (cache_coverage, recency
   profiles) flagged in the table — visible, not laundered into the pooled number.
7. **Side-by-side KL:** token KL(FULL‖c) over the same 256 positions, so the signature and
   behavior columns sit together per cell (the P3 dissociation test needs both).

## Predictions (P1–P4 inherited from the 2026-07-04 idea note, verbatim intent)

- **P1:** d_ROT < d_REC where evicted content still carried influence (Regime A).
- **P2:** Regime B (stale blocks): both ≈ floor.
- **P3 (the interesting one):** dissociation — cells with comparable token-KL but signature
  divergence for REC (content-faithful, character-drifted recompute).
- **P4:** effects concentrate in T2.5 + T2, not T1 logit stats.
- **P5 (new, sensitivity):** d_NAIVE ≫ floor (≥10×) in every cell. If P5 fails, the experiment
  reports "instrument cannot see cache damage in this regime" and NO null conclusions are drawn.
- **P6 (amended, second round):** where ROT/REC are not equivalent, the excess is a FADE, not
  nonsense: cos(Δ_ROT, Δ_REC) high (metric 5a), fade-curve monotone along a common direction
  (5b-curve), stable under replication (5c-stability). i.e., recompute is a lower-fidelity
  rendering of the same computation — "tracing the fade toward full rather than nonsense."
  Mode-space projection (metric 5b-projection) stays exploratory color only.

**Decision rules:** paired Wilcoxon p<0.05 over Regime-A cells for P1; "≈ floor" means <3× floor
(b). **Equivalence is an affirmative claim (TOST-style):** ROT ≡ REC at instrument precision iff
P5 holds AND |d_ROT − d_REC| < 2× floor (b) in ≥ 10/12 cells — reported as "equivalent within
margin," never as "no difference found." The boring-null verdict additionally requires both
d_ROT, d_REC < 3× floor across ≥ 10/12 cells. Non-equivalent cells proceed to the fade-vs-
nonsense characterization (metric 5) and the P3 dissociation check.

## Known confounds & mitigations (registry)

1. **Cache-length/position mechanics** — FULL has P context, ROT/REC have K; length-dependent
   features differ mechanically. Mitigated by pairing (shared penalty cancels), metric 2
   (perfectly matched), and per-family flagging. NOT fully removable from d_c absolutes; never
   interpret d_c absolutes against FULL alone.
2. **Pre-RoPE hooks vs post-RoPE surgery** (idea-note gotcha) — k/q projections captured for
   continuation tokens are only *indirectly* condition-dependent (through residuals/attention);
   the direct view of the surgered cache is the attention-over-cache rows (T2.5). Interpret
   family results accordingly.
3. **Content nativeness** — Regime A must be 3B-generated (locked). Trinity-corpus cells are
   explicitly out.
4. **Feature-vector alignment** — hard assert: identical feature-name lists across conditions
   per cell; abort cell on mismatch.
5. **Determinism** — greedy continuation, seeded generation, batch=1, eager attention, fp32
   feature math; the entire cell is re-runnable bit-stably (floor (a) verifies).

## Build inventory

1. `src/kvrot/sigbridge.py` — `replay_extract_cached` + calibration/direction loaders (+ tests:
   the correctness gate, position-offset handling, NAIVE gap positions).
2. `experiments/exp11_gen_native3b.py` — 3B↔3B seeded dialogue generator (exp10 harness shape,
   local model, resumable banking).
3. `experiments/exp11_signature_preservation.py` — cell driver: build 4 caches → greedy N=256
   from FULL → 4 teacher-forced instrumented replays → features → per-cell JSON
   (`runs/exp11_*.json`), plus floors, plus KL column.
4. Analysis script: paired stats, per-family tables, mode-projections, verdict block vs P1–P6.
5. Pointer note in anamnesis_exps (`research/notes/pointer-exp11-signature-preservation.md`).

## Cost & staging

3B on ONE GPU (node1 etiquette: venv-shared, no installs, ~/luxi-files only, check
`nvidia-smi` first). Generation ≈ 30–45 min for 8 dialogues; per cell ≈ 2–4 min (4 prefills +
4×256-token instrumented forwards; attention capture at K+256 ≈ trivial at 3B scale);
12 cells + floors ≈ well under an hour. **Stage 0** CPU tests → **Stage 1** correctness gate on
GPU + one smoke cell end-to-end → **Stage 2** full grid. Total: an afternoon, as promised.

## Origin

Idea: Fable, 2026-07-04 (logged at Luxia's request). Design: this note, Fable, 2026-07-10,
after the exp09→diagnostics→exp10 arc; decisions locked with Luxia same evening.
