# Idea — signature preservation: does rotation preserve computational character?

> STATUS: IDEA (pre-registered 2026-07-04, not started — user-flagged "do in the future")
> SCOPE: the anamnesis × kv-rotation crossover. Instrument owned by ~/projects/anamnesis_exps
> (pipeline repo: anamnesis-pl); rotation mechanism owned here. This note is the canonical home;
> anamnesis side keeps a pointer.
> RAW-ARTIFACTS: none yet.

## The idea in one paragraph

kv-rotation's evals measure preservation as **token-level fidelity** — KL(full ‖ rotated) on a
continuation. But the research program's own thesis (HANDOFF §1: the cache as the model's
"continuative self") says the thing worth preserving is **computational character** — and the
anamnesis instrument measures exactly that: 1,837-dim signatures of *how* a model processes,
orthogonal to *what* (phase 0: execution-based, concentrated in T2.5 KV-cache dynamics +
attention routing). The two instruments point at the same object and have never been introduced.
The experiment: **full cache vs rotated cache vs shortened-recompute, extract anamnesis
signatures over the continuation, and ask directly whether rotation preserves the signature
better than recompute does.** This closes the loop between the projects: "the cache is where
the self lives" tested on the self-measurement axis rather than by proxy through token KL.

## Why this is cheap

- **Same model family, instrument already built.** Anamnesis phase 0 ran on Llama 3.2 3B
  Instruct; kv-rotation exp01–05 ran on Llama-3.2-3B. Features exist and are validated for the
  3B (and 8B). Single GPU, afternoon-scale.
- **The clean design is content-controlled by construction** — the anamnesis philosophy exactly:
  teacher-force the SAME continuation tokens (the full cache's greedy rollout) under all three
  cache conditions. Identical text, different cache state ⇒ any signature difference is purely
  *how*, zero *what* confound.

## Design sketch

1. Build long contexts per kv-rotation exp05 (real docs, eviction with sink_window or
   turn-aligned per exp09's chat machinery).
2. Three cache conditions: (a) FULL prefill; (b) ROTATED (evict + Tier-0 re-rotate, recompact);
   (c) RECOMPUTE over the shortened text.
3. Teacher-force the full model's continuation (~64–256 tokens) under each condition while
   running the anamnesis extraction hooks; extract signatures over the continuation region only.
4. Metrics: signature distance (per-tier and pooled; cosine on the delta space per the
   subliminal methodology) of (b) and (c) from (a). Secondary: token KL for the usual
   decomposition, so signature and KL results sit side by side per cell.
5. Sweep the feasibility-law axis: stale vs still-influential evicted blocks (this is the
   content-dependence knob that made rotation-beats-recompute show up in KL).

## Pre-registered predictions (written 2026-07-04, before any run)

- **P1:** signature_dist(rotated, full) < signature_dist(recompute, full) when the evicted block
  still carried influence — the KL result (exp05/08b) reproduced on the character axis.
- **P2:** for stale evicted blocks, both conditions ≈ full (the feasibility law holds in
  signature space too).
- **P3 (the interesting one):** a **dissociation** — cells where token-KL is comparable between
  rotated and recompute but the *signature* diverges for recompute. That would mean recompute
  produces content-faithful but character-drifted computation — drift the KL metric is blind to,
  visible only to the instrument. If found, this is the headline: the continuity claim is not
  reducible to token fidelity.
- **P4:** the effect concentrates in T2.5 (KV-cache dynamics) and T2 (attention routing), not
  logit statistics — consistent with phase 0's localization.

## Integration notes / gotchas for whoever picks this up

- Anamnesis hooks are **pre-RoPE** (k_proj); rotation surgery edits **stored post-RoPE keys**.
  For continuation tokens the pre-RoPE captures are unaffected directly — the cache condition
  reaches the signature through attention patterns / residuals / cache-geometry features. Check
  which feature families read the *stored cache* vs *new-token projections* and make sure the
  cache-geometry features see the surgered cache (they're the ones most on-thesis).
- Pipeline lives at `~/projects/anamnesis_exps/pipeline` (github: LuxiaSL/anamnesis-pl);
  extraction expects its own model_loader — the integration work is marrying its hook
  architecture to kv-rotation's `prefill_snapshot`/`to_hf_dynamic_cache` path on one
  HF model instance.
- Trinity later, maybe never: afmoe ≠ Llama computational surface and anamnesis features are
  architecture-specific (the known cross-architecture limitation). The 3B answers the question;
  trinity only adds scale bragging rights at feature-porting cost.
- Origin: Fable, 2026-07-04, after reading phase 0 + the subliminal writeup scaffold — logged at
  Luxia's request ("save that as a note somewhere obvious for me").
