# Design — exp09 diagnostics (D1/D2/D3/D5): why does chat-shaped eviction break the mechanism?

> STATUS: pre-registered 2026-07-10, written BEFORE the GPU runs (same day as exp09).
> SCOPE: diagnose exp09's P1 refutation — mech-KL(short‖rot) 0.2–1.3 on chat vs ≤5e-4
> on raw docs (exp07/08b) at matched L and evict-frac. Driver:
> `experiments/exp09_diagnostics.py` (single trinity load, all cells chained).
> Raw artifacts land in `runs/exp09diag*.{json,npz,log}`; findings go into the
> exp09 journal draft's "Diagnostics (same day)" section.

## 0. Pre-run discovery that reframes everything (tokenizer-only, no GPU)

Rendering the exact exp09 configs through the trinity tokenizer shows the evicted
turns are **consecutive and their spans back-to-back**: `oldest_turns_to_evict`
takes turns 1..13 and the evicted tokens form ONE contiguous block —
[32, 5199) of P=8921 (frac 0.579) at 8k, [32, 10231) of 17528 (0.582) at 16k.
**exp09's eviction is NOT interleaved.** The draft journal's "interleaved
early-to-mid turns" interpretation is wrong and will be corrected. Geometrically,
exp09 ≈ exp08 (kept prefix of 32 system tokens vs 4 sinks; one contiguous interior
block; kept tail). What actually changed vs exp08b is the **content bundle**:
chat template scaffolding, low-entropy assistant acks, the generation prompt, and
the measurement point (an assistant *reply* instead of doc continuation).

Second pre-run observation: the sweep's "granularity effect" is **confounded with
achieved evict fraction** (whole-turn rounding): t8 evicts 0.62, t12 0.58, t24
0.54, t48 will evict 0.52. Since mech-KL also rises steeply in ef (0.46 → 0.68 →
1.06 at 8k for ef 0.32/0.58/0.74), part of the t8→t24 improvement is just the
frac curve. D5 quantifies this.

## 1. D1 — KV geometry: measure the baked-in contamination directly

**What it is.** For the exp09 main cell, compare the rotated-survivor snapshot
against a fresh recompute of the identical shortened context, tensor-by-tensor:
per-layer (60) cosine similarity of keys and values separately, per-token, split
sliding/RoPE (45) vs global/NoPE (15), profiled by token distance to the evicted
block's seam.

**Why it's sharp.** The rotated survivor KV is *identical* to the pre-eviction
full-prefill KV except for the (bit-exact) key re-rotation on RoPE layers: values
are copied verbatim everywhere, NoPE keys are untouched by reindex. So
rot-vs-short divergence is exactly **"KV computed with the evicted block present"
vs "KV recomputed without it"** — the baked-in contamination itself, not a
mechanism defect. exp08b showed this contamination is *behaviourally benign* on
raw docs (mech-KL ≤5e-4, and it's the very thing that makes rotation closer to
full than recompute). exp09 says on chat it is behaviourally loud. D1 shows where
it lives; computed at every D2 corner for free (both snapshots exist), so we also
get contamination-at-matched-geometry, chat vs raw.

**Pre-registered expectations:**
- E1.1 Survivors *before* the block (the 32 system tokens) match to numerical
  noise (cos ≈ 1; residual = prefill-length kernel nondeterminism — this is the
  noise floor and we report it).
- E1.2 Contamination is worst for survivors immediately **after** the seam and
  decays with distance; on sliding layers it should largely die past ~4096
  original-token distance (window), on NoPE/global layers it persists broadly.
- E1.3 Keys and values divergence are comparable (both come from the same
  contaminated hidden states); NoPE-key divergence ≈ value divergence.
- E1.4 The discriminating one: contamination magnitude (cosines) will be
  **similar for chat and raw-doc at matched geometry** (corners b vs c), while
  downstream mech-KL differs by ~3 orders — i.e. the pathology is not "chat
  contaminates more" but "chat *reads* the contamination at decode time"
  (attention onto contaminated survivors / sensitivity of the reply
  distribution). If instead chat cosines are dramatically lower, the story is
  contamination-magnitude and Tier-3 selective recompute targets shift.

## 2. D2 — 2×2 factor isolation: content-bundle × eviction-geometry

| | contiguous exp08-style geometry | exp09 turn-mask geometry |
|---|---|---|
| **raw-doc content** | (a) exp08b — BANKED: mech-KL 3.8e-4/2.1e-4 @8k, 1.4e-4/4.7e-4 @16k | (c) NEW |
| **chat content** | (d) NEW | (b) exp09 — BANKED: mech-KL 0.684 @8k, 0.414 @16k |

- **(c) raw + exp09 geometry:** tokenize the same eval corpus (same doc order,
  same joiner) to exactly the exp09 cell's C, transplant exp09's *exact* keep
  mask (system-prefix 32 + block + tail become arbitrary doc offsets), same
  KL(full‖rot)/KL(full‖short)/KL(short‖rot) decomposition on a 32-token greedy
  continuation. Holds prose and geometry fixed; removes template/acks/reply.
- **(d) chat + exp08 geometry:** the exp09 conversation and its full-cache reply,
  but keep = 4 sinks + everything after a contiguous block of the *same evicted
  token count* starting at token 4 (`sink_window` policy verbatim). This slices
  the system prompt (tokens 4–32 die) and cuts mid-turn at the far end — exactly
  what exp08's geometry does to a chat, documented as such. Same K as (b).

**Pre-registered expectations.** Given §0 (geometry is nearly matched already),
we predict the verdict is **content-bundle, not geometry**: (c) comes back low
(mech-KL ≤ ~1e-3, near exp08b) and (d) comes back high (same order as exp09,
0.3–1.0). If (c) blows up instead, the driver is geometry after all (kept-prefix
width or seam placement) and the interleaving post-mortem reopens. If both are
intermediate, it's an interaction and D1's per-corner geometry arbitrates.

## 3. D3 — per-step KL: is the drift real or a post-EOS artifact?

**What.** Record per-decode-step KL (all three pairs) for the exp09 main cells
(both L) plus the first `<|im_end|>` (id 3) position in the full model's
reference reply; report means split before/after the first im_end and the
within-reply drift profile. (exp09's reference reply closed the turn after ~17
tokens and then emitted junk — part of the 32-token KL window is post-EOS.)

**Pre-registered decision rule.** If pre-im_end mech-KL < 0.05 (≥10× below the
pooled 0.68), P1's refutation is substantially a post-EOS measurement artifact
and the exp09 headline gets revised (though 0.05 is still ~100× raw-doc — the
finding would shrink, not vanish). If pre-im_end mech-KL ≳ 0.1, the refutation
stands as written. Expectation (weakly held): drift is real pre-EOS but the
pooled numbers are inflated ~2–5× by the junk tail; within the live reply, KL
grows with step index (each step conditions on more shared teacher-forced
context — mild decline is also plausible; no strong prior).

## 4. D5 — granularity extension: turns=48

One more sweep cell: `--turns 48`, ef=0.5, both lengths, with recompute —
extends t8/t12/t24 (mech-KL 1.30/1.28 → 0.68/0.41 → 0.27/0.41 @8k/16k).

**Pre-registered discriminating prediction (see §0 confound).** Achieved evict
frac at t48 is 0.516. Under the *frac-only* hypothesis (granularity is
epiphenomenal; interpolate the ef sweep), t48 @8k lands mech-KL ≈ 0.5–0.6. Under
the *granularity* hypothesis (draft pattern 2 taken at face value), the trend
continues down: ≈ 0.15 or lower. These differ by ~4×; the result adjudicates.
Note t48's C overshoots (≈10.4k for the 8k target) — reported, not chased.

## 5. Protocol notes

- Single driver, one trinity load (`--device-map balanced --mem-frac 0.70`,
  ~354 s), cells chained; per-cell try/except so one failure doesn't kill the
  chain. gen=32, doc=0, sinks=4, turns=12 everywhere except D5.
- All KV-geometry stats computed on-GPU in-process; only reduced arrays saved
  (per-layer × per-token cosine curves as fp16, keep sets, block bounds —
  ~1–2 MB npz per corner). Raw KV never hits disk.
- No recall probes (P2 is settled; saves time).
- Reference continuations are the full cache's own greedy reply per context —
  same continuation-source caveat as exp05/09 (fine: all comparisons share it).
