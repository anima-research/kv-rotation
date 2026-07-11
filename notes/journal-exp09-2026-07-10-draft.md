# exp09 — chat-shaped eval (turn-aligned eviction), first run — 2026-07-10 (DRAFT)

> Runner: agent session, node1 (8x B200, otherwise idle; Heimdall queue empty before
> and during). Design + pre-registered predictions: `notes/design-chat-eval.md`.
> Raw artifacts: `runs/exp09_smoke_3b.{json,log}`, `runs/exp09_trinity_main.{json,log}`,
> sweep files `runs/exp09_trinity_{ef30,ef70,t8,t24}.{json,log}` (local copies pulled
> from node1 `~/luxi-files/kv-rotation/runs/`).

## Headline

**P3 confirmed, loudly — and it is mechanism error, not info loss (P1 REFUTED).**
On trinity, total reply drift under turn-aligned oldest-turn eviction is
KL(full‖rot) ≈ 0.47–0.58 — four orders of magnitude above exp08's raw-doc ~5e-5 at
matched L and evict-frac. The recompute decomposition pins it on the mechanism:
info-loss KL(full‖short) is small (0.047 @8k, 0.158 @16k) while mech-KL(short‖rot)
is 0.68 / 0.41. For the first time in the project, **rotation is the outlier**:
the shortened recompute sits near full, rotation sits far from both. The raw-doc
"mechanism is free" result does NOT carry to chat-shaped, mid-context, turn-aligned
eviction. Recall, however, is perfectly clean (P2 ✓), and nothing crashed (P4 ✓).

## What ran

- Code: exp09 as staged 2026-07-03, plus two small additions made today before
  running (no protocol changes): (1) `--out PATH` JSON persistence in
  `experiments/exp09_trinity_chat.py` (rows + config + failures + summary);
  (2) a transformers-5.x compat fix in `kvrot.chat._render_ids` —
  `apply_chat_template(tokenize=True)` now returns a BatchEncoding/dict in 5.x,
  which the incremental renderer iterated as string keys → `ValueError: too many
  dimensions 'str'` on the first smoke attempt. Fixed by unwrapping `"input_ids"`;
  suite still 42/42 CPU-green.
- Smoke: Llama-3.2-3B-Instruct, 1 GPU, `--lengths 2048 --turns 8 --gen 16`.
- Main: Trinity-Large-Preview, 8 GPUs, `--lengths 8192 16384 --turns 12
  --device-map balanced --mem-frac 0.70 --with-recompute`. Load 354 s; total 1171 s.
- Sweeps (same base flags): `--evict-frac 0.3`, `--evict-frac 0.7`,
  `--turns 8`, `--turns 24`. [numbers below]

## Probe-style decision (pre-registered, design note §4.5)

Smoke decided: **plain-text probes retained** (exp08-comparable). On the 3B
instruct, full-cache recall logprobs on bare-text Q&A suffixes were near-zero
(vault −0.017, locker −0.115) with correct recall on both — not noisy, no need for
chat-templated probes. Trinity concurred (full lp −0.04…−0.17, all recalled).

## Smoke (3B, L≈2495, 8 user turns, evict 9 turns = 1491 toks, keep 1003/2494)

- KL(full‖rot) over 16 reply toks = **1.32e-1**, top1 94% — already ~100× above the
  3B's raw-doc exp01 numbers (1e-4…1e-3): the P3 signal was visible pre-trinity.
- vault (RETAINED): full lp −0.017 → rot −0.011, preserved ✓
- locker (EVICTED): full lp −0.115 → rot −8.32, dropped ✓
- rotation surgery 13.6 ms.

## Main (trinity, 12 user turns, evict-frac 0.5 → 13 turns evicted, both cells ✓ guards)

| cell | keep | KL(full‖rot) | top1 | info-loss KL(full‖short) | MECH KL(short‖rot) | vault (ret) lp full→rot | locker (ev) lp full→rot | t_rot | t_recompute |
|---|---|---|---|---|---|---|---|---|---|
| L≈8922 | 3754/8921 | 5.81e-1 | 91% | 4.68e-2 | **6.84e-1** | −0.043→−0.009 ✓ | −0.061→−11.47 ✓ | 44.1 ms | 31.7 s (718×) |
| L≈17529 | 7329/17528 | 4.69e-1 | 78% | 1.58e-1 | **4.14e-1** | −0.073→−0.013 ✓ | −0.171→−10.63 ✓ | 60.9 ms | 62.1 s (1020×) |

## Pre-registered verdicts

- **P1 (mechanism, predicted mech-KL ≤ ~1e-3): REFUTED.** mech-KL = 0.68 @8k /
  0.41 @16k — three orders above prediction and above every raw-doc measurement
  (exp07/08b ≤5e-4). Turn-aligned chat eviction is NOT "nothing special to the
  rotation math" in effect: the *math* is still bit-exact (CPU tests), but reusing
  survivor KV after mid-context whole-turn eviction diverges hard from a clean
  recompute on the next-turn reply. See "interpretation" below.
- **P2 (recall): CONFIRMED.** Retained vault passcode recalls under rotation ≈ full
  (rot lp −0.009/−0.013, in fact slightly *better* than full; well within the ±0.5
  registered band); evicted locker drops to lp ≈ −11 (faithful forgetting). 2/2 cells,
  plus smoke.
- **P3 (total drift higher than raw-doc): CONFIRMED, ~1e4×.** KL(full‖rot) 0.47–0.58
  vs exp08's ~5e-5 at matched L/evict-frac. Per the registered decision rule (≥5–10×
  = the finding), oldest-turn eviction is NOT deployment-ready for chat shape as-is;
  importance-aware (Tier 2) / turn-summary policies are the indicated follow-up.
- **P4 (template safety): CONFIRMED.** Trinity's `chat_template.jinja` is
  prefix-stable (no assert), spans aligned, per-layer-gated reindex ran clean
  (applies_rope=45/60), no NoPE corruption signatures, both guards passed.

## Interpretation (why the mechanism breaks here and not on raw docs)

> ⚠ SUPERSEDED same day: the "interleaved turns" story below is WRONG (the evicted
> turns form one contiguous block) — see "Diagnostics (same day)" at the end of this
> file for the corrected, measured explanation.

The triangle inequality of the three KLs tells the story: full≈short (0.047),
rot far from short (0.68) AND far from full (0.58). Rotation is not "closer to
full than recompute" in this regime — the exp05/07/08b continuity advantage
inverts. Candidate explanation (not yet isolated): raw-doc evals evicted one
contiguous *early* block, so surviving keys' baked-in attention onto the evicted
content was mostly stale; turn-aligned eviction removes *interleaved* early-to-mid
turns whose immediate successors (assistant acks, next user turns) sit well inside
the 4096-token sliding window and reference the removed turns. The survivors carry
strong baked-in influence from evicted neighbours, and the next-turn reply (which
attends back into the conversation) exposes it. Follow-ups: per-step KL to localize
drift within the reply; Tier-2 importance eviction; Tier-3 selective recompute of
high-deviation survivors (this is now motivated by data, not hygiene).

## Anomalies / caveats

1. **Degenerate reference reply.** Trinity's full-cache greedy 32-token reply ends
   the turn after ~17 tokens ("…next section when ready.<|im_end|>") then emits
   im_end/begin_of_text junk; part of the KL window is post-EOS. This does NOT
   explain P1's refutation — the recompute tracks full through the same positions
   (0.047) while rotation doesn't — but absolute KL magnitudes should be read with
   that in mind. Follow-up: per-step KL split at the first im_end, or --gen 16.
   (The low-entropy templated assistant acks are the design note's known caveat.)
2. **Small disk offload persisted** at `balanced --mem-frac 0.70` ("Some parameters
   … offloaded to the disk"; GPU allocs 71+82×6+123 ≈ 686 GiB < 727). exp08b's
   recipe didn't fully eliminate offload this time. Recompute timings (31.7/62.1 s)
   still in the expected GPU-resident order; rotation timings unaffected.
3. **Evict-frac overshoot:** requested 0.5, whole-turn granularity delivered 0.58
   at both lengths (5167/8921, 10199/17528). Expected behaviour, worth remembering
   when comparing across evict-frac cells.
4. Achieved lengths overshoot targets (~9k for 8192, ~17.5k for 16384) — the
   4-chars/token heuristic plus template scaffolding; reported as designed.
5. transformers 5.x `apply_chat_template` API drift (BatchEncoding) — fixed in
   `chat.py`, see "what ran". Also assorted harmless FutureWarnings (rope
   validation, afmoe `input_embeds`) and torchao .so load failures (pre-existing
   venv noise, non-fatal).

## Timing (rotation vs recompute)

Rotation surgery: 44 ms @8k / 61 ms @16k (13 whole turns evicted, 0 tokens
recomputed) vs shortened-prefill recompute 31.7 s / 62.1 s → **718× / 1020×**
(HF naive model-parallelism, same caveat as exp08b). Trinity load: 354 s.

## Sweeps

All 4 sweep runs (8 cells) exited 0, **zero failures** — the anticipated ef=0.7
fact-placement guard trip never occurred; the vault survived on token budget in
every cell. Recall was 16/16 across the whole experiment (vault retained ✓,
locker faithfully dropped ✓ in every cell including smoke and main).

| run | L | keep | ev turns | KL(full‖rot) | top1 | info KL(full‖short) | MECH KL(short‖rot) | t_rot | t_rec |
|---|---|---|---|---|---|---|---|---|---|
| ef30 | 8922 | 6054 | 7 | 0.371 | 94% | 0.029 | 0.463 | 66 ms | 54.0 s |
| ef30 | 17529 | 11792 | 7 | 0.151 | 94% | 0.025 | 0.207 | 102 ms | 44.3 s |
| ef50 (main) | 8922 | 3754 | 13 | 0.581 | 91% | 0.047 | 0.684 | 44 ms | 31.7 s |
| ef50 (main) | 17529 | 7329 | 13 | 0.469 | 78% | 0.158 | 0.414 | 61 ms | 62.1 s |
| ef70 | 8922 | 2282 | 17 | 0.900 | 72% | 0.307 | 1.06 | 40 ms | 32.7 s |
| ef70 | 17529 | 4369 | 17 | 0.500 | 78% | 0.371 | 0.859 | 33 ms | 55.3 s |
| t8 | 8762 | 3327 | 9 | 0.431 | 88% | 0.241 | 1.30 | 39 ms | 31.7 s |
| t8 | 17372 | 6542 | 9 | 0.718 | 75% | 0.336 | 1.28 | 51 ms | 50.5 s |
| t24 | 9425 | 4357 | 25 | 0.291 | 94% | 0.031 | 0.265 | 47 ms | 54.4 s |
| t24 | 18028 | 8304 | 25 | 0.434 | 91% | 0.102 | 0.414 | 64 ms | 40.2 s |

Patterns:

1. **Drift scales with evict fraction**, in both components: mech-KL rises
   0.21–0.46 (ef30) → 0.41–0.68 (ef50) → 0.86–1.06 (ef70), and info-loss KL rises
   with it (0.03 → 0.05–0.16 → 0.31–0.37). Mechanism error dominates info loss in
   every cell (ratio ~3–16×) — P1's refutation is not an artifact of one operating
   point.
2. **Turn granularity matters a lot at fixed ef=0.5**: fewer/bigger turns (t8:
   mech-KL 1.28–1.30) are much worse than many/smaller turns (t24: 0.27–0.41),
   with t12 in between. Finer-grained eviction is gentler on survivor KV —
   consistent with the interleaving story (big evicted blocks leave more baked-in
   references in their immediate survivors). t24 also posts the best top1
   (91–94%). This is the most actionable sweep result: granularity is a free
   policy knob.
3. Rotation surgery stayed 33–102 ms against 32–62 s recompute across all cells
   (~500–1300×, same HF naive-MP caveat).

## Machine state

Sweeps ran under nohup on node1 (`runs/exp09_sweeps.sh`, log `runs/exp09_sweeps.log`,
`ALL_SWEEPS_DONE` marker written). Final state verified 2026-07-10 ~06:15: no
exp09/trinity processes remaining, `nvidia-smi` compute-apps empty — node left
clean. All artifacts rsynced back to local `runs/`. (Session note: the local VPN
dropped twice during the runs; the nohup chain on node1 was unaffected.)

---

## Diagnostics (same day) — why chat-shaped eviction breaks the mechanism

> Pre-registered design: `notes/design-diagnostics-2026-07-10.md` (written before the
> GPU runs, after a tokenizer-only geometry check). Driver:
> `experiments/exp09_diagnostics.py` — single trinity load (`balanced --mem-frac 0.70`),
> all cells chained, total 1974 s, zero cell failures. Raw artifacts:
> `runs/exp09diag.{json,log}`, KV-geometry curves `runs/exp09diag_geo_*.npz`
> (per-layer × per-token rot-vs-short cosines, fp16), 3B smoke
> `runs/exp09diag_smoke_3b.json`. All local; node1 left clean, `runs/DIAG_DONE` written.

### Correction first: exp09's eviction was NEVER interleaved

Tokenizer-only check before any GPU work: `oldest_turns_to_evict` takes *consecutive*
turns and turn spans are back-to-back, so the evicted turns form ONE contiguous token
block — [32, 5199) of P=8921 @8k, [32, 10231) of 17528 @16k, contiguous in every
config. The draft interpretation above ("removes interleaved early-to-mid turns…") is
**wrong** and is withdrawn. exp09's geometry is exp08's geometry (kept prefix of 32
system tokens instead of 4 sinks, one interior block, kept tail). Second pre-run
observation: whole-turn rounding makes achieved evict-frac fall with turn count
(t8 0.62, t12 0.58, t24 0.54, t48 0.52), partially confounding the sweep's
"granularity" pattern with the evict-frac pattern.

### D2 — the 2×2 verdict: content-bundle, not geometry

Mech-KL(short‖rot), reply/continuation of 32 greedy tokens, matched evicted-token
counts (same K within a row):

| | contiguous sink4 mask (exp08 geometry) | exp09 turn-mask |
|---|---|---|
| **raw doc** | (a) exp08b banked: 2.1e-4–3.8e-4 @8k, 1.4e-4–4.7e-4 @16k | (c) **1.0e-4** @8922, **7.2e-5** @17529, top1 100% |
| **chat** | (d) **0.516** @8922, **0.392** @17529 | (b) **0.665** @8922, **0.274** @17529 (rerun of exp09 main; orig 0.684/0.414 — MoE run-to-run jitter) |

Corner (c) transplants exp09's *exact* keep mask onto the same corpus prose tokenized
raw (same C, same joiner): mech-KL collapses to exp08b levels — a ~4000–6000× drop
with geometry held bit-identical. Corner (d) forces exp08's exact `sink_window`
geometry onto the chat cache (system prompt tokens [4,32) evicted, far edge cutting
mid-turn): mech-KL stays at exp09 levels. **Both length cells agree. The driver is
the content bundle (chat template + boilerplate acks + reply-style measurement
point), not eviction geometry** — and turn-alignment isn't protective either: (d)
butchers the template mid-turn and is no worse than (b).

### D1 — KV geometry: contamination is real, similar in both regimes, and only chat *reads* it

Rotated-survivor KV is identical to the full-prefill KV except exact key re-rotation
(values verbatim, NoPE keys untouched), so rot-vs-short divergence measures the
**baked-in contamination** — "KV computed with the evicted block present vs recomputed
without it" — directly. Cosine similarity by token distance d past the seam
(mean over layers in group; keys K / values V; @8922, turn-mask):

| zone | chat K slide/glob | chat V | raw K slide/glob | raw V |
|---|---|---|---|---|
| pre-block (32 toks) | 0.998 / 0.996 | 0.992 | 0.999 / 0.999 | 0.997 |
| d[0,16) | 0.850 / **0.726** | 0.74 | 0.844 / **0.725** | 0.78 |
| d[16,64) | 0.855 / 0.726 | 0.77 | 0.918 / 0.822 | 0.87 |
| d[64,256) | 0.954 / 0.885 | 0.92 | 0.958 / 0.910 | 0.93 |
| d[1024,4096) | 0.976 / 0.956 | 0.96 | 0.988 / 0.975 | 0.97 |

- **E1.1 ✓** pre-block survivors sit at the numerical noise floor (0.992–0.999).
- **E1.2 ✓** contamination is worst immediately after the seam and decays
  monotonically; it never fully dies (0.95–0.99 even at d≥4096, 16k cell).
- **E1.3 ✓** values diverge as much as keys, and V-slide ≈ V-glob in every bin (values
  come from the same hidden states — a good internal consistency check).
- **Pre-registration REFUTED in direction:** global **NoPE keys are the MOST
  contaminated group everywhere** (they integrate the whole context; sliding keys
  only carry window-local information). Worst layers are stable across all corners:
  late global layers for keys (L59, L55, L35, L31, L51 ≈ 0.68–0.81 near-seam), late
  sliding layers for values (L56, L58, L53, L49, L46).
- **E1.4 ✓ — the headline.** At d[0,16) chat and raw contamination are *identical*
  (0.850/0.726 vs 0.844/0.725). Chat decays a bit slower (~1–3% lower cosine in
  mid-distance bins), but that's nowhere near the 3.5–4 orders separating their
  mech-KLs. **The pathology is not contamination magnitude; it's readout.** A chat
  reply is a global operation over the conversational scaffold (acks, turn structure,
  "running summary" semantics) and attends into the contaminated near-seam survivors;
  greedy doc continuation reads mostly clean local prose. The feasibility law
  generalizes: mechanism error ≈ future attention mass onto *contaminated survivors*,
  the same way info-loss ≈ attention onto *evicted content*.

### D3 — per-step KL: real drift, inflated ~4× by the post-EOS tail

First `<|im_end|>` (id 3) lands at step 16 of 32 in every chat cell (same degenerate
reply as the main run). Mech-KL(short‖rot) split pre/post first-im_end:

| cell | pooled | pre-im_end (live reply) | post-im_end (junk tail) |
|---|---|---|---|
| chat turn-mask @8922 | 0.665 | **0.172** | 1.224 |
| chat sink4 @8922 | 0.516 | 0.244 | 0.824 |
| chat turn-mask @17529 | 0.274 | **0.129** | 0.438 |
| chat sink4 @17529 | 0.392 | 0.087 | 0.737 |
| t48 @10432 / @19030 | 0.380 / 0.365 | 0.198 / 0.026 | 0.585 / 0.750 |

**P1's refutation stands under the pre-registered rule** (pre-im_end ≥ 0.1 for the
main cells): the honest headline number shrinks from ~0.7 to **~0.13–0.17, still
≥1000× the raw-doc mechanism error** measured through the *same* pipeline (corner (c)
per-step KLs are flat ~1e-5–1e-4 at every step). Within the live reply the profile is
spiky, not monotone: large KL at content-bearing decision steps (0.2–0.9), ≈0 at
template-forced tokens. The single largest divergence everywhere is **step 17 — the
first post-im_end token** (spikes 4–9 nats): what the rotated cache wants to do
*after* the turn closes diverges hardest. Deployment note: a serving stack that stops
at im_end never samples the worst region, but the pre-im_end drift alone is already
disqualifying for oldest-turn eviction. Caveat: single conversation per cell and
spike-dominated 17-step means → per-cell pre-im_end differences (e.g. 0.087 vs 0.244)
are noise; the chat-vs-raw orders-of-magnitude gap is the robust signal.

### D5 — granularity: the benefit saturates (and was partly the evict-frac curve)

t48 cells: mech-KL **0.380** @C=10432 (achieved ef 0.516), **0.365** @C=19030 (0.517).
The sweep curve t8 → t12 → t24 → t48 is 1.30 → 0.68 → 0.27 → **0.38** @8k-scale:
**non-monotone — the improvement stops at t24.** Neither pre-registered hypothesis
wins cleanly: the frac-only interpolation predicted ~0.5–0.6 (t48 lands below it, so
granularity is not epiphenomenal), the naive granularity trend predicted ≤0.15 (t48
lands well above). Finer-than-t24 granularity buys nothing; residual mech-KL at
ef≈0.5 plateaus around 0.3–0.4 regardless. Consistent with D1: t48's near-seam
contamination is actually the *worst* measured (K-glob 0.634 at d[0,16)) — many small
turns put ack/turn-boundary scaffolding (the contamination-heavy tokens) right at the
seam. "Granularity is a free policy knob" (sweep pattern 2 above) is hereby demoted:
it helps from t8 to t24, then stops.

### What actually breaks (synthesis)

Eviction bakes contamination into every survivor computed after the evicted block —
that is true on raw docs and chat alike, to nearly identical magnitude (D1), and it
is the same contamination that makes rotation *closer to full* than a recompute when
it's benign (exp05/08b). What chat changes is **what the next tokens read**: an
assistant reply is a low-entropy, template-locked, globally-attending operation over
the conversational scaffold, so its next-token distribution is exquisitely sensitive
to perturbations in the near-seam survivors (acks referencing evicted turns, turn
headers, the "running summary" frame), while a doc continuation reads recent clean
prose and doesn't care (D2: swapping content moves mech-KL ~4000×; swapping geometry
moves it ~1.5×). The drift is real within the live reply (D3: ~0.13–0.17, ≥1000×
raw-doc), worst at the first post-reply decision, and not fixable by finer turn
granularity (D5 plateau). Indicated fixes, in order: **Tier-3 selective recompute of
the near-seam survivor band** (d<256 covers the worst contamination for ~7% of K at
8k — and D1 says where: late global-layer keys, late sliding-layer values), Tier-2
importance eviction to stop evicting still-referenced turns, or turn-summarization
that replaces rather than amputates. Contamination-aware serving beats
contamination-free serving — but only if the reply isn't about to read the wound.

### Machine state (diagnostics)

Run under nohup on node1 (`runs/exp09diag.log`), EXIT_CODE=0, 1974 s wall, zero cell
failures. GPUs verified clean after the run (`nvidia-smi` compute-apps empty, no
stray processes); all artifacts rsynced to local `runs/`;
`~/luxi-files/kv-rotation/runs/DIAG_DONE` written 2026-07-10T16:14Z for the exp10
agent. Heimdall queue was empty at launch. Same small disk-offload warning as the
main run (`balanced --mem-frac 0.70`, g7 at 123 GiB); rotation timings unaffected.

## exp10 — naturalized dialogue (same day): the drift was the scaffold, not the shape

> Runner: exp10 agent session, node1 (8x B200, taken after DIAG_DONE). Design +
> pre-registered expectations: `notes/design-natural-chat.md` (incl. §6b readout
> addendum and §6c two-stage course correction, both written before any run).
> Raw artifacts: `runs/exp10_stage1_ef50.{json,log}` (local copies pulled),
> data: `data/backrooms_convs_2026-07-10.jsonl` (converter:
> `experiments/exp10_convert_corpus.py`), eval: `experiments/exp10_natural_chat.py`.

### Headline

**exp09's P3/P1-refutation does NOT survive naturalization — H-content confirmed,
decisively.** On naturalized backrooms dialogue (kv_perturb corpus, 82–85%
model-generated, multiparty group chat rendered through the same chat template with
the same turn-aligned oldest-first eviction, same facts, same probes, ef=0.5),
mech-KL(short‖rot) collapses from exp09's 0.41–0.68 to **0.039–0.042 (means, n=8)**
— 16x @8k / 11x @16k below exp09-main, 6–11x below even the granularity-matched t24
comparator, past the pre-registered ≥5x H-content threshold, and below the
diagnostics live-reply anchor (0.17/0.13). In **all 16 of 16 cells** rotation is
*closer to full than the clean recompute is* (KL(full‖rot) < KL(full‖short)) — the
raw-doc continuity advantage (exp05/07/08b) reappears on chat-shaped contexts once
the content is natural. Recall was 16/16 clean (vault preserved at full-cache level,
locker faithfully dropped to lp ≈ −8…−12.6). Zero failures, zero guard trips.

Verdict on the exp09 story: the mechanism was never "broken by chat shape" — it was
broken by what the measured reply was *reading*: exp09's low-entropy templated
acknowledgment scaffold (consistent with the same-day diagnostics: content swap
moves mech-KL ~4000x, geometry ~1.5x). On naturalized dialogue, turn-aligned
oldest-turn eviction + Tier-0 rotation is back in the deployable regime:
mech ≈ info-loss ≈ a few e-2, top1 91–92%, surgery 10–23 ms vs 13–40 s recompute
(~1200–3600x, HF naive-MP caveat as always).

### Data (stage 1: existing kv_perturb backrooms corpus)

57 transcripts survive at `data/backrooms_corpus/{chat,prefill}/*/transcript.json`
(rsynced from node2:/models/kv_perturb_experiment/tier2_runs/ — node2's models/
partition is NOT backed up; these are the only copies. **Provenance correction
made during this session:** the bot "Cogito" is Cogito v2.1 671B
(`/models/cogito-671b-v2.1`, a DeepSeek-V3-arch derivative, served TP16 across
both nodes; the "trinity-probe" tag in the generation logs is the serving
infra's provider label, not the model) — NOT Trinity-Large-Preview.
Interlocutors: Opus-3 auditor, Opus-4.7 seeds, lyra (human-roleplay), others.
~27k approx tokens each, 82–85% model-generated).
Conversion = kv_perturb's own `to_format_b` mapping (bot msgs → raw assistant
turns; consecutive non-bot msgs → one `speaker: content` user turn; no system
message — source had none; sinks kept token-wise). Facts spliced post-hoc,
byte-identical strings/probes to exp09/exp08. Headline cells: the first 8
chat-condition conversations x L∈{8192,16384} x ef=0.5, --with-recompute,
--device-map even --mem-frac 0.95.

### Results (per cell; KL over 32 teacher-forced reply tokens)

| conv (chat-) | L | keep | evict turns | toks/turn | KL(full‖rot) | top1 | info KL(f‖s) | MECH KL(s‖r) | t_rot | t_rec |
|---|---|---|---|---|---|---|---|---|---|---|
| 0006 | 8588 | 4113/8587 | 1 | 477 | 1.55e-2 | 88% | 6.38e-2 | 4.95e-2 | 23 ms | 39.3 s |
| 0006 | 17686 | 8719/17685 | 20 | 376 | 3.10e-2 | 91% | 3.29e-2 | 3.26e-2 | 15 ms | 40.4 s |
| 0012 | 9158 | 4370/9157 | 3 | 509 | 2.83e-2 | 91% | 7.83e-2 | 4.40e-2 | 11 ms | 32.6 s |
| 0012 | 16458 | 8068/16457 | 15 | 261 | 2.27e-2 | 88% | 7.06e-2 | 5.39e-2 | 14 ms | 26.1 s |
| 0013 | 8507 | 4175/8506 | 5 | 370 | 1.42e-2 | 97% | 3.26e-2 | 2.62e-2 | 10 ms | 36.7 s |
| 0013 | 16457 | 7970/16456 | 23 | 265 | 2.47e-2 | 97% | 4.18e-2 | 4.36e-2 | 14 ms | 33.1 s |
| 0028 | 8530 | 4039/8529 | 2 | 316 | 3.50e-2 | 94% | 4.13e-2 | 2.95e-2 | 11 ms | 13.5 s |
| 0028 | 16490 | 7980/16489 | 27 | 250 | 4.21e-2 | 81% | 5.56e-2 | 3.75e-2 | 14 ms | 28.9 s |
| 0039 | 9029 | 4265/9028 | 3 | 334 | 5.16e-2 | 91% | 7.99e-2 | 5.03e-2 | 11 ms | 39.3 s |
| 0039 | 16681 | 8190/16680 | 25 | 265 | 2.31e-2 | 94% | 4.16e-2 | 4.35e-2 | 14 ms | 32.1 s |
| 0056 | 8735 | 3967/8734 | 2 | 336 | 6.81e-2 | 88% | 8.39e-2 | 2.95e-2 | 10 ms | 13.4 s |
| 0056 | 16837 | 8122/16836 | 26 | 234 | 3.85e-2 | 94% | 6.42e-2 | 3.61e-2 | 14 ms | 31.9 s |
| 0070 | 8760 | 4295/8759 | 3 | 438 | 1.94e-2 | 94% | 6.23e-2 | 5.92e-2 | 10 ms | 34.5 s |
| 0070 | 17075 | 8335/17074 | 20 | 300 | 1.31e-2 | 94% | 4.22e-2 | 3.93e-2 | 14 ms | 38.5 s |
| 0075 | 8705 | 4108/8704 | 2 | 622 | 2.64e-2 | 94% | 4.30e-2 | 4.56e-2 | 11 ms | 27.1 s |
| 0075 | 16693 | 8187/16692 | 12 | 303 | 2.09e-2 | 91% | 2.54e-2 | 2.16e-2 | 14 ms | 38.2 s |

Aggregates (mean/median across n=8 conversations):

| L | KL(full‖rot) | top1 | info KL(full‖short) | MECH KL(short‖rot) |
|---|---|---|---|---|
| 8192 | 3.23e-2 / 2.73e-2 | 92% | 6.06e-2 / 6.30e-2 | **4.17e-2 / 4.48e-2** |
| 16384 | 2.70e-2 / 2.39e-2 | 91% | 4.68e-2 / 4.20e-2 | **3.85e-2 / 3.84e-2** |

### Pre-registered verdicts (design-natural-chat.md §6/§6b)

- **E1 (decision variable): H-content CONFIRMED.** mech 0.042/0.039 vs exp09-t12
  0.684/0.414 (16x/11x) and vs t24 0.265/0.414 (6.4x/10.7x) — all ≥5x. Also below
  the §6b readout anchor (0.17/0.13): a fully-natural reply over fully-natural
  context is cleaner still than exp09's live-reply-pre-im_end slice.
- **E2 (recall): CONFIRMED 16/16.** Vault rot-lp −0.05…−0.14 (≈ or better than
  full); locker dropped to −8.2…−12.6. One footnote: chat-0013@8k's FULL cache
  itself missed greedy locker recall (lp −1.13) — natural multiparty context makes
  bare-text probe recall slightly harder for the reference too; rotation-vs-full
  comparisons unaffected.
- **E3 (info-loss rises, mech falls — dissociation): CONFIRMED.** info-loss
  0.047–0.061 (means) is same-order-but-higher than exp09-main @8k (0.047) while
  mech collapsed ~15x; in 13/16 cells mech < info. Forgetting natural turns costs
  information; re-rotating survivors costs almost nothing.
- **E4 (reply health): CONFIRMED.** first-eos = −1 (no eos) in all 16 reference
  replies; per-step KL series flat (no post-degeneracy inflation). exp09 caveat #1
  does not apply to these numbers.
- **E5 (safety): CONFIRMED.** Prefix-stability held, applies_rope=45/60, guards
  clean, no failures.

### Caveats

1. **Multiparty, not two-party.** Stage-1 content is group chat with a human and
   several Claude-family models; ~48% of tokens are Trinity's own. Clean two-party
   Trinity↔DeepSeek dialogue is stage 2 (`design-natural-chat.md` §6c), which also
   removes any Claude-flavored register.
2. **Granularity differs both ways**: grouped seed blocks make the 8k cells evict
   1–5 huge turns (~4.5k tokens as one block — t8-like, exp09's *worst* shape, yet
   mech stayed ≤0.06), while 16k cells evict 12–27 mixed turns. The effect is flat
   across that whole range — consistent with diagnostics D5 (granularity plateau)
   and further evidence the exp09 sweep's granularity slope was also readout-driven.
3. Facts are still English passcode sentences spliced into a stylized register —
   mildly out-of-register, kept byte-identical to exp09/08 for comparability.
4. approx_tokens (chars/4) undercounts trinity tokens by ~10–15% on this corpus
   (27k approx → 26.5–27.5k source chars rendered to >17.7k template tokens for
   16k cuts with room to spare); all 57 convs support both cells.
5. **The content is NOT Trinity-authored** (see provenance correction above):
   the bot turns were generated by Cogito 671B, the rest by Claude-family models
   and a human. Generated July 2026 ⇒ cannot be in Trinity's training data. This
   *strengthens* the H-content reading — even non-self-generated natural dialogue
   collapses the drift ~15x, so the effect is natural-vs-templated register, not
   self-generation — but it means the "Trinity evaluating its own voice" axis is
   untested in stage 1; that is precisely what stage 2 (Trinity↔DeepSeek fresh
   generation, Trinity writing its own turns) adds.

### Infrastructure notes (hard-won, apply to any future trinity run)

- **accelerate's `balanced`/`auto` placement is untrustworthy for afmoe**: at
  mem-frac 0.70/0.85/0.95 it repeatedly stacked ~2x layers (~176 GB) on GPU 7,
  then silently disk-offloaded the tail (layers.59, norm, rotary_emb, lm_head) —
  "Some parameters are on the meta device" — wedging decode below ~0.6 tok/s
  (this also retroactively explains exp09's "small disk offload persisted"
  anomaly #2 and its 240 ms cold rotation). Fix: `--device-map even`
  (`kvrot.natural.trinity_even_device_map`, explicit per-layer dict: embed+7 on
  g0, 7–8/GPU middle, 7+norm+rotary+lm_head on g7 → ~9–59B params/GPU, zero
  offload) plus `kvrot.natural.assert_no_offload` (per-device param counts logged,
  hard error on any disk/cpu/meta placement) now wired into both exp10 scripts.
  Load: 190 s.
- transformers' load-time warning "generation flags are not valid:
  ['temperature','top_p']" comes from the checkpoint's generation_config and is
  inert for exp10 (all sampling is manual nucleus in `TrinityTurnGenerator`;
  `model.generate()` is never called).
- node2's vLLM (deepseek-v3-base, /models/DeepSeek-V3-Base, max_model_len 16384)
  is reachable and fast (0.9–2.2 s per ~200-token completion) but NOT
  seed-deterministic under continuous batching — stage-2 transcripts are canonical
  as banked jsonl, seeds recorded for provenance only.

### Machine state

Run under nohup on node1 (`runs/exp10_stage1_ef50.log`), 4233 s wall, exit clean,
zero cell failures. GPUs verified drained after the run (compute-apps empty, all
8 at 0 MiB); artifacts rsynced local. Stage-2 (fresh Trinity↔DeepSeek generation
via the probe-server path, `backrooms_gen.py` adaptation) is designed and
pre-registered but not yet run.

## exp10 — stage 2 (fresh Trinity↔DeepSeek generation): BLOCKED by a Trinity chat pathology

> Attempted same day after stage 1. Code: `experiments/exp10_natural_gen2.py`
> (HTTP-only two-party generation: Trinity via a temporary vLLM server on
> node1:8001, DeepSeek-V3-Base via node2's production vLLM), `TrinityChatClient`
> + `TrinityTurnGenerator` in `experiments/exp10_natural_gen.py`. Logs:
> `runs/exp10_gen2.log`, `runs/trinity_vllm.log`.

### What worked

- **Serving Trinity is viable and fast.** node1's kimi/glm5/mistral conda envs
  carry vLLM 0.16 with afmoe support; `vllm serve /models/Trinity-Large-Preview
  --served-model-name trinity -tp 8 --max-model-len 32768 --gpu-mem 0.85` came up
  in ~5 min and generated at **~100 tok/s** (vs the in-process HF path's 0.2 tok/s
  under the same even placement — HF naive-MP single-GPU-at-a-time decode is
  simply unusable for generation; use the server). This is a reusable recipe for
  any future trinity generation.
- node2's DeepSeek path ran flawlessly at ~95 tok/s throughout.
- The two-party growth loop, per-turn banking, sentence-trim of cap-truncated
  turns, and prompt windowing all worked; conversations grew coherently and
  quickly to ~5k render tokens.

### What broke (systematic)

**Trinity-Large-Preview degenerates into a special-token repetition basin in
plain deep multi-turn chat.** At ~5k–5.6k accumulated tokens (turn ~15) every one
of the 8 conversations hit a state where Trinity emits 450 tokens that decode to
pure `<|begin_of_text|>` repetition (finish_reason=length, `message.content`
empty after special-token stripping). It is **context-locked, not stochastic**:
once a conversation reaches the basin, all 3 seed-retries fail at that exact
context, and the failure recurs across all 8 topics. Shallow context (≤~3 turns)
generates perfectly coherent, on-register prose.

Sampling-parameter fixes do NOT work (verified at the failing context, 3 seeds
each): `repetition_penalty` 1.05–1.15, `frequency_penalty` 0.3, `bad_words`
["<|begin_of_text|>"], and OpenAI `logit_bias` {0: −100} on the BOS token — the
last merely displaces the loop onto other special tokens (content still empty).
This is a model/inference-format instability, not a decoding-noise problem.

### Root cause & the documented fix (for the next agent)

kv_perturb's `backrooms_gen.py` (which generated the stage-1 corpus) deliberately
does **not** use plain chat format for the bot. It uses a **CLI-simulation
prefill format** (`to_format_a`: one assistant turn carrying the whole transcript
with `<|EOT|>` delimiters, system = "The system is in CLI simulation mode.") plus
**burst generation** (generate many turns in one completion, parse afterward) —
with an explicit code comment that "Cogito refuses to roleplay" in the chat
format. That is the same class of pathology observed here, solved there by the
prefill framing. Stage 2, done properly, should generate via node1's Trinity vLLM
server using kv_perturb's prefill/burst path (`/v1/completions` on the rendered
CLI-prefill prompt with `<|EOT|>` stops), not `/v1/chat/completions`. That
machinery is ~2000 lines in `backrooms_gen.py` and adapting it is the concrete
stage-2 task; it was out of budget this session.

### Why this does not weaken the exp10 conclusion

Stage 1 already answers the pre-registered question (E1: H-content confirmed,
≥5x threshold cleared 6–16x over, 16/16 cells) on **naturalized model-generated
dialogue through the identical eval path**. Stage 2's marginal value is narrower:
(a) clean two-party structure vs stage-1's multiparty group chat, and (b) Trinity
reading its *own* generated voice vs Cogito's. Neither is load-bearing for the
"naturalization collapses the exp09 drift" finding — which stands. Stage 2 remains
worth doing for completeness once the prefill-format generator is ported.

### Machine state

gen2 driver and the temporary Trinity vLLM server (node1:8001) both stopped; its
8 TP workers were orphaned by the parent kill and reaped explicitly by pid.
Final state verified: `nvidia-smi` compute-apps empty, all 8 GPUs at 0 MiB, no
stray processes. Node2's production DeepSeek server was only ever queried over
HTTP and was never touched. Scratch generation files removed from node1;
`runs/exp10_gen2.log` + `runs/trinity_vllm.log` pulled local for the record.
