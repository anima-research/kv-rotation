# Design — exp10: naturalized dialogue (backrooms-style Trinity ↔ DeepSeek)

> STATUS: DESIGNED 2026-07-10 (pre-registered before any generation or GPU run).
> SCOPE: reruns exp09's turn-aligned eviction+rotation eval on NATURALIZED
> conversations instead of synthetic template-scaffolded ones, per user direction:
> "stop using adhoc data — do backrooms-style generation with Trinity so the
> content is naturalized." Code: `src/kvrot/natural.py`,
> `experiments/exp10_natural_gen.py` (generation),
> `experiments/exp10_natural_chat.py` (eval), `tests/test_natural.py`.
> Companion: `notes/design-chat-eval.md` (exp09), `notes/journal-exp09-2026-07-10-draft.md`.

## 1. The question exp10 answers

exp09 refuted P1 loudly: on chat-shaped contexts, mech-KL(short‖rot) came back
0.41–0.68 at ef=0.5 — three orders above every raw-doc measurement. But exp09's
conversations were synthetic: real prose chunk-fed by templated user leads, with
three rotating low-entropy assistant acknowledgments ("Noted. I'm tracking…").
The design note's own honest caveat: that measures chat-**shaped** contexts, not
chat-**distributed** content. Two live hypotheses:

- **H-content:** the synthetic scaffolding inflated the drift — canned acks are
  low-entropy attractors, chunk-boundaries are unnatural, and the "reply" being
  measured was itself degenerate (exp09 caveat #1). Natural dialogue → materially
  lower mech-KL.
- **H-structure:** the contamination is content-independent — survivors near the
  eviction boundary carry baked-in attention onto interleaved evicted turns no
  matter what the turns say. Natural dialogue → numbers hold.

## 2. Data: model-generated backrooms conversations

Two AI systems in open-ended dialogue — curious, associative, no assistant-persona
scaffolding, no human, no task. This is the naturalized register: every turn is
written by a live model conditioned on the whole (windowed) conversation so far.

- **Speakers.** `Trinity-Large-Preview` (HF in-process on node1, the model under
  eval — its own turns are maximally in-distribution for it) alternating with
  `deepseek-v3-base` served by node2's production vLLM (`/models/DeepSeek-V3-Base`,
  max_model_len 16384; identity confirmed via `/v1/models` 2026-07-10 and recorded
  in run metadata).
- **N = 8 independent conversations**, distinct hand-written seed topics/openers
  (memory-as-riverbed, reading-without-remembering, cities-as-organisms,
  translation & the shape of thought, deep time & fragile records, play & rules,
  ship-of-Theseus for software selves, silence in music). Opener ≈150–200 words,
  developed, sets a substantial-turn norm (verified 2026-07-10: a developed opener
  pulls DeepSeek turns from ~70 to ~200+ tokens).
- **Roles for the trinity render:** DeepSeek turns = `user`, Trinity turns =
  `assistant`, plus a short scene-setting `system` line (also holds the sinks and
  is eviction-protected, as in exp09). The opener is DeepSeek's first turn
  (hand-written, flagged `generated=false` in metadata).
- **DeepSeek side:** base model → raw `/v1/completions` with a dialogue-transcript
  prompt (`Trinity: …\n\nDeepSeek: …`) and stop sequences `\nTrinity:`/`\nDeepSeek:`
  — the classic backrooms idiom. max_tokens 450, temperature 0.9, top_p 0.95,
  seed recorded per request. Prompt windowed to fit node2's 16384 cap: keep the
  preamble, drop oldest turns wholesale behind an `[earlier conversation omitted]`
  marker (char-budgeted, shrink-and-retry on context-length 400s). Request rate:
  sequential, retry with backoff.
- **Trinity side:** in-process sampled decode (temperature 0.9, top_p 0.95,
  max_new_tokens 450, stop at eos/im_end), torch seed fixed per (conv, turn).
  KV carried across turns within a conversation via an exact token-prefix check
  (fall back to full re-prefill on any mismatch — correctness never depends on
  the carry).
- **Turn lengths are the models' choice** under the 450 cap (~150–400 target set
  by register, not forced): turns <30 tokens are retried up to 3× (new seed),
  then the longest attempt is accepted and flagged. Cross-speaker label bleed
  ("Trinity:" appearing mid-turn) is truncated at the offending line.
- **Growth target:** each conversation grows until its trinity-chat-template
  render is ≥ **18,500 tokens** and ends on a DeepSeek (user) turn — so both the
  8k and 16k eval cells cut at turn boundaries from the same conversation.
  Safety cap 200 turns.
- **Banking / resumability:** every turn is appended to
  `data/natural_turns_2026-07-10.jsonl` the moment it lands (typed records:
  speaker, model identity, sampling params, seed, token counts, timestamps);
  a crash loses nothing and a rerun resumes mid-conversation. A finalize step
  compacts to `data/natural_convs_2026-07-10.jsonl` (one conversation per line).

### Reproducibility caveat (pre-registered)

node2's vLLM is NOT seed-deterministic (verified 2026-07-10: identical request +
seed twice → different text; expected under continuous batching). Therefore the
**saved jsonl transcript is the canonical artifact** — seeds and params are
recorded for provenance, not for bit-reproduction. The eval itself is fully
deterministic given the jsonl.

## 3. Fact probes (P2 machinery, unchanged)

Probes are spliced into the transcript **at eval time, post-generation** (never
shown to either generator), for two reasons: (a) it prevents either model from
repeating a passcode into a turn with the opposite eviction fate, which would
break the retain/drop design; (b) it exactly matches exp09, where the canned acks
likewise never referenced the facts. Fact strings, passcodes, probe Q&A suffixes,
and placement rule are **byte-identical to exp09**:

- `"By the way, my locker code is 62483."` appended to the **first user turn**
  (the opener) → destined for EVICTION.
- `"Also, note this down: the vault passcode is 73914."` appended to the
  **second-to-last user turn of the cut** → RETAINED (survives on token budget
  at ef=0.5, protected in spirit by proximity + `protect_last=2`, with the same
  placement guard exp09 used: abort the cell if the evict set disagrees).
- Recall via `answer_logprob` with exp09's plain-text Q&A suffixes.
- Guard: scan the transcript for either passcode outside its planted turn; abort
  the cell if found (cannot happen given post-hoc splicing, but cheap to assert).

## 4. Eval protocol (identical decomposition to exp09)

Per conversation × per length L ∈ {8192, 16384}:

1. **Cut** the conversation at a turn boundary: smallest message-prefix whose
   trinity render (with generation prompt) is ≥ L and ends on a user turn —
   same overshoot-and-report behaviour as exp09.
2. Plant the two facts (§3) on a copy; render via `turn_token_spans` (same
   prefix-stability assert).
3. Turn-aligned eviction: `oldest_turns_to_evict(target=ef·P, protect_last=2)`
   + `turn_keep_indices(sinks=4)`; Tier-0 re-rotation with recompaction.
4. Reference = full cache's own greedy 32-token next-turn reply; teacher-force
   under rotation → KL(full‖rot) + top1. With `--with-recompute`: shortened
   prefill → info-loss KL(full‖short) and MECH KL(short‖rot).
5. Recall probes for both facts, full vs rotation.
6. Timings: rotation surgery vs recompute prefill.

New instrumentation (addresses exp09 caveat #1, additive only): the **per-step
KL series** and the index of the first eos/im_end in the reference reply are
stored per row, so drift can be split at the degeneracy point post hoc.

Cells: ef=0.5 on all 8 conversations × both lengths (16 rows) is the headline;
ef ∈ {0.3, 0.7} on a subset if time. Aggregate = mean and median across
conversations per (L, ef) cell — exp09 had one conversation per cell; exp10's
N=8 gives a spread for free.

## 5. Comparability covariates (pre-registered, not tunable after the fact)

- **Turn granularity.** Natural turns will run ~150–400 tokens; exp09-main (t12)
  had ~700 (8k) / ~1400 (16k) tokens per turn, and exp09 showed granularity
  matters (t8 mech 1.3 vs t24 0.27–0.41). The honest comparator set is therefore
  the whole exp09 granularity family {t8, t12, t24}, with **t24 (~380/750
  toks/turn) the nearest neighbour**, not t12 alone. Achieved tokens-per-turn is
  reported per cell.
- **Evict-frac overshoot.** Whole-turn granularity overshoots ef; finer natural
  turns should overshoot *less* than exp09's 0.5→0.58. Achieved keep-fraction
  reported per cell.
- **Reply register.** exp09's measured reply was a templated ack that went
  degenerate after ~17 tokens; exp10's reply is a live conversational turn.
  This is part of the treatment (naturalization), not a confound to remove —
  the per-step KL series quantifies it.

## 6. Pre-registered expectations

- **E1 (the decision variable): mech-KL(short‖rot) at ef=0.5.**
  - If **H-content** (synthetic scaffolding inflated drift): natural dialogue
    posts mech-KL **≥5× lower** than the granularity-matched exp09 comparator
    (t24: 0.27 @8k / 0.41 @16k) — i.e. ≲0.05, heading back toward raw-doc
    territory. Verdict: exp09's P3 was partly an artifact; re-run policy
    conclusions on natural data.
  - If **H-structure** (contamination is content-independent): mech-KL within
    **~2× either way** of the t24 comparator (~0.13–0.85). Verdict: exp09's
    finding stands on naturalized data — turn-aligned oldest-turn eviction
    genuinely needs Tier-2/Tier-3 help in chat deployments.
  - In between (5×–2×): partial inflation; report as graded, no clean verdict.
- **E2 (recall):** unchanged from exp09 — vault recalls under rotation ≈ full
  (lp within ±0.5), locker drops to ≈ chance, in every cell. Any leak/loss is a
  bug in the cut/plant logic before it is a finding.
- **E3 (info-loss):** KL(full‖short) should *rise* vs exp09 (natural turns
  actually reference each other, so forgetting them costs more information than
  forgetting chunk-feeding scaffold). If info-loss rises while mech-KL falls,
  that dissociation is itself evidence for H-content.
- **E4 (reply health):** natural greedy replies degenerate later (first eos
  beyond token 24 on most cells) — checked via the new per-step series.
- **E5 (safety):** prefix-stability assert, per-layer RoPE gating, and guards
  behave exactly as in exp09 (no new mechanism is being tested).

## 6b. Addendum 2026-07-10 (post-diagnostics, pre-generation)

Recorded BEFORE any exp10 generation or eval ran, but AFTER the design above was
written: the parallel diagnostics session on exp09 reported that the drift
pathology is **readout, not contamination magnitude** — contamination at the
eviction seam is essentially identical for chat-shaped and raw-doc contexts;
swapping the *content* being read out moves mech-KL ~4000×, while geometry moves
it only ~1.5×; and on a live (non-templated) reply, exp09's mech-KL pre-im_end
was already down to 0.17/0.13. This sharpens E1's fork rather than changing it:
exp10's naturalized dialogue removes exactly the low-entropy templated-scaffold
readout that diagnostics implicates, so **H-content is now the favoured
hypothesis, with a quantitative anchor: expect mech-KL around ~0.1–0.2 or below
at ef=0.5** (the diagnostics live-reply numbers), rather than exp09's 0.41–0.68.
If exp10 instead reproduces ~0.4+, that would contradict the readout story and
point back at structure. E2–E5 unchanged.

## 6c. Addendum 2026-07-10 (course correction: two stages, corpus first)

User direction via coordinator, recorded before any exp10 GPU eval ran:
kv_perturb_exp already solved backrooms generation for this purpose, and its
surviving transcripts make the naturalization question answerable WITHOUT fresh
generation. exp10 is therefore two pre-registered stages:

**Stage 1 (headline): existing kv_perturb corpus.** 57 naturalized backrooms
transcripts (`data/backrooms_corpus/{chat,prefill}/*/transcript.json`, rsynced
from node2:/models/kv_perturb_experiment/tier2_runs/): multiparty Discord-style
group chats, ~27k approx tokens each, 82–85% model-generated. Bot "Cogito" =
Cogito v2.1 671B (DeepSeek-V3-arch; the "trinity-probe" provider tag is the
serving infra's label — provenance corrected 2026-07-10 during the session,
before stage-2 ran; the eval target model remains Trinity-Large-Preview). Conversion to eval form uses
kv_perturb's own `to_format_b` mapping (bot msgs → raw assistant turns;
consecutive non-bot msgs → one user turn of `speaker: content` lines;
alternation guaranteed; NO system message — the source had none, sinks are
protected token-wise). This holds exp09's SHAPE fixed (Trinity chat template,
turn-aligned oldest-first eviction, same facts/probes/decomposition) and swaps
only CONTENT — the cleanest possible test of H-content vs H-structure. Facts
are spliced post-hoc exactly as §3. Headline cells: 8 chat-condition
conversations × L ∈ {8192, 16384} × ef=0.5 with recompute; prefill-condition
runs (finer, burstier turns: ~70 vs ~300 toks/msg) become a granularity
contrast if time permits. Caveat to report: this content is Trinity+Opus-era
group chat (multiple speakers, one a human), not clean two-party AI dialogue —
that axis arrives in stage 2.

**Stage 2: fresh Trinity ↔ DeepSeek-V3 generation** (the original ask, §2),
now to be adapted from kv_perturb's `backrooms_gen.py` rather than written
fresh: keep its formats + turn-boundary outputs, swap the Opus interlocutor for
node2's DeepSeek API, and prefer its Trinity probe-server generation path
(OpenAI-compatible server on node1) over in-process HF if the server recipe is
revivable — in-process HF decode measured ~sub-1 tok/s under disk offload, and
the offload/do-sample fixes below apply only if in-process generation is still
needed. Trinity prompting lore inherited from backrooms_gen.py: Trinity balks
at the canonical ChapterX CLI system message; use the SHORT variant
("The system is in CLI simulation mode.").

Expectations E1–E5 and the 6b anchor apply to BOTH stages unchanged. Stage-1
results are reported as soon as they exist, not held for stage 2.

**Stage-2 status (2026-07-10, after stage 1):** attempted, BLOCKED. Serving Trinity via vLLM on node1:8001 works (~100 tok/s, vs 0.2 tok/s in-process HF), but Trinity-Large-Preview degenerates into a `<|begin_of_text|>` special-token repetition basin in plain `/v1/chat/completions` multi-turn chat at ~5k tokens (context-locked; unfixable by repetition_penalty/frequency_penalty/bad_words/logit_bias — all verified). This is exactly the pathology kv_perturb's `backrooms_gen.py` avoids by using CLI-simulation *prefill* format + burst generation rather than chat format. Correct stage-2 path: generate via the Trinity vLLM server using kv_perturb's prefill/burst `/v1/completions` idiom (not chat). Deferred — stage 1 already answers E1. See journal 'exp10 — stage 2' for the full diagnosis.

**Run-hygiene decisions (2026-07-10, pre-stage-1):** (a) any disk/cpu/meta
offload is now a hard error (`kvrot.natural.assert_no_offload`, per-device
param counts logged) — observed offload at mem-frac 0.70/0.85 wedged decode;
generation/eval run at `--mem-frac 0.95` on an empty node (~170 GiB/GPU cap).
(b) Stage-2 in-process sampling is manual nucleus (temperature 0.9, top_p 0.95,
per-turn seeds) via TrinityTurnGenerator — model.generate() is never called;
transformers' load-time "generation flags are not valid" warning refers to the
checkpoint's generation_config and is inert. (c) Throughput rule: if a
~250-token turn exceeds ~3–4 min GPU-resident, reconsider turn lengths / conv
count / serving path rather than letting generation run for days.

## 7. Run plan

Phase 1 (local, CPU): generation + eval scripts, `kvrot/natural.py`, CPU tests
(transcript windowing, jsonl round-trip, cut selection, alternation, splicing).
node2 API smoke: done 2026-07-10 (see §2 caveat).

Phase 2 (node1, only after `~/luxi-files/kv-rotation/runs/DIAG_DONE` exists AND
nvidia-smi compute-apps is empty — a diagnostics agent owns the GPUs until then):

```bash
# sync (no --delete), then a tiny generation smoke (1 conv, low caps) to measure
# trinity decode tok/s, then the full nohup generation run:
PYTHONPATH=src ... exp10_natural_gen.py --convs all --target-tokens 18500 \
    --turns-log runs/exp10_gen.log   # banks every turn; resumable
# eval (per exp08b recipe):
PYTHONPATH=src ... exp10_natural_chat.py --convs data/natural_convs_2026-07-10.jsonl \
    --lengths 8192 16384 --evict-frac 0.5 --with-recompute \
    --device-map balanced --mem-frac 0.70 --out runs/exp10_ef50.json
# if time: --evict-frac 0.3 / 0.7 on a 3-conversation subset.
```

Deliverable: journal section "exp10 — naturalized dialogue" with the same-format
table and a direct exp09-vs-exp10 verdict against E1.
