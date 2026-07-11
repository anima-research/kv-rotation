# Design — chat-shaped eval (exp09): turn-aligned eviction on multi-turn chat

> STATUS: STAGED 2026-07-03 (designed + CPU-tested; NOT yet run — GPUs busy)
> SCOPE: closes HANDOFF §9.1a, the realism frontier. Code: `src/kvrot/chat.py`,
> `experiments/exp09_trinity_chat.py`, `tests/test_chat.py` (16 CPU tests green).
> RAW-ARTIFACTS: none yet — will land in notes/journal.md when exp09 runs.

## 1. Why this is the next experiment

Every result so far (exp01–08b) evicts token ranges from **raw documents**. The
deployment story is a rolling **chat**: the window fills, whole early *turns* get
popped, and the cache must keep behaving as if the conversation were intact. That
shape differs from raw docs in ways that could matter:

- **Template structure.** Role headers/separators recur at every boundary; on the
  instruct Preview they're strongly in-distribution and likely attention-heavy.
  Evicting whole turns removes them cleanly; the raw-doc evals never tested this.
- **Turn-aligned granularity.** Eviction boundaries coincide with semantic units
  (a whole exchange disappears), not arbitrary token offsets.
- **Conversational referencing.** Later turns refer back ("as I said", the running
  summary), so the *feasibility law* (drift ≈ future attention onto the evicted
  block) gets a harder, more realistic test than book prose.

## 2. Data: synthesized-from-real-prose conversations (and why not OASST2)

No clean turn-data jsonl exists in kotodama, node1 runs `HF_HUB_OFFLINE=1`, and
OASST2 here is only a log. Instead `kvrot.chat.synthesize_conversations` builds
deterministic conversations from the **real eval corpus** (`data/eval_docs.jsonl`):
the user feeds successive chunks of a real document; the assistant replies with
short templated acknowledgments. Real prose fills the KV; the turn scaffolding is
canned; zero RNG, zero network, zero LLM calls to build data.

*Honest caveat:* assistant turns are low-entropy boilerplate, so this measures
chat-**shaped** contexts, not chat-**distributed** content. If exp09 shows anything
surprising, the follow-up is generating the assistant turns with trinity itself
(one prefill pass, then freeze) or importing a real chat dump. For the mechanism
question — does whole-turn eviction + re-rotation stay faithful under a chat
template — shape is the variable that changed, and this isolates it.

## 3. Turn spans and the prefix-stability assert

`turn_token_spans` renders `messages[:i]` incrementally through
`tokenizer.apply_chat_template` and takes differences as spans, asserting each
render is a strict token-prefix of the next (`TemplateNotPrefixStableError`
otherwise — e.g. templates that inject message counts or move the generation
prompt). The final generation header belongs to **no** span: it tees up the reply
under measurement and must never be evicted with the last turn.

Trinity ships `chat_template.jinja` (Preview = the chat variant); standard
append-loop templates satisfy the assert. If trinity's doesn't, the assert fires
on the very first cell and the fix is bespoke span handling, not silent misalignment.

## 4. Protocol (per cell: one length L × one conversation)

1. Synthesize ≈L-token conversation (`--turns` user turns; ~4 chars/token chunk
   heuristic; report achieved C, don't chase L). Plant two synthetic passcodes:
   - `locker 62483` in the **first user turn** → will be EVICTED (policy cost).
   - `vault 73914` in the **second-to-last user turn** → RETAINED (protected by
     `protect_last=2`). Same passcode as exp08 for cross-experiment comparability.
2. Full prefill → snapshot (the reference cache).
3. **Turn-aligned eviction**: `oldest_turns_to_evict` takes whole early turns
   (system prompt + last 2 turns protected) until ~`--evict-frac` (default 0.5)
   of prefill tokens are freed; `turn_keep_indices` + sinks → keep set;
   Tier-0 re-rotation with **recompaction** (the deployment recipe).
4. **Continuation drift**: teacher-force the full cache's own greedy next-turn
   reply (32 tokens — real reply, not a passcode) under rotation →
   `KL(full ‖ rotation)` + top-1. With `--with-recompute`, also the shortened
   recompute → the info-loss / mechanism-error decomposition of exp05/07.
5. **Recall probes**: plain-text Q&A suffixes (exp08 pattern) for both facts —
   retained must recall, evicted must drop. *(Simplification: probes are not
   chat-templated; if the instruct model reads noisy on bare text, switch to
   rendering the probe as a proper user turn — costs one extra prefill of the
   question tokens per condition, no other change.)*

## 5. Pre-registered predictions (written before any run)

- **P1 (mechanism):** mech-KL(short ‖ rot) stays ≤ ~1e-3 — turn-aligned boundaries
  are nothing special to the rotation math; exp07/08's near-error-free result carries.
- **P2 (recall):** retained vault passcode recalls under rotation ≈ full (top1,
  lp within ~0.5); evicted locker code drops to chance. Same as exp08 but through
  the template.
- **P3 (the one that could surprise):** total drift KL(full ‖ rot) on the reply
  runs HIGHER than exp07/08's raw-doc numbers at matched L and evict-frac, because
  an assistant reply attends back into the conversation (incl. evicted turns) more
  than book-prose continuation attends to earlier book text. If it comes back
  ≥5–10× exp08's ~5e-5, that's the finding: chat needs importance-aware (Tier 2)
  or turn-summary policies, not just oldest-turn. If it's flat, oldest-turn
  eviction is deployment-ready for this shape.
- **P4 (template safety):** no NoPE/sliding-layer surprises — span-level eviction
  goes through the same per-layer-gated `reindex` as everything else.

## 6. Run plan (when GPUs free up)

```bash
# sync per HANDOFF §4, then smoke on the 3B (1 GPU, minutes):
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src ... exp09_trinity_chat.py \
    --model /models/llama-3.2-3b-instruct --lengths 2048 --turns 8
# trinity (8 GPUs, nvidia-smi first; balanced placement per exp08b):
PYTHONPATH=src ... exp09_trinity_chat.py --lengths 8192 16384 --turns 12 \
    --device-map balanced --mem-frac 0.70 --with-recompute
```

Sweeps after the first pass: `--evict-frac {0.3, 0.5, 0.7}`, `--turns {8, 24}`
(many small turns vs few big ones — granularity effect), longer L per §9.1b.
