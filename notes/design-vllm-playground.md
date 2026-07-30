# Design: Trinity interactive playground with KV rotation via vLLM KV-connector injection

*Status: **VALIDATED** 2026-07-29 — all three validation agents (vLLM-on-node1
recon, kvrot codebase audit, upstream-vLLM design review) returned; every recon
question R1–R12 answered with file:line evidence and folded in (see §12
validation log). Verdict: Option B viable in the pinned vLLM 0.16.0, all four
potential design-killers retired. Remaining pre-implementation open question:
Preview-vs-TrueBase target (§1, needs Luxia). R13 (prompt_logprobs in the 0.16
server) checks during Phase 1. Author: session w/ Luxia.*

---

## 1. Goal

An interactive chat playground where a human talks to **Trinity-Large**
(afmoe, ~389B MoE, 60 layers = 45 SWA(W=4096) + 15 full-attention NoPE) with
**KV-cache rotation live**: when the rolling context exceeds a budget, evict
turn-aligned blocks and re-rotate survivors *in the serving cache*, instead of
recomputing. Decode speed must stay at vLLM-server rates (~100 tok/s measured,
`runs/trinity_vllm.log`), because the in-process HF path decodes at ~0.2 tok/s
(journal-exp09/exp10) and is unusable interactively.

Selectable per session (or live via REPL command):

| Policy | Meaning |
|---|---|
| `none` | unbounded cache, no eviction (reference behaviour) |
| `recompute` | on budget breach, truncate transcript + full re-prefill (status-quo baseline) |
| `oldest` | naive oldest-block eviction + re-rotation (no sink protection) |
| `sink_rotate` | Tier 0+1: keep sinks, evict oldest post-sink turn block, re-rotate survivors |

(Tier 2 importance needs attention capture — not available server-side; out of
scope v1. Tier 3 selective recompute: out of scope v1, but the injection
machinery built here is its prerequisite.)

**Target checkpoint (RESOLVED 2026-07-29):** Preview vanished from node1's
local `/models/` but lives on the shared fs — **`/net/node2/models/Trinity-Large-Preview`**
(verified complete: 31 safetensors shards, afmoe config with explicit
`layer_types` (45/15), 8 KV heads, W=4096, bf16, `chat_template.jinja` → the
instruct variant). **Playground targets Preview from that path.** Caveat: the
~727 GiB load crosses NFS instead of local disk — expect slower cold start than
the 190 s local-disk figure; the ~5 min vLLM bring-up from exp10 may stretch.
`/models/Trinity-Large-TrueBase` (local disk, 8k ctx, base model, has a
production serve script) remains the fallback/ablation target. exp07–10
baselines (Preview) transfer.

**Non-goals (v1):** production-grade in-place paged rotation kernel (this design
extracts→rotates→re-injects between turns, which is the first milestone of
backlog item #7, not the whole port); gap-mode positions (see §5.3); multi-user
serving; Tier 2/3; web UI.

## 2. Why KV injection is required (no API-level shortcut exists)

A rotated cache is **not equal to any prompt's prefill**: survivors' KV carries
the evicted block's baked-in influence (the project's core result, exp05/07/08b).
Therefore no manipulation of prompts, `/v1/completions`, or prefix caching can
reproduce rotation. The rotated tensors must physically enter vLLM's paged KV
cache. Ways in, ranked:

- **B. KV-connector injection (CHOSEN, this spec).** vLLM v1's KV-transfer
  connector interface (the LMCache / disaggregated-prefill mechanism) is
  config-pluggable by class/module path → **zero modifications to vLLM source**,
  safe for the shared conda envs on node1.
- **C. In-process engine + `collective_rpc` worker surgery (FALLBACK).** Run
  `LLM`/AsyncLLMEngine in our driver, mutate cache tensors per-worker via vLLM's
  custom-RPC extension point, reconciling block tables ourselves. More invasive
  bookkeeping; use only if the 0.16 connector API cannot express save+load of
  arbitrary externally-modified KV.
- **A. Patch vLLM internals in our own env (LAST RESORT).** Own copy of the env
  (Luxia authorizes a copy into `/models/`), patch block manager / attention.
  Most power, most maintenance, slowest loop.

## 3. Architecture overview (Option B)

```
┌────────────┐   token ids + session ctl   ┌──────────────────────────────┐
│ REPL client │ ───────────────────────────▶│ driver (our python process)  │
│ (terminal)  │ ◀─────────────────────────── │  - transcript/token ledger   │
└────────────┘        reply stream          │  - turn spans + evict planner │
                                            │  - drift/timing readout      │
                                            └────────────┬─────────────────┘
                                                         │ requests (+ per-request
                                                         │ connector metadata)
                                            ┌────────────▼─────────────────┐
                                            │ vLLM v1 engine, TP=8, afmoe  │
                                            │  KvrotConnector (ours):      │
                                            │   worker-side, per TP shard  │
                                            │   - SAVE: request KV → store │
                                            │   - SURGERY: evict+rerotate  │
                                            │   - LOAD: store → new blocks │
                                            └──────────────────────────────┘
```

- **One vLLM request per turn.** The driver holds the canonical token-level
  transcript. Each request's prompt is exactly `surviving_tokens + new_turn_tokens`.
- **Connector, worker-side, per TP rank:** after a turn's decode finishes, save
  that request's KV (all layers, this rank's KV-head shard) into a GPU/pinned-CPU
  session store keyed by session id. When the budget trips, the driver computes an
  eviction plan (keep-indices + recompacted positions — identical across ranks)
  and ships it via connector metadata; each rank applies `evict` + RoPE-gated
  re-rotation to its stored shard. On the next request the connector reports the
  survivors as externally-matched tokens, vLLM allocates blocks for them, the
  connector copies the rotated KV in, and vLLM prefills **only the new turn's
  suffix** before decoding.
- **Surgery runs inside the workers** on their own shards (never ship ~2 GB
  through the driver). The plan (indices/positions) is tiny and identical
  everywhere; rotation math is per-head-independent so TP sharding is free.
- **Reuse, not rewrite:** `kvrot.rope.reindex_keys` (shape contract `[..., S,
  head_dim]` — per-rank shards like `[1, seq, head_dim]` work as-is, audited),
  `kvrot.eviction` (keep-indices/new-positions), and `chat.py`'s *planning* half
  (`TurnSpan`, `oldest_turns_to_evict`, `turn_keep_indices` — framing-agnostic).
  The span-*recovery* half (`turn_token_spans`) is chat-template-bound and does
  NOT apply to prefill/burst streams; a marker-offset span ledger is new code
  (small: the driver owns the token ledger, so spans are append-time arithmetic
  emitting `TurnSpan`s). The connector adapts layouts (paged block →
  `[heads, seq, head_dim]` contiguous → back) and must carry the per-layer NoPE
  gate itself when calling `reindex_keys` directly (or wrap each layer in a
  1-layer `KVSnapshot`, which is legal).

## 4. Per-turn flow (detail)

1. User types turn *t*. Driver renders it to token ids (prefill/burst
   CLI-simulation format — **not** the chat template; see §6.2), appends to ledger.
2. Budget check on `len(survivor_tokens) + len(new_tokens)`:
   - Under budget → no surgery. Connector loads the saved KV for the survivors
     verbatim (it is byte-identical to what vLLM would have; positions unchanged).
   - Over budget → driver computes turn-aligned eviction plan: protected sinks
     (first N tokens incl. BOS/system preamble), evict oldest post-sink whole-turn
     span(s) until under budget, recompact positions `0..N-1`. Plan → connector
     metadata; workers apply surgery to the session store.
3. Request submitted with prompt = survivors + new turn. Connector claims
   survivors as matched (`get_num_new_matched_tokens` → survivor count), loads
   rotated KV into the allocated blocks; vLLM computes the suffix + decodes reply.
4. During/after decode, connector saves the request's full KV (survivors' rotated
   KV + newly computed suffix/decode KV) back to the session store; the ledger
   appends the reply tokens (from `return_token_ids`, the exp11 idiom).
5. Driver prints per-turn stats: cache len, evicted spans, surgery+load ms,
   decode tok/s. Transcript + eviction history banked to jsonl for offline replay.

**Per-turn overhead budget:** Trinity KV ≈ 60 layers × 8 KV-heads × 128 ×
2(K,V) × 2B ≈ **240 KiB/token** whole-model → ~2 GB at 8k ctx, ~245 MB per rank.
Save+load are GPU-local copies per rank (~sub-second at B200 bandwidth even with
paging overhead); rotation itself is O(ms) (29–52 ms measured HF-side at 8–16k,
exp08). Seconds-scale worst case per turn — invisible next to ~100 tok/s decode.

## 5. Design details & invariants

### 5.1 TP sharding
GQA 8 KV heads / TP=8 → 1 KV head per rank per layer. **R1 — CONFIRMED**
(afmoe.py:203 `num_kv_heads = max(1, total // tp_size)`). Eviction is per-token
(rank-independent), rotation is per-head (rank-local). The eviction plan
broadcast is identical for all ranks.

**Per-rank cache layout (recon):** fused K/V, one tensor per layer, backend-
dependent shape — FlashAttention `(2, num_blocks, block_size, kv_heads,
head_dim)` vs FlashInfer `(num_blocks, 2, block_size, kv_heads, head_dim)`;
default `block_size=16` on CUDA. The layout adapter must handle both; Gate 0
logs the selected backend.

**Backend constraint for hybrid models (found live, 2026-07-29):** with a
connector configured, HMA is off → all layers share ONE KV-cache group, and
**FlashInfer's metadata builder refuses mixed sliding windows per group**
(`flashinfer.py:885` "Window left is not the same for all layers") — trinity
(45×SWA + 15×full) dies at engine warmup. **FlashAttention is required**
(`VLLM_ATTENTION_BACKEND=FLASH_ATTN`): it applies the window per layer at
kernel-call time and merely disables AOT scheduling for mixed groups
(flash_attn.py:364). Uniform-layer models (the 3B) run fine on either backend.
Do NOT use `disable_sliding_window` (changes attention semantics). Connectors receive the raw per-layer tensors via
`register_kv_caches(dict[str, Tensor])` and write via slot-mapping indexing
(steal `ExampleConnector.inject_kv_into_layer` / `extract_kv_from_layer`).

### 5.2 Per-layer RoPE gating (the NoPE trap)
afmoe applies RoPE **only on sliding layers**; the 15 global layers are NoPE and
their keys must pass through surgery byte-identical (rotating them corrupts).
`ArchSpec.applies_rope` already encodes this from HF config. **R2 — CONFIRMED:**
vLLM's in-tree afmoe (`model_executor/models/afmoe.py:211,245-254,283-285`)
creates `rotary_emb` only when `layer_types[i] == "sliding_attention"` and
applies it only on those layers — identical gating to HF, driven by the same
`config.layer_types` field (present explicitly in trinity's config.json,
45/15 split verified).

**Config-only trap (audited):** the map must be built via
`AutoConfig.from_pretrained(path, trust_remote_code=True)` — `AfmoeConfig`
*derives* `layer_types` from `global_attn_every_n_layers` when config.json omits
it, and `ArchSpec.from_hf_config` silently falls back to all-`full_attention`
(→ zero rotation, wrong) on a raw dict missing `layer_types`. `inv_freq` is
config-only clean: `RoPESpec.build_inv_freq` (θ=10000, no scaling for trinity);
no model load needed in the connector.

### 5.3 Positions: recompact-only
vLLM assigns positions = token index within the request sequence, so injected KV
must be rotated to recompacted positions `0..N-1`. Gap-mode (preserving absolute
positions) is **not expressible** without engine changes — acceptable: exp01
showed recompact ≈ gap within scaled range, and every trinity experiment
(exp07–09) ran recompact. (Upstream confirms: no absolute/gap-position notion
exists in the v1 request path; the open RFC vllm#25672 proposes exactly the
gap direction — non-contiguous reuse with preserved positions — but is
unimplemented with no maintainer engagement. Recompact-only is right today.)

### 5.4 Sliding-window layers (the subtlest trap)
Recompaction shrinks positional distances → keys formerly outside a query's 4096
window can legitimately **re-enter** the window (HF replicates this: masks are
distance-based). Two failure modes, both ruled out by recon:
- **Storage — R3 CONFIRMED, RESOLVED FOR FREE:** the hybrid manager does free
  out-of-window blocks on SWA layers (`SlidingWindowManager.remove_skipped_blocks`),
  and afmoe does register per-layer windows — BUT vLLM 0.16 **auto-disables HMA
  whenever `kv_transfer_config` is set** (config/vllm.py:1070-1087, "turn HMA
  off for connector unless specifically enabled"). With HMA off, every
  `SlidingWindowSpec` is unified to `FullAttentionSpec` and nothing is freed.
  Rules: do NOT subclass `SupportsHMA` (opting back in), and still pass
  `--disable-hybrid-kv-cache-manager` explicitly for self-documentation.
- **Kernel semantics — R4 CONFIRMED:** windows are applied by relative token
  index within the request (`flash_attn` `window_size=(W-1, 0)` over in-request
  positions; FlashInfer `window_left` equivalently), and vLLM's positions ARE
  the in-request indices `0..N-1` — exactly consistent with recompacted
  positions. (Also independently confirms §5.3: no gap-position notion exists
  anywhere in the request path.)

### 5.5 Prefix caching / APC
**R5 — CONFIRMED; APC-off is a hard requirement, not a maybe.** APC is on by
default in 0.16 and local prefix hits are counted *before* the connector is
consulted (`scheduler.py:598-631`) — stale unrotated APC blocks would shadow the
connector for the tokens they cover. Worse (upstream critique): blocks we fill
with *rotated* KV get token-hashed into the APC pool and could later serve as
"hits" for look-alike prefixes. Run every server with
`--no-enable-prefix-caching`; then the connector governs the whole prefix.

### 5.6 Connector API surface (0.16, pinned on node1)
**R6 — CONFIRMED on all five sub-points** (node1 recon + upstream v0.16.0 tag,
independently): (a) a connector can claim arbitrary matched tokens for a fresh
request with no real store — in-tree existence proof: `DecodeBenchConnector`
claims-and-fills dummy blocks; the scheduler accepts the count as-is (no hash
re-check); (b) `start_load_kv` fires before model execution and
`ExampleConnector.inject_kv_into_layer` shows the write: a plain slot-mapping
scatter into the paged tensor, no kernel needed; (c) save hooks
(`save_kv_layer`/`wait_for_save` via the `@maybe_transfer_kv_layer` decorator)
fire on **every** forward pass, decode included — see §5.7; (d) out-of-tree
loading via `KVTransferConfig(kv_connector="KvrotConnector",
kv_connector_module_path="kvrot_vllm.connector", kv_role="kv_both")` — verified
live-importable in the glm5 env, zero installs, our repo on `PYTHONPATH`;
(e) per-request metadata: `SamplingParams.extra_args["kv_transfer_params"]`
(offline) / top-level `kv_transfer_params` field (OpenAI server) reaches
`request.kv_transfer_params` on the scheduler-side connector — session id +
eviction plan ride here — and `request_finished`'s returned dict is echoed back
in the response (free channel for per-turn connector stats → §6 readout).

**Claiming rules (recon-derived, load-bearing):**
- **Never claim all tokens of a request** — `assert num_new_tokens > 0`
  (scheduler.py:668), no backstop. Guaranteed structurally by the new-turn
  suffix; assert anyway.
- **Claim block-aligned counts**: round survivor count down to a multiple of
  `block_size` (16) and let vLLM recompute the ≤15-token ragged tail *from the
  rotated prefix* (rolling_replay semantics). Gate 2's HF oracle must replicate
  this tail-recompute (or show it sits below the Gate-1 floor); the per-turn
  readout logs claimed-vs-survivor deltas.
- Connector constructor is **3-arg** in 0.16 (`vllm_config, role,
  kv_cache_config=None`) — older 2-arg examples break on init.

**R7 — answered, demoted to trivia:** per-request params exist, so no
side-channel is needed. (If ever used: 0.16 rewrites request ids internally to
`{external_id}-{8char}` — prefix-match, and only AsyncLLM/server accept caller
ids, not the sync `LLM` API.)

### 5.7 Save-side completeness
**R8 — answered; the "fallback" is promoted to the primary v1 design.** The API
fully supports decode-token saving (hooks fire every step; `request_finished` →
delay-free → copy blocks in `get_finished` is the NIXL pattern; in-tree
`OffloadingConnector` stores decode blocks as they complete — `ExampleConnector`'s
prompt-only saving is a policy choice, not an API limit). But decode-step block
copying adds bookkeeping and failure modes for marginal gain, so **v1 saves
prefill KV only** (visible to `save_kv_layer` during each request's prefill:
that covers the injected survivors plus the new turn's tokens) **and never
claims the previous reply's tokens as matched** — vLLM recomputes that short
reply suffix from the injected rotated prefix. Semantically identical to HF
`rolling_replay` (recomputed suffix KV on top of a rotated prefix); cost ≈ one
reply-length prefill per turn. Decode-token capture via
`request_finished`/`get_finished` (delay-free, the NIXL pattern) is the Phase-5
optimization, implemented only if per-turn timing shows it matters.

### 5.8 Numerics
**R9 — CONFIRMED:** `--kv-cache-dtype auto` resolves to the model dtype = bf16
for trinity (config `dtype: bfloat16`, `quantization_config: None` — neither
fp8 trap fires). Extract→rotate→inject round-trips losslessly; rotation in fp32
then cast back, matching `kvrot.rope`. Leave `auto`, never pass fp8.

### 5.9 Chat-format pathology
Trinity-Preview degenerates into a `<|begin_of_text|>` repetition basin in plain
chat-template multi-turn at ~5k tokens (exp10 stage-2, context-locked, sampling
penalties don't fix it). With the target now TrueBase (§1) the pathology is
moot — a base model has no chat template — but the conclusion is unchanged
either way: the playground renders turns in the
**CLI-simulation prefill format** with `<|EOT|>` delimiters — the idiom lives in
**`/home/luxia/projects/kv_perturb_exp/backrooms_gen.py`** (separate repo,
2,041 lines: `to_format_a` :318, `PREFILL_MSG_DELIMITER` :114) and was **never
adapted into kvrot** (exp10 stage-2's Trinity client used the chat template —
that IS the degenerating path; the journal deferred the adaptation). We adapt
only the prefill framing + stop-sequence handling for one-reply-per-request
interactive use (~80–150 lines), not the burst/parse/quality-gate machinery.
Turn spans tracked by a new marker-offset ledger (§3), not `chat.py`'s
template-diff method.

**R10 — ANSWERED (codebase audit):** the client to lift is **exp11's
`VllmTurnGenerator`** (`exp11_gen_native3b.py:498` — raw `/v1/completions`,
prompt as token ids, `return_token_ids` verified against vLLM 0.16, per-turn API
seeds, retry/backoff); plus `TurnRecord`/`append_turn_record` banking
(`natural.py`), `trim_to_sentence`, `served_model_id`. exp10's
`TrinityTurnGenerator` is the HF in-process path (0.2 tok/s — design reference
only); `TrinityChatClient` (gen2.py) is the chat-completions client that hit the
basin. No turn-span bookkeeping exists for raw completions streams — new code.

## 6. Playground UX

Terminal REPL (runs on node1 or locally against ssh-tunneled port):
- `/policy <name>`, `/budget <tokens>`, `/sinks <n>`, `/evict` (force), `/stats`,
  `/save`, `/quit`; plain text = next user turn.
- Per-turn readout: `[cache 6832/8192 | evicted 2 turns (1204 tok) | surgery 38ms
  | load 410ms | 97 tok/s]`.
- Every session banks `{transcript tokens, turn spans, eviction events, seeds,
  policy}` to `runs/playground/*.jsonl`.

### 6.1 Fidelity metrics are OFFLINE, by design
A live HF shadow oracle is impossible: vLLM at `--gpu-mem 0.85` leaves no VRAM
for a 727 GiB HF copy, and the HF path is 0.2 tok/s anyway. Instead:
`experiments/exp12_replay_session.py` (new) replays a banked session through the
HF harness offline — full-context vs rotated teacher-forced KL / top-1 per turn
— giving per-session drift reports after the fact. Live readout is
timing/counters only. Honest split.

**Replay primitives (audited):** the replay must apply the **exact banked
eviction plans**, not recompute them — `evict`/`reindex` accept explicit
keep-indices and position vectors, so per event it's
`reindex(evict(snap, banked_keep), banked_new_pos, inv_freq)`. Assembly from
`prefill_snapshot` + `harness._teacher_forced_logits` + `DriftReport`;
`rolling_replay` computes its own plans internally and is a loop *template*
only, not a callable. exp09's `evaluate_cell` is the single-event version to
generalize to a multi-event per-turn loop (~200–300 lines, mostly assembly).

### 6.2 Turn framing
Prefill/burst CLI-sim format per §5.9. Sinks = BOS + system/scene preamble
(protect entirely). Eviction unit = whole turns (user+assistant pair spans).

## 7. Correctness gates (sequenced; each blocks the next)

- **Gate 0 — round-trip identity:** save a request's KV, re-inject it verbatim
  for the continuation request (no surgery), decode. Must match an uninterrupted
  session's logits/tokens within kernel-nondeterminism tolerance. Proves the
  connector plumbing alone doesn't perturb.

  **Silent-no-op hardening (upstream critique — the primary Gate-0 failure
  mode is "silently identical because nothing loaded"):** (i) the connector
  counts layer loads/saves per step and hard-fails on zero (vllm#18489: an
  `attn_metadata is None` guard can silently skip the hooks); (ii) run gates
  with `kv_load_failure_policy="fail"` — the default `"recompute"` would mask a
  broken load as perfect parity; (iii) default piecewise compilation only, no
  full-cudagraph flags (connector hooks are a Python decorator around
  `unified_attention`; graph capture is known-hostile to such interception —
  LMCache CacheBlend excludes graph mode too); (iv) log the selected attention
  backend + KV tensor shape at startup (FA vs FlashInfer layouts, §5.1); (v)
  assert at startup that HMA was in fact disabled (§5.4).
- **Gate 1 — cross-stack parity (full cache):** same prompt in vLLM vs HF
  (greedy, teacher-forced comparison on a fixed continuation): establishes the
  vLLM↔HF numerical noise floor that Gate 2 is judged against.
- **Gate 2 — rotated parity (the real gate):** rotated-in-vLLM vs rotated-in-HF
  (identical eviction plan) teacher-forced comparison on the same continuation.
  Mechanism error added by the vLLM path must sit at the Gate-1 noise floor.

**Gate metric reality check (audit finding):** full-distribution KL is NOT
obtainable from vLLM's completions API — it returns top-k `logprobs` and
`prompt_logprobs` (teacher-forcing: send `continuation` as prompt suffix with
`max_tokens=0`-style scoring), not `[steps, vocab]` logits. Gates 1–2 are
therefore defined over: **per-token chosen-token logprob deltas** (via
`prompt_logprobs`) + **top-k overlap/agreement** vs the HF oracle's same
quantities, with the HF-vs-HF full-KL run alongside as the interpretive anchor.
(Full-logit capture engine-side via the connector is possible but is scope
creep — revisit only if the logprob-based gate proves too blunt. FLOOR
discipline per exp11: same-path-twice floor + cross-stack floor.) Verify
`prompt_logprobs` is exposed in the pinned 0.16 server (**R13** — check during
Phase 1 recon; exp11 already verified `return_token_ids`).
- **Gate 3 — end-to-end soak:** scripted multi-turn session (synthesized turns,
  passcode needle in a retained + an evicted turn, exp08/09 style), rolling
  eviction over ≥10 pops; offline replay (§6.1) must reproduce the exp09-scale
  drift numbers; needle recall must match rotation expectations (retained ≈
  full, evicted forgotten).
- Gates run on the **3B first** end-to-end (1 GPU vLLM + 1 GPU HF oracle,
  cheap, fast iteration — the 3B is a *validation vehicle* here, not the demo
  target), then Trinity (Gate 1/2 need the HF oracle run serially before/after
  the vLLM run — same GPUs, sequential, offline).

## 8. Risk register

*(Post-recon revision: the four design-killers — connector claim/load, SWA
freeing, metadata channel, RoPE gating — are all RETIRED with file:line
evidence. What remains is the quiet-failure class.)*

| Risk | Severity | Status / Mitigation |
|---|---|---|
| ~~0.16 connector can't claim matched tokens w/o real store~~ | ~~fatal~~ | **RETIRED** — `DecodeBenchConnector` existence proof; scheduler accepts count as-is (R6a) |
| ~~SWA out-of-window KV freed by hybrid manager~~ | ~~high~~ | **RETIRED** — HMA auto-disables when a connector is configured (R3); don't subclass `SupportsHMA`; assert at startup |
| ~~No per-request metadata channel~~ | ~~medium~~ | **RETIRED** — `kv_transfer_params` verified both directions (R6e) |
| ~~vLLM afmoe RoPE gating differs from HF~~ | ~~high~~ | **RETIRED** — identical gating verified (R2); Gate 2 still catches residuals |
| ~~fp8 KV default / APC interference~~ | ~~med/low~~ | **RETIRED** — bf16 under `auto` (R9); APC-off hard requirement (R5) |
| **Silent no-op loads** (attn_metadata guard, vllm#18489) | **high** | Gate-0 instrumentation: per-layer load/save counters, hard-fail on zero |
| **`kv_load_failure_policy: recompute` masks injection bugs** | **high (gates)** | run gates with `"fail"`; production keeps `recompute` |
| Block-alignment bookkeeping (claim = align-down(N), ragged tail recomputed) | medium | encode in ledger from day 1; Gate-2 oracle replicates tail-recompute; log claimed-vs-survivor |
| CUDA-graph modes bypass connector hooks | medium | piecewise compilation only; documented in run recipes |
| ~~Target checkpoint: Preview gone~~ | ~~medium~~ | **RETIRED** — Preview verified complete at `/net/node2/models/Trinity-Large-Preview` (§1); NFS load-time caveat noted |
| Backend-dependent KV tensor layout (FA vs FlashInfer) | low | adapter handles both shapes; log at Gate 0 |
| chat-format degeneration basin | low (was high) | moot on TrueBase (base model); prefill format is the natural interface anyway (§5.9) |
| shared-box contention (8 GPUs; a production TrueBase server may be running via `start_trinity.sh`/pt-llm) | operational | nvidia-smi + check port before serving; coordinate with box owners; `--gpu-mem` conservative |
| conda env mutation risk | policy | activation-only use of **glm5** (§9); if patching needed, copy env to `/models/` (Luxia-authorized); never touch originals |

## 9. Environment strategy

1. **Default: activation-only reuse of `glm5`** (the shared conda env:
   vLLM 0.16.0, torch 2.9.1, transformers 5.3.0, clean). Recon verdicts on the
   alternatives: **kimi** has an md5-identical vllm tree but auto-loads
   `vllm_probe_plugin` (emotion-probe steering) into every vLLM process via
   `vllm.general_plugins` — excluded; **mistral** is vLLM 0.11.2 (the "0.16"
   memory was wrong) — excluded; **pt-llm** (0.14.1) is what
   `/models/start_trinity.sh` uses — not ours to touch. Zero installs; our code
   via `PYTHONPATH`; connector loads out-of-tree via `kv_connector_module_path`
   (**R6d — verified live-importable in glm5**).
2. **If patching vLLM becomes necessary** (Options A, or C needing edits): copy
   the chosen env wholesale to `/models/envs/kvrot-vllm/` (space is there;
   Luxia-authorized) and run from the copy. Prefer `conda create --clone -p`
   semantics over raw `cp -a` if prefix-baking bites; verify the copy serves
   before relying on it. Originals are never modified.
3. Local dev box: connector module gets CPU unit tests (layout adapters,
   plan application on fake block tables) in `tests/` like everything else.

## 10. Deliverables & plan

| Phase | Deliverable | Est. |
|---|---|---|
| 0. Recon (this validation pass) | R1–R12 answered with file:line evidence; spec revised | 1 day |
| 1. Connector skeleton | `src/kvrot_vllm/connector.py` (+ layout adapters, session store, CPU tests) | 1–2 days |
| 2. Gates 0–2 on the 3B | `experiments/exp12_vllm_gates.py`; numbers banked | 1 day |
| 3. Gates 0–2 on Trinity | same script, trinity config | 0.5 day (GPU availability) |
| 4. Playground client + Gate 3 | `scripts/playground_chat.py`, `exp12_replay_session.py` | 1–1.5 days |
| 5. Journal + RESULTS entry | exp12 section, timing table incl. first honest vLLM-stack rotation-vs-recompute ratio | 0.5 day |

**R11 — ANSWERED.** Patterns to steal, all at the v0.16.0 tag:
`ExampleConnector` (the renamed SharedStorageConnector) for
`inject_kv_into_layer`/`extract_kv_from_layer` (slot-mapping scatter/gather,
reshape paged tensor to `[2, pages*page_size, -1]`) and the claim-align-down
idiom; `DecodeBenchConnector` as the cleanest claim-and-fill skeleton;
`OffloadingConnector` for the (Phase-5-only) block-granular decode-save
pattern. LMCache's MP connector subclasses `SupportsHMA` for hybrid models —
the template for a future HMA-on Tier-3 variant, explicitly NOT for v1.
Session store default: **pinned CPU** (`kv_buffer_device="cpu"`; ~245 MB/rank
moves in single-digit ms over DMA) — GPU-resident is an optimization.

**R12 — ANSWERED: nothing upstream does cached-key re-rotation; we don't
conflict, and three open unimplemented vLLM RFCs (sink-aware eviction #36311,
retention priorities #37003, non-contiguous reuse #25672) show demand without
supply.** LMCache CacheBlend heals cross-attention by *selective recompute*
(our Tier 3 analog — layerwise mode, no CUDA graphs), not re-rotation; aligns
with, doesn't supersede. Closest published system: **Leyline (arXiv
2606.01065) — δ-rotation reanchoring of cached keys in a serving stack; read
before Phase 1** and cite in the journal alongside SparseX (2606.06256) and
KVLink (2502.16002).

**Version landscape warning:** v0.16.0 (2026-02-25) is ~10 releases behind
current (0.26.0, 2026-07-25; ~2-week cadence). `docs.vllm.ai/latest` describes
a different tree — **always read the v0.16.0 tag**. Layout drift vs older
tutorials: attention code lives under `model_executor/layers/attention/`
(incl. `kv_transfer_utils.py`), config under `vllm/config/`,
`shared_storage_connector.py` → `example_connector.py`.

## 11. References

- `HANDOFF.md` §3 (afmoe arch), §9.7 (vLLM port backlog entry)
- `notes/journal-exp09-2026-07-10-draft.md` (even device-map, 0.2 tok/s HF; vLLM
  serve recipe ~100 tok/s; chat degeneration diagnosis)
- `notes/journal-exp11-2026-07-10-draft.md` (3B decode tolls; vLLM 355 tok/s;
  token-id round-trip idiom; correctness-gate pattern)
- `src/kvrot/{rope,snapshot,eviction,harness,chat,natural}.py`
- `experiments/exp11_gen_native3b.py` (`VllmTurnGenerator` — the client lift:
  token-ids in, `return_token_ids` out, verified on vLLM 0.16);
  `experiments/exp10_natural_gen2.py` (`TrinityChatClient` — the chat-template
  path that degenerates; cautionary reference only); `runs/trinity_vllm.log`
- `/home/luxia/projects/kv_perturb_exp/backrooms_gen.py` (CLI-sim prefill
  format source — adapt, don't import; separate repo)
- vLLM v1 KV-connector interface + LMCache (recon to pin exact paths/versions)

---

## 12. Validation log

- **2026-07-29, GATES 0–2 GREEN ON TRINITY (the target).** Job
  `exp12-trinity-gates-fa3` via heimdall, node1, TP=8, Preview read over IB
  NFS (server up in ~2.5 min — IB NFS beats the 190 s local-disk figure).
  ctx 8192 / evict 2048 post-sink / gen 48, real corpus
  (`runs/exp12_gates_trinity.json`):
  - **Gate 0 PASS**: 48/48 tokens identical, MAD 2.2e-5.
  - **Gate 1 floor**: 2.6e-5, HF greedy match 100%.
  - **Gate 2 PASS**: rotated parity MAD **1.3e-5 = 0.51× the floor** — the
    vLLM rotation path (45-layer re-rotation + 15-layer NoPE passthrough,
    per-rank shards) measures BELOW cross-stack noise at 389B. 100%
    teacher-forced greedy agreement. Clean-recompute yardstick: 1.1e-4.
  - **Timing:** rotated request wall **1.05 s** end-to-end at 8k ctx
    (plan surgery + inject + 16-token tail prefill + 48-token decode) on the
    serving stack whose HF-naive equivalent recompute was 20–29 s.
  - Serving-stack gauntlet getting here (each fix banked in the job script +
    §5.1): node2 envs flashinfer-mismatched (cubin 0.6.1/0.6.4 vs python
    0.6.3 — both broken, use node1 glm5); FlashInfer refuses mixed windows
    in the HMA-off single group → `--attention-backend FLASH_ATTN` (0.16
    removed the env var; CLI flag only); custom all-reduce kernel crash on
    8×B200 at warmup → `--disable-custom-all-reduce`; `grep|head` under
    pipefail SIGPIPE-kills a healthy job (exit 141) → slice+`tail`.
    **Zero failures in the rotation machinery itself.**
  **Substrate verified end-to-end on the real target. Next: the playground
  REPL (Phase 4) rides on exactly this server configuration.**

- **2026-07-29, IMPLEMENTATION + GATES 0–2 GREEN ON THE 3B.**
  `src/kvrot_vllm/{core,connector}.py` built and live-tested against vLLM
  0.16.0 (glm5 env, out-of-tree load, zero installs). 120 CPU tests green
  (incl. bit-exact surgery parity vs the KVSnapshot path and both paged
  layouts — node1's stack turned out to be **FlashInfer layout, kv_axis=1**,
  the case where a naive reshape-scatter silently no-ops). Gate results
  (`runs/exp12_gates_3b.json`, ctx 1024 / evict 256 post-sink / gen 48,
  real eval-corpus context):
  - **Gate 0 PASS**: 48/48 greedy tokens identical after save→re-inject;
    logprob MAD 2.7e-3 ≈ kernel batch-composition noise.
  - **Gate 1 floor**: cross-stack MAD 2.3e-3, HF greedy match 100%.
  - **Gate 2 PASS**: rotated-in-vLLM vs rotated-in-HF MAD 4.2e-3 = **1.8×
    the Gate-1 floor** (2× margin), HF teacher-forced greedy match **100%**
    over all 48 rotated-generated tokens. The vLLM rotation path adds no
    error beyond cross-stack kernel noise.
  - Caught + fixed en route: `RoPESpec.from_hf_config` mis-read `rope_theta`
    on transformers 5.x (relocated into the rope dict; Llama-3.2's θ=500000
    read as 10000 → first Gate-2 run failed at MAD 0.83, 365× floor). Fix +
    regression tests + a connector-init cross-check against transformers'
    `ROPE_INIT_FUNCTIONS` (fails loudly on table mismatch or
    attention_scaling ≠ 1). A wrong-but-static table passes the homomorphism
    assert — only the cross-check or Gate 2 catches it.
  - Known benign metric quirk: `greedy_first_divergence_vs_hf_rollout` is
    noisy (batched teacher-forcing vs incremental decode → tie-flips at
    ~4e-3 logit deltas); teacher-forced greedy match is the meaningful
    agreement metric.
  - Timing note (3B, 1 GPU): rotated-request wall 0.28 s end-to-end incl.
    plan surgery + CPU-store round trip at 752 claimed tokens.
  Next: same gates on Trinity-Preview (`/net/node2/models/`, -tp 8), then
  the playground REPL (Phase 4).

- **2026-07-29, codebase audit (agent 2/3): PASSED with corrections, applied
  above.** Confirmed: `reindex_keys` per-rank shard compat ([..., S, d]
  contract, fp32 internal + cast back), surgery accepts banked explicit plans,
  single-layer snapshots legal, KV arithmetic (240 KiB/tok), recompact-only
  consistent with all trinity experiments (exp07/08/09). Corrected: prefill/
  burst provenance (§5.9), client lift target → exp11 `VllmTurnGenerator`
  (§5.9/§11), chat.py reuse split into planner-vs-recovery (§3), replay
  primitives named + `rolling_replay` demoted to template (§6.1), Gate 1–2
  metric redefined over `prompt_logprobs`/top-k (§7, new R13), config-only
  `applies_rope` trap (§5.2). Net-new code inventory: connector 400–600 lines,
  gates 250–400, replay 200–300, REPL 300–400, renderer+ledger ~150–250,
  CPU tests 150–250.
- **2026-07-29, node1 vLLM recon (agent 1/3): OPTION B CONFIRMED VIABLE in the
  pinned 0.16.0.** R1/R2/R3/R4/R5/R6(a–e)/R9 all CONFIRMED with file:line
  evidence (envs: glm5 recommended; kimi carries a steering plugin; mistral is
  0.11.2 not 0.16). Big wins: HMA auto-disables under any connector config
  (SWA-freeing risk retired for free); `DecodeBenchConnector` proves
  claim-without-store; `kv_connector_module_path` out-of-tree loading verified
  live in glm5. New facts folded in: block_size=16 claim alignment, fused-KV
  backend-dependent layouts, request-id rewrite wrinkle, no KV-edit API exists
  (grep clean). **Discovery: `/models/Trinity-Large-Preview` is gone; only
  TrueBase remains** (confirmed by main session `ls` — spec retargeted, §1).
  vLLM source mirror: `/tmp/claude-output/vllm016/vllm/` (local),
  logs `/tmp/claude-output/node1-*.log`.
- **2026-07-29, upstream vLLM/LMCache review (agent 3/3): OPTION B CONFIRMED,
  "better-supported by upstream 0.16 than the spec assumes."** Independently
  verified the same API facts at the v0.16.0 git tag. Adversarial findings
  folded in: silent-no-op loads as the primary Gate-0 failure mode (vllm#18489),
  `kv_load_failure_policy="fail"` for gates, no-full-cudagraph rule, APC-off
  promoted to hard requirement (injected blocks re-enter the hash pool),
  claim-align-down + tail-recompute bookkeeping, §5.7 fallback promoted to
  primary v1 design, request-id side-channel deleted, pinned-CPU session store
  default. Prior art: Leyline (2606.01065) closest published system; vLLM RFCs
  #25672/#36311/#37003 confirm demand, no supply. All verdicts cross-checked
  consistent with the node1 recon (independent methods, same conclusions).
