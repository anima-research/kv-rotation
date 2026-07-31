# API contract — KV cache control (KvrotConnector) + playground

Two layers. **Layer 1** is the raw primitive: per-request cache control on a
vLLM server running the `KvrotConnector`. **Layer 2** is the playground REST
API, which wraps Layer 1 with a token ledger, turn-aligned eviction planning,
and chat framing — use it unless you specifically want to drive the cache
yourself.

Validated end-to-end by exp12 gates (rotated-vs-HF-oracle parity below
cross-stack kernel noise at 389B/TP8; `RESULTS.md` Phase 5).

---

## Layer 1 — vLLM completions with cache control

### Server

Start vLLM with the connector (exact flags in
`scripts/serve_playground_trinity.sh`; the non-negotiables):

```
--kv-transfer-config '{"kv_connector": "KvrotConnector",
    "kv_connector_module_path": "kvrot_vllm.connector", "kv_role": "kv_both"}'
--no-enable-prefix-caching --disable-hybrid-kv-cache-manager
--attention-backend FLASH_ATTN        # required for hybrid-window models (trinity)
```

### Request

Standard `POST /v1/completions`, plus three things:

```jsonc
{
  "model": "trinity",
  "prompt": [128000, 882, ...],        // TOKEN IDS, not text (see invariants)
  "max_tokens": 256, "temperature": 0.8,
  "stop_token_ids": [128009],
  "return_token_ids": true,            // get the reply as ids for your ledger
  "kv_transfer_params": {
    "kvrot": {
      "session_id": "my-session-1",    // names this session's KV store
      "plan": {                        // OPTIONAL — evict + re-rotate first
        "keep": [0, 1, 2, 3, 2052, 2053, ...],
        "src_len": 8192
      }
    }
  }
}
```

### Semantics

The connector keeps a per-`session_id` KV **store** on the server (one shard
per TP rank, host RAM). The contract:

1. **The store is always the token-identical prefix of your next prompt.**
   After every request, store := that request's full prompt (all prompt-token
   KV is saved at prefill; **generated tokens are never stored**).
2. **Reuse**: on the next request whose prompt extends the store, the
   connector silently claims `floor(min(store_len, len(prompt)-1) / 16) * 16`
   tokens — those skip prefill entirely. Everything after the claim
   (including the previous reply's tokens, which you must re-send yourself)
   is recomputed normally.
3. **Eviction + rotation** (`plan`): before loading, the connector keeps only
   the store rows in `keep` (strictly-increasing indices into the store),
   recompacts positions to `0..K-1`, and exactly re-rotates the surviving
   keys (RoPE layers only; NoPE layers pass through byte-identical). Your
   prompt for that request must then be exactly
   `[store tokens selected by keep] + new tokens`.
4. **`src_len` is a safety interlock**: it must equal the server-side store
   length (== your previous prompt length). Mismatch ⇒ error (strict mode) or
   clean recompute (production mode) — never silent garbage.
5. The response echoes `kv_transfer_params.kvrot = {session_id,
   stored_tokens, claimed_tokens}` — assert `claimed_tokens > 0` when you
   expected reuse.

### Invariants you must hold (or the math stops being exact)

- **Token ids, not text.** Retokenization at text boundaries breaks the
  prefix-identity requirement. Tokenize once, keep a ledger, send ids.
- Prompts must extend / match the store per rules 1–3. Never claim a prompt
  the store can't back.
- One in-flight request per session.
- Stores are LRU-capped (`kvrot_max_sessions`, default 4–6) and die with the
  server — a vanished store just means a full re-prefill, not corruption.
- Positions are recompacted (`0..K-1`); gap-preserving positions are not
  expressible in vLLM.

### Minimal working sequence

```bash
# 1) establish: full prefill, store <- prompt (claims 0)
curl -s $VLLM/v1/completions -d '{"model":"trinity","prompt":'"$CTX_IDS"',
  "max_tokens":64,"return_token_ids":true,
  "kv_transfer_params":{"kvrot":{"session_id":"demo"}}}'

# 2) extend: prompt = CTX_IDS + reply_ids + more_ids  -> claims ~len(CTX_IDS)
# 3) evict+rotate: keep sinks + tail of the store, prompt = kept ids + new ids
curl -s $VLLM/v1/completions -d '{"model":"trinity","prompt":'"$KEPT_PLUS_NEW"',
  "max_tokens":64,"return_token_ids":true,
  "kv_transfer_params":{"kvrot":{"session_id":"demo",
    "plan":{"keep":'"$KEEP"',"src_len":'"$PREV_PROMPT_LEN"'}}}}'
```

---

## Layer 2 — playground REST API (`http://node2:2222`)

Does all of the above for you (ledger, ChatML framing, turn-aligned plans,
store-coordinate bookkeeping). If `KVROT_TOKEN` is set, send it as
`X-Kvrot-Token` or `?token=`.

| Endpoint | Body | Does |
|---|---|---|
| `GET /api/health` | — | vLLM/model status |
| `POST /api/sessions` | `{config?, preamble?}` | new session |
| `GET /api/sessions` | — | list sessions/branches |
| `GET /api/sessions/{id}` | — | full state (turns w/ evicted flags, events) |
| `POST .../turns` | `{text, ab?}` | send a turn; `ab:true` also answers from a recompute-policy control (display-only, side-by-side) |
| `POST .../seed` | `{template, doc_index?, target_tokens, depths?}` | fill a fresh session with a long doc as evictable turns + planted needles; returns the needle table |
| `POST .../config` | `{policy?, budget?, ...}` | live-switch policy/budget |
| `POST .../evict` | — | force one eviction round now |
| `POST .../fork` | `{name?}` | independent branch (fresh store) |
| `POST .../reroll` | — | resample last reply (full KV reuse); alternatives kept |
| `POST .../variant` | `{variant: n}` | switch the tail reply's active variant |
| `GET .../export` / `POST /api/sessions/import` | — / `{export}` | save/restore a session |

`config`: `policy` (`sink_rotate` \| `oldest` \| `recompute` \| `none`),
`budget` (eviction trigger, tokens), `num_sink_tokens`, `max_reply_tokens`,
`temperature`, `render_mode` (`auto`/`chat`/`prefill`).

Turn responses carry `stats`: `wall_s`, `claimed_tokens` (KV reused from the
rotated store), `prompt_tokens`, `evicted_turns/tokens`.

### What to expect (the honest mechanics)

Rotation preserves the evicted context's **influence**, not its verbatim
content: facts inside evicted turns are genuinely forgotten (correct
behaviour); facts in retained regions survive rotation exactly; and the
conversation's continuity survives eviction measurably better than a clean
recompute (`RESULTS.md` exp05/08b/11/12). The A/B toggle shows exactly this,
per message, live.
