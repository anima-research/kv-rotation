# HANDOFF — KV-cache rotation project

*Read this first. Written for the next agent/session picking this up cold. Last updated
2026-06-15. Companion docs: `notes/feasibility.md` (synthesis), `notes/journal.md`
(chronological log with every number), `notes/design-eviction.md` (policy/sync/matrix design),
`notes/references.md` (literature), `notes/status-stage1.md` (manager-facing report).*

---

## 0. TL;DR of where we are

We're prototyping **accurate KV-cache rotation**: when a long/agentic session must drop its
oldest context to reclaim space, instead of invalidating the cache (the status quo) we
**surgically evict those tokens and cheaply re-rotate the survivors' positions**, reusing their
KV. Validated on Llama-3.2-3B (exp01–05) and brought up end-to-end on the real target,
**Trinity-Large-Preview** (afmoe, ~389B MoE), exp06 smoke + exp07 (real long-context matrix).

**Core result so far:** rotation reproduces a clean recompute almost exactly (mechanism error
~0.01 nats on the 3B) at **~64× lower cost**, and for the "behave as if full context were present"
goal it is *closer to full than a recompute is* — because survivors still carry the evicted block's
baked-in influence. **On trinity at scale (exp07, 8k/16k real context) the mechanism stays
near-error-free (mech-KL ≤3e-4, same tiny order as the information loss), and trinity even tolerates
naive sink-dropping** — the favourable case. CPU correctness tests prove the rotation math is
bit-exact (26/26 green).

Status: Phase 1 complete on the 3B; **trinity validated at scale** (exp07 decomposition +
mechanism near-error-free/naive-tolerant; exp08 **recall preserved at scale** — retained needle
recalls under rotation ≈ full, 100% top1, evicted faithfully dropped; exp08b **clean perf +
decomposition** on a no-offload load — mech-KL ≤5e-4, rotation closer to full than recompute for
evicted content at both lengths, warm rotation 29–51 ms vs recompute 20–29 s = ~560–720× in HF
naive-MP). Committed + pushed to `git@github.com:anima-research/kv-rotation.git`. Trinity feasibility
is essentially settled; the open frontier is realism (chat-shaped eval), exactness (Tier 3), and the
production vLLM/TP port (where the true speedup ratio gets measured).

---

## 1. The idea & the reframing (read this carefully — it's the whole project)

**Problem.** Today any change to the prompt prefix invalidates the KV cache from that point on,
forcing a recompute. That kills prompt caching for exactly the workload we care about: rolling
context, where you pop early turns to make room.

**The load-bearing reframing — "exact relative to *what*?"**
- **Exact vs. the *full* prompt** (behave as if the dropped context were still present, just
  reclaim the space) — **THIS is the goal.** It matches the manager's framing of the cache as
  the model's "continuative self": keep the *influence* of old context, shrink its footprint.
- **Exact vs. the *shortened* prompt** (behave as if the early tokens never existed) — provably
  needs a near-full recompute, and it *discards* the old info. NOT the goal. Don't chase it.

**The feasibility law (organises everything):**
> drift vs. full-prompt ≈ the future attention mass that *would* have landed on the evicted block.

So feasibility is **content-dependent**: dropping stale context is near-free; dropping
still-referenced context costs. Corollaries: **evict by importance, not age**, and **never drop
the first few "sink" tokens** (they hold disproportionate attention; dropping them is
catastrophic).

**The drift decomposition (how we measure honestly, exp05/exp07):**
- `KL(full ‖ shortened-recompute)` = **information loss** of forgetting the block (a policy
  choice; fine if stale).
- `KL(shortened-recompute ‖ rotation)` = **mechanism error** — how far our cheap reuse is from an
  actual clean recompute. **This is the verdict on the mechanism.** Want ≈ 0.
- `KL(full ‖ rotation)` = total drift.
- Surprising empirical result: rotation is *closer to full* than the recompute is — it retains
  the evicted block's baked-in influence that a recompute throws away. So for continuity,
  rotation isn't just cheaper than recompute, it's *better*.

**North-star metric:** per-step `KL(p_full ‖ p_rotated)` over a teacher-forced continuation +
top-1 agreement + needle recall. **NOT perplexity** (the consolidation literature uses ppl; we
deliberately don't — behavioural drift is the honest signal).

---

## 2. The mechanism stack (tiers)

| Tier | What | Cost | Status |
|---|---|---|---|
| 0 | RoPE re-rotation of survivors (`R(−k)`) | ~free | ✅ proven bit-exact (tests) |
| 1 | sink-aware eviction (keep first N tokens) | ~free | ✅ |
| 2 | importance eviction (H2O accumulated attention) | ~free | ✅ on Llama; ⚠ trinity needs `output_attentions` (afmoe forward may not thread it — check) |
| 3 | selective recompute (CacheBlend-style, high-KV-deviation tokens) | compute | ⛔ not built |
| 4 | learned consolidation (Gist/Cartridges) | training | ⛔ not built |

Tier 0 detail: in HF/vLLM/SGLang the stored key is **pre-rotated** by absolute position, so to
move a token you read its stored K, multiply by `R(Δpos)`, write back; **V is position-free**.
Re-rotation is exact iff `inv_freq` is position-independent (true for standard + llama3 + linear
scaling; we assert the rotation homomorphism at load to catch dynamic-NTK). **Re-rotation must be
per-layer-gated** (see trinity NoPE below).

---

## 3. Trinity architecture (the target — critical details)

`Trinity-Large-Preview` = **afmoe**, ~389B MoE. 60 layers: **`[sliding×3, full]×15` → 45
sliding-window (W=4096) + 15 full-attention.** GQA 8 KV heads, head_dim 128, θ=10000, **no rope
scaling**, muP, 256k context. Uses a plain `DynamicCache` with **mask-enforced** sliding windows.

**THE critical finding (from reading `reference/modeling_afmoe.py`):** RoPE is applied **only on
the sliding layers** — `apply_rotary_pos_emb` is gated by `is_local_attention`. **The 15 global
layers are NoPE** (no position encoding). Consequences:
- Re-rotating the global layers would *corrupt* them (rotating never-rotated keys). So
  `ArchSpec.applies_rope` is per-layer (afmoe → sliding-only; everything else → all layers), and
  `snapshot.reindex` only rotates RoPE layers, leaving NoPE keys byte-for-byte.
- This makes trinity *easier*: 15 NoPE layers need no positional fix on eviction (just drop
  keys); 45 sliding layers re-rotate, and evicting beyond their 4096 window is exact.

`Trinity-Large-TrueBase` = the base model, **8k context** (good for an in-distribution base check,
too short for long-context). Preview ships a `chat_template.jinja` → it's the instruct/chat
variant and the deployment target.

---

## 4. Environment & how to run (IMPORTANT: node1 is shared)

**Local dev box** (`/home/luxia/projects/kv-rotation`): no GPU, CPU torch, `uv`-managed. Run the
correctness tests here: `uv run pytest` (26/26). Edit code here.

**node1** (`ssh node1`): **SHARED machine, 8× B200 (183 GB each).** Read the global rules in
`~/.claude/CLAUDE.md`. Hard constraints:
- **NEVER** `pip install`/`conda install` outside an explicitly activated venv; **never** write
  outside `~/luxi-files/`. A stray `~/.local/` install poisons every conda env on the box.
- Use the **shared venv read-only**: `~/luxi-files/.venv-shared` (Python 3.12, **torch
  2.11+cu128, transformers 5.3.0, accelerate 1.13**). It has no `pip`; don't install into it.
- We run via `PYTHONPATH=src` against that venv's python — no install of our package needed.

**Models:** `/models/llama-3.2-3b-instruct`, `/models/Trinity-Large-Preview`,
`/models/Trinity-Large-TrueBase`. **Data:** `/models/kotodama-data/deduped/*.jsonl` (the kotodama
*training* corpus — see memorization caveat §6). Curated eval subset:
`~/luxi-files/kv-rotation/data/eval_docs.jsonl` (25 diverse docs; rebuild with
`scripts/sample_eval_corpus.py`).

**Workflow (edit → sync → run):**
```bash
# from the dev box, after editing:
rsync -az --exclude .venv --exclude .git --exclude __pycache__ --exclude .pytest_cache \
  --exclude runs --exclude uv.lock --exclude reference --exclude data \
  /home/luxia/projects/kv-rotation/ node1:~/luxi-files/kv-rotation/

# Llama experiments (single GPU, be a good citizen):
ssh node1 'cd ~/luxi-files/kv-rotation && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/luxi-files/.venv-shared/bin/python \
  experiments/exp01_rotation_drift.py --model /models/llama-3.2-3b-instruct ...'

# trinity (ALL 8 GPUs, device_map=auto; check nvidia-smi is free first!):
ssh node1 'cd ~/luxi-files/kv-rotation && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONUNBUFFERED=1 PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ~/luxi-files/.venv-shared/bin/python experiments/exp08_trinity_recall.py --lengths 8192 16384'
```
Run long jobs in the background and read the output file on completion (filter the
`Loading weights`/`it/s]` progress-bar spam with `grep -avE`). **Always `nvidia-smi` before
grabbing all 8 GPUs** — others use this box.

Gotchas learned (exp08):
- **Buffering:** piping python through `tee`/a file block-buffers stdout, so `print()` progress is
  invisible until the run ends (looks "stuck"). Always set `PYTHONUNBUFFERED=1`; exp08 also flushes
  + stamps phases `[t+…s]`. Loading a ~727 GiB model from disk is single-threaded (GPU 0%, load 1.0)
  and takes minutes — watch the `Loading checkpoint shards` bar; that's normal, not a hang.
- **Memory:** trinity weights ≈ 727 GiB on-GPU; `device_map="auto"` places them **imbalanced** (g7
  ~164/178, g0–g6 ~80/178), causing a small disk offload even at `--mem-frac 0.95`. It's imbalance,
  not capacity — ~600 GiB free aggregate; 16k activations add only +2–4 GiB. For long contexts /
  clean recompute timing, rebalance (`device_map="balanced"` or `--mem-frac 0.98`) first.
- The `max_memory` cap is now sized from each GPU's **free** memory (`load_model(max_memory_frac=…)`,
  default 0.92), so it adapts to contention on the shared box instead of over-promising.

---

## 5. The 5 transformers 4.57→5.3 compat shims (all in `harness.load_model`)

afmoe targets transformers 4.57; the shared venv is 5.3. None touch model math; all verified
(homomorphism assert + live-model `inv_freq`). If trinity loading breaks after a transformers
upgrade, look here first:
1. **rope_scaling parse** — 5.x rewrites null → `{"rope_type":"default"}`; `RoPESpec` maps
   default/None → none.
2. **token-id backfill** — 5.x raises on unset config attrs; backfill pad/bos/eos.
3. **pass `config=cfg`** to `from_pretrained` so backfills take effect.
4. **plain RoPE as `linear`/factor=1.0** — sidesteps 5.x routing `"default"` rope through a
   module method (`compute_default_rope_parameters`) that 4.x rotary classes lack (hits both
   `__init__` and `_init_weights`). Identical math.
5. **`device_map="auto"` `max_memory` cap (0.80×VRAM)** — default packs weights to the brim and
   OOMs the forward by ~36 MiB; reserve headroom.

Also: **cross-device surgery** — trinity is sharded across 8 GPUs, so `snapshot.evict/reindex`
and `from_hf_cache` align position/index tensors to each layer's device. And transformers 5.x
cache API is `cache.layers[i].keys/.values` (not `key_cache`).

---

## 6. Caveats that affect interpretation

- **Memorization.** The eval corpus is kotodama *training* data (`train.provenance.json` lists
  all `deduped/*` as training sources). Trinity's own data is proprietary/unknown but likely
  overlaps these common public sets. ⇒ **absolute** drift may read artificially low; trust the
  **relative** comparisons (method vs method) and the **synthetic needle** (a made-up passcode
  the model can't have memorized) for recall.
- **Continuation source bias (exp05).** "rotation closer to full than recompute" was measured on
  a continuation generated by the *full* model, which favors KV-sharing variants. A
  neutral-source rerun is queued to size the pure effect (direction is sound regardless).
- **Synthetic vs real context.** exp01–06 used repetitive synthetic filler → under-estimates
  drift (redundant text makes eviction trivially cheap). exp07 switched to real long docs (pg19
  books etc.); prefer real content going forward.
- **Base vs instruct / chat shape.** Deployment is rolling *chat*; our contexts are raw docs
  (mild OOD on the instruct Preview). The fully-faithful eval is chat-formatted multi-turn with
  turn-aligned eviction — not yet done (no clean turn-data jsonl in kotodama; OASST2 only a log).

---

## 7. Code map (`src/kvrot/`)

- `rope.py` — Tier 0: `reindex_keys` (apply `R(Δ)` to stored keys), `default_inv_freq`,
  `llama3_inv_freq`, `assert_rotation_homomorphism` (exactness guard).
- `snapshot.py` — `KVSnapshot` (per-layer K/V/positions/layer_types/**applies_rope**) + surgery:
  `evict`, `reindex` (per-layer RoPE-gated), HF adapters (`from_hf_cache`, `to_hf_dynamic_cache`).
  Device-sharding-aware.
- `eviction.py` — policies → keep-indices: `compute_keep_indices` (none/oldest/sink_window),
  `importance_keep_indices` (Tier 2), `new_positions_for` (recompact vs gap).
- `metrics.py` — `stepwise_kl`, `top1_agreement`, `DriftReport`, `kv_deviation` (for Tier 3).
- `config.py` — pydantic `RoPESpec`/`ArchSpec`/`EvictionSpec`; `ArchSpec.from_hf_config` sets
  `applies_rope` (afmoe→sliding-only).
- `harness.py` — real-model runner: `load_model` (the 5 shims), `prefill_snapshot`,
  `run_trial`, `rolling_replay`, `answer_logprob`, `prefill_with_importance`, decode helpers.
- `data.py` — context loaders: `read_texts`/`load_text_context` (txt/dir/jsonl/json),
  `insert_needle`, `synthetic_context` (fallback).

`tests/` (26, CPU, no model): rope exactness, snapshot surgery (incl. NoPE gating), eviction,
metrics, config, data. `scripts/sample_eval_corpus.py` builds the eval subset on node1.
`reference/` (gitignored) holds the vendored afmoe source for inspection.

---

## 8. Experiments (all in `experiments/`, results in `notes/journal.md`)

- **exp01** drift vs full (Llama): sink+rot near-exact (KL ~1e-4–1e-3, 100% top1); naive
  catastrophic (KL 2.6–4.5). recompact≈gap within scaled range.
- **exp02** recall: retained facts survive (−0.025 ≈ full −0.029); dropping live facts loses them.
- **exp03** long-horizon rollout: drift bounded over repeated pops; recompaction caps positions.
- **exp04** Tier-2 importance: keeps a mid-context fact age-based eviction drops; KL is
  content-dependent.
- **exp05** the matrix + decomposition: mechanism error 0.014; rotation closer to full (0.027)
  than recompute (0.091); 64× cheaper.
- **exp06** trinity smoke: loads on 8 GPUs, applies_rope=45/60 ✓, homomorphism ✓, runs.
- **exp07** trinity real-data matrix (8k/16k real pg19): mechanism near-error-free at 389B
  (mech-KL ≤3e-4 ≈ info-loss); **naive/oldest safe at scale** (no sink fragility); rotation
  34–48 ms vs disk-offloaded recompute 24–42 s. Needle was evicted → tested forgetting, not recall;
  continuity-beats-recompute held at 8k, flipped at 16k (evicted block too stale to matter).
- **exp08** trinity recall sweep (rotation-only): needle at depth 0.80 (**retained**) recalls under
  rotation (−0.12/−0.15 ≈ full) with 100% top1 and KL 4–6e-5; depth 0.40 (**evicted**) faithfully
  drops it (−14/−15). Closes exp07's recall gap. Mem instrumented: `auto` overloads g7 (164/178)
  while g0–g6 sit ~80/178 → small disk offload (imbalance, not capacity); 16k activations +2–4 GiB,
  so huge headroom for longer contexts. Rotation surgery 29–52 ms, GPU-resident.
- **exp08b** clean perf + decomposition (`--device-map balanced --mem-frac 0.70 --with-recompute`):
  no disk offload (g7 164→123). Mech-KL ≤5e-4; **rotation closer to full than recompute for evicted
  content at both lengths** (exp07's 16k flip was the disk-confounded run). Warm rotation 29–51 ms vs
  GPU-resident recompute 20–29 s = **~560–720×** — but that's HF *naive* model-parallelism (1 GPU at
  a time); production TP ratio TBD. Cold first-call inflates (240 ms); report warm. Placement lever:
  cap per-GPU near the even-split, not the max.

---

## 9. Next steps (prioritized backlog)

1. **Pick a frontier — trinity feasibility is settled** (exp01–08b: faithful, recall-exact,
   continuity-preserving, ~3 orders cheaper). Remaining, in rough value order:
   (a) **Chat-shaped eval** — turn-aligned eviction on multi-turn chat (the real deployment shape;
       our biggest realism gap — all evals are raw docs). Needs a clean turn-data source.
   (b) **Longer contexts** (32k/64k/128k toward 256k) — exp08 shows memory is *not* the constraint
       (~600 GiB free aggregate; use `--device-map balanced --mem-frac 0.70`); stresses SWA-exactness.
   (c) **Tier 3 selective recompute** — drive drift→0 by recomputing high-deviation survivors, ideally
       only the 15 global layers. (d) **vLLM/TP port** — the only way to measure the production speedup
       (HF's ~560–720× is naive model-parallelism).
2. **Chat-shaped eval** — fetch OASST2 or chat-wrap docs as turns; turn-aligned eviction (the
   real deployment shape). No clean turn-data in kotodama yet.
3. **Neutral-continuation rerun of exp05** — size the "rotation > recompute for continuity"
   effect without the full-model-continuation bias.
4. **Tier 3 selective recompute** (CacheBlend-style) — recompute high-KV-deviation survivors,
   ideally only on the 15 global layers (where contamination concentrates). Drives drift→0 at
   bounded cost.
5. **Tier 2 on trinity** — needs `output_attentions`; check whether afmoe's forward threads it
   (it may not collect attentions — its `MoeModelOutputWithPast` returns None for them).
6. **SWA-exactness isolation** — measure per-layer that sliding layers with the evicted block
   beyond the 4096 window are bit-exact (the "75% of layers free" claim).
7. **vLLM port** (production): in-place KV mutate API + Δ-rotation kernel keyed on slot_mapping +
   APC re-hashing after position shift (or bypass APC for rotated blocks). Study LMCache/CacheBlend.

---

## 10. Quick start for the next agent

```bash
cd /home/luxia/projects/kv-rotation
uv run pytest                      # 26/26, proves the rotation math on CPU
sed -n '1,80p' notes/journal.md    # the chronological story + numbers
# then sync + run on node1 per §4. nvidia-smi before grabbing all 8 GPUs.
```
Memory file: `~/.claude/projects/-home-luxia-projects-kv-rotation/memory/` has the project memo.
Git: initialized, staged, **not committed** (user's call). When asked, the scaffold+notes are a
clean first commit.
