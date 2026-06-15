# References (verified literature sweep, 2026-06)

Grouped by tier. Numbers are from primary sources; **⚠ flags** mark claims the research
pass could not fully verify (treat as approximate).

## Tier 0/1 — RoPE re-rotation, sinks, training-free eviction

- **RoFormer / RoPE** — relative-position property ⟨f(q,m),f(k,n)⟩ = g(q,k,m−n); rotations
  compose (R(a)R(b)=R(a+b)), which is what makes re-rotation exact.
  https://arxiv.org/abs/2104.09864
- **StreamingLLM** — keep ~4 attention-sink tokens + rolling window; re-index positions
  *within the cache*. Window-only collapses (5.40→5158 ppl, Llama-2-13B); with sinks
  stable to 4M tokens; first token >50% attention mass.
  https://arxiv.org/abs/2309.17453 · github.com/mit-han-lab/streaming-llm
- **LM-Infinite** — Λ-mask (global start tokens + local window + distance clamp);
  generalises to 200M tokens, 2.7× speedup. https://arxiv.org/abs/2308.16137
- **Massive Activations** — a few residual dims (esp. BOS/first token) hold 10³–10⁶×
  values, the mechanism behind sinks. https://arxiv.org/abs/2402.17762
- **H2O** — keep recent + "heavy hitter" tokens; <20% budget ≈ full quality.
  https://arxiv.org/abs/2306.14048
- **SnapKV** — recent observation window votes for important prefix tokens; 1024-tok cache
  beats H2O-4096 on 11/16 LongBench. https://arxiv.org/abs/2404.14469
- **PyramidKV** — per-layer budget (more shallow, less deep); matches full at ~12% cache.
  https://arxiv.org/abs/2406.02069
- **TOVA** (https://arxiv.org/abs/2401.06104) · **Scissorhands**
  (https://arxiv.org/abs/2305.17118) · **FastGen** (https://arxiv.org/abs/2310.01801) ·
  **Keyformer** (https://arxiv.org/abs/2403.09054) · **Quest** (query-aware page select,
  *no eviction*; passkey 99% vs ~1% for evictors, https://arxiv.org/abs/2406.10774) ·
  **Ada-KV** (https://arxiv.org/abs/2407.11550).
- ⚠ **Pitfalls of KV-cache compression** (https://arxiv.org/abs/2510.00231) — eviction →
  silent instruction-skipping / system-prompt leakage (IFEval; per-number deltas not
  extracted). The "position-id re-indexing" section attribution was **not** confirmed.

## Tier 3 — selective recompute toward exactness

- **CacheBlend** — reuse multi-chunk KV, recompute top high-KV-deviation tokens
  (default 15%) → ≤0.002 F1 loss, 2.2–3.3× TTFT, 2.8–5× throughput. EuroSys'25.
  https://arxiv.org/abs/2405.16444
- **EPIC** — position-independent chunk reuse; recompute ~constant <20 boundary tokens →
  0–7% drop; up to 8× TTFT. ICML'25. https://arxiv.org/abs/2410.15332
- **Prompt Cache** (modular prompt KV, https://arxiv.org/abs/2311.04934) · **RAGCache**
  (exact prefix-path reuse only, https://arxiv.org/abs/2404.12457) · **KVLink** (link
  tokens + fine-tune; recompute ~5 tok/doc, https://arxiv.org/abs/2502.16002) · **APE**
  (training-free parallel encoding; 98% RAG / 93% ICL retained,
  https://arxiv.org/abs/2502.05431) · **Block-Attention** (RoPE re-rotation + FT; ≤1% gap,
  16% drop without FT, https://arxiv.org/abs/2409.15355) · **CacheGen**
  (https://arxiv.org/abs/2310.07240).
- Selector signal: **KV deviation** ‖reused − true KV‖ (CacheBlend's HKVD) proxies
  attention-output deviation. No method gives a *provable* error bound — all empirical on
  QA/summarisation F1, and all target RAG chunk-fusion, not prefix-pop.

## Tier 4 — information-preserving consolidation (training)

- **Cartridges** — per-corpus KV trained via self-study/context-distillation; **38.6×
  memory / 26.4× throughput at ICL parity**; per-context offline training.
  https://arxiv.org/abs/2506.06266 · github.com/HazyResearch/cartridges
  (⚠ training cost not disclosed)
- **Activation Beacon** — beacon tokens condense per-layer activations; **8× KV mem / 2×
  speed at parity**, <9h on 8×A800, drop-in. https://arxiv.org/abs/2401.03462
- **Compressed Context Memory** — online K/V compression via conditional LoRA; 5× at
  full-context quality, streaming. https://arxiv.org/abs/2312.03414
- **Gist tokens** (https://arxiv.org/abs/2304.08467) · **AutoCompressor**
  (https://arxiv.org/abs/2305.14788) · **ICAE** (4×→99% recon,
  https://arxiv.org/abs/2307.06945) · **500xCompressor** (6–480×, 62–73% retained @500×,
  https://arxiv.org/abs/2408.03094) · **RMT** (https://arxiv.org/abs/2207.06881) ·
  **Memorizing Transformers** (https://arxiv.org/abs/2203.08913) · **Landmark Attention**
  (https://arxiv.org/abs/2305.16300) · **CEPE** (https://arxiv.org/abs/2402.16617).
- Gap: none address RoPE position re-indexing of compressed/evicted prefixes; "drift" is
  measured as ppl/accuracy, not behavioural/agentic continuity.

## Systems

- **vLLM** — RoPE applied in the model layer **before** the cache write
  (`LlamaAttention.forward`: `q,k = rotary_emb(...)` → `self.attn`); FlashAttn backend
  writes/reads keys as-is (`reshape_and_cache_flash`). No in-place block-mutate API
  (only evict/reset); APC hashes blocks by token-ids+parent (position shift breaks
  prefix-sharing → need position-epoch/salt). ⚠ exact `attention/layer.py` path,
  FlashInfer path unverified.
- **SGLang** — most hackable: `MHATokenToKVPool.set_kv_buffer(layer, loc, k, v)` direct
  slot write; RadixCache stores indices, not tensors. Stored K also pre-rotated.
- **LMCache / CacheBlend** — closest existing prior art (non-prefix KV reuse + selective
  recompute in-engine). https://github.com/LMCache/LMCache
- **OpenVINO** note: cache re-rotation is exact only for standard linear/llama3 RoPE;
  dynamic-NTK variants degrade (we guard via `assert_rotation_homomorphism`).

## Exactness regimes (where prefix removal is genuinely exact, no recompute)

1. **Sliding-window layer**, evicted block ≥ W before oldest survivor → provably
   un-attended (Mistral W=4096; trinity SWA layers). Per-layer.
2. **Pure additive linear attention** (no decay/gating): state = Σ φ(k)ᵀv is subtractable
   exactly. Breaks with any decay/gate (RetNet γ, GLA/Mamba-2 data-dependent decay,
   RWKV). N/A for our RoPE-softmax targets, noted for completeness.
3. **RoPE positional component**, any softmax model: re-rotation exact (Tier 0). Content
   is *not* corrected — that's the measured residual.
