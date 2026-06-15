"""exp07 — trinity eviction matrix on REAL long context (run on node1, 8 GPUs).

The first real-data, at-scale test on the deployment target. For each context length (swept
across trinity's 4096 sliding-window boundary) it:
  * loads real text from the curated corpus (data/eval_docs.jsonl),
  * inserts a SYNTHETIC needle (made-up passcode) at a depth — memorization-proof recall,
  * runs the decomposition: KL(full‖shortened) = info loss, KL(shortened‖rotation) = mechanism
    error (the verdict), KL(full‖rotation) = total, plus naive (drop-sinks),
  * times rotation vs the recompute it replaces.

Memorization caveat: corpus docs are kotodama training data, so absolute drift may read low;
the relative comparisons + the synthetic needle are the trustworthy signals.

Usage:
    python experiments/exp07_trinity_matrix.py --model /models/Trinity-Large-Preview \
        --source data/eval_docs.jsonl --lengths 8192 16384 --evict-frac 0.6
"""

from __future__ import annotations

import argparse
import time

import torch


def evaluate_length(lm, source, L, args) -> None:
    from kvrot.config import EvictionSpec
    from kvrot.data import insert_needle, load_text_context
    from kvrot.eviction import compute_keep_indices
    from kvrot.harness import (
        _greedy_decode,
        _teacher_forced_logits,
        answer_logprob,
        prefill_snapshot,
    )
    from kvrot.metrics import stepwise_kl
    from kvrot.snapshot import evict as evict_kv
    from kvrot.snapshot import reindex

    tok, dev, inv = lm.tokenizer, lm.device, lm.inv_freq
    PASS = "73914"
    needle = tok(f" Note: the vault passcode is {PASS}. ", add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    base = load_text_context(source, tok, max(8, L - needle.shape[1]), device=dev, doc_index=args.doc)
    ctx, n_start, n_end = insert_needle(base, needle, args.needle_depth)
    C = ctx.shape[1]

    suffix = tok("\n\nQuestion: What is the vault passcode?\nAnswer: The vault passcode is", add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    target = tok(f" {PASS}", add_special_tokens=False, return_tensors="pt").input_ids.to(dev)

    prefill_ids, last_tok = ctx[:, :-1], ctx[:, -1:]
    P = prefill_ids.shape[1]
    evict_n = int(args.evict_frac * P)
    keep = compute_keep_indices(
        P, EvictionSpec(policy="sink_window", num_sink_tokens=args.sinks, evict_count=evict_n)
    ).to(dev)
    K = int(keep.shape[0])
    needle_evicted = not bool(((keep >= n_start) & (keep < n_end)).any().item())

    print(f"\n== L≈{C} (doc {args.doc}) | keep {K}/{P} | evict {evict_n} after {args.sinks} sinks "
          f"| needle@{args.needle_depth:.2f} evicted={needle_evicted} ==")

    ref_cont, full_logits = _greedy_decode(lm, prefill_ids, last_tok, args.gen)
    probe = torch.cat([last_tok, ref_cont[:, :-1]], dim=1)

    s0 = prefill_snapshot(lm, prefill_ids)
    torch.cuda.synchronize(); t = time.perf_counter()
    snap_rot = reindex(evict_kv(s0, keep), torch.arange(K, device=dev), inv)
    torch.cuda.synchronize(); t_rot = time.perf_counter() - t

    shortened = prefill_ids[:, keep]
    torch.cuda.synchronize(); t = time.perf_counter()
    snap_short = prefill_snapshot(lm, shortened)
    torch.cuda.synchronize(); t_short = time.perf_counter() - t

    keep_naive = torch.arange(P - K, P, device=dev)
    snap_naive = reindex(evict_kv(s0, keep_naive), torch.arange(K, device=dev), inv)

    short_logits = _teacher_forced_logits(lm, snap_short, probe)
    rot_logits = _teacher_forced_logits(lm, snap_rot, probe)
    naive_logits = _teacher_forced_logits(lm, snap_naive, probe)

    def kl(a, b):
        return stepwise_kl(a, b).mean().item()

    print(f"  info-loss  KL(full ‖ shortened)   = {kl(full_logits, short_logits):.3e}")
    print(f"  MECH error KL(shortened ‖ rotation) = {kl(short_logits, rot_logits):.3e}   <-- verdict")
    print(f"  total      KL(full ‖ rotation)     = {kl(full_logits, rot_logits):.3e}")
    print(f"  naive      KL(full ‖ naive)        = {kl(full_logits, naive_logits):.3e}")
    for name, snap in (("full", s0), ("shortened", snap_short), ("rotation", snap_rot)):
        lp, rec = answer_logprob(lm, snap, suffix, target)
        print(f"  recall {name:<10} logprob={lp:8.3f}  {'YES' if rec else 'no'}")
    print(f"  cost: rotation {t_rot*1e3:7.1f} ms (0 toks) vs recompute {t_short*1e3:8.1f} ms "
          f"({K} toks) = {t_short/max(t_rot,1e-9):.1f}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Trinity-Large-Preview")
    ap.add_argument("--source", default="data/eval_docs.jsonl")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--doc", type=int, default=0)
    ap.add_argument("--needle-depth", type=float, default=0.4)
    ap.add_argument("--evict-frac", type=float, default=0.6)
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--gen", type=int, default=16)
    args = ap.parse_args()

    from kvrot.harness import load_model

    lm = load_model(args.model, device_map="auto")
    a = lm.arch
    print(f"loaded {args.model}: {a.num_hidden_layers}L ({len(a.sliding_layers)} sliding W={a.sliding_window} "
          f"+ {len(a.full_attention_layers)} full), applies_rope={sum(a.applies_rope)}/{a.num_hidden_layers}")
    for L in args.lengths:
        try:
            evaluate_length(lm, args.source, L, args)
        except Exception as e:  # keep going across lengths even if one OOMs
            print(f"  L={L} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
