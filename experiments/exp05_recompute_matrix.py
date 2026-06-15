"""exp05 — the comparison matrix (run on node1).

Decomposes drift into *information loss* (forgetting [1]) vs *mechanism error* (reuse
contamination), by adding the missing baseline: a fresh recompute of the shortened context.

References & variants, all teacher-forced on the SAME continuation:
  full              — no eviction (carries [1]).
  shortened/no-KV   — fresh prefill of the survivors (clean KV; [1] gone). The "correct
                      answer" once we've decided to drop [1].
  rotation          — survivors' KV reused from the full pass + re-rotated (our mechanism).
  no-rotate         — evict but leave positions (gap).
  naive             — drop the sinks too (the broken baseline).

Decomposition:
  KL(full ‖ shortened)  = information loss of forgetting [1]   (policy budget)
  KL(shortened ‖ rotation) = mechanism error / contamination  (the verdict on rotation)
  KL(full ‖ rotation)   = total drift

A passcode fact is planted in the *evicted* span, so full recalls it, shortened correctly
forgets it, and rotation should match shortened (faithful forgetting). We also time the
rotation surgery vs the recompute it replaces.

Usage:
    python experiments/exp05_recompute_matrix.py --model /models/llama-3.2-3b-instruct \
        --context-len 512 --gen 24 --evict 128
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-len", type=int, default=512)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--evict", type=int, default=128)
    args = ap.parse_args()

    from kvrot.config import EvictionSpec
    from kvrot.eviction import compute_keep_indices, new_positions_for
    from kvrot.harness import (
        _greedy_decode,
        _teacher_forced_logits,
        answer_logprob,
        load_model,
        prefill_snapshot,
    )
    from kvrot.metrics import stepwise_kl
    from kvrot.snapshot import evict, reindex

    lm = load_model(args.model)
    tok, dev, inv = lm.tokenizer, lm.device, lm.inv_freq

    def enc(s, bos):
        return tok(s, return_tensors="pt", add_special_tokens=bos).input_ids.to(dev)

    PASSCODE = "48127"
    pre = enc("We begin a long session and record the following note. ", bos=True)
    fact = enc(f"The vault passcode is {PASSCODE}. ", bos=False)
    fill = enc(
        "The team proceeds through the task, noting observations and revisiting points "
        "as discussion continues over many turns. ",
        bos=False,
    )
    reps = max(1, (args.context_len - pre.shape[1] - fact.shape[1]) // fill.shape[1] + 1)
    context_ids = torch.cat([pre, fact, fill.repeat(1, reps)], dim=1)[:, : args.context_len]
    C = context_ids.shape[1]
    fact_start, fact_end = pre.shape[1], pre.shape[1] + fact.shape[1]

    suffix = enc("\nQuestion: What is the vault passcode?\nAnswer: The vault passcode is", bos=False)
    target = enc(f" {PASSCODE}", bos=False)

    prefill_ids, last_tok = context_ids[:, :-1], context_ids[:, -1:]
    P = prefill_ids.shape[1]
    keep = compute_keep_indices(
        P, EvictionSpec(policy="sink_window", num_sink_tokens=args.sinks, evict_count=args.evict)
    ).to(dev)
    K = int(keep.shape[0])
    evicted = sorted(set(range(P)) - set(keep.tolist()))
    fact_evicted = all(i in set(evicted) for i in range(fact_start, min(fact_end, P)))
    print(f"loaded {args.model}; C={C}, evict [{args.sinks},{args.sinks+args.evict}) → keep {K}; "
          f"fact span=[{fact_start},{fact_end}) evicted={fact_evicted}\n")

    G = args.gen
    ref_cont, full_logits = _greedy_decode(lm, prefill_ids, last_tok, G)
    probe = torch.cat([last_tok, ref_cont[:, :-1]], dim=1)

    # full snapshot (the pre-existing cache; not part of per-evict cost)
    s0 = prefill_snapshot(lm, prefill_ids)

    # --- rotation: time only the surgery (evict + re-rotate) ---
    torch.cuda.synchronize(); t = time.perf_counter()
    snap_rot = reindex(evict(s0, keep), torch.arange(K, device=dev), inv)
    torch.cuda.synchronize(); t_surgery = time.perf_counter() - t

    # --- shortened recompute / no-KV: time the fresh prefill the mechanism avoids ---
    shortened_prefill = prefill_ids[:, keep]
    torch.cuda.synchronize(); t = time.perf_counter()
    snap_short = prefill_snapshot(lm, shortened_prefill)
    torch.cuda.synchronize(); t_recompute = time.perf_counter() - t

    snap_norot = reindex(evict(s0, keep), new_positions_for(keep, s0.positions[0], recompact=False), inv)
    keep_naive = torch.arange(P - K, P, device=dev)
    snap_naive = reindex(evict(s0, keep_naive), torch.arange(K, device=dev), inv)

    short_logits = _teacher_forced_logits(lm, snap_short, probe)
    rot_logits = _teacher_forced_logits(lm, snap_rot, probe)
    norot_logits = _teacher_forced_logits(lm, snap_norot, probe)
    naive_logits = _teacher_forced_logits(lm, snap_naive, probe)

    def kl(a, b):
        return stepwise_kl(a, b).mean().item()

    print("== drift decomposition (mean KL over continuation) ==")
    print(f"  information loss   KL(full ‖ shortened)   = {kl(full_logits, short_logits):.3e}")
    print(f"  MECHANISM error    KL(shortened ‖ rotation) = {kl(short_logits, rot_logits):.3e}   <-- verdict on rotation")
    print(f"  total drift        KL(full ‖ rotation)     = {kl(full_logits, rot_logits):.3e}")
    print(f"  no-rotate (gap)    KL(shortened ‖ no-rot)  = {kl(short_logits, norot_logits):.3e}")
    print(f"  naive (drop sinks) KL(full ‖ naive)        = {kl(full_logits, naive_logits):.3e}")

    print("\n== recall of the planted (evicted) fact ==")
    for name, snap in (("full", s0), ("shortened", snap_short), ("rotation", snap_rot)):
        lp, rec = answer_logprob(lm, snap, suffix, target)
        print(f"  {name:<10} logprob={lp:8.3f}  recall={'YES' if rec else 'no'}")

    print("\n== cost (per eviction) ==")
    print(f"  rotation surgery (reuse {K} survivors): {t_surgery*1e3:7.2f} ms,  tokens recomputed = 0")
    print(f"  shortened recompute / no-KV          : {t_recompute*1e3:7.2f} ms,  tokens recomputed = {K}")
    print(f"  speedup (recompute / rotation)       : {t_recompute / max(t_surgery,1e-9):6.1f}x")


if __name__ == "__main__":
    main()
