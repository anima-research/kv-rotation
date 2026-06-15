"""exp04 — Tier 2: importance-aware eviction vs. contiguous oldest-drop (run on node1).

At an *equal* kept-token budget, compare three eviction strategies:
  * oldest       — keep the last K tokens (drops sinks + everything old).
  * sink_window  — keep 4 sinks + the most recent K-4 (drops the middle).
  * importance   — keep sinks + a recent window + the highest-attention "heavy hitters".

We plant a fact in the *middle* of the context (so oldest/sink_window drop it) and measure,
per strategy: mean KL vs full on a generic continuation, recall of the planted fact, and
whether the fact's tokens were actually retained.

Hypothesis: importance lowers generic-continuation KL (keeps salient tokens). Open
question: does accumulated-attention importance *catch* a one-off mid-context fact? (It is
reactive, not predictive — we measure rather than assume.)

Usage:
    python experiments/exp04_importance.py --model /models/llama-3.2-3b-instruct
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--filler", type=int, default=200, help="approx filler tokens on each side")
    ap.add_argument("--gen", type=int, default=16)
    args = ap.parse_args()

    from kvrot.config import EvictionSpec
    from kvrot.eviction import compute_keep_indices, importance_keep_indices
    from kvrot.harness import (
        _greedy_decode,
        answer_logprob,
        load_model,
        prefill_with_importance,
    )
    from kvrot.metrics import stepwise_kl
    from kvrot.snapshot import evict, reindex

    lm = load_model(args.model, attn_implementation="eager")
    tok, dev = lm.tokenizer, lm.device

    def enc(s, bos):
        return tok(s, return_tensors="pt", add_special_tokens=bos).input_ids.to(dev)

    PASSCODE = "48127"
    pre = enc("We begin a long session. ", bos=True)
    fill = enc(
        "The team continues working through the task, recording observations and "
        "revisiting earlier notes as the discussion proceeds. ",
        bos=False,
    )
    reps = max(1, args.filler // fill.shape[1] + 1)
    fill_block = fill.repeat(1, reps)
    fact = enc(f"The vault passcode is {PASSCODE}. ", bos=False)

    # context: pre + filler + FACT (middle) + filler
    context_ids = torch.cat([pre, fill_block, fact, fill_block], dim=1)
    C = context_ids.shape[1]
    fact_start = pre.shape[1] + fill_block.shape[1]
    fact_end = fact_start + fact.shape[1]

    suffix = enc("\nQuestion: What is the vault passcode?\nAnswer: The vault passcode is", bos=False)
    target = enc(f" {PASSCODE}", bos=False)

    print(f"loaded {args.model} (eager); context={C}, fact span=[{fact_start},{fact_end}) "
          f"(~{fact_start/C:.0%} through)\n")

    s0, imp = prefill_with_importance(lm, context_ids)
    inv = lm.inv_freq

    # full-context references
    ref_cont, ref_logits = _greedy_decode(lm, context_ids[:, :-1], context_ids[:, -1:], args.gen)
    probe = torch.cat([context_ids[:, -1:], ref_cont[:, :-1]], dim=1)
    full_lp, full_recall = answer_logprob(lm, s0, suffix, target)
    print(f"  full context: recall_logprob={full_lp:.3f} ({'YES' if full_recall else 'no'})\n")

    from kvrot.harness import _teacher_forced_logits

    def evaluate(keep, label):
        keep = keep.to(dev)
        snap = reindex(evict(s0.clone(), keep), torch.arange(keep.shape[0], device=dev), inv)
        kl = stepwise_kl(ref_logits, _teacher_forced_logits(lm, snap, probe)).mean().item()
        lp, recall = answer_logprob(lm, snap, suffix, target)
        fact_kept = bool(((keep >= fact_start) & (keep < fact_end)).any().item())
        print(f"  {label:<14} kept={keep.shape[0]:>4}/{C}  mean_KL={kl:.3e}  "
              f"recall_lp={lp:8.3f} ({'YES' if recall else 'no '})  fact_retained={fact_kept}")

    for frac in (0.4, 0.6):
        K = int(frac * C)
        print(f"-- budget K={K} ({frac:.0%} of {C}) --")
        evaluate(compute_keep_indices(C, EvictionSpec(policy="oldest", evict_count=C - K)), "oldest")
        evaluate(
            compute_keep_indices(C, EvictionSpec(policy="sink_window", num_sink_tokens=4, evict_count=C - K)),
            "sink_window",
        )
        evaluate(
            importance_keep_indices(imp, num_sink_tokens=4, recent_window=K // 2, budget=K),
            "importance",
        )
        print()


if __name__ == "__main__":
    main()
