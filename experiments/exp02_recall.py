"""exp02 — recall/continuity probe (run on node1).

KL-vs-full (exp01) shows the *continuation* matches; it does not show whether information
survives eviction. This probe plants a fact ("the vault passcode is 48127") at a
controlled position and measures the model's log-prob (and greedy recall) of the answer
under full vs. rotated cache.

Two questions:
  Probe 1 (retained fact): keep the fact in the sink region, evict an *unrelated* later
    span. Does recall survive? (sink-aware should ≈ full; naive — which drops the sinks
    and the fact — should fail.) → "eviction of unrelated content doesn't damage kept info"
  Probe 2 (evicted fact): evict the span *containing* the fact. Recall should be lost —
    quantifying the cost of dropping still-live info (the feasibility law). → "evict by
    importance, not age."

Usage:
    python experiments/exp02_recall.py --model /models/llama-3.2-3b-instruct --filler 600
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--filler", type=int, default=600, help="approx filler tokens after the fact")
    args = ap.parse_args()

    from kvrot.harness import answer_logprob, load_model, prefill_snapshot
    from kvrot.rope import default_inv_freq  # noqa: F401  (kept for reference)
    from kvrot.snapshot import evict, reindex

    lm = load_model(args.model)
    tok = lm.tokenizer
    dev = lm.device

    PASSCODE = "48127"
    preamble = "We begin a long session. Note this important detail. "
    fact = f"The vault passcode is {PASSCODE}. "
    filler_unit = (
        "The team continues working through the task, recording observations and "
        "revisiting earlier notes as the discussion proceeds. "
    )

    def enc(s: str, bos: bool) -> torch.Tensor:
        return tok(s, return_tensors="pt", add_special_tokens=bos).input_ids.to(dev)

    # Build context piecewise so we know the fact's token span exactly.
    pre_ids = enc(preamble, bos=True)
    fact_ids = enc(fact, bos=False)
    fact_start = pre_ids.shape[1]
    fact_end = fact_start + fact_ids.shape[1]

    filler_ids = enc(filler_unit, bos=False)
    reps = max(1, args.filler // filler_ids.shape[1] + 1)
    filler_all = filler_ids.repeat(1, reps)

    context_ids = torch.cat([pre_ids, fact_ids, filler_all], dim=1)
    C = context_ids.shape[1]

    suffix = enc("\nQuestion: What is the vault passcode?\nAnswer: The vault passcode is", bos=False)
    target = enc(f" {PASSCODE}", bos=False)

    print(f"loaded {args.model}; context={C} tokens; fact span=[{fact_start},{fact_end}); "
          f"target tokens={target.tolist()}")

    s0 = prefill_snapshot(lm, context_ids)
    inv = lm.inv_freq

    def score(keep_indices: torch.Tensor | None, label: str) -> None:
        if keep_indices is None:
            snap, kept = s0, C
        else:
            keep_indices = keep_indices.to(dev)
            new_pos = torch.arange(keep_indices.shape[0], device=dev)  # recompact
            snap = reindex(evict(s0.clone(), keep_indices), new_pos, inv)
            kept = keep_indices.shape[0]
        lp, greedy = answer_logprob(lm, snap, suffix, target)
        print(f"  {label:<46} kept={kept:>4}/{C}  logprob(answer)={lp:8.3f}  recall={'YES' if greedy else 'no '}")

    ar = torch.arange(C)

    print("\n-- baseline --")
    score(None, "full context (reference)")

    print("\n-- Probe 1: fact RETAINED, evict unrelated later span --")
    ev_start = fact_end + 8
    ev_len = max(64, (C - ev_start) // 2)
    keep_sink = torch.cat([ar[:ev_start], ar[ev_start + ev_len:]])
    score(keep_sink, f"sink-aware (keep fact, drop [{ev_start},{ev_start+ev_len}))")
    # naive 'oldest': drop the leading block, which includes the fact AND the sinks
    keep_naive = ar[ev_len + 8:]
    score(keep_naive, f"naive oldest (drops first {ev_len+8}, incl. fact+sinks)")

    print("\n-- Probe 2: fact EVICTED (dropped span covers it) --")
    keep_drop_fact = torch.cat([ar[:4], ar[fact_end + 8:]])  # keep 4 sinks, drop 4..fact_end+8
    score(keep_drop_fact, "sink-aware but fact is in the dropped span")


if __name__ == "__main__":
    main()
