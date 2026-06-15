"""exp03 — long-horizon multi-eviction rollout (run on node1).

Simulates a rolling session: generate G tokens while a bounded cache repeatedly pops its
oldest (post-sink) tokens. Compares against a full-context reference fed the same tokens,
measuring whether drift *accumulates* over many evictions and whether re-rotation keeps
absolute positions bounded.

Key comparison: recompact (renumber survivors to be contiguous) vs no-recompact (leave a
positional gap). The single-eviction exp01 found them ~equal; here we expect recompact to
keep max-position ≈ budget while no-recompact grows unboundedly with session length — the
real reason re-rotation matters for long-running sessions.

Usage:
    python experiments/exp03_long_horizon.py --model /models/llama-3.2-3b-instruct \
        --context-len 256 --gen 512 --budget 320 --evict 128
"""

from __future__ import annotations

import argparse

import torch


def build_context(tok, n_tokens, device):
    text = "Begin a long rolling session. " + (
        "The agent works step by step, noting observations and revisiting earlier points "
        "as the conversation grows ever longer over many turns. " * 16
    )
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < n_tokens:
        text += "The session continues with further discussion and incremental detail. " * 8
        ids = tok(text, return_tensors="pt").input_ids
    return ids[:, :n_tokens].to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-len", type=int, default=256)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--budget", type=int, default=320)
    ap.add_argument("--evict", type=int, default=128)
    ap.add_argument("--sinks", type=int, default=4)
    args = ap.parse_args()

    from kvrot.harness import _greedy_decode, load_model, rolling_replay
    from kvrot.metrics import stepwise_kl

    lm = load_model(args.model)
    ctx = build_context(lm.tokenizer, args.context_len, lm.device)
    G = args.gen
    print(f"loaded {args.model}; context={ctx.shape[1]}, gen={G}, "
          f"budget={args.budget}, evict_block={args.evict}, sinks={args.sinks}")

    # full-context reference: greedy decode, recording the fed sequence + logits
    ref_cont, ref_logits = _greedy_decode(lm, ctx[:, :-1], ctx[:, -1:], G)
    feed = torch.cat([ctx[:, -1:], ref_cont[:, :-1]], dim=1)  # length G
    print(f"reference: full cache grows {ctx.shape[1]-1} -> {ctx.shape[1]-1+G}\n")

    half = G // 2
    for recompact in (True, False):
        roll_logits, max_pos, n_ev = rolling_replay(
            lm, ctx, feed, budget=args.budget, evict_block=args.evict,
            sinks=args.sinks, recompact=recompact,
        )
        kl = stepwise_kl(ref_logits, roll_logits)
        tag = "recompact " if recompact else "no-recompact"
        print(
            f"  {tag}: mean_KL={kl.mean():.3e}  early(0:{half})={kl[:half].mean():.3e}  "
            f"late({half}:)={kl[half:].mean():.3e}  max_KL={kl.max():.3e}  "
            f"evictions={n_ev}  max_position={max_pos}"
        )

    print("\nnote: full reference keeps all tokens; bounded cache stays ~budget. "
          "max_position shows the unbounded-growth problem recompaction solves.")


if __name__ == "__main__":
    main()
