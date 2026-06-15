"""exp01 — behavioural drift vs. full prompt under prefix eviction (run on node1).

Reproduces the core Phase-1 result on a real model: that naive prefix popping (which
drops attention sinks) blows up KL-vs-full, while sink-aware eviction + RoPE
re-rotation keeps it small. Sweeps how much prefix we drop.

Usage (inside the node1 venv, see NODE1.md):

    python experiments/exp01_rotation_drift.py \
        --model /models/llama-3.2-3b-instruct --context-len 512 --gen 32 \
        --evict 64 128 256

Then point ``--model /models/Trinity-Large-Preview`` to test the hybrid-SWA target.
"""

from __future__ import annotations

import argparse

import torch


def build_context(tok, n_tokens: int, device) -> torch.Tensor:
    """A long, mildly self-referential context so early tokens carry real information.

    Fills up to ``n_tokens`` (the planted facts live in the first ~30 tokens, i.e. the
    sink region we always retain — useful for a later continuity probe)."""
    preamble = (
        "We are running a long agentic session. Remember these facts for later: "
        "the project codename is HALCYON, the budget ceiling is 4096 units, and the "
        "primary contact is Dr. Okonkwo. "
    )
    chunk = (
        "The agent proceeds step by step, taking notes and revisiting earlier decisions "
        "as the context grows. Each turn adds detail while older turns remain relevant. "
    ) * 16
    text = preamble
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < n_tokens:
        text += chunk
        ids = tok(text, return_tensors="pt").input_ids
    return ids[:, :n_tokens].to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-len", type=int, default=512)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--evict", type=int, nargs="+", default=[64, 128, 256])
    args = ap.parse_args()

    from kvrot.config import EvictionSpec
    from kvrot.harness import load_model, run_trial

    lm = load_model(args.model)
    print(f"loaded {args.model}")
    print(
        f"  layers={lm.arch.num_hidden_layers} kv_heads={lm.arch.num_kv_heads} "
        f"head_dim={lm.arch.head_dim} sliding_window={lm.arch.sliding_window} "
        f"full_attn_layers={len(lm.arch.full_attention_layers)}/{lm.arch.num_hidden_layers}"
    )

    ctx = build_context(lm.tokenizer, args.context_len, lm.device)
    print(f"context tokens: {ctx.shape[1]}, generating {args.gen}\n")

    for ev in args.evict:
        variants = {
            f"oldest    (evict {ev}, drops sinks)": EvictionSpec(policy="oldest", evict_count=ev),
            f"sink+rot  (evict {ev}, {args.sinks} sinks)": EvictionSpec(
                policy="sink_window", num_sink_tokens=args.sinks, evict_count=ev, recompact=True
            ),
            f"sink,nogap(evict {ev}, no recompact)": EvictionSpec(
                policy="sink_window", num_sink_tokens=args.sinks, evict_count=ev, recompact=False
            ),
        }
        reports = run_trial(lm, ctx, n_steps=args.gen, variants=variants)
        for rep in reports.values():
            print("  " + rep.one_line())
        print()


if __name__ == "__main__":
    main()
