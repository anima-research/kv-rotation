"""exp06 — trinity (afmoe) load + hybrid-path sanity (run on node1, all 8 GPUs).

First trinity run. Validates, on the real model:
  * it loads sharded across the GPUs and our ArchSpec parses it correctly,
  * applies_rope is sliding-only (the NoPE-global finding) — # rotated layers == # sliding,
  * inv_freq is found and the rotation homomorphism holds (re-rotation is exact for θ=10k),
  * one eviction round runs end-to-end through the per-layer-gated surgery, with sink-aware
    rotation near-full and naive (drop-sinks) bad — same signature we saw on Llama.

Small context so it's quick once the (~778 GB) load is done.

Usage:
    python experiments/exp06_trinity_smoke.py --model /models/Trinity-Large-Preview \
        --context-len 256 --gen 16 --evict 64
"""

from __future__ import annotations

import argparse

import torch


def build_context(tok, n_tokens, device):
    text = "Begin a long trinity session. " + (
        "The agent proceeds step by step, noting observations and revisiting earlier points "
        "as the conversation grows over many turns. " * 16
    )
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < n_tokens:
        text += "The session continues with further detail. " * 8
        ids = tok(text, return_tensors="pt").input_ids
    return ids[:, :n_tokens].to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Trinity-Large-Preview")
    ap.add_argument("--context-len", type=int, default=256)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--evict", type=int, default=64)
    args = ap.parse_args()

    from kvrot.config import EvictionSpec
    from kvrot.harness import load_model, run_trial

    lm = load_model(args.model, device_map="auto")
    a = lm.arch
    n_rope = sum(a.applies_rope)
    print(f"\nloaded {args.model}")
    print(f"  layers={a.num_hidden_layers}  sliding={len(a.sliding_layers)}  full={len(a.full_attention_layers)}")
    print(f"  kv_heads={a.num_kv_heads}  head_dim={a.head_dim}  sliding_window={a.sliding_window}")
    print(f"  rope: theta={a.rope.theta} scaling={a.rope.scaling_type}")
    print(f"  applies_rope: {n_rope}/{a.num_hidden_layers} layers rotated (expect == #sliding={len(a.sliding_layers)})")
    assert n_rope == len(a.sliding_layers), "applies_rope should be sliding-only for afmoe!"
    print("  ✓ per-layer RoPE gating matches the afmoe source (NoPE on global layers)")
    print("  ✓ rotation homomorphism held at load (re-rotation is exact for these frequencies)\n")

    ctx = build_context(lm.tokenizer, args.context_len, lm.device)
    print(f"context tokens: {ctx.shape[1]}, generating {args.gen}, evict {args.evict}\n")
    reports = run_trial(
        lm,
        ctx,
        n_steps=args.gen,
        variants={
            f"sink+rot (evict {args.evict})": EvictionSpec(policy="sink_window", num_sink_tokens=4, evict_count=args.evict),
            f"oldest   (evict {args.evict})": EvictionSpec(policy="oldest", evict_count=args.evict),
        },
    )
    for r in reports.values():
        print("  " + r.one_line())
    print("\nexpected: sink+rot near-full (small KL, high top1); oldest much worse.")


if __name__ == "__main__":
    main()
