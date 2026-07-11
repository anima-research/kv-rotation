"""exp09 — chat-shaped eval: turn-aligned eviction on multi-turn conversations (STAGED).

The realism frontier (HANDOFF §9.1a). Every prior experiment evicted token ranges from
raw documents; deployment is a rolling *chat* where whole early turns are popped. This
experiment runs the drift/recall/decomposition protocol on chat-template-rendered
multi-turn conversations with TURN-ALIGNED eviction:

  * conversations are synthesized deterministically from the real eval corpus
    (kvrot.chat.synthesize_conversations — real prose in the KV, templated scaffolding);
  * spans per turn come from incremental chat-template rendering with a
    prefix-stability assert (kvrot.chat.turn_token_spans);
  * eviction takes whole early turns (system prompt + last 2 turns protected),
    then Tier-0 re-rotation with recompaction — the deployment recipe;
  * two synthetic passcode facts are planted: one in a turn that will be EVICTED,
    one in a turn that stays RETAINED. Rotation must drop the first and keep the
    second, while the teacher-forced next-turn reply stays near the full cache
    (KL + top1), with the optional shortened-recompute yardstick for the
    info-loss/mechanism-error decomposition.

STATUS: staged 2026-07-03 (GPUs busy) — logic is CPU-tested via tests/test_chat.py;
this script has NOT yet run against trinity. Design + predictions:
notes/design-chat-eval.md.

Usage (node1, all 8 GPUs — check nvidia-smi first; sync per HANDOFF §4):
    python experiments/exp09_trinity_chat.py --lengths 8192 16384 --turns 12
    # add --with-recompute for the decomposition yardstick (GPU-resident, use
    #   --device-map balanced --mem-frac 0.70 per exp08b to avoid disk offload)
    # smoke test on the 3B first (1 GPU):
    python experiments/exp09_trinity_chat.py --model /models/llama-3.2-3b-instruct \
        --lengths 2048 --turns 8 --gen 16
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import time
from pathlib import Path
from typing import Any

import torch

# Live output on long runs (see exp08 for the war story): flush every print and
# stamp phases with wall-clock. Pair with PYTHONUNBUFFERED=1 on the command line.
print = functools.partial(builtins.print, flush=True)
_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[t+{time.perf_counter() - _T0:7.1f}s] {msg}")


def gpu_mem_report(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    g = 2**30
    parts = []
    for i in range(torch.cuda.device_count()):
        free_i, total_i = torch.cuda.mem_get_info(i)
        alloc = torch.cuda.memory_allocated(i)
        parts.append(f"g{i} alloc{alloc / g:.0f} free{free_i / g:.0f}/{total_i / g:.0f}")
    print(f"  [mem {tag}] " + " | ".join(parts))


PASS_EVICTED = "62483"   # synthetic — can't be training-data recall
PASS_RETAINED = "73914"  # exp08's passcode, reused for cross-experiment comparability


def build_conversation(tokenizer, docs: list[str], L: int, n_turns: int, doc_index: int):
    """Deterministic chat at ≈L tokens with both facts planted. Returns
    (messages, ctx_ids [1,T], spans, evicted_fact_turn, retained_fact_turn)."""
    from kvrot.chat import plant_fact, synthesize_conversations, turn_token_spans

    # ~4 chars/token heuristic; report the achieved length, don't chase it.
    chars = max(200, int(4 * L / max(n_turns, 1)))
    messages = synthesize_conversations(
        docs, n_user_turns=n_turns, chars_per_chunk=chars, doc_index=doc_index
    )
    # Facts: turn 1 = first user turn (oldest evictable — WILL be evicted);
    # the second-to-last user turn is protected by protect_last=2 (RETAINED).
    user_turns = [i for i, m in enumerate(messages) if m["role"] == "user"]
    evicted_turn, retained_turn = user_turns[0], user_turns[-2]
    plant_fact(messages, evicted_turn, f"By the way, my locker code is {PASS_EVICTED}.")
    plant_fact(messages, retained_turn, f"Also, note this down: the vault passcode is {PASS_RETAINED}.")

    ctx, spans = turn_token_spans(tokenizer, messages, add_generation_prompt=True)
    return messages, ctx, spans, evicted_turn, retained_turn


def evaluate_cell(lm, docs: list[str], L: int, args) -> dict:
    from kvrot.chat import oldest_turns_to_evict, turn_keep_indices
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
    messages, ctx, spans, ev_turn, ret_turn = build_conversation(
        tok, docs, L, args.turns, args.doc
    )
    ctx = ctx.to(dev)
    C = ctx.shape[1]
    prefill_ids, last_tok = ctx[:, :-1], ctx[:, -1:]
    P = prefill_ids.shape[1]

    # Turn-aligned eviction: whole early turns until ~evict_frac of the prefill is freed.
    evict_turns = oldest_turns_to_evict(
        spans, target_tokens=int(args.evict_frac * P), protect_last=2
    )
    if ev_turn not in evict_turns or ret_turn in evict_turns:
        raise RuntimeError(
            f"fact placement vs eviction mismatch (evicted={evict_turns}, "
            f"facts at {ev_turn}/{ret_turn}) — adjust --turns/--evict-frac"
        )
    keep = turn_keep_indices(spans, evict_turns, P, num_sink_tokens=args.sinks).to(dev)
    K = int(keep.shape[0])
    n_evicted_turns = len(evict_turns)

    print(f"\n== chat L≈{C} ({args.turns} user turns, doc {args.doc}) | "
          f"evict {n_evicted_turns} turns = {P - K} toks | keep {K}/{P} | "
          f"locker@turn{ev_turn} EVICTED, vault@turn{ret_turn} RETAINED ==")

    # Reference: the FULL cache writing its own next-turn reply.
    ref_cont, full_logits = _greedy_decode(lm, prefill_ids, last_tok, args.gen)
    probe = torch.cat([last_tok, ref_cont[:, :-1]], dim=1)
    reply_preview = tok.decode(ref_cont[0, : min(24, ref_cont.shape[1])])
    print(f"  full-cache reply opens: {reply_preview!r}")

    # Rotation path: full prefill -> whole-turn evict -> re-rotate (recompact) -> reuse.
    torch.cuda.reset_peak_memory_stats()
    s0 = prefill_snapshot(lm, prefill_ids)
    gpu_mem_report(f"after prefill P={P}")
    torch.cuda.synchronize(); t = time.perf_counter()
    snap_rot = reindex(evict_kv(s0, keep), torch.arange(K, device=dev), inv)
    torch.cuda.synchronize(); t_rot = time.perf_counter() - t

    rot_logits = _teacher_forced_logits(lm, snap_rot, probe)
    kl_full_rot = stepwise_kl(full_logits, rot_logits).mean().item()
    top1 = (full_logits.argmax(-1) == rot_logits.argmax(-1)).float().mean().item()

    # Recall probes (plain-text suffix, exp08-comparable — see design note §5 for
    # the fully-templated-probe variant if this reads noisy on the instruct model).
    probes = {
        "vault (RETAINED)": (
            "\n\nQuestion: What is the vault passcode?\nAnswer: The vault passcode is",
            f" {PASS_RETAINED}", True,
        ),
        "locker (EVICTED)": (
            "\n\nQuestion: What is the locker code?\nAnswer: The locker code is",
            f" {PASS_EVICTED}", False,
        ),
    }
    row = dict(L=C, P=P, K=K, n_evicted_turns=n_evicted_turns, kl=kl_full_rot,
               top1=top1, t_rot=t_rot)
    print(f"  KL(full ‖ rotation) over {args.gen} reply toks = {kl_full_rot:.3e}   "
          f"top1 {top1 * 100:.0f}%")
    for name, (q, target_text, want_kept) in probes.items():
        suffix = tok(q, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        target = tok(target_text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        lp_full, rec_full = answer_logprob(lm, s0, suffix, target)
        lp_rot, rec_rot = answer_logprob(lm, snap_rot, suffix, target)
        ok = rec_rot if want_kept else not rec_rot
        verdict = ("preserved ✓" if want_kept and rec_rot else
                   "LOST (kept!) ✗" if want_kept else
                   "dropped (faithful) ✓" if not rec_rot else "LEAKED (evicted!) ✗")
        print(f"  recall {name}: full lp={lp_full:8.3f} {'Y' if rec_full else 'n'} | "
              f"rotation lp={lp_rot:8.3f} {'Y' if rec_rot else 'n'}  <-- {verdict}")
        key = "ret" if want_kept else "ev"
        row.update({f"lp_full_{key}": lp_full, f"lp_rot_{key}": lp_rot,
                    f"rec_full_{key}": rec_full, f"rec_rot_{key}": rec_rot,
                    f"ok_{key}": ok})
    print(f"  rotation surgery: {t_rot * 1e3:7.1f} ms ({n_evicted_turns} whole turns, "
          f"0 tokens recomputed)")

    if args.with_recompute:  # optional yardstick — the decomposition, not the object of study
        shortened = prefill_ids[:, keep]
        torch.cuda.synchronize(); t = time.perf_counter()
        snap_short = prefill_snapshot(lm, shortened)
        torch.cuda.synchronize(); t_short = time.perf_counter() - t
        short_logits = _teacher_forced_logits(lm, snap_short, probe)
        kl_fs = stepwise_kl(full_logits, short_logits).mean().item()
        kl_sr = stepwise_kl(short_logits, rot_logits).mean().item()
        print(f"  [recompute yardstick] info-loss KL(full‖short)={kl_fs:.3e}  "
              f"MECH KL(short‖rot)={kl_sr:.3e}  "
              f"cost {t_short * 1e3:8.1f} ms = {t_short / max(t_rot, 1e-9):.0f}x rotation")
        row.update(kl_fs=kl_fs, kl_sr=kl_sr, t_short=t_short)

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Trinity-Large-Preview")
    ap.add_argument("--source", default="data/eval_docs.jsonl")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--turns", type=int, default=12, help="user turns per conversation")
    ap.add_argument("--doc", type=int, default=0)
    ap.add_argument("--evict-frac", type=float, default=0.5,
                    help="fraction of prefill tokens to free via whole-turn eviction")
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--gen", type=int, default=32,
                    help="teacher-forced reply tokens (longer than exp08: replies, not passcodes)")
    ap.add_argument("--mem-frac", type=float, default=0.92)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--with-recompute", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="write results (rows + config + summary) to this JSON file")
    args = ap.parse_args()

    from kvrot.data import read_texts
    from kvrot.harness import load_model

    docs = read_texts(args.source)
    log(f"loaded {len(docs)} eval docs from {args.source}")
    log(f"loading {args.model} (several minutes for trinity — watch the shards bar)...")
    lm = load_model(args.model, device_map=args.device_map, max_memory_frac=args.mem_frac)
    a = lm.arch
    log(f"model loaded: {a.num_hidden_layers}L, applies_rope="
        f"{sum(a.applies_rope)}/{a.num_hidden_layers}")
    gpu_mem_report("after load")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for L in args.lengths:
        log(f"cell L={L} starting")
        try:
            rows.append(evaluate_cell(lm, docs, L, args))
        except Exception as e:  # keep going across cells
            print(f"  L={L} FAILED: {type(e).__name__}: {e}")
            failures.append({"L": L, "error": f"{type(e).__name__}: {e}"})

    summary_lines: list[str] = []
    print("\n==== summary (chat-shaped, turn-aligned) ====")
    for r in rows:
        ok = r.get("ok_ret") and r.get("ok_ev")
        line = (f"L≈{r['L']:>6} ({r['n_evicted_turns']} turns evicted, keep {r['K']}/{r['P']}): "
                f"KL {r['kl']:.2e}  top1 {r['top1'] * 100:3.0f}%  "
                f"retained {'Y' if r.get('rec_rot_ret') else 'n'} / "
                f"evicted {'leak!' if r.get('rec_rot_ev') else 'dropped'}  "
                f"rot {r['t_rot'] * 1e3:.0f}ms  {'✓' if ok else '✗ UNEXPECTED'}")
        print("  " + line)
        summary_lines.append(line)

    if args.out is not None:
        payload: dict[str, Any] = {
            "experiment": "exp09_trinity_chat",
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
            "rows": rows,
            "failures": failures,
            "summary": summary_lines,
        }
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
            log(f"results written to {args.out}")
        except OSError as e:  # never lose a finished run to a bad path
            print(f"  WARNING: could not write --out {args.out}: {e}")
            print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
