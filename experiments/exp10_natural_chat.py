"""exp10 (eval) — exp09's turn-aligned eviction+rotation eval on NATURALIZED dialogue.

Identical protocol and decomposition to exp09 (see notes/design-chat-eval.md §4 and
notes/design-natural-chat.md §4), with the conversations loaded from the jsonl that
exp10_natural_gen.py produced instead of synthesized from templates:

  * per conversation x per length L: cut at the smallest user-terminated turn
    boundary whose trinity render >= L (exp09's overshoot-and-report behaviour);
  * plant the SAME two passcode facts with the SAME strings, placement rule and
    guards as exp09 (locker -> first user turn, EVICTED; vault -> second-to-last
    user turn of the cut, RETAINED), spliced post-generation so neither generator
    ever saw a passcode;
  * whole-turn oldest-first eviction (protect system + last 2 turns, 4 sinks),
    Tier-0 re-rotation with recompaction; KL(full||rot) + top1 on the full cache's
    own greedy 32-token next-turn reply; --with-recompute adds the info-loss /
    mechanism-error decomposition; recall probes for both facts.

Additive instrumentation over exp09 (caveat #1 there): the per-step KL series and
the index of the first eos in the reference reply are stored per row, so drift can
be split at the degeneracy point post hoc.

Usage (node1, GPUs free — per exp08b recipe):
    python experiments/exp10_natural_chat.py --convs data/natural_convs_2026-07-10.jsonl \
        --lengths 8192 16384 --with-recompute --device-map balanced --mem-frac 0.95 \
        --out runs/exp10_ef50.json
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

print = functools.partial(builtins.print, flush=True)
_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[t+{time.perf_counter() - _T0:7.1f}s] {msg}")


# Byte-identical to exp09 (cross-experiment comparability).
PASS_EVICTED = "62483"
PASS_RETAINED = "73914"
FACT_EVICTED = f"By the way, my locker code is {PASS_EVICTED}."
FACT_RETAINED = f"Also, note this down: the vault passcode is {PASS_RETAINED}."


def cut_and_plant(tokenizer, conv, L: int):
    """Cut the conversation at a turn boundary >= L and plant both facts.

    Returns (messages, ctx_ids [1,T], spans, evicted_fact_turn, retained_fact_turn,
    rendered_tokens). Mirrors exp09.build_conversation's contract.
    """
    from kvrot.chat import plant_fact, turn_token_spans
    from kvrot.natural import choose_cut_prefix, find_passcode_leaks

    full_messages = conv.messages()
    n_msgs, rendered = choose_cut_prefix(tokenizer, full_messages, L, end_role="user")
    messages = [dict(m) for m in full_messages[:n_msgs]]  # deep-enough copy (str contents)

    user_turns = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_turns) < 4:
        raise RuntimeError(f"cut has only {len(user_turns)} user turns — too short to plant facts")
    evicted_turn, retained_turn = user_turns[0], user_turns[-2]
    plant_fact(messages, evicted_turn, FACT_EVICTED)
    plant_fact(messages, retained_turn, FACT_RETAINED)

    for code, planted in ((PASS_EVICTED, evicted_turn), (PASS_RETAINED, retained_turn)):
        leaks = find_passcode_leaks(messages, code, planted)
        if leaks:
            raise RuntimeError(f"passcode {code} leaked into messages {leaks}")

    ctx, spans = turn_token_spans(tokenizer, messages, add_generation_prompt=True)
    return messages, ctx, spans, evicted_turn, retained_turn, rendered


def evaluate_cell(lm, conv, L: int, args) -> dict:
    """One (conversation, length) cell — exp09's evaluate_cell on natural data."""
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
    messages, ctx, spans, ev_turn, ret_turn, rendered = cut_and_plant(tok, conv, L)
    ctx = ctx.to(dev)
    C = ctx.shape[1]
    prefill_ids, last_tok = ctx[:, :-1], ctx[:, -1:]
    P = prefill_ids.shape[1]
    n_user_turns = sum(1 for m in messages if m["role"] == "user")
    non_system = [sp for sp in spans if sp.role != "system"]
    toks_per_turn = (sum(sp.n_tokens for sp in non_system) / max(len(non_system), 1))

    evict_turns = oldest_turns_to_evict(
        spans, target_tokens=int(args.evict_frac * P), protect_last=2
    )
    if ev_turn not in evict_turns or ret_turn in evict_turns:
        raise RuntimeError(
            f"fact placement vs eviction mismatch (evicted={evict_turns}, "
            f"facts at {ev_turn}/{ret_turn})"
        )
    keep = turn_keep_indices(spans, evict_turns, P, num_sink_tokens=args.sinks).to(dev)
    K = int(keep.shape[0])
    n_evicted_turns = len(evict_turns)

    print(f"\n== {conv.conv_id} @ L={L}: C≈{C} ({n_user_turns} user turns, "
          f"~{toks_per_turn:.0f} toks/turn) | evict {n_evicted_turns} turns = {P - K} toks | "
          f"keep {K}/{P} ({K / P:.2f}) | locker@t{ev_turn} EVICTED, vault@t{ret_turn} RETAINED ==")

    # Reference: the FULL cache writing its own greedy next-turn reply.
    ref_cont, full_logits = _greedy_decode(lm, prefill_ids, last_tok, args.gen)
    probe = torch.cat([last_tok, ref_cont[:, :-1]], dim=1)
    print(f"  full-cache reply opens: {tok.decode(ref_cont[0, : min(24, ref_cont.shape[1])])!r}")
    eos_ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for t in ("<|im_end|>",):
        try:
            i = tok.convert_tokens_to_ids(t)
            if i is not None and i >= 0 and i != getattr(tok, "unk_token_id", None):
                eos_ids.add(int(i))
        except Exception:
            pass
    first_eos = next((j for j, t_ in enumerate(ref_cont[0].tolist()) if t_ in eos_ids), -1)

    # Rotation path: full prefill -> whole-turn evict -> re-rotate (recompact) -> reuse.
    torch.cuda.reset_peak_memory_stats()
    s0 = prefill_snapshot(lm, prefill_ids)
    torch.cuda.synchronize(); t = time.perf_counter()
    snap_rot = reindex(evict_kv(s0, keep), torch.arange(K, device=dev), inv)
    torch.cuda.synchronize(); t_rot = time.perf_counter() - t

    rot_logits = _teacher_forced_logits(lm, snap_rot, probe)
    kl_steps = stepwise_kl(full_logits, rot_logits)
    kl_full_rot = kl_steps.mean().item()
    top1 = (full_logits.argmax(-1) == rot_logits.argmax(-1)).float().mean().item()

    row: dict[str, Any] = dict(
        conv_id=conv.conv_id, L_target=L, L=C, P=P, K=K,
        keep_frac=K / P, n_user_turns=n_user_turns, toks_per_turn=toks_per_turn,
        n_evicted_turns=n_evicted_turns, kl=kl_full_rot, top1=top1, t_rot=t_rot,
        first_eos=first_eos, kl_steps=[round(float(x), 6) for x in kl_steps.tolist()],
    )
    print(f"  KL(full ‖ rotation) over {args.gen} reply toks = {kl_full_rot:.3e}   "
          f"top1 {top1 * 100:.0f}%   (first eos in ref reply @ {first_eos})")

    # Recall probes — exp09's exact plain-text suffixes.
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
                    f"rec_full_{key}": rec_full, f"rec_rot_{key}": rec_rot, f"ok_{key}": ok})
    print(f"  rotation surgery: {t_rot * 1e3:7.1f} ms ({n_evicted_turns} whole turns)")

    if args.with_recompute:  # the decomposition yardstick
        shortened = prefill_ids[:, keep]
        torch.cuda.synchronize(); t = time.perf_counter()
        snap_short = prefill_snapshot(lm, shortened)
        torch.cuda.synchronize(); t_short = time.perf_counter() - t
        short_logits = _teacher_forced_logits(lm, snap_short, probe)
        kl_fs_steps = stepwise_kl(full_logits, short_logits)
        kl_sr_steps = stepwise_kl(short_logits, rot_logits)
        kl_fs, kl_sr = kl_fs_steps.mean().item(), kl_sr_steps.mean().item()
        print(f"  [recompute yardstick] info-loss KL(full‖short)={kl_fs:.3e}  "
              f"MECH KL(short‖rot)={kl_sr:.3e}  "
              f"cost {t_short * 1e3:8.1f} ms = {t_short / max(t_rot, 1e-9):.0f}x rotation")
        row.update(kl_fs=kl_fs, kl_sr=kl_sr, t_short=t_short,
                   kl_fs_steps=[round(float(x), 6) for x in kl_fs_steps.tolist()],
                   kl_sr_steps=[round(float(x), 6) for x in kl_sr_steps.tolist()])
    return row


def aggregate(rows: list[dict], key: str) -> dict[str, float] | None:
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return None
    return {"mean": statistics.fmean(vals),
            "median": statistics.median(vals), "n": len(vals)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Trinity-Large-Preview")
    ap.add_argument("--convs", type=Path, default=Path("data/natural_convs_2026-07-10.jsonl"))
    ap.add_argument("--conv-ids", nargs="*", default=None,
                    help="subset of conv_ids to evaluate (default: all)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--evict-frac", type=float, default=0.5)
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--mem-frac", type=float, default=0.95,
                    help="fraction of each GPU's FREE memory for weights; lower values "
                         "let accelerate spill to disk (hard error via assert_no_offload)")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--with-recompute", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from kvrot.harness import load_model
    from kvrot.natural import load_convs_any

    convs = load_convs_any(args.convs)  # stage-1 corpus or stage-2 generated schema
    if args.conv_ids:
        convs = [c for c in convs if c.conv_id in set(args.conv_ids)]
    if not convs:
        raise SystemExit(f"no conversations selected from {args.convs}")
    log(f"loaded {len(convs)} conversations from {args.convs}: "
        f"{[c.conv_id for c in convs]}")

    log(f"loading {args.model} (several minutes for trinity — watch the shards bar)...")
    device_map: Any = args.device_map
    if device_map == "even":  # explicit per-layer map; accelerate's balancer misplaces afmoe
        from kvrot.natural import trinity_even_device_map
        device_map = trinity_even_device_map()
        log("using explicit even device map (accelerate balanced misplaces afmoe)")
    lm = load_model(args.model, device_map=device_map, max_memory_frac=args.mem_frac)
    a = lm.arch
    log(f"model loaded: {a.num_hidden_layers}L, applies_rope="
        f"{sum(a.applies_rope)}/{a.num_hidden_layers}")
    from kvrot.natural import assert_no_offload
    assert_no_offload(lm.model)  # disk/cpu offload invalidates timings; fail loudly

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for conv in convs:
        for L in args.lengths:
            try:
                rows.append(evaluate_cell(lm, conv, L, args))
            except Exception as e:  # keep going across cells
                print(f"  {conv.conv_id} L={L} FAILED: {type(e).__name__}: {e}")
                failures.append({"conv_id": conv.conv_id, "L": L,
                                 "error": f"{type(e).__name__}: {e}"})

    summary_lines: list[str] = []
    print("\n==== summary (naturalized dialogue, turn-aligned) ====")
    for r in rows:
        ok = r.get("ok_ret") and r.get("ok_ev")
        line = (f"{r['conv_id']} L≈{r['L']:>6} ({r['n_evicted_turns']} turns evicted, "
                f"keep {r['K']}/{r['P']}): KL {r['kl']:.2e}  top1 {r['top1'] * 100:3.0f}%  "
                + (f"mech {r['kl_sr']:.2e}  info {r['kl_fs']:.2e}  " if "kl_sr" in r else "")
                + f"retained {'Y' if r.get('rec_rot_ret') else 'n'} / "
                f"evicted {'leak!' if r.get('rec_rot_ev') else 'dropped'}  "
                f"{'✓' if ok else '✗ UNEXPECTED'}")
        print("  " + line)
        summary_lines.append(line)

    agg: dict[str, Any] = {}
    print("\n==== per-length aggregates (across conversations) ====")
    for L in args.lengths:
        lr = [r for r in rows if r["L_target"] == L]
        cell = {k: aggregate(lr, k) for k in ("kl", "top1", "kl_fs", "kl_sr",
                                              "keep_frac", "toks_per_turn")}
        agg[str(L)] = cell
        if lr:
            def fm(k: str) -> str:
                c = cell[k]
                return f"{c['mean']:.3e}/{c['median']:.3e}" if c else "-"
            print(f"  L={L} (n={len(lr)}): KL mean/med {fm('kl')}  "
                  f"mech {fm('kl_sr')}  info {fm('kl_fs')}  "
                  f"top1 {cell['top1']['mean'] * 100:.0f}%  "
                  f"toks/turn {cell['toks_per_turn']['mean']:.0f}")

    if args.out is not None:
        payload: dict[str, Any] = {
            "experiment": "exp10_natural_chat",
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
            "conv_meta": [{"conv_id": c.conv_id,
                           "schema": type(c).__name__,
                           "topic": getattr(c, "topic", None),
                           "condition": getattr(c, "condition", None),
                           "bot_name": getattr(c, "bot_name", None),
                           "source_path": getattr(c, "source_path", None),
                           "approx_tokens": getattr(c, "approx_tokens", None),
                           "generated_pct": getattr(c, "generated_pct", None),
                           "trinity_model": getattr(c, "trinity_model", None),
                           "deepseek_model": getattr(c, "deepseek_model", None),
                           "render_tokens": getattr(c, "trinity_render_tokens", None),
                           "n_messages": len(c.messages())}
                          for c in convs],
            "rows": rows, "aggregates": agg, "failures": failures,
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
