"""exp08 — trinity recall preservation + clean rotation timing (rotation-only).

The exp02 analog on the real target, and the clean-timing follow-up to exp07. exp07's needle
landed in the *evicted* zone, so it only tested faithful *forgetting*. Here we sweep the needle
across evicted AND retained regions to show the rotation mechanism:
  * PRESERVES recall when the fact is kept (rotation recall ≈ full), and
  * faithfully DROPS it when the fact is evicted (the honest policy cost),
all with NO recompute in the measured path. Rotation (evict + re-rotate, reuse KV) is the object
of study; a shortened-context recompute is only an optional yardstick (--with-recompute), kept off
by default so its (disk-offload-prone) cost never pollutes the headline.

Also: instruments per-GPU memory (so we can size max_memory from the real footprint instead of
guessing) and times the rotation surgery cleanly (GPU-resident tensor ops, not exp07's
disk-offloaded recompute baseline).

Usage (node1, all 8 GPUs — check nvidia-smi first):
    python experiments/exp08_trinity_recall.py --lengths 8192 16384 --depths 0.4 0.8
    # add --with-recompute to also print the shortened-recompute comparison + decomposition
"""

from __future__ import annotations

import argparse
import builtins
import functools
import time

import torch

# Live output: piping python through `tee`/a file block-buffers stdout, so progress prints don't
# appear until the buffer fills (or the run ends). Force every print to flush, and stamp phases with
# elapsed wall-clock so a long run is trackable in real time. (Pair with PYTHONUNBUFFERED=1 on the
# command line as belt-and-suspenders.)
print = functools.partial(builtins.print, flush=True)
_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[t+{time.perf_counter() - _T0:7.1f}s] {msg}")


def gpu_mem_report(tag: str) -> None:
    """Per-GPU allocated / peak / free — so we can see the real footprint and any contention."""
    if not torch.cuda.is_available():
        return
    g = 2**30
    parts = []
    for i in range(torch.cuda.device_count()):
        free_i, total_i = torch.cuda.mem_get_info(i)
        alloc = torch.cuda.memory_allocated(i)
        peak = torch.cuda.max_memory_allocated(i)
        parts.append(f"g{i} alloc{alloc / g:.0f} peak{peak / g:.0f} free{free_i / g:.0f}/{total_i / g:.0f}")
    print(f"  [mem {tag}] " + " | ".join(parts))


def evaluate_cell(lm, source, L: float, depth: float, args) -> dict:
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
    PASS = "73914"  # synthetic — model can't have memorized it (guards against training-data recall)
    needle = tok(f" Note: the vault passcode is {PASS}. ", add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    base = load_text_context(source, tok, max(8, int(L) - needle.shape[1]), device=dev, doc_index=args.doc)
    ctx, n_start, n_end = insert_needle(base, needle, depth)
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
    needle_kept = bool(((keep >= n_start) & (keep < n_end)).any().item())

    print(f"\n== L≈{C} (doc {args.doc}) | keep {K}/{P} | evict {evict_n} after {args.sinks} sinks "
          f"| needle@{depth:.2f} span[{n_start}:{n_end}] KEPT={needle_kept} ==")

    # Reference: the FULL cache generating its own continuation (this is the legitimate
    # exact-vs-full reference, NOT a shortened recompute).
    ref_cont, full_logits = _greedy_decode(lm, prefill_ids, last_tok, args.gen)
    probe = torch.cat([last_tok, ref_cont[:, :-1]], dim=1)

    # Rotation path (the object of study): full prefill -> evict -> re-rotate -> reuse KV.
    torch.cuda.reset_peak_memory_stats()
    s0 = prefill_snapshot(lm, prefill_ids)
    gpu_mem_report(f"after prefill P={P}")
    torch.cuda.synchronize(); t = time.perf_counter()
    snap_rot = reindex(evict_kv(s0, keep), torch.arange(K, device=dev), inv)
    torch.cuda.synchronize(); t_rot = time.perf_counter() - t

    rot_logits = _teacher_forced_logits(lm, snap_rot, probe)
    kl_full_rot = stepwise_kl(full_logits, rot_logits).mean().item()
    top1 = (full_logits.argmax(-1) == rot_logits.argmax(-1)).float().mean().item()

    lp_full, rec_full = answer_logprob(lm, s0, suffix, target)
    lp_rot, rec_rot = answer_logprob(lm, snap_rot, suffix, target)

    print(f"  KL(full ‖ rotation) over {args.gen} toks = {kl_full_rot:.3e}   top1 {top1 * 100:.0f}%")
    print(f"  recall full      logprob={lp_full:8.3f}  {'YES' if rec_full else 'no'}")
    verdict = ("preserved ✓" if (needle_kept and rec_rot) else
               "LOST (kept!) ✗" if needle_kept else
               "dropped (faithful)" if not rec_rot else "?")
    print(f"  recall rotation  logprob={lp_rot:8.3f}  {'YES' if rec_rot else 'no'}   <-- {verdict}")
    print(f"  rotation surgery: {t_rot * 1e3:7.1f} ms (0 tokens recomputed, GPU-resident)")

    row = dict(L=C, depth=depth, kept=needle_kept, K=K, P=P, kl=kl_full_rot, top1=top1,
               lp_full=lp_full, lp_rot=lp_rot, rec_full=rec_full, rec_rot=rec_rot, t_rot=t_rot)

    if args.with_recompute:  # optional yardstick only — not the thing under study
        from kvrot.snapshot import reindex as _reindex
        shortened = prefill_ids[:, keep]
        torch.cuda.synchronize(); t = time.perf_counter()
        snap_short = prefill_snapshot(lm, shortened)
        torch.cuda.synchronize(); t_short = time.perf_counter() - t
        short_logits = _teacher_forced_logits(lm, snap_short, probe)
        lp_short, rec_short = answer_logprob(lm, snap_short, suffix, target)
        kl_fs = stepwise_kl(full_logits, short_logits).mean().item()
        kl_sr = stepwise_kl(short_logits, rot_logits).mean().item()
        print(f"  [recompute yardstick] info-loss KL(full‖short)={kl_fs:.3e}  "
              f"MECH KL(short‖rot)={kl_sr:.3e}  recall short lp={lp_short:.3f} "
              f"{'YES' if rec_short else 'no'}")
        print(f"  [recompute yardstick] cost: recompute {t_short * 1e3:8.1f} ms ({K} toks) "
              f"= {t_short / max(t_rot, 1e-9):.0f}x rotation")
        row.update(kl_fs=kl_fs, kl_sr=kl_sr, lp_short=lp_short, t_short=t_short)

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Trinity-Large-Preview")
    ap.add_argument("--source", default="data/eval_docs.jsonl")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.4, 0.8],
                    help="needle depths; <evict-frac lands in the evicted zone, >evict-frac is retained")
    ap.add_argument("--doc", type=int, default=0)
    ap.add_argument("--evict-frac", type=float, default=0.6)
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--mem-frac", type=float, default=0.92,
                    help="fraction of each GPU's FREE memory to budget for weights")
    ap.add_argument("--with-recompute", action="store_true",
                    help="also run the shortened-context recompute as a comparison yardstick")
    args = ap.parse_args()

    from kvrot.harness import load_model

    log(f"loading {args.model} (reads the full model from disk — several minutes for trinity; "
        "watch the 'Loading checkpoint shards' bar)...")
    lm = load_model(args.model, device_map="auto", max_memory_frac=args.mem_frac)
    log("model loaded")
    a = lm.arch
    print(f"loaded {args.model}: {a.num_hidden_layers}L ({len(a.sliding_layers)} sliding "
          f"W={a.sliding_window} + {len(a.full_attention_layers)} full), "
          f"applies_rope={sum(a.applies_rope)}/{a.num_hidden_layers}")
    gpu_mem_report("after load")

    rows = []
    for L in args.lengths:
        for depth in args.depths:
            log(f"cell L={L} depth={depth} starting")
            try:
                rows.append(evaluate_cell(lm, args.source, L, depth, args))
            except Exception as e:  # keep going across cells even if one OOMs
                print(f"  L={L} depth={depth} FAILED: {type(e).__name__}: {e}")

    # Headline summary: did rotation preserve recall for KEPT facts and drop it for EVICTED ones?
    print("\n==== summary ====")
    for r in rows:
        tag = "kept" if r["kept"] else "evicted"
        ok = (r["rec_rot"] if r["kept"] else not r["rec_rot"])
        print(f"  L≈{r['L']:>6} depth {r['depth']:.2f} [{tag:>7}]: "
              f"recall full {'Y' if r['rec_full'] else 'n'} / rot {'Y' if r['rec_rot'] else 'n'} "
              f"(lp {r['lp_rot']:7.2f})  KL {r['kl']:.2e}  top1 {r['top1'] * 100:3.0f}%  "
              f"rot {r['t_rot'] * 1e3:.0f}ms  {'✓' if ok else '✗ UNEXPECTED'}")


if __name__ == "__main__":
    main()
