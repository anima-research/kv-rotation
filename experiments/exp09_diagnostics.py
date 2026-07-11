"""exp09 diagnostics — D1 KV-geometry, D2 2x2 corners, D3 per-step KL, D5 turns=48.

Why: exp09 found mech-KL(short‖rot) 0.2–1.3 under chat-shaped turn-aligned eviction vs
<=5e-4 on raw docs (exp07/08b) at matched L/evict-frac. This driver diagnoses why, per
the pre-registered design in notes/design-diagnostics-2026-07-10.md. Single trinity
load; all cells chained; per-cell try/except so one failure doesn't kill the chain.

Cells per length L:
  chat — the exp09 main conversation (turns=12):
           corner (b) turn-aligned keep mask  -> decomposition + per-step KLs (D3)
                                                 + KV-geometry npz (D1)
           corner (d) exp08-style sink4+contiguous block, same evicted-token count
                                              -> decomposition + geometry
  raw  — same corpus tokenized raw to the same C, exp09's keep mask transplanted:
           corner (c)                         -> decomposition + geometry
  t48  — D5 granularity cell (turns=48, ef=0.5): decomposition + geometry.

D1 note (why rot-vs-short KV divergence == baked-in contamination): the rotated
survivor KV is IDENTICAL to the pre-eviction full-prefill KV except for the bit-exact
key re-rotation on RoPE layers — values verbatim, NoPE keys untouched. So the
divergence measured here is "KV computed with the evicted block present" vs "KV
recomputed without it". Stats are computed on-GPU; only reduced arrays are saved
(per-layer x per-token cosine curves, fp16). Raw KV never hits disk.

Usage (node1, 8 GPUs — nvidia-smi + Heimdall queue must be clear first):
    PYTHONUNBUFFERED=1 PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      ~/luxi-files/.venv-shared/bin/python experiments/exp09_diagnostics.py \
      --lengths 8192 16384 --device-map balanced --mem-frac 0.70 \
      --out runs/exp09diag.json --outdir runs
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

print = functools.partial(builtins.print, flush=True)
_T0 = time.perf_counter()

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for exp09_trinity_chat
from exp09_trinity_chat import build_conversation, gpu_mem_report  # noqa: E402


def log(msg: str) -> None:
    print(f"[t+{time.perf_counter() - _T0:7.1f}s] {msg}")


# ---------------------------------------------------------------- D1 geometry --

@torch.no_grad()
def kv_geometry(snap_rot, snap_short) -> dict[str, np.ndarray]:
    """Per-layer, per-token cosine similarity between rotated-survivor KV and the
    fresh shortened-recompute KV. Keys and values separately; mean over kv-heads.
    Returns fp16 arrays [num_layers, K] — small enough to bank whole curves."""
    import torch.nn.functional as F

    nl, K = snap_rot.num_layers, snap_rot.seq_len()
    cos_k = torch.empty(nl, K, dtype=torch.float16)
    cos_v = torch.empty(nl, K, dtype=torch.float16)
    for li in range(nl):
        kr = snap_rot.keys[li]
        dev = kr.device
        ks = snap_short.keys[li].to(dev)
        vr, vs = snap_rot.values[li], snap_short.values[li].to(dev)
        # [B, H, S, D] -> cosine over D -> [B, H, S] -> mean over heads -> [S]
        cos_k[li] = F.cosine_similarity(kr.float(), ks.float(), dim=-1)[0].mean(0).half().cpu()
        cos_v[li] = F.cosine_similarity(vr.float(), vs.float(), dim=-1)[0].mean(0).half().cpu()
    return {"cos_k": cos_k.numpy(), "cos_v": cos_v.numpy()}


def geometry_summary(
    geo: dict[str, np.ndarray], keep_np: np.ndarray, block: tuple[int, int],
    applies_rope: list[bool],
) -> dict[str, float]:
    """Reduce the geometry curves to a few pre-registered scalars for the JSON row.
    Zones (original token coordinates): pre-block survivors (expect ~1.0 = noise
    floor), near-seam (first 256 after block end), far (>2048 after block end)."""
    sl = np.asarray(applies_rope, dtype=bool)  # sliding/RoPE layers
    pre = keep_np < block[0]
    dist = keep_np - block[1]  # negative for pre-block
    near = (dist >= 0) & (dist < 256)
    far = dist >= 2048
    out: dict[str, float] = {}
    for what in ("cos_k", "cos_v"):
        arr = geo[what].astype(np.float32)
        for lname, lmask in (("slide", sl), ("glob", ~sl)):
            for zname, zmask in (("pre", pre), ("near", near), ("far", far)):
                if lmask.any() and zmask.any():  # e.g. no NoPE layers on llama
                    out[f"{what[4:]}_{lname}_{zname}"] = float(arr[lmask][:, zmask].mean())
    return out


# ------------------------------------------------------------------- corners --

@torch.no_grad()
def run_corner(
    lm, s0, prefill_ids, keep, probe, full_logits, *,
    tag: str, block: tuple[int, int], outdir: Path, gen: int,
) -> dict[str, Any]:
    """One eviction geometry on one prefilled context: rotate, recompute the identical
    shortened context, decompose the drift per step, and bank the KV-geometry npz."""
    from kvrot.harness import _teacher_forced_logits, prefill_snapshot
    from kvrot.metrics import stepwise_kl
    from kvrot.snapshot import evict as evict_kv
    from kvrot.snapshot import reindex

    dev = lm.device
    K, P = int(keep.shape[0]), int(prefill_ids.shape[1])

    torch.cuda.synchronize(); t = time.perf_counter()
    snap_rot = reindex(evict_kv(s0, keep), torch.arange(K, device=dev), lm.inv_freq)
    torch.cuda.synchronize(); t_rot = time.perf_counter() - t
    rot_logits = _teacher_forced_logits(lm, snap_rot, probe)

    torch.cuda.synchronize(); t = time.perf_counter()
    snap_short = prefill_snapshot(lm, prefill_ids.index_select(1, keep))
    torch.cuda.synchronize(); t_short = time.perf_counter() - t
    short_logits = _teacher_forced_logits(lm, snap_short, probe)

    ps_fr = [float(x) for x in stepwise_kl(full_logits, rot_logits).tolist()]
    ps_fs = [float(x) for x in stepwise_kl(full_logits, short_logits).tolist()]
    ps_sr = [float(x) for x in stepwise_kl(short_logits, rot_logits).tolist()]
    top1 = float((full_logits.argmax(-1) == rot_logits.argmax(-1)).float().mean().item())

    geo = kv_geometry(snap_rot, snap_short)
    keep_np = keep.cpu().numpy().astype(np.int32)
    gsum = geometry_summary(geo, keep_np, block, lm.arch.applies_rope)
    npz_path = outdir / f"exp09diag_geo_{tag}.npz"
    try:
        np.savez_compressed(
            npz_path, cos_k=geo["cos_k"], cos_v=geo["cos_v"], keep=keep_np,
            block=np.asarray(block, dtype=np.int64),
            applies_rope=np.asarray(lm.arch.applies_rope, dtype=bool),
        )
    except OSError as e:  # never lose the run to a bad path
        print(f"  WARNING: could not write {npz_path}: {e}")

    row: dict[str, Any] = dict(
        tag=tag, P=P, K=K, block=list(block), gen=gen,
        kl_full_rot=float(np.mean(ps_fr)), kl_full_short=float(np.mean(ps_fs)),
        kl_short_rot=float(np.mean(ps_sr)), top1=top1,
        ps_kl_full_rot=ps_fr, ps_kl_full_short=ps_fs, ps_kl_short_rot=ps_sr,
        t_rot=t_rot, t_short=t_short, geometry=gsum, geo_npz=str(npz_path),
    )
    print(f"  [{tag}] keep {K}/{P} block=[{block[0]},{block[1]}) | "
          f"KL(f‖r)={row['kl_full_rot']:.3e} info KL(f‖s)={row['kl_full_short']:.3e} "
          f"MECH KL(s‖r)={row['kl_short_rot']:.3e} top1={top1 * 100:.0f}% | "
          f"rot {t_rot * 1e3:.0f}ms recompute {t_short:.1f}s")
    print("  [geometry] " + "  ".join(f"{k}={v:.4f}" for k, v in gsum.items()))

    del snap_rot, snap_short, rot_logits, short_logits
    return row


def imend_split(row: dict[str, Any], first_imend: int) -> None:
    """D3: annotate a corner row with pre/post first-im_end KL means. Step t predicts
    reply token t; the im_end prediction itself is 'live', so pre = steps [0, i]."""
    row["first_imend"] = first_imend
    for key in ("ps_kl_full_rot", "ps_kl_full_short", "ps_kl_short_rot"):
        ps = row[key]
        if 0 <= first_imend < len(ps) - 1:
            row[key.replace("ps_", "") + "_pre_imend"] = float(np.mean(ps[: first_imend + 1]))
            row[key.replace("ps_", "") + "_post_imend"] = float(np.mean(ps[first_imend + 1 :]))


# --------------------------------------------------------------------- cells --

@torch.no_grad()
def cell_chat(lm, docs, L, args, *, n_turns: int, with_d: bool, label: str):
    """Chat context at ~L tokens: corner (b) turn-mask (+D3), optionally corner (d)
    sink4+contiguous. Returns (rows, C, keep_cpu, block) — keep/block are reused by
    the raw-transplant cell."""
    from kvrot.chat import oldest_turns_to_evict, turn_keep_indices
    from kvrot.harness import _greedy_decode, prefill_snapshot

    tok, dev = lm.tokenizer, lm.device
    messages, ctx, spans, ev_turn, ret_turn = build_conversation(tok, docs, L, n_turns, args.doc)
    ctx = ctx.to(dev)
    C = int(ctx.shape[1])
    prefill_ids, last_tok = ctx[:, :-1], ctx[:, -1:]
    P = int(prefill_ids.shape[1])

    evict_turns = oldest_turns_to_evict(spans, target_tokens=int(args.evict_frac * P), protect_last=2)
    evset = set(evict_turns)
    ev_spans = [sp for sp in spans if sp.index in evset]
    contiguous = all(a.end == b.start for a, b in zip(ev_spans, ev_spans[1:]))
    block = (ev_spans[0].start, ev_spans[-1].end)
    keep = turn_keep_indices(spans, evict_turns, P, num_sink_tokens=args.sinks).to(dev)
    K = int(keep.shape[0])
    print(f"\n== {label} C={C} ({n_turns} user turns) | evict {len(evict_turns)} turns = "
          f"{P - K} toks, block [{block[0]},{block[1]}) contiguous={contiguous} | keep {K}/{P} ==")

    ref_cont, full_logits = _greedy_decode(lm, prefill_ids, last_tok, args.gen)
    ids = ref_cont[0].tolist()
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    first_imend = ids.index(im_end_id) if im_end_id in ids else -1
    print(f"  full-cache reply opens: {tok.decode(ids[:24])!r}  (first im_end at step {first_imend})")

    s0 = prefill_snapshot(lm, prefill_ids)
    gpu_mem_report(f"after prefill P={P}")
    probe = torch.cat([last_tok, ref_cont[:, :-1]], dim=1)

    rows = []
    row_b = run_corner(lm, s0, prefill_ids, keep, probe, full_logits,
                       tag=f"{label}_turnmask_C{C}", block=block, outdir=args.outdir, gen=args.gen)
    row_b.update(C=C, contiguous=contiguous, n_evicted_turns=len(evict_turns),
                 corner="b" if n_turns == args.turns else "d5",
                 n_turns=n_turns, ref_ids=ids, ref_text=tok.decode(ids))
    imend_split(row_b, first_imend)
    rows.append(row_b)

    if with_d:  # corner (d): exp08's sink_window geometry on the same chat cache
        E = P - K  # match the evicted-token count exactly (=> same K)
        keep_d = torch.cat([torch.arange(args.sinks, device=dev),
                            torch.arange(args.sinks + E, P, device=dev)])
        row_d = run_corner(lm, s0, prefill_ids, keep_d, probe, full_logits,
                           tag=f"{label}_sink{args.sinks}_C{C}", block=(args.sinks, args.sinks + E),
                           outdir=args.outdir, gen=args.gen)
        row_d.update(C=C, corner="d", n_turns=n_turns,
                     note="chat content, exp08 sink_window geometry: system prompt tokens "
                          f"[{args.sinks},32) evicted, far edge cuts mid-turn")
        imend_split(row_d, first_imend)
        rows.append(row_d)

    keep_cpu = keep.cpu()
    del s0, probe, full_logits, ref_cont, ctx, prefill_ids
    torch.cuda.empty_cache()
    return rows, C, keep_cpu, block


@torch.no_grad()
def cell_raw(lm, C: int, keep_cpu, block, args):
    """Corner (c): raw-doc content (same corpus, same joiner, same C) with exp09's
    exact keep mask transplanted. Continuation = the raw full cache's own greedy."""
    from kvrot.data import load_text_context
    from kvrot.harness import _greedy_decode, prefill_snapshot

    tok, dev = lm.tokenizer, lm.device
    ctx = load_text_context(args.source, tok, C, device=dev, doc_index=args.doc)
    prefill_ids, last_tok = ctx[:, :-1], ctx[:, -1:]
    P = int(prefill_ids.shape[1])
    keep = keep_cpu.to(dev)
    print(f"\n== raw-doc transplant C={C} | exp09 keep mask reused: keep "
          f"{int(keep.shape[0])}/{P}, block [{block[0]},{block[1]}) ==")

    ref_cont, full_logits = _greedy_decode(lm, prefill_ids, last_tok, args.gen)
    print(f"  full-cache continuation opens: {tok.decode(ref_cont[0, :24])!r}")
    s0 = prefill_snapshot(lm, prefill_ids)
    probe = torch.cat([last_tok, ref_cont[:, :-1]], dim=1)

    row = run_corner(lm, s0, prefill_ids, keep, probe, full_logits,
                     tag=f"raw_turnmask_C{C}", block=block, outdir=args.outdir, gen=args.gen)
    row.update(C=C, corner="c", ref_text=tok.decode(ref_cont[0]))

    del s0, probe, full_logits, ref_cont, ctx, prefill_ids
    torch.cuda.empty_cache()
    return [row]


# ---------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Trinity-Large-Preview")
    ap.add_argument("--source", default="data/eval_docs.jsonl")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--d5-turns", type=int, default=48)
    ap.add_argument("--doc", type=int, default=0)
    ap.add_argument("--evict-frac", type=float, default=0.5)
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--mem-frac", type=float, default=0.70)
    ap.add_argument("--device-map", default="balanced")
    ap.add_argument("--cells", nargs="+", default=["chat", "raw", "t48"],
                    choices=["chat", "raw", "t48"])
    ap.add_argument("--out", type=Path, default=Path("runs/exp09diag.json"))
    ap.add_argument("--outdir", type=Path, default=Path("runs"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    from kvrot.data import read_texts
    from kvrot.harness import load_model

    docs = read_texts(args.source)
    log(f"loaded {len(docs)} eval docs from {args.source}")
    log(f"loading {args.model} (several minutes — watch the shards bar)...")
    lm = load_model(args.model, device_map=args.device_map, max_memory_frac=args.mem_frac)
    a = lm.arch
    log(f"model loaded: {a.num_hidden_layers}L, applies_rope="
        f"{sum(a.applies_rope)}/{a.num_hidden_layers}")
    gpu_mem_report("after load")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for L in args.lengths:
        chat_ctx: tuple | None = None
        if "chat" in args.cells:
            log(f"cell chat L={L} starting (corners b + d, D1 + D3)")
            try:
                r, C, keep_cpu, block = cell_chat(lm, docs, L, args, n_turns=args.turns,
                                                  with_d=True, label="chat")
                rows += r
                chat_ctx = (C, keep_cpu, block)
            except Exception as e:
                print(f"  chat L={L} FAILED: {type(e).__name__}: {e}")
                failures.append({"cell": "chat", "L": L, "error": f"{type(e).__name__}: {e}"})
        if "raw" in args.cells:
            log(f"cell raw L={L} starting (corner c)")
            try:
                if chat_ctx is None:
                    raise RuntimeError("raw cell needs the chat cell's C/keep/block "
                                       "(run with cells including 'chat')")
                rows += cell_raw(lm, *chat_ctx, args)
            except Exception as e:
                print(f"  raw L={L} FAILED: {type(e).__name__}: {e}")
                failures.append({"cell": "raw", "L": L, "error": f"{type(e).__name__}: {e}"})
        if "t48" in args.cells:
            log(f"cell t48 L={L} starting (D5, turns={args.d5_turns})")
            try:
                r, _, _, _ = cell_chat(lm, docs, L, args, n_turns=args.d5_turns,
                                       with_d=False, label="t48")
                rows += r
            except Exception as e:
                print(f"  t48 L={L} FAILED: {type(e).__name__}: {e}")
                failures.append({"cell": "t48", "L": L, "error": f"{type(e).__name__}: {e}"})

    print("\n==== summary (diagnostics) ====")
    summary_lines: list[str] = []
    for r in rows:
        line = (f"{r['tag']:<28} keep {r['K']}/{r['P']}: KL(f‖r) {r['kl_full_rot']:.2e}  "
                f"info {r['kl_full_short']:.2e}  MECH {r['kl_short_rot']:.2e}  "
                f"top1 {r['top1'] * 100:3.0f}%"
                + (f"  mech pre/post-imEnd {r.get('kl_short_rot_pre_imend'):.2e}/"
                   f"{r.get('kl_short_rot_post_imend'):.2e}"
                   if r.get("kl_short_rot_pre_imend") is not None else ""))
        print("  " + line)
        summary_lines.append(line)

    payload: dict[str, Any] = {
        "experiment": "exp09_diagnostics",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "rows": rows,
        "failures": failures,
        "summary": summary_lines,
    }
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        log(f"results written to {args.out}")
    except OSError as e:
        print(f"  WARNING: could not write --out {args.out}: {e}")
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
