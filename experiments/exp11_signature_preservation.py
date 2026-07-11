"""exp11 (cells) — signature preservation under KV eviction+rotation.

Design contract: notes/design-signature-preservation-2026-07-10.md (incl. both
2026-07-10 amendments). Per cell:

    1. build the context (Regime A: banked 3B-native dialogue through the chat
       template; Regime B: raw eval docs, exp08 contiguous-early-block geometry);
    2. FULL prefill → frozen greedy continuation (N=256, generated once);
    3. caches per condition: FULL / ROT (evict + Tier-0 re-rotate, recompacted) /
       REC (fresh prefill of the identical shortened text) / NAIVE (evict WITHOUT
       re-rotation — survivors keep original positions, continuation resumes at
       survivor-max+1);
    4. teacher-force the identical continuation against each cache through the
       instrumented bridge (kvrot.sigbridge.replay_extract_cached) → full v3
       signature (2,713-dim, feature-name alignment hard-asserted per cell);
    5. floors: (a) FULL replayed twice (numerics), (b) incremental decode-path
       FULL replay (path floor — every distance is reported against this);
    6. side-by-side token KL(FULL‖c) + top1 from the same replay logits;
    7. swept cells (2 dialogue + 1 doc): ROT/REC additionally at evict-frac
       {0.25, 0.75} for the fade curve, plus one ROT/REC replay repetition for
       the stability read (metric 5).

Everything (full signature vectors included) is banked to --out so the entire
geometry is computable in exp11_analysis.py without re-running.

Stage 1 (gate): --gate-only runs the correctness gate + one smoke cell with the
P5 tripwire (d_NAIVE ≥ 10× path floor) and exits. Stage 2: full grid.

Usage (node1, ONE free GPU, anamnesis-pl staged read-only):
    CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=~/luxi-files/anamnesis-pl-exp11 \
    python experiments/exp11_signature_preservation.py \
        --model /models/Llama-3.2-3B-Instruct \
        --convs data/native3b_convs_2026-07-10.jsonl \
        --docs data/eval_docs.jsonl \
        --calib-dir exp11_calibration \
        --out runs/exp11_grid.json
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

print = functools.partial(builtins.print, flush=True)
_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[t+{time.perf_counter() - _T0:7.1f}s] {msg}")


SWEEP_EFS = (0.25, 0.75)  # base evict-frac comes from --evict-frac (default 0.5)


def cond_name(base: str, ef: float, rep: bool = False) -> str:
    return f"{base}@{ef:.2f}" + ("#rep" if rep else "")


# ── Model + instrument (ONE model instance serving both worlds) ──────────────────


def build_instrument(args):
    """anamnesis LoadedModel (hooks) + kvrot LoadedModel view (surgery) on one model."""
    from anamnesis.config import MODEL_PRESETS, ModelConfig
    from anamnesis.extraction.model_loader import load_model as anamnesis_load

    from kvrot.config import ArchSpec
    from kvrot.harness import LoadedModel as KvLoadedModel
    from kvrot.harness import _find_inv_freq
    from kvrot.rope import assert_rotation_homomorphism

    preset = MODEL_PRESETS["3b"]
    # Deviation from the anamnesis 3B preset (fp16), per the exp11 design contract: bf16
    # to match every kv-rotation experiment. Recorded in run config.
    model_config = ModelConfig(
        model_id=args.model,
        torch_dtype="bfloat16",
        attn_implementation="eager",   # required by both attention capture and hooks
        device_map=args.device_map,
        num_layers=preset.num_layers,
        hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads,
        num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
    )
    loaded = anamnesis_load(
        model_config,
        sampled_layers=preset.sampled_layers,
        register_gate_hooks=True,                 # v3 gate features
        key_layers=list(range(preset.num_layers)),  # all-layer keys (v3 replay surface)
        value_layers=None,
        query_layers=None,
    )
    loaded.disable_hooks()  # enabled only inside instrumented replays

    arch = ArchSpec.from_hf_config(loaded.model.config)
    inv_freq = _find_inv_freq(loaded.model, arch)
    assert_rotation_homomorphism(inv_freq)
    device = next(loaded.model.parameters()).device
    kv_lm = KvLoadedModel(
        model=loaded.model, tokenizer=loaded.tokenizer, arch=arch,
        inv_freq=inv_freq.to(device), device=device,
    )
    return loaded, kv_lm, preset


# ── Context builders ──────────────────────────────────────────────────────────────


def dialogue_context(tokenizer, conv, device):
    """Full banked conversation → (ctx_ids [1,P], spans). Ends with the generation
    prompt (the model is about to write the next persona-B turn — the natural
    continuation point the greedy rollout starts from)."""
    from kvrot.chat import turn_token_spans

    messages = conv.messages()
    ctx, spans = turn_token_spans(tokenizer, messages, add_generation_prompt=True)
    return ctx.to(device), spans


def doc_context(tokenizer, docs_path: Path, doc_index: int, n_tokens: int, device):
    from kvrot.data import load_text_context

    return load_text_context(
        str(docs_path), tokenizer, n_tokens, device=device, doc_index=doc_index
    )


def keep_for_dialogue(spans, p_len: int, ef: float, sinks: int) -> tuple[torch.Tensor, int]:
    """Turn-aligned oldest-first eviction (protect system + last 2 turns, keep sinks)."""
    from kvrot.chat import oldest_turns_to_evict, turn_keep_indices

    evict_turns = oldest_turns_to_evict(
        spans, target_tokens=int(ef * p_len), protect_last=2
    )
    if not evict_turns:
        raise RuntimeError(f"no evictable turns at ef={ef} (protections consumed everything)")
    keep = turn_keep_indices(spans, evict_turns, p_len, num_sink_tokens=sinks)
    return keep, len(evict_turns)


def keep_for_doc(p_len: int, ef: float, sinks: int) -> tuple[torch.Tensor, int]:
    """exp08 geometry: keep sinks, evict ONE contiguous early block, keep the tail."""
    from kvrot.config import EvictionSpec
    from kvrot.eviction import compute_keep_indices

    spec = EvictionSpec(
        policy="sink_window", num_sink_tokens=sinks, evict_count=int(ef * p_len)
    )
    return compute_keep_indices(p_len, spec), 1


# ── Reference continuation (frozen greedy rollout from FULL) ─────────────────────


@torch.no_grad()
def greedy_continuation(kv_lm, ctx_ids: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Greedy N-token rollout from a fresh FULL prefill of ctx_ids. Returns [1, N]."""
    from transformers import DynamicCache

    out = kv_lm.model(input_ids=ctx_ids, past_key_values=DynamicCache(), use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :].float()
    pos = ctx_ids.shape[1]
    cont: list[int] = []
    for _ in range(n_steps):
        nxt = int(logits.argmax(-1).item())
        cont.append(nxt)
        step = kv_lm.model(
            input_ids=torch.tensor([[nxt]], device=kv_lm.device),
            past_key_values=cache,
            use_cache=True,
            position_ids=torch.tensor([[pos]], device=kv_lm.device),
            cache_position=torch.tensor([pos], device=kv_lm.device),
        )
        cache = step.past_key_values
        logits = step.logits[:, -1, :].float()
        pos += 1
    del cache
    torch.cuda.empty_cache()
    return torch.tensor([cont], dtype=torch.long, device=kv_lm.device)


def first_eos_index(tokenizer, cont_ids: torch.Tensor) -> int:
    eos: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos.add(int(tokenizer.eos_token_id))
    for t in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"):
        try:
            i = tokenizer.convert_tokens_to_ids(t)
        except Exception:
            i = None
        if i is not None and i >= 0 and i != getattr(tokenizer, "unk_token_id", None):
            eos.add(int(i))
    return next((j for j, t in enumerate(cont_ids[0].tolist()) if t in eos), -1)


# ── KL column ─────────────────────────────────────────────────────────────────────


def kl_and_top1(ref_logits: list[np.ndarray], other_logits: list[np.ndarray]) -> dict[str, Any]:
    """Token KL(FULL‖c) + top1 over the replay's T transition positions."""
    ref = torch.from_numpy(np.stack(ref_logits)).float()
    oth = torch.from_numpy(np.stack(other_logits)).float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    oth_logp = torch.log_softmax(oth, dim=-1)
    kl = (ref_logp.exp() * (ref_logp - oth_logp)).sum(dim=-1)
    top1 = (ref.argmax(-1) == oth.argmax(-1)).float().mean().item()
    return {
        "mean": float(kl.mean().item()),
        "max": float(kl.max().item()),
        "top1": float(top1),
        "steps": [round(float(x), 6) for x in kl.tolist()],
    }


# ── One cell ──────────────────────────────────────────────────────────────────────


def run_cell(
    loaded,
    kv_lm,
    calib,
    extraction_config,
    family_config,
    *,
    cell_id: str,
    regime: str,
    ctx_ids: torch.Tensor,
    spans,
    args,
    sweep: bool,
) -> dict[str, Any]:
    from transformers import DynamicCache

    from kvrot.harness import prefill_snapshot
    from kvrot.sigbridge import (
        assert_feature_alignment,
        compute_signature,
        replay_extract_cached,
    )
    from kvrot.snapshot import evict as evict_kv
    from kvrot.snapshot import reindex, to_hf_dynamic_cache

    t_cell = time.perf_counter()
    p_len = int(ctx_ids.shape[1])
    device = kv_lm.device
    base_ef = args.evict_frac

    # 2) frozen greedy continuation from FULL
    cont_ids = greedy_continuation(kv_lm, ctx_ids, args.gen)
    f_eos = first_eos_index(loaded.tokenizer, cont_ids)
    cont_head = loaded.tokenizer.decode(cont_ids[0, :48].tolist())
    log(f"[{cell_id}] P={p_len} N={args.gen} first_eos={f_eos} cont: {cont_head[:120]!r}")

    # 3) FULL snapshot (surgery source) — hooks stay disabled during prefills
    s0 = prefill_snapshot(kv_lm, ctx_ids)

    efs = [base_ef] + (list(SWEEP_EFS) if sweep else [])
    keeps: dict[float, torch.Tensor] = {}
    ef_meta: dict[str, Any] = {}
    for ef in efs:
        if regime == "dialogue":
            keep, n_units = keep_for_dialogue(spans, p_len, ef, args.sinks)
        else:
            keep, n_units = keep_for_doc(p_len, ef, args.sinks)
        keeps[ef] = keep.to(device)
        ef_meta[f"{ef:.2f}"] = {
            "K": int(keep.shape[0]),
            "evicted_tokens": p_len - int(keep.shape[0]),
            "keep_frac": round(int(keep.shape[0]) / p_len, 4),
            "n_evicted_units": n_units,
        }

    # Build every replay job: (name, snapshot-or-None, offset, incremental)
    def rot_snap(ef: float):
        k = keeps[ef]
        return reindex(evict_kv(s0.clone(), k), torch.arange(int(k.shape[0]), device=device),
                       kv_lm.inv_freq)

    def rec_snap(ef: float):
        return prefill_snapshot(kv_lm, ctx_ids[:, keeps[ef]])

    def naive_snap(ef: float):
        return evict_kv(s0.clone(), keeps[ef])  # positions carried unchanged (the gap)

    jobs: list[tuple[str, Any, int, bool]] = []
    jobs.append(("FULL", s0, p_len, False))
    jobs.append(("FLOOR_A", s0, p_len, False))          # numerics floor: same path twice
    jobs.append(("FLOOR_B", s0, p_len, True))           # path floor: incremental twin
    for ef in efs:
        sr = rot_snap(ef)
        jobs.append((cond_name("ROT", ef), sr, sr.seq_len(), False))
        se = rec_snap(ef)
        jobs.append((cond_name("REC", ef), se, se.seq_len(), False))
    sn = naive_snap(base_ef)
    jobs.append((cond_name("NAIVE", base_ef), sn, sn.next_position(), False))
    if sweep:  # stability repeats at the base operating point (metric 5-stability)
        jobs.append((cond_name("ROT", base_ef, rep=True), rot_snap(base_ef), None, False))
        jobs.append((cond_name("REC", base_ef, rep=True), rec_snap(base_ef), None, False))

    signatures: dict[str, list[float]] = {}
    kls: dict[str, Any] = {}
    timing: dict[str, float] = {}
    ref_names: list[str] | None = None
    ref_slices: dict[str, tuple[int, int]] | None = None
    full_logits: list[np.ndarray] | None = None

    for name, snap, offset, incremental in jobs:
        t0 = time.perf_counter()
        if offset is None:  # repeats: same offset as the base condition
            offset = snap.seq_len() if name.startswith(("ROT", "REC")) else p_len
        cache = to_hf_dynamic_cache(snap)
        raw = replay_extract_cached(
            loaded, cache, cont_ids, int(offset),
            positional_means=calib.positional_means, incremental=incremental,
        )
        del cache
        result = compute_signature(raw, extraction_config, family_config, calib)
        if ref_names is None:
            ref_names, ref_slices = list(result.feature_names), dict(result.tier_slices)
            full_logits = raw.logits
        else:
            assert_feature_alignment(ref_names, list(result.feature_names), f"{cell_id}/{name}")
            kls[name] = kl_and_top1(full_logits, raw.logits)
        signatures[name] = [float(x) for x in np.asarray(result.features, dtype=np.float32)]
        del raw
        timing[name] = round(time.perf_counter() - t0, 2)
        d = (
            float(np.linalg.norm(
                np.asarray(signatures[name], dtype=np.float64)
                - np.asarray(signatures["FULL"], dtype=np.float64)
            ))
            if name != "FULL"
            else 0.0
        )
        kl_str = f" KL={kls[name]['mean']:.3e}" if name in kls else ""
        log(f"  [{cell_id}] {name}: d_from_FULL={d:.4f}{kl_str} ({timing[name]:.1f}s)")

    del s0
    torch.cuda.empty_cache()

    sig_full = np.asarray(signatures["FULL"], dtype=np.float64)
    d_from_full = {
        n: float(np.linalg.norm(np.asarray(s, dtype=np.float64) - sig_full))
        for n, s in signatures.items()
        if n != "FULL"
    }
    return {
        "cell_id": cell_id,
        "regime": regime,
        "P": p_len,
        "N": args.gen,
        "first_eos": f_eos,
        "cont_head": cont_head,
        "cont_ids": cont_ids[0].tolist(),
        "base_ef": base_ef,
        "sweep": sweep,
        "sinks": args.sinks,
        "efs": ef_meta,
        "signatures": signatures,
        "d_from_full": d_from_full,
        "kl": kls,
        "timing_s": {**timing, "cell_total": round(time.perf_counter() - t_cell, 1)},
        "_feature_names": ref_names,       # hoisted to run level by the caller
        "_tier_slices": ref_slices,
    }


def p5_tripwire(cell: dict[str, Any], base_ef: float) -> tuple[bool, str]:
    """d_NAIVE must exceed the path floor by ≥10× for null-readings to count (P5)."""
    floor_b = cell["d_from_full"]["FLOOR_B"]
    d_naive = cell["d_from_full"][cond_name("NAIVE", base_ef)]
    ratio = d_naive / max(floor_b, 1e-12)
    msg = (
        f"P5 check [{cell['cell_id']}]: d_NAIVE={d_naive:.4f}, path floor(b)={floor_b:.4f} "
        f"→ ratio {ratio:.1f}x (need ≥10x)"
    )
    return ratio >= 10.0, msg


# ── Main ──────────────────────────────────────────────────────────────────────────


def write_out(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, default=str) + "\n")
        tmp.replace(path)
    except OSError as e:
        print(f"WARNING: could not write {path}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="exp11 signature-preservation cell driver")
    ap.add_argument("--model", default="/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--convs", type=Path, default=Path("data/native3b_convs_2026-07-10.jsonl"))
    ap.add_argument("--docs", type=Path, default=Path("data/eval_docs.jsonl"))
    ap.add_argument("--calib-dir", type=Path, default=Path("exp11_calibration"),
                    help="dir with positional_means.npz + pca_model.pkl (3B, read-only)")
    ap.add_argument("--anamnesis-path", default=None,
                    help="anamnesis-pl checkout (fallback: ANAMNESIS_PL_PATH, known paths)")
    ap.add_argument("--out", type=Path, default=Path("runs/exp11_grid.json"))
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--evict-frac", type=float, default=0.5)
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--doc-cells", type=int, default=4)
    ap.add_argument("--doc-tokens", type=int, default=4608)
    ap.add_argument("--conv-ids", nargs="*", default=None)
    ap.add_argument("--sweep-dialogue", type=int, default=2,
                    help="first N dialogue cells get the evict-frac fade sweep")
    ap.add_argument("--sweep-docs", type=int, default=1)
    ap.add_argument("--gate-tokens", type=int, default=2048,
                    help="context length for the correctness gate (memory-bounded)")
    ap.add_argument("--gate-rtol", type=float, default=1e-4)
    ap.add_argument("--gate-atol", type=float, default=1e-6)
    ap.add_argument("--skip-gate", action="store_true",
                    help="ONLY after the gate has passed in a previous run on this box")
    ap.add_argument("--gate-only", action="store_true",
                    help="stage 1: gate + one smoke cell (P5 tripwire), then exit")
    ap.add_argument("--no-p5-abort", action="store_true")
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    from kvrot.sigbridge import ensure_anamnesis

    used_path = ensure_anamnesis(args.anamnesis_path)
    log(f"anamnesis-pl: {used_path}")

    from kvrot.natural import load_convs_any
    from kvrot.sigbridge import (
        correctness_gate,
        load_3b_calibration,
        v3_extraction_config,
        v3_family_config,
    )

    calib = load_3b_calibration(args.calib_dir)
    log(f"calibration: {calib.source_dir} (positional_means "
        f"{calib.positional_means.shape}, pca "
        f"{getattr(calib.pca_components, 'shape', type(calib.pca_components).__name__)})")
    extraction_config = v3_extraction_config("3b")
    family_config = v3_family_config("3b")

    log(f"loading {args.model} (bf16, eager, v3 hook surface)...")
    loaded, kv_lm, preset = build_instrument(args)
    log(f"loaded on {kv_lm.device}; sampled_layers={preset.sampled_layers}")

    convs = load_convs_any(args.convs) if args.convs.exists() else []
    if args.conv_ids:
        convs = [c for c in convs if c.conv_id in set(args.conv_ids)]
    log(f"dialogue conversations: {[c.conv_id for c in convs]}")
    if not convs:
        raise SystemExit(f"no conversations at {args.convs} — run exp11_gen_native3b.py first")

    payload: dict[str, Any] = {
        "experiment": "exp11_signature_preservation",
        "design": "notes/design-signature-preservation-2026-07-10.md",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model_dtype": "bfloat16 (deviation from anamnesis 3B preset fp16, per design)",
        "sweep_efs": list(SWEEP_EFS),
        "calibration_dir": calib.source_dir,
        "gate": None,
        "cells": [],
        "failures": [],
    }

    # ── Stage 1: correctness gate ──
    if not args.skip_gate:
        from kvrot.natural import choose_cut_prefix

        conv0 = convs[0]
        n_msgs, rendered = choose_cut_prefix(
            loaded.tokenizer, conv0.messages(), args.gate_tokens, end_role="user"
        )
        from kvrot.chat import turn_token_spans

        gate_ctx, _ = turn_token_spans(
            loaded.tokenizer, conv0.messages()[:n_msgs], add_generation_prompt=True
        )
        gate_ctx = gate_ctx.to(kv_lm.device)
        gate_cont = greedy_continuation(kv_lm, gate_ctx, args.gen)
        log(f"gate: ctx {gate_ctx.shape[1]} tok (cut of {conv0.conv_id}), cont {args.gen} tok")
        t0 = time.perf_counter()
        gate = correctness_gate(
            loaded, gate_ctx, gate_cont, extraction_config, family_config, calib,
            rtol=args.gate_rtol, atol=args.gate_atol,
        )
        print(gate.summary() + f"  ({time.perf_counter() - t0:.0f}s)")
        payload["gate"] = {
            "passed": gate.passed, "names_equal": gate.names_equal,
            "n_features": gate.n_features, "n_violations": gate.n_violations,
            "max_abs_diff": gate.max_abs_diff, "max_rel_diff": gate.max_rel_diff,
            "rtol": gate.rtol, "atol": gate.atol,
            "worst": gate.worst, "ctx_tokens": int(gate_ctx.shape[1]),
        }
        write_out(args.out, payload)
        if not gate.passed:
            raise SystemExit(
                "CORRECTNESS GATE FAILED — the injection path perturbs the signature. "
                "No cells were run (spec: gate must pass first). See --out for details."
            )
        del gate_ctx, gate_cont
        torch.cuda.empty_cache()

    # ── Cell plan ──
    plan: list[dict[str, Any]] = []
    for i, conv in enumerate(convs):
        plan.append({
            "cell_id": f"dlg-{conv.conv_id}", "regime": "dialogue", "conv": conv,
            "sweep": i < args.sweep_dialogue,
        })
    if args.docs.exists():
        for d in range(args.doc_cells):
            plan.append({
                "cell_id": f"doc-{d:02d}", "regime": "doc", "doc_index": d,
                "sweep": d < args.sweep_docs,
            })
    else:
        log(f"WARNING: {args.docs} not found — running dialogue cells only")
    if args.gate_only:
        plan = plan[:1]
        log("--gate-only: running ONE smoke cell after the gate, then exiting")

    feature_names: list[str] | None = None
    for cell_idx, spec in enumerate(plan):
        cid = spec["cell_id"]
        try:
            if spec["regime"] == "dialogue":
                ctx, spans = dialogue_context(loaded.tokenizer, spec["conv"], kv_lm.device)
            else:
                ctx = doc_context(
                    loaded.tokenizer, args.docs, spec["doc_index"], args.doc_tokens,
                    kv_lm.device,
                )
                spans = None
            cell = run_cell(
                loaded, kv_lm, calib, extraction_config, family_config,
                cell_id=cid, regime=spec["regime"], ctx_ids=ctx, spans=spans,
                args=args, sweep=spec["sweep"],
            )
            if feature_names is None:
                feature_names = cell.pop("_feature_names")
                payload["feature_names"] = feature_names
                payload["tier_slices"] = {
                    k: list(v) for k, v in cell.pop("_tier_slices").items()
                }
                log(f"feature space: {len(feature_names)} dims, "
                    f"families: {list(payload['tier_slices'])}")
            else:
                from kvrot.sigbridge import assert_feature_alignment

                assert_feature_alignment(feature_names, cell.pop("_feature_names"), cid)
                cell.pop("_tier_slices")
            ok, msg = p5_tripwire(cell, args.evict_frac)
            cell["p5_ok"] = ok
            payload["cells"].append(cell)
            write_out(args.out, payload)
            log(("  " if ok else "  !! ") + msg)
            # Hard abort ONLY on the first (smoke) cell — the staging plan's P5 gate.
            # Later cells record the flag and keep the grid alive; the analysis
            # excludes P5-failing cells from any null interpretation per spec.
            if not ok and cell_idx == 0 and not args.no_p5_abort:
                payload["failures"].append({"cell": cid, "error": "P5 tripwire: " + msg})
                write_out(args.out, payload)
                raise SystemExit(
                    "P5 SENSITIVITY GATE FAILED on the smoke cell — the instrument does "
                    "not see gross cache damage (NAIVE) above 10x the path floor in this "
                    "regime. Stopping per spec; no null conclusions may be drawn. "
                    "(--no-p5-abort to override.)"
                )
        except SystemExit:
            raise
        except Exception as e:  # keep the grid alive across cell failures
            import traceback

            traceback.print_exc()
            payload["failures"].append({"cell": cid, "error": f"{type(e).__name__}: {e}"})
            write_out(args.out, payload)

    n_ok = len(payload["cells"])
    log(f"done: {n_ok} cells, {len(payload['failures'])} failures → {args.out}")


if __name__ == "__main__":
    main()
