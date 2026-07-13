"""exp11 (follow-up 2) — PER-FEATURE decomposition INSIDE T3.

Motivation (Luxia → imago's precise questions, 2026-07-12): the family decomposition
(`exp11_family_decomp.py`) established that T3 residual-PCA carries the ROT-vs-REC
contamination separation by its own per-feature weight (+1.23 log2, 12/12), and it
sub-decomposed T3 by PCA layer and snapshot. What it did NOT emit is the FULL per-feature
ranking of T3's 1,250 dims with per-feature significance — `top_gap_features` was capped at
k=8 and carries no p-value. This script answers the two precise questions directly:

  Q1: were there SPECIFIC T3 features that were competitive (carry the gap by themselves)?
  Q2: what is the per-feature breakdown across T3?

Method — training-free, consistent with the family decomposition (NOT the anamnesis
encoder-CV removal-cost, which needs many labeled samples and cannot run per-feature on
n=12 paired aggregate cells; see the journal "No encoders" note):

  * Standardize per feature: Δz = (sig_cond − sig_FULL) ⊘ FULL_scale (factor_directions_3b).
  * Per T3 feature j, per cell c: abs-excess  g_cj = |Δz_REC| − |Δz_ROT|  (>0 ⇒ recompute
    perturbs this feature further from FULL than rotation does — the contamination axis).
    Descriptive effect size also reported as median per-cell log2(|Δz_REC|/|Δz_ROT|).
  * Per-feature EXACT sign-flip p (paired, one-sided mean(g)>0), vectorized over all 1,250
    features at once, chunked to bound memory; n capped at 20 (exact enumeration).
  * Multiple-comparison control across the 1,250 T3 features: Benjamini–Hochberg FDR.
  * Breakdowns: by PCA layer, by snapshot, by component index, and the full ranked table.

Runs on the banked aggregate grid (CPU, ~seconds) — no GPU, no regeneration. Both the main
grid (dialogue n=8, all n=12) and the pooled confirmatory grid (dialogue n=20) are analyzed.

Usage:
    .venv/bin/python experiments/exp11_t3_perfeature.py \
        --run runs/exp11_grid.json --pooled runs/exp11_pooled.json \
        --factor-directions ~/projects/anamnesis_exps/outputs/analysis/v3_audit/factor_directions_3b.npz \
        --out runs/exp11_t3_perfeature.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the vetted helpers from the family decomposition (unit-tested).
from exp11_family_decomp import (
    CellDeltas,
    T3_RE,
    guarded_scale,
    load_cell_deltas,
    signflip_p_one_sided,
)

SCALE_EPS = 1e-9
CHUNK = 128  # feature-chunk for the vectorized sign-flip (bounds the (2^n, chunk) buffer)


def t3_index(names: list[str]) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Indices of T3 residual-PCA features (pca_L{layer}_t{snap}_c{comp}) + parsed keys."""
    idx: list[int] = []
    keys: list[tuple[int, int, int]] = []
    for i, n in enumerate(names):
        m = T3_RE.match(n)
        if m:
            idx.append(i)
            keys.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if not idx:
        raise SystemExit("no T3 (pca_L*_t*_c*) features found in this run")
    return np.asarray(idx, dtype=np.int64), keys


def abs_excess_matrix(cells: list[CellDeltas], idx: np.ndarray) -> np.ndarray:
    """(n_cells, n_feat) per-cell |Δz_REC| − |Δz_ROT| on the feature subset."""
    return np.stack(
        [np.abs(c.dz["REC"][idx]) - np.abs(c.dz["ROT"][idx]) for c in cells]
    ).astype(np.float64)


def perfeature_signflip(g: np.ndarray) -> np.ndarray:
    """Exact one-sided sign-flip p (mean(g)>0) for every column of g (n_cells, n_feat).

    Vectorized: enumerate all 2^n sign vectors once, matmul against feature chunks.
    Returns p per feature; n capped at 20 (matches the family-decomp enumeration bound).
    """
    n, nfeat = g.shape
    if n == 0:
        return np.ones(nfeat)
    if n > 20:
        raise ValueError(f"exact sign-flip enumeration capped at n=20, got n={n}")
    bits = np.arange(2**n, dtype=np.int64)[:, None]
    signs = np.where((bits >> np.arange(n)) & 1, 1.0, -1.0)  # (2^n, n)
    obs = g.mean(axis=0)  # (n_feat,)
    p = np.empty(nfeat, dtype=np.float64)
    for s in range(0, nfeat, CHUNK):
        e = min(s + CHUNK, nfeat)
        null = signs @ g[:, s:e] / n  # (2^n, chunk)
        p[s:e] = (null >= obs[s:e] - 1e-15).mean(axis=0)
    return p


def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, float]:
    """Benjamini–Hochberg: boolean reject mask + the p threshold. NaN p's never reject."""
    p = np.asarray(pvals, dtype=np.float64)
    finite = np.isfinite(p)
    m = int(finite.sum())
    reject = np.zeros_like(p, dtype=bool)
    if m == 0:
        return reject, float("nan")
    order = np.argsort(np.where(finite, p, np.inf))
    ranked = p[order][:m]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    if not passed.any():
        return reject, 0.0
    kmax = int(np.max(np.nonzero(passed)))
    pcut = float(ranked[kmax])
    reject[finite & (p <= pcut)] = True
    return reject, pcut


def safe_log2_ratio(num: float, den: float) -> float:
    if den < SCALE_EPS or num < SCALE_EPS:
        return float("nan")
    return float(np.log2(num / den))


def analyze_grid(
    cells: list[CellDeltas], idx: np.ndarray, keys: list[tuple[int, int, int]], names: list[str]
) -> dict[str, Any]:
    """Per-feature T3 stats for a set of cells, split by regime where each subset has ≥1 cell."""
    out: dict[str, Any] = {}
    regimes = {
        "dlg": [c for c in cells if c.regime == "dialogue"],
        "doc": [c for c in cells if c.regime == "doc"],
        "all": list(cells),
    }
    feat_names = [names[int(i)] for i in idx]

    for label, group in regimes.items():
        if not group or len(group) > 20:
            out[label] = {"n_cells": len(group), "skipped": len(group) > 20}
            continue
        g = abs_excess_matrix(group, idx)  # (n_cells, n_feat)
        med = np.median(g, axis=0)
        n_pos = (g > 0).sum(axis=0)
        p = perfeature_signflip(g)
        reject, pcut = bh_fdr(p, q=0.05)

        # descriptive median log2 ratio (per-cell, guarded, then median over cells)
        rot = np.stack([np.abs(c.dz["ROT"][idx]) for c in group])
        rec = np.stack([np.abs(c.dz["REC"][idx]) for c in group])
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log2(rec / rot)
        lr[~np.isfinite(lr)] = np.nan
        med_log2 = np.nanmedian(lr, axis=0)

        out[label] = {
            "n_cells": len(group),
            "n_features": int(idx.size),
            "bh_q": 0.05,
            "bh_pcut": pcut,
            "n_sig_bh": int(reject.sum()),
            "n_rec_further_majority": int((n_pos > len(group) / 2).sum()),
            "n_all_cells_rec_further": int((n_pos == len(group)).sum()),
            "_per_feature": {  # arrays aligned to idx order (kept for ranking; trimmed on write)
                "name": feat_names,
                "median_absdz_excess": med.tolist(),
                "median_log2_ratio": [None if np.isnan(x) else float(x) for x in med_log2],
                "n_cells_pos": n_pos.tolist(),
                "p_signflip": p.tolist(),
                "bh_reject": reject.tolist(),
            },
        }
    return out


def ranked_table(block: dict[str, Any], top: int) -> list[dict[str, Any]]:
    pf = block["_per_feature"]
    med = np.asarray(pf["median_absdz_excess"])
    order = np.argsort(-med)
    rows = []
    for j in order[:top]:
        rows.append(
            {
                "name": pf["name"][j],
                "median_absdz_excess": round(float(pf["median_absdz_excess"][j]), 4),
                "median_log2_ratio": pf["median_log2_ratio"][j],
                "n_cells_pos": int(pf["n_cells_pos"][j]),
                "p_signflip": round(float(pf["p_signflip"][j]), 6),
                "bh_sig": bool(pf["bh_reject"][j]),
            }
        )
    return rows


def group_breakdown(
    block: dict[str, Any], keys: list[tuple[int, int, int]], which: str
) -> dict[str, dict[str, Any]]:
    """Aggregate the per-feature significance by layer / snapshot / component."""
    pf = block["_per_feature"]
    med = np.asarray(pf["median_absdz_excess"])
    reject = np.asarray(pf["bh_reject"])
    key_of = {"layer": 0, "snapshot": 1, "component": 2}[which]
    agg: dict[Any, list[int]] = {}
    for j, k in enumerate(keys):
        agg.setdefault(k[key_of], []).append(j)
    prefix = {"layer": "L", "snapshot": "t", "component": "c"}[which]
    result: dict[str, dict[str, Any]] = {}
    for kv, members in sorted(agg.items()):
        mm = np.asarray(members)
        result[f"{prefix}{kv:02d}" if which != "component" else f"{prefix}{kv}"] = {
            "n": int(mm.size),
            "n_sig_bh": int(reject[mm].sum()),
            "mean_median_excess": round(float(med[mm].mean()), 4),
            "max_median_excess": round(float(med[mm].max()), 4),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="exp11 T3 per-feature decomposition")
    ap.add_argument("--run", type=Path, default=Path("runs/exp11_grid.json"))
    ap.add_argument("--pooled", type=Path, default=Path("runs/exp11_pooled.json"))
    ap.add_argument(
        "--factor-directions",
        type=Path,
        default=Path(
            "~/projects/anamnesis_exps/outputs/analysis/v3_audit/factor_directions_3b.npz"
        ).expanduser(),
    )
    ap.add_argument("--anamnesis-path", type=str, default=None)
    ap.add_argument("--top", type=int, default=40, help="rows printed / banked in ranked table")
    ap.add_argument("--out", type=Path, default=Path("runs/exp11_t3_perfeature.json"))
    args = ap.parse_args()

    if not args.factor_directions.exists():
        sys.exit(f"factor directions npz not found: {args.factor_directions}")
    fd = np.load(args.factor_directions, allow_pickle=True)
    fd_names = [str(x) for x in fd["FULL_names"]]
    scale = guarded_scale(fd["FULL_scale"])

    report: dict[str, Any] = {"grids": {}, "meta": {}}
    for tag, path in (("main", args.run), ("pooled", args.pooled)):
        if not path.exists():
            print(f"[skip] {tag}: {path} not found")
            continue
        data = json.loads(path.read_text())
        names: list[str] = data["feature_names"]
        if fd_names != list(names):
            sys.exit(f"feature-name alignment FAILED for {tag} (hard requirement)")
        idx, keys = t3_index(names)
        cells = load_cell_deltas(data["cells"], scale)
        n_dlg = sum(1 for c in cells if c.regime == "dialogue")
        print(
            f"\n{'='*78}\n{tag} grid: {len(cells)} cells ({n_dlg} dlg / {len(cells)-n_dlg} doc), "
            f"T3 = {idx.size} features\n{'='*78}"
        )
        block = analyze_grid(cells, idx, keys, names)

        grid_out: dict[str, Any] = {"n_cells": len(cells), "regimes": {}}
        for label in ("dlg", "all"):
            b = block.get(label, {})
            if b.get("skipped") or b.get("n_cells", 0) == 0:
                # pooled 'all'=24 exceeds the exact-enum cap → dialogue is the reportable set
                continue
            print(
                f"\n--- {label} (n={b['n_cells']}) ---  "
                f"BH-significant T3 features: {b['n_sig_bh']}/{b['n_features']} "
                f"(p≤{b['bh_pcut']:.4g}); majority-REC-further {b['n_rec_further_majority']}; "
                f"all-cells-REC-further {b['n_all_cells_rec_further']}"
            )
            table = ranked_table(b, args.top)
            print(f"  top {min(args.top,10)} by median |Δz_REC|−|Δz_ROT|:")
            for r in table[:10]:
                lr = "  nan" if r["median_log2_ratio"] is None else f"{r['median_log2_ratio']:+.2f}"
                sig = "*" if r["bh_sig"] else " "
                print(
                    f"    {r['name']:<20} excess={r['median_absdz_excess']:+.4f} "
                    f"log2={lr}  {r['n_cells_pos']:>2}/{b['n_cells']}  p={r['p_signflip']:.4f}{sig}"
                )
            by_layer = group_breakdown(b, keys, "layer")
            by_snap = group_breakdown(b, keys, "snapshot")
            print("  by layer   :", {k: f"{v['n_sig_bh']}/{v['n']}" for k, v in by_layer.items()})
            print("  by snapshot:", {k: f"{v['n_sig_bh']}/{v['n']}" for k, v in by_snap.items()})
            # trim the heavy per-feature arrays from the persisted block; keep the ranked table
            b_out = {k: v for k, v in b.items() if k != "_per_feature"}
            b_out["ranked_top"] = table
            b_out["by_layer"] = by_layer
            b_out["by_snapshot"] = by_snap
            b_out["by_component"] = group_breakdown(b, keys, "component")
            grid_out["regimes"][label] = b_out
        report["grids"][tag] = grid_out

    report["meta"] = {
        "statistic": "per-feature abs-excess g = |Δz_REC| − |Δz_ROT|, exact one-sided sign-flip",
        "standardization": "Δz = (sig − sig_FULL) ⊘ FULL_scale (calibration per-feature std)",
        "multiple_comparison": "Benjamini-Hochberg FDR q=0.05 across the T3 feature set",
        "note": "anamnesis encoder-CV removal-cost is family-level and needs labeled samples; "
        "n=12/20 paired aggregate cells support the training-free per-feature sign-flip only",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
