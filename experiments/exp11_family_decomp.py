"""exp11 (follow-up) — feature-FAMILY decomposition of the ROT-vs-REC separation.

Motivation (Luxia, 2026-07-10, same night as the grid): the per-tier readout in
`exp11_analysis.py` said "the effect lives in T3 residual-PCA" — but T3 is 1,250 of the
2,713 dims. Tiers are lossy groupings, and a family can win a pooled statistic by PURE
FEATURE MASS rather than per-feature strength. This script re-does the decomposition the
way the anamnesis June-15 machinery does it (anamnesis-pl `analysis/v3_audit/`:
`feature_decomp.py` standalone/LOFO/removal-cost, `surface_fusion_loso.py` unique-vs-
redundant, `feature_map.py` source×method×depth vocabulary), adapted to exp11's n=12
paired-cell geometry:

  1. MASS CORRECTION — everything is computed in per-feature STANDARDIZED space
     (Δz = (sig_c − sig_FULL) ⊘ FULL_scale from `factor_directions_3b.npz`, the repo's
     calibration-based per-feature scale; means cancel in the difference) and every
     family statistic is a PER-FEATURE mean, not a sum:
     RMS_F = ‖Δz_F‖ / √|F|. A 1,250-feature family cannot win by count. Family size is
     reported next to every statistic. Robustness: the whole ranking is recomputed under
     a floor-based per-feature scale (per-feature RMS of the banked FLOOR_B deltas).
  2. UNIQUE vs REDUNDANT — leave-one-family-out (LOFO) recomputation of the REC-vs-ROT
     separation on the complement of each family ("does removing T3 kill the gap?") and
     on the family alone ("does T3 alone carry it?"), following feature_decomp.py's
     standalone/LOFO shape. No training involved — these are recomputed paired
     distances, so n=12 is fine; inference is EXACT sign-flip enumeration (2^n).
  3. FINER-THAN-TIER GROUPING — the decomposition is repeated at the feature_map
     vocabulary level (legacy families, source, source×band, method) plus a T3-internal
     sub-decomposition by PCA layer and by snapshot index.
  4. PER-FAMILY FLOORS — each family's condition RMS is reported as a ratio to that
     family's own FLOOR_B (path-floor replay) RMS in the same cell.
  5. NAIVE AXIS PER FAMILY — cos(Δz_NAIVE, Δz_ROT) / cos(Δz_NAIVE, Δz_REC) /
     cos(Δz_ROT, Δz_REC) restricted to each family: is the headline contamination axis
     (NAIVE colinear with ROT, REC off-axis) family-uniform or driven by specific
     families?

No encoders: the banked grid stores one aggregate 2,713-dim signature per condition per
cell (no per-step feature rows), and n=12 cells cannot support cell-level encoder CV.
See the journal note for the per-step recapture option.

Ownership: anamnesis-pl is a read-only PYTHONPATH dependency (feature_map only; pure
numpy/pydantic). Nothing there is modified.

Usage:
    .venv/bin/python experiments/exp11_family_decomp.py \
        --run runs/exp11_grid.json \
        --factor-directions ~/projects/anamnesis_exps/outputs/analysis/v3_audit/factor_directions_3b.npz \
        --out runs/exp11_family_decomp.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CONDS = ("ROT", "REC", "NAIVE")
MECHANICAL_RE = re.compile(r"cache_coverage|recency|lookback")
T3_RE = re.compile(r"^pca_L(\d+)_t(\d+)_c(\d+)$")
SCALE_EPS = 1e-9


# ── pure helpers (unit-tested in tests/test_exp11_family_decomp.py) ────────────────


def guarded_scale(scale: np.ndarray, eps: float = SCALE_EPS) -> np.ndarray:
    """Per-feature scale with degenerate (≈0) entries replaced by 1.0 (no rescale).

    Mirrors sigbridge.mode_projection's guard; returns a copy.
    """
    s = np.asarray(scale, dtype=np.float64).copy()
    bad = np.abs(s) < eps
    s[bad] = 1.0
    return s


def rms(v: np.ndarray) -> float:
    """Root-mean-square over features: the mass-corrected magnitude (per-feature mean,
    not sum — ‖v‖/√n)."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(v * v)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(a @ b / (na * nb))


def signflip_p_one_sided(g: Sequence[float]) -> float:
    """Exact sign-flip permutation p-value for mean(g) > 0 (paired, small n).

    Enumerates all 2^n sign assignments of the paired differences; p = fraction of
    assignments whose mean ≥ the observed mean. The observed assignment is included,
    so p ≥ 1/2^n (valid test). n is capped at 20 to bound memory.
    """
    g_arr = np.asarray(list(g), dtype=np.float64)
    n = g_arr.size
    if n == 0:
        return 1.0
    if n > 20:
        raise ValueError(f"exact enumeration capped at n=20, got {n}")
    signs = np.array(
        [[1.0 if (i >> k) & 1 else -1.0 for k in range(n)] for i in range(2**n)]
    )
    dist = signs @ g_arr / n
    obs = float(g_arr.mean())
    return float((dist >= obs - 1e-15).mean())


def holm_adjust(pvals: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family-wise, monotone, capped at 1)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = min(1.0, running)
    return adj


def safe_ratio(num: float, den: float) -> float:
    """num/den with a guarded denominator; NaN when the denominator is degenerate."""
    if den < 1e-15:
        return float("nan")
    return num / den


# ── groupings ───────────────────────────────────────────────────────────────────────


def build_groupings(
    names: list[str], tier_slices: Mapping[str, Sequence[int]]
) -> dict[str, dict[str, np.ndarray]]:
    """All family groupings as {grouping_name: {family_name: index_array}}.

    tier          — the run's banked tier_slices (back-compat with exp11_analysis)
    family        — feature_map legacy families (finer: splits T2, isolates hand fams)
    source        — feature_map Source axis (substrate being read)
    source_band   — Source × depth band (early/mid/late)
    method        — feature_map Method axis (operator type)
    t3_layer      — T3 residual-PCA sub-decomposed by layer
    t3_snapshot   — T3 residual-PCA sub-decomposed by snapshot index (gen position)
    """
    from anamnesis.analysis.feature_map import MODEL_LAYERS, FeatureMap

    fm = FeatureMap(names, MODEL_LAYERS["3b"])
    unc = fm.unclassified()
    if unc:
        raise SystemExit(f"feature_map failed to classify {len(unc)} features: {unc[:5]}")

    out: dict[str, dict[str, np.ndarray]] = {}
    out["tier"] = {
        fam: np.arange(a, b) for fam, (a, b) in sorted(tier_slices.items(), key=lambda kv: kv[1][0])
    }

    def collect(keyer) -> dict[str, np.ndarray]:
        d: dict[str, list[int]] = {}
        for i, t in enumerate(fm.tags):
            k = keyer(t)
            if k is not None:
                d.setdefault(k, []).append(i)
        return {k: np.asarray(v) for k, v in sorted(d.items(), key=lambda kv: -len(kv[1]))}

    out["family"] = collect(lambda t: t.family)
    out["source"] = collect(lambda t: t.source.value)
    out["source_band"] = collect(
        lambda t: f"{t.source.value}/{t.band.value if t.band else 'none'}"
    )
    out["method"] = collect(lambda t: t.method.value)

    t3_layer: dict[str, list[int]] = {}
    t3_snap: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        m = T3_RE.match(n)
        if m:
            t3_layer.setdefault(f"pca_L{int(m.group(1)):02d}", []).append(i)
            t3_snap.setdefault(f"pca_t{m.group(2)}", []).append(i)
    out["t3_layer"] = {k: np.asarray(v) for k, v in sorted(t3_layer.items())}
    out["t3_snapshot"] = {k: np.asarray(v) for k, v in sorted(t3_snap.items())}
    return out


# ── per-cell standardized deltas ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class CellDeltas:
    """Standardized deltas for one cell at its base evict-fraction."""

    cell_id: str
    regime: str
    dz: dict[str, np.ndarray]  # cond -> Δz = (sig_cond − sig_FULL) ⊘ scale
    dz_floor: np.ndarray       # FLOOR_B path-floor delta, same standardization


def load_cell_deltas(
    cells_raw: list[dict[str, Any]], scale: np.ndarray
) -> list[CellDeltas]:
    out: list[CellDeltas] = []
    for raw in cells_raw:
        ef = raw["base_ef"]
        sigs = raw["signatures"]
        full = np.asarray(sigs["FULL"], dtype=np.float64)
        dz: dict[str, np.ndarray] = {}
        for cond in CONDS:
            key = f"{cond}@{ef:.2f}"
            if key not in sigs:
                raise SystemExit(f"{raw['cell_id']}: missing condition {key}")
            dz[cond] = (np.asarray(sigs[key], dtype=np.float64) - full) / scale
        floor = (np.asarray(sigs["FLOOR_B"], dtype=np.float64) - full) / scale
        out.append(
            CellDeltas(cell_id=raw["cell_id"], regime=raw["regime"], dz=dz, dz_floor=floor)
        )
    return out


# ── family statistics ────────────────────────────────────────────────────────────────


def gap_stats(cells: list[CellDeltas], idx: np.ndarray) -> dict[str, Any]:
    """REC-vs-ROT separation on a feature subset: per-cell log2(RMS_REC/RMS_ROT),
    favorable-cell counts and exact sign-flip p, split by regime."""
    per_cell: dict[str, float] = {}
    for c in cells:
        r_rot, r_rec = rms(c.dz["ROT"][idx]), rms(c.dz["REC"][idx])
        lr = safe_ratio(r_rec, r_rot)
        per_cell[c.cell_id] = float(np.log2(lr)) if np.isfinite(lr) and lr > 0 else float("nan")
    out: dict[str, Any] = {"per_cell_log2": per_cell}
    for label, group in (
        ("dlg", [c for c in cells if c.regime == "dialogue"]),
        ("doc", [c for c in cells if c.regime == "doc"]),
        ("all", cells),
    ):
        g = np.asarray([per_cell[c.cell_id] for c in group], dtype=np.float64)
        g = g[np.isfinite(g)]
        if g.size == 0:
            out[label] = {"n": 0}
            continue
        out[label] = {
            "n": int(g.size),
            "mean_log2": float(g.mean()),
            "median_log2": float(np.median(g)),
            "n_rec_further": int((g > 0).sum()),
            "p_signflip": signflip_p_one_sided(g),
        }
    return out


def family_row(
    cells: list[CellDeltas], idx: np.ndarray, names: list[str]
) -> dict[str, Any]:
    """One family's full statistics block (mass-corrected, floored, NAIVE-axis)."""
    row: dict[str, Any] = {"n": int(idx.size)}
    row["n_mechanical"] = int(sum(bool(MECHANICAL_RE.search(names[i])) for i in idx))

    floor_rms = {c.cell_id: rms(c.dz_floor[idx]) for c in cells}
    for cond in CONDS:
        vals = np.asarray([rms(c.dz[cond][idx]) for c in cells])
        ratios = np.asarray(
            [safe_ratio(rms(c.dz[cond][idx]), floor_rms[c.cell_id]) for c in cells]
        )
        finite = ratios[np.isfinite(ratios)]
        row[f"rms_{cond}"] = {
            "mean": float(vals.mean()),
            "xfloor_median": float(np.median(finite)) if finite.size else float("nan"),
            "xfloor_min": float(finite.min()) if finite.size else float("nan"),
            "xfloor_max": float(finite.max()) if finite.size else float("nan"),
        }
    row["rms_floor_mean"] = float(np.mean(list(floor_rms.values())))
    row["gap"] = gap_stats(cells, idx)

    cos_rows: dict[str, dict[str, list[float]]] = {
        "cos_NAIVE_ROT": {"dlg": [], "doc": []},
        "cos_NAIVE_REC": {"dlg": [], "doc": []},
        "cos_ROT_REC": {"dlg": [], "doc": []},
    }
    for c in cells:
        reg = "dlg" if c.regime == "dialogue" else "doc"
        cos_rows["cos_NAIVE_ROT"][reg].append(cosine(c.dz["NAIVE"][idx], c.dz["ROT"][idx]))
        cos_rows["cos_NAIVE_REC"][reg].append(cosine(c.dz["NAIVE"][idx], c.dz["REC"][idx]))
        cos_rows["cos_ROT_REC"][reg].append(cosine(c.dz["ROT"][idx], c.dz["REC"][idx]))
    for k, per in cos_rows.items():
        row[k] = {
            reg: {
                "median": float(np.median(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
            for reg, v in per.items()
            if v
        }
    return row


def top_gap_features(
    cells: list[CellDeltas], idx: np.ndarray, names: list[str], k: int = 8
) -> list[dict[str, Any]]:
    """Features (within a family) most consistently carrying the REC>ROT excess:
    ranked by median over cells of |Δz_REC| − |Δz_ROT|."""
    diffs = np.stack(
        [np.abs(c.dz["REC"][idx]) - np.abs(c.dz["ROT"][idx]) for c in cells]
    )  # (n_cells, n_feat)
    med = np.median(diffs, axis=0)
    order = np.argsort(-med)[:k]
    return [
        {
            "name": names[int(idx[j])],
            "median_absdz_excess": float(med[j]),
            "n_cells_positive": int((diffs[:, j] > 0).sum()),
        }
        for j in order
    ]


# ── main ─────────────────────────────────────────────────────────────────────────────


def analyze(
    cells: list[CellDeltas],
    groupings: dict[str, dict[str, np.ndarray]],
    names: list[str],
    n_features: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    # A. standardized global headline (reference row for every LOFO below)
    all_idx = np.arange(n_features)
    report["global"] = family_row(cells, all_idx, names)

    # B. per-grouping family tables + Holm across families within a grouping
    report["groupings"] = {}
    for gname, fams in groupings.items():
        block: dict[str, Any] = {"families": {}}
        for fam, idx in fams.items():
            block["families"][fam] = family_row(cells, idx, names)
        pv = {
            fam: r["gap"]["all"].get("p_signflip", 1.0)
            for fam, r in block["families"].items()
            if r["gap"]["all"].get("n", 0) > 0
        }
        adj = holm_adjust(pv)
        for fam, r in block["families"].items():
            if fam in adj:
                r["gap"]["all"]["p_holm"] = adj[fam]
        report["groupings"][gname] = block

    # C. LOFO (unique vs redundant) on tier + family + source groupings
    report["lofo"] = {}
    for gname in ("tier", "family", "source"):
        fams = groupings[gname]
        block = {}
        for fam, idx in fams.items():
            comp = np.setdiff1d(all_idx, idx, assume_unique=False)
            block[fam] = {
                "n_family": int(idx.size),
                "n_complement": int(comp.size),
                "gap_without_family": gap_stats(cells, comp),
                "gap_family_alone": gap_stats(cells, idx),
            }
        report["lofo"][gname] = block

    # D. top gap-carrying features for the biggest families
    report["top_features"] = {}
    for fam, idx in groupings["family"].items():
        report["top_features"][fam] = top_gap_features(cells, idx, names)

    return report


def print_tables(report: dict[str, Any], groupings: dict[str, dict[str, np.ndarray]]) -> None:
    def fmt_gap(g: dict[str, Any]) -> str:
        if g.get("n", 0) == 0:
            return "-"
        star = "*" if g.get("p_holm", g["p_signflip"]) < 0.05 else " "
        return (
            f"{g['median_log2']:+.2f} ({g['n_rec_further']}/{g['n']}, "
            f"p={g['p_signflip']:.4f}{star})"
        )

    g = report["global"]
    print("\n==== A. standardized global headline (Δz ⊘ FULL_scale, mass-corrected) ====")
    print(
        f"  all {g['n']} dims: RMS×floor median  ROT {g['rms_ROT']['xfloor_median']:.1f}  "
        f"REC {g['rms_REC']['xfloor_median']:.1f}  NAIVE {g['rms_NAIVE']['xfloor_median']:.1f}"
    )
    print(
        f"  REC-vs-ROT gap  dlg {fmt_gap(g['gap']['dlg'])}   doc {fmt_gap(g['gap']['doc'])}   "
        f"all {fmt_gap(g['gap']['all'])}"
    )
    print(
        "  cos(Δz_NAIVE,Δz_ROT) med dlg/doc: "
        f"{g['cos_NAIVE_ROT']['dlg']['median']:.3f}/{g['cos_NAIVE_ROT']['doc']['median']:.3f}"
        "   cos(Δz_NAIVE,Δz_REC): "
        f"{g['cos_NAIVE_REC']['dlg']['median']:.3f}/{g['cos_NAIVE_REC']['doc']['median']:.3f}"
        "   cos(Δz_ROT,Δz_REC): "
        f"{g['cos_ROT_REC']['dlg']['median']:.3f}/{g['cos_ROT_REC']['doc']['median']:.3f}"
    )

    for gname in ("tier", "family", "source", "source_band", "method", "t3_layer", "t3_snapshot"):
        fams = report["groupings"][gname]["families"]
        print(f"\n==== B. grouping = {gname} (mass-corrected; * = Holm<0.05 within grouping) ====")
        print(
            f"  {'family':<22} {'n':>5} {'mech':>4} | {'ROT×fl':>7} {'REC×fl':>7} {'NAI×fl':>7} | "
            f"{'gap dlg':>22} {'gap doc':>22} {'gap all':>24} | {'cosNR':>6} {'cosNE':>6} {'cosRE':>6}"
        )
        order = sorted(
            fams.items(),
            key=lambda kv: -(kv[1]["gap"]["all"].get("median_log2", 0.0) or 0.0),
        )
        for fam, r in order:
            cnr = r["cos_NAIVE_ROT"]["dlg"]["median"]
            cne = r["cos_NAIVE_REC"]["dlg"]["median"]
            cre = r["cos_ROT_REC"]["dlg"]["median"]
            print(
                f"  {fam:<22} {r['n']:>5} {r['n_mechanical']:>4} | "
                f"{r['rms_ROT']['xfloor_median']:>7.1f} {r['rms_REC']['xfloor_median']:>7.1f} "
                f"{r['rms_NAIVE']['xfloor_median']:>7.1f} | "
                f"{fmt_gap(r['gap']['dlg']):>22} {fmt_gap(r['gap']['doc']):>22} "
                f"{fmt_gap(r['gap']['all']):>24} | {cnr:>6.2f} {cne:>6.2f} {cre:>6.2f}"
            )

    print("\n==== C. LOFO — does removing the family kill the REC-vs-ROT separation? ====")
    for gname in ("tier", "family", "source"):
        print(f"  --- {gname} ---")
        print(
            f"  {'family':<22} {'n':>5} | {'without family (all cells)':>28} | "
            f"{'family alone (all cells)':>28}"
        )
        for fam, r in report["lofo"][gname].items():
            print(
                f"  {fam:<22} {r['n_family']:>5} | "
                f"{fmt_gap(r['gap_without_family']['all']):>28} | "
                f"{fmt_gap(r['gap_family_alone']['all']):>28}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="exp11 family decomposition (mass-corrected)")
    ap.add_argument("--run", type=Path, default=Path("runs/exp11_grid.json"))
    ap.add_argument(
        "--factor-directions",
        type=Path,
        default=Path(
            "~/projects/anamnesis_exps/outputs/analysis/v3_audit/factor_directions_3b.npz"
        ).expanduser(),
    )
    ap.add_argument(
        "--anamnesis-path", type=str, default=None, help="local anamnesis-pl checkout"
    )
    ap.add_argument("--out", type=Path, default=Path("runs/exp11_family_decomp.json"))
    args = ap.parse_args()

    try:
        from kvrot.sigbridge import ensure_anamnesis

        ensure_anamnesis(args.anamnesis_path)
    except (ImportError, SystemExit) as e:
        sys.exit(f"anamnesis-pl not importable (needed for feature_map): {e}")

    data = json.loads(args.run.read_text())
    names: list[str] = data["feature_names"]
    tier_slices: dict[str, list[int]] = data["tier_slices"]

    if not args.factor_directions.exists():
        sys.exit(f"factor directions npz not found: {args.factor_directions}")
    fd = np.load(args.factor_directions, allow_pickle=True)
    fd_names = [str(x) for x in fd["FULL_names"]]
    if fd_names != list(names):
        sys.exit(
            f"feature-name alignment FAILED: run has {len(names)} names, "
            f"FULL_names has {len(fd_names)} (hard requirement — aborting)"
        )
    scale = guarded_scale(fd["FULL_scale"])
    n_guarded = int((np.abs(np.asarray(fd["FULL_scale"], dtype=np.float64)) < SCALE_EPS).sum())

    groupings = build_groupings(names, tier_slices)
    cells = load_cell_deltas(data["cells"], scale)
    n_dlg = sum(1 for c in cells if c.regime == "dialogue")
    print(
        f"exp11 family decomposition: {len(cells)} cells ({n_dlg} dlg / "
        f"{len(cells) - n_dlg} doc), {len(names)} features, "
        f"{n_guarded} degenerate FULL_scale entries guarded"
    )

    report = analyze(cells, groupings, names, len(names))

    # robustness: floor-based per-feature scale (instrument's own noise as the ruler)
    floor_deltas = np.stack([c.dz_floor for c in cells]) * scale  # back to raw units
    floor_scale_raw = np.sqrt(np.mean(floor_deltas**2, axis=0))
    n_dead = int((floor_scale_raw < SCALE_EPS).sum())
    floor_scale = floor_scale_raw.copy()
    floor_scale[floor_scale < SCALE_EPS] = float(
        np.median(floor_scale[floor_scale >= SCALE_EPS])
    )
    cells_fs = load_cell_deltas(data["cells"], floor_scale)
    rob: dict[str, Any] = {"n_features_floor_scale_imputed": n_dead, "groupings": {}}
    for gname in ("tier", "family", "source"):
        rob["groupings"][gname] = {
            fam: {
                "n": int(idx.size),
                "gap_all": gap_stats(cells_fs, idx)["all"],
            }
            for fam, idx in groupings[gname].items()
        }
    rob["global_gap_all"] = gap_stats(cells_fs, np.arange(len(names)))["all"]
    report["robustness_floor_scale"] = rob

    print_tables(report, groupings)

    print("\n==== robustness: floor-based per-feature scale ====")
    print(f"  ({n_dead} features with ~zero floor variance imputed to median floor RMS)")
    for gname in ("family", "source"):
        print(f"  --- {gname} ---")
        for fam, r in sorted(
            rob["groupings"][gname].items(),
            key=lambda kv: -(kv[1]["gap_all"].get("median_log2", 0.0) or 0.0),
        ):
            ga = r["gap_all"]
            if ga.get("n", 0):
                print(
                    f"  {fam:<22} n={r['n']:>5}  median_log2={ga['median_log2']:+.2f}  "
                    f"({ga['n_rec_further']}/{ga['n']}, p={ga['p_signflip']:.4f})"
                )

    report["meta"] = {
        "run": str(args.run),
        "factor_directions": str(args.factor_directions),
        "n_cells": len(cells),
        "n_features": len(names),
        "n_scale_guarded": n_guarded,
        "standardization": "Δz = (sig_cond − sig_FULL) ⊘ FULL_scale (calibration per-feature std)",
        "family_statistic": "RMS = ‖Δz_F‖/√|F| (per-feature mean — mass-corrected)",
        "inference": "exact sign-flip enumeration over paired per-cell log2(RMS_REC/RMS_ROT)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
