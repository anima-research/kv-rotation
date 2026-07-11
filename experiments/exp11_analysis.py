"""exp11 (analysis) — signature preservation: stats, tables, and P1–P6 verdicts.

Consumes the cell driver's --out JSON (full signature vectors banked per cell) and
computes everything in the amended design note
(notes/design-signature-preservation-2026-07-10.md, incl. both 2026-07-10 amendments):

  metric 1   paired d_ROT vs d_REC (Wilcoxon; scipy if available, exact sign-flip
             enumeration otherwise), per regime;
  metric 2   dist(sig_ROT, sig_REC) directly;
  metric 3   floors (a) numerics, (b) path — every distance reported as ratio to (b);
  metric 4   P5 sensitivity (d_NAIVE ≥ 10× floor b per cell);
  metric 5   fade-vs-nonsense (PRIMARY small-magnitude characterization):
             5a colinearity cos(Δ_ROT, Δ_REC); 5b-curve evict-frac fade curve on the
             swept cells; 5c-stability replication of Δ direction; 5d NAIVE placement
             relative to the fade axis;
  metric 5c' cross-cell direction consistency (SECONDARY) vs floor-delta and
             feature-permutation nulls;
  metric 5b' mode-space projection (EXPLORATORY) with its three nulls + NAIVE
             calibration read (hard FULL_names alignment assert; HAND fallback);
  metric 6   per-family breakdown with length-mechanical features flagged;
  metric 7   token-KL column next to the signature column (P3 dissociation).

Decision rules implemented verbatim: P1 Wilcoxon p<0.05 (Regime A); "≈ floor" = <3×
floor (b); TOST-style equivalence: P5 holds AND |d_ROT − d_REC| < 2× floor (b) in
≥10/12 cells; boring null additionally both d < 3× floor (b) in ≥10/12.

Usage:
    python experiments/exp11_analysis.py --run runs/exp11_grid.json \
        --factor-directions exp11_calibration/factor_directions_3b.npz \
        --out runs/exp11_analysis.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

MECHANICAL_RE = re.compile(r"cache_coverage|recency|lookback")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(a @ b / (na * nb))


# ── Wilcoxon (scipy → exact sign-flip fallback) ──────────────────────────────────


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    sx = x[order]
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def wilcoxon_paired(x: list[float], y: list[float]) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank of y vs x. Reports two-sided p and the one-sided
    p for median(y − x) > 0 (P1's direction with x = d_ROT, y = d_REC)."""
    d = np.asarray(y, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    d = d[d != 0]
    n = len(d)
    if n < 2:
        return {"method": "degenerate", "n": n, "p_two_sided": 1.0, "p_greater": 1.0}
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon

        two = scipy_wilcoxon(y, x, alternative="two-sided", zero_method="wilcox")
        gt = scipy_wilcoxon(y, x, alternative="greater", zero_method="wilcox")
        return {
            "method": "scipy.wilcoxon",
            "n": n,
            "stat": float(two.statistic),
            "p_two_sided": float(two.pvalue),
            "p_greater": float(gt.pvalue),
        }
    except ImportError:
        pass
    ranks = _avg_ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    if n <= 20:  # exact enumeration of all sign assignments
        dist = np.zeros(2**n)
        for i, signs in enumerate(itertools.product((0.0, 1.0), repeat=n)):
            dist[i] = float(np.asarray(signs) @ ranks)
        p_ge = float((dist >= w_plus - 1e-12).mean())
        p_le = float((dist <= w_plus + 1e-12).mean())
        return {
            "method": "exact sign-flip",
            "n": n,
            "stat": w_plus,
            "p_two_sided": min(1.0, 2 * min(p_ge, p_le)),
            "p_greater": p_ge,
        }
    mu = ranks.sum() / 2
    sigma = float(np.sqrt((ranks**2).sum() / 4))
    z = (w_plus - mu) / max(sigma, 1e-12)
    from math import erf

    phi = lambda v: 0.5 * (1 + erf(v / np.sqrt(2)))  # noqa: E731
    return {
        "method": "normal approx",
        "n": n,
        "stat": w_plus,
        "p_two_sided": 2 * (1 - phi(abs(z))),
        "p_greater": 1 - phi(z),
    }


# ── Per-cell geometry ─────────────────────────────────────────────────────────────


class Cell:
    def __init__(self, raw: dict[str, Any], names: list[str], slices: dict[str, list[int]]):
        self.raw = raw
        self.id: str = raw["cell_id"]
        self.regime: str = raw["regime"]
        self.base_ef: float = raw["base_ef"]
        self.sweep: bool = raw.get("sweep", False)
        self.p5_ok: bool = raw.get("p5_ok", True)
        self.names = names
        self.slices = slices
        self.sigs = {k: np.asarray(v, dtype=np.float64) for k, v in raw["signatures"].items()}
        self.full = self.sigs["FULL"]

    def key(self, base: str, ef: float | None = None, rep: bool = False) -> str:
        ef = self.base_ef if ef is None else ef
        return f"{base}@{ef:.2f}" + ("#rep" if rep else "")

    def delta(self, cond: str) -> np.ndarray:
        return self.sigs[cond] - self.full

    def d(self, cond: str) -> float:
        return float(np.linalg.norm(self.delta(cond)))

    @property
    def floor_a(self) -> float:
        return float(np.linalg.norm(self.sigs["FLOOR_A"] - self.full))

    @property
    def floor_b(self) -> float:
        return float(np.linalg.norm(self.sigs["FLOOR_B"] - self.full))

    def ratio(self, cond: str) -> float:
        return self.d(cond) / max(self.floor_b, 1e-12)

    def kl(self, cond: str) -> float:
        entry = self.raw.get("kl", {}).get(cond)
        return float(entry["mean"]) if entry else float("nan")


def family_breakdown(cell: Cell, cond: str, names: list[str]) -> dict[str, Any]:
    """Metric 6: per-family Δ share + cosine distance, mechanical features flagged."""
    delta = cell.delta(cond)
    sig_c, sig_f = cell.sigs[cond], cell.full
    total_sq = float(delta @ delta)
    out: dict[str, Any] = {}
    for fam, (a, b) in sorted(cell.slices.items(), key=lambda kv: kv[1][0]):
        dl = delta[a:b]
        out[fam] = {
            "d": float(np.linalg.norm(dl)),
            "share_of_delta_sq": float(dl @ dl / max(total_sq, 1e-24)),
            "cos_dist": 1.0 - cosine(sig_c[a:b], sig_f[a:b]),
            "n": b - a,
        }
    mech = np.array([bool(MECHANICAL_RE.search(n)) for n in names])
    dl = delta[mech]
    out["_mechanical_flagged"] = {
        "d": float(np.linalg.norm(dl)),
        "share_of_delta_sq": float(dl @ dl / max(total_sq, 1e-24)),
        "n": int(mech.sum()),
        "note": "length-mechanical features (cache_coverage/recency/lookback) — "
                "shown, not laundered into the pooled number",
    }
    return out


# ── Mode projection (exploratory 5b') ────────────────────────────────────────────


def mode_projection_block(
    cells: list[Cell], names: list[str], fd_path: Path, rng: np.random.Generator
) -> dict[str, Any]:
    from kvrot.sigbridge import load_factor_directions, mode_projection

    block: dict[str, Any] = {"status": "ok", "space": None}
    fd = None
    for space in ("FULL", "HAND"):
        try:
            cand = load_factor_directions(fd_path, space=space)
            if list(names) == list(cand.names):
                fd, block["space"] = cand, space
                break
            block[f"{space}_mismatch"] = (
                f"run names ({len(names)}) != {space}_names ({len(cand.names)})"
            )
        except Exception as e:  # noqa: BLE001
            block[f"{space}_error"] = str(e)
    if fd is None:
        block["status"] = "SKIPPED — no banked direction space aligns with the run's "
        block["status"] += "feature names (hard alignment requirement; spec 5b)"
        return block

    d_dims = len(fd.names)
    per_cell: dict[str, Any] = {}
    fracs = {"ROT": [], "REC": [], "NAIVE": [], "FLOOR_B": []}
    for cell in cells:
        entry = {}
        for base in ("ROT", "REC", "NAIVE"):
            mp = mode_projection(
                cell.delta(cell.key(base)).astype(np.float32), names, fd
            )
            entry[base] = {
                "mode_fraction": mp.mode_fraction,
                "proj_w": mp.proj_w,
                "centroid_cosines": mp.centroid_cosines,
            }
            fracs[base].append(mp.mode_fraction)
        mp_f = mode_projection(
            (cell.sigs["FLOOR_B"] - cell.full).astype(np.float32), names, fd
        )
        entry["FLOOR_B"] = {"mode_fraction": mp_f.mode_fraction}
        fracs["FLOOR_B"].append(mp_f.mode_fraction)
        per_cell[cell.id] = entry

    # Nulls: (i) floor deltas (above), (ii) isotropic 4/D, (iii) feature permutation.
    n_perm = 200
    perm_fracs: list[float] = []
    q, _ = np.linalg.qr(fd.W.T)
    scale = np.where(np.abs(fd.scale) < 1e-12, 1.0, fd.scale)
    for cell in cells:
        dz = cell.delta(cell.key("REC")) / scale
        for _ in range(n_perm // max(len(cells), 1) + 1):
            dz_p = rng.permutation(dz)
            pr = q.T @ dz_p
            perm_fracs.append(float(pr @ pr / max(dz_p @ dz_p, 1e-24)))
    block.update({
        "per_cell": per_cell,
        "mean_mode_fraction": {k: float(np.mean(v)) for k, v in fracs.items() if v},
        "nulls": {
            "floor_b_mean": float(np.mean(fracs["FLOOR_B"])) if fracs["FLOOR_B"] else None,
            "isotropic": 4.0 / d_dims,
            "feature_permutation_mean": float(np.mean(perm_fracs)),
            "feature_permutation_p95": float(np.quantile(perm_fracs, 0.95)),
        },
        "note": "EXPLORATORY: a null here is never evidence against character drift "
                "(spec 5b). NAIVE row is the calibration read.",
    })
    return block


# ── Main ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="exp11 analysis + verdicts")
    ap.add_argument("--run", type=Path, default=Path("runs/exp11_grid.json"))
    ap.add_argument("--factor-directions", type=Path,
                    default=Path("exp11_calibration/factor_directions_3b.npz"))
    ap.add_argument("--out", type=Path, default=Path("runs/exp11_analysis.json"))
    ap.add_argument("--seed", type=int, default=20260710)
    args = ap.parse_args()

    data = json.loads(args.run.read_text())
    names: list[str] = data["feature_names"]
    slices: dict[str, list[int]] = data["tier_slices"]
    cells = [Cell(c, names, slices) for c in data["cells"]]
    rng = np.random.default_rng(args.seed)
    dlg = [c for c in cells if c.regime == "dialogue"]
    doc = [c for c in cells if c.regime == "doc"]
    report: dict[str, Any] = {
        "run": str(args.run),
        "n_cells": len(cells),
        "n_features": len(names),
        "gate": data.get("gate"),
        "cells": {},
    }

    # ── Per-cell table (metrics 1–4, 5a, 7) ──
    print(f"\n==== exp11 per-cell table ({len(cells)} cells, {len(names)}-dim signature) ====")
    hdr = (f"{'cell':<28} {'P':>5} {'K':>5} | {'floor_b':>9} | "
           f"{'d_ROT':>9} {'xfl':>6} | {'d_REC':>9} {'xfl':>6} | {'d_NAIVE':>9} {'xfl':>7} | "
           f"{'d(R,R)':>8} {'cosΔΔ':>6} | {'KL_ROT':>9} {'KL_REC':>9} {'KL_NAIVE':>9} | P5")
    print(hdr)
    for c in cells:
        rot, rec, nai = c.key("ROT"), c.key("REC"), c.key("NAIVE")
        d_rr = float(np.linalg.norm(c.sigs[rot] - c.sigs[rec]))
        cos_deltas = cosine(c.delta(rot), c.delta(rec))
        k_len = c.raw["efs"][f"{c.base_ef:.2f}"]["K"]
        row = {
            "regime": c.regime, "P": c.raw["P"], "K": k_len,
            "first_eos": c.raw.get("first_eos"),
            "floor_a": c.floor_a, "floor_b": c.floor_b,
            "d_ROT": c.d(rot), "d_REC": c.d(rec), "d_NAIVE": c.d(nai),
            "ratio_ROT": c.ratio(rot), "ratio_REC": c.ratio(rec), "ratio_NAIVE": c.ratio(nai),
            "dist_ROT_REC": d_rr, "cos_dROT_dREC": cos_deltas,
            "kl_ROT": c.kl(rot), "kl_REC": c.kl(rec), "kl_NAIVE": c.kl(nai),
            "kl_floor_b": c.kl("FLOOR_B"),
            "p5_ok": c.p5_ok,
            "families_ROT": family_breakdown(c, rot, names),
            "families_REC": family_breakdown(c, rec, names),
            "families_NAIVE": family_breakdown(c, nai, names),
        }
        report["cells"][c.id] = row
        print(f"{c.id:<28} {row['P']:>5} {k_len:>5} | {c.floor_b:>9.4f} | "
              f"{row['d_ROT']:>9.4f} {row['ratio_ROT']:>6.1f} | "
              f"{row['d_REC']:>9.4f} {row['ratio_REC']:>6.1f} | "
              f"{row['d_NAIVE']:>9.4f} {row['ratio_NAIVE']:>7.1f} | "
              f"{d_rr:>8.4f} {cos_deltas:>6.3f} | "
              f"{row['kl_ROT']:>9.3e} {row['kl_REC']:>9.3e} {row['kl_NAIVE']:>9.3e} | "
              f"{'ok' if c.p5_ok else 'FAIL'}")

    # ── Pooled per-family shares (P4) ──
    print("\n==== per-family share of Δ² (pooled across cells) ====")
    fam_pool: dict[str, dict[str, list[float]]] = {}
    for c in cells:
        for base in ("ROT", "REC", "NAIVE"):
            fams = report["cells"][c.id][f"families_{base}"]
            for fam, v in fams.items():
                fam_pool.setdefault(fam, {}).setdefault(base, []).append(
                    v["share_of_delta_sq"]
                )
    report["family_shares_mean"] = {
        fam: {b: float(np.mean(vals)) for b, vals in per.items()}
        for fam, per in fam_pool.items()
    }
    for fam, per in sorted(report["family_shares_mean"].items()):
        line = "  ".join(f"{b}={v:.3f}" for b, v in sorted(per.items()))
        print(f"  {fam:<24} {line}")

    # ── Metric 1: paired Wilcoxon per regime ──
    print("\n==== metric 1: paired d_ROT vs d_REC (Wilcoxon) ====")
    report["wilcoxon"] = {}
    for label, group in (("dialogue", dlg), ("doc", doc), ("all", cells)):
        if len(group) >= 2:
            w = wilcoxon_paired([c.d(c.key("ROT")) for c in group],
                                [c.d(c.key("REC")) for c in group])
            report["wilcoxon"][label] = w
            print(f"  {label:<9} n={w['n']}: p_two={w['p_two_sided']:.4f} "
                  f"p(d_REC>d_ROT)={w['p_greater']:.4f} [{w['method']}]")

    # ── Decision rules: TOST equivalence + boring null ──
    n_total = len(cells)
    p5_all = all(c.p5_ok for c in cells)
    equiv_cells = [
        c.id for c in cells
        if c.p5_ok and abs(c.d(c.key("ROT")) - c.d(c.key("REC"))) < 2 * c.floor_b
    ]
    both_floor_cells = [
        c.id for c in cells
        if c.p5_ok and c.ratio(c.key("ROT")) < 3 and c.ratio(c.key("REC")) < 3
    ]
    equivalent = p5_all and len(equiv_cells) >= min(10, n_total)
    boring_null = equivalent and len(both_floor_cells) >= min(10, n_total)
    report["decision"] = {
        "p5_all_cells": p5_all,
        "equiv_margin_cells": equiv_cells,
        "n_equiv": len(equiv_cells),
        "both_below_3x_floor_cells": both_floor_cells,
        "tost_equivalent": equivalent,
        "boring_null": boring_null,
        "rule": "ROT≡REC iff P5 holds AND |d_ROT−d_REC|<2×floor(b) in ≥10/12; "
                "boring null adds both d<3×floor(b) in ≥10/12",
    }

    # ── Metric 5: fade vs nonsense (swept cells) ──
    print("\n==== metric 5: fade vs nonsense (swept cells) ====")
    report["fade"] = {}
    for c in [c for c in cells if c.sweep]:
        entry: dict[str, Any] = {}
        for base in ("ROT", "REC"):
            efs = sorted({c.base_ef, 0.25, 0.75})
            deltas = {ef: c.delta(c.key(base, ef)) for ef in efs
                      if c.key(base, ef) in c.sigs}
            if len(deltas) < 3:
                continue
            axis_ef = max(deltas)
            axis = deltas[axis_ef] / max(np.linalg.norm(deltas[axis_ef]), 1e-12)
            mags = {f"{ef:.2f}": float(np.linalg.norm(dl)) for ef, dl in deltas.items()}
            projs = {f"{ef:.2f}": float(dl @ axis) for ef, dl in deltas.items()}
            coss = {f"{ef:.2f}": cosine(dl, deltas[axis_ef]) for ef, dl in deltas.items()}
            sorted_efs = sorted(deltas)
            mono_mag = all(
                mags[f"{a:.2f}"] <= mags[f"{b:.2f}"] + 1e-9
                for a, b in zip(sorted_efs, sorted_efs[1:])
            )
            mono_proj = all(
                projs[f"{a:.2f}"] <= projs[f"{b:.2f}"] + 1e-9
                for a, b in zip(sorted_efs, sorted_efs[1:])
            )
            entry[base] = {
                "magnitudes": mags, "proj_on_ef075_axis": projs,
                "cos_to_ef075": coss, "monotone_magnitude": mono_mag,
                "monotone_projection": mono_proj,
            }
            # 5c-stability: replication of the base-ef delta direction
            rep_key = c.key(base, rep=True)
            if rep_key in c.sigs:
                d0, d1 = c.delta(c.key(base)), c.sigs[rep_key] - c.full
                entry[base]["stability_cos"] = cosine(d0, d1)
                entry[base]["stability_dist_vs_floor_a"] = float(
                    np.linalg.norm(c.sigs[rep_key] - c.sigs[c.key(base)])
                    / max(c.floor_a, 1e-12)
                )
        # 5d: NAIVE on/off the (REC) fade axis
        if "REC" in entry:
            axis = c.delta(c.key("REC", 0.75))
            entry["naive_cos_to_rec_fade_axis"] = cosine(c.delta(c.key("NAIVE")), axis)
            entry["naive_cos_to_rot_fade_axis"] = cosine(
                c.delta(c.key("NAIVE")), c.delta(c.key("ROT", 0.75))
            )
        report["fade"][c.id] = entry
        for base, v in entry.items():
            if isinstance(v, dict) and "magnitudes" in v:
                print(f"  {c.id} {base}: |Δ|={v['magnitudes']} cos→ef.75={ {k: round(x,3) for k,x in v['cos_to_ef075'].items()} } "
                      f"mono_mag={v['monotone_magnitude']} mono_proj={v['monotone_projection']}"
                      + (f" stab_cos={v['stability_cos']:.4f}" if "stability_cos" in v else ""))
        if "naive_cos_to_rec_fade_axis" in entry:
            print(f"  {c.id} NAIVE cos to fade axis: REC {entry['naive_cos_to_rec_fade_axis']:.3f} "
                  f"ROT {entry['naive_cos_to_rot_fade_axis']:.3f}")

    # ── Metric 5c': cross-cell direction consistency (SECONDARY) ──
    print("\n==== metric 5c (secondary): cross-cell direction consistency ====")
    report["cross_cell_consistency"] = {}
    if len(dlg) >= 3:
        def mean_pairwise_cos(vecs: list[np.ndarray]) -> float:
            pairs = [cosine(a, b) for a, b in itertools.combinations(vecs, 2)]
            return float(np.mean(pairs))

        for base in ("ROT", "REC"):
            vecs = [c.delta(c.key(base)) for c in dlg]
            stat = mean_pairwise_cos(vecs)
            floor_stat = mean_pairwise_cos([c.sigs["FLOOR_B"] - c.full for c in dlg])
            n_perm = 500
            perm = np.empty(n_perm)
            for i in range(n_perm):
                perm[i] = mean_pairwise_cos([rng.permutation(v) for v in vecs])
            p = float((perm >= stat).mean())
            report["cross_cell_consistency"][base] = {
                "mean_pairwise_cos": stat, "floor_null": floor_stat,
                "perm_null_mean": float(perm.mean()),
                "perm_null_p95": float(np.quantile(perm, 0.95)), "p_perm": p,
            }
            print(f"  Δ_{base} (dialogue): {stat:.4f}  floor-null {floor_stat:.4f}  "
                  f"perm p={p:.3f}")

    # ── Metric 5b': mode projection (EXPLORATORY) ──
    print("\n==== metric 5b (exploratory): mode-space projection ====")
    if args.factor_directions.exists():
        block = mode_projection_block(cells, names, args.factor_directions, rng)
        report["mode_projection"] = block
        if block.get("status") == "ok":
            mm = block["mean_mode_fraction"]
            nl = block["nulls"]
            print(f"  space={block['space']}  mean mode_fraction: "
                  + "  ".join(f"{k}={v:.5f}" for k, v in mm.items()))
            print(f"  nulls: floor={nl['floor_b_mean']}  iso={nl['isotropic']:.5f}  "
                  f"perm_mean={nl['feature_permutation_mean']:.5f} "
                  f"(p95 {nl['feature_permutation_p95']:.5f})")
        else:
            print(f"  {block['status']}")
    else:
        report["mode_projection"] = {"status": f"SKIPPED — {args.factor_directions} missing"}
        print(f"  skipped: {args.factor_directions} not found")

    # ── Verdicts ──
    print("\n==== VERDICTS (P1–P6, decision rules per amended spec) ====")
    verdicts: dict[str, str] = {}

    w_dlg = report["wilcoxon"].get("dialogue", {})
    p1_sig = w_dlg.get("p_greater", 1.0) < 0.05
    p1_dir = (np.median([c.d(c.key("REC")) - c.d(c.key("ROT")) for c in dlg]) > 0
              if dlg else False)
    verdicts["P1"] = (
        f"{'CONFIRMED' if (p1_sig and p1_dir) else 'NOT CONFIRMED'} — "
        f"d_ROT < d_REC paired over Regime A: one-sided p={w_dlg.get('p_greater', float('nan')):.4f} "
        f"(rule: p<0.05), median(d_REC−d_ROT)="
        f"{np.median([c.d(c.key('REC')) - c.d(c.key('ROT')) for c in dlg]) if dlg else float('nan'):.4f}"
    )

    if doc:
        doc_near_floor = [c.id for c in doc
                          if c.ratio(c.key("ROT")) < 3 and c.ratio(c.key("REC")) < 3]
        verdicts["P2"] = (
            f"{'CONFIRMED' if len(doc_near_floor) == len(doc) else 'PARTIAL/REFUTED'} — "
            f"doc cells with both ROT,REC <3× floor(b): {len(doc_near_floor)}/{len(doc)} "
            f"({doc_near_floor})"
        )
    else:
        verdicts["P2"] = "NO DOC CELLS RUN"

    # P3: comparable token-KL but REC signature divergence (operationalization stated).
    p3_cells = []
    for c in cells:
        kl_r, kl_e = c.kl(c.key("ROT")), c.kl(c.key("REC"))
        if not (np.isfinite(kl_r) and np.isfinite(kl_e)) or max(kl_r, kl_e) <= 0:
            continue
        comparable_kl = max(kl_r, kl_e) / max(min(kl_r, kl_e), 1e-12) < 2.0
        sig_diverges = (c.d(c.key("REC")) - c.d(c.key("ROT"))) > 2 * c.floor_b
        if comparable_kl and sig_diverges:
            p3_cells.append(c.id)
    verdicts["P3"] = (
        f"{'DISSOCIATION FOUND' if p3_cells else 'no dissociation'} — cells with "
        f"KL_ROT≈KL_REC (<2× apart) but d_REC−d_ROT>2×floor(b): {p3_cells or 'none'} "
        "(operationalization chosen at analysis time; spec left thresholds open)"
    )

    fam = report["family_shares_mean"]
    t2x = {b: fam.get("tier2", {}).get(b, 0) + fam.get("tier2_5", {}).get(b, 0)
           for b in ("ROT", "REC", "NAIVE")}
    t1 = {b: fam.get("tier1", {}).get(b, 0) for b in ("ROT", "REC", "NAIVE")}
    p4_holds = all(t2x[b] > t1[b] for b in ("ROT", "REC"))
    verdicts["P4"] = (
        f"{'CONFIRMED' if p4_holds else 'NOT CONFIRMED'} — mean Δ² share "
        f"T2+T2.5 vs T1: ROT {t2x['ROT']:.3f}/{t1['ROT']:.3f}, "
        f"REC {t2x['REC']:.3f}/{t1['REC']:.3f} "
        "(note: T3 and hand families carry the rest; see family table)"
    )

    p5_fails = [c.id for c in cells if not c.p5_ok]
    verdicts["P5"] = (
        f"{'CONFIRMED' if not p5_fails else 'FAILED'} — d_NAIVE ≥10× floor(b) in "
        f"{n_total - len(p5_fails)}/{n_total} cells"
        + (f"; FAILING: {p5_fails} — null readings excluded there" if p5_fails else "")
    )

    fade_summary = []
    for cid, entry in report.get("fade", {}).items():
        for base in ("ROT", "REC"):
            if base in entry and "magnitudes" in entry[base]:
                v = entry[base]
                min_cos = min(x for k, x in v["cos_to_ef075"].items())
                fade_summary.append(
                    (cid, base, v["monotone_magnitude"] and v["monotone_projection"],
                     min_cos, v.get("stability_cos"))
                )
    colin = [report["cells"][c.id]["cos_dROT_dREC"] for c in cells]
    n_mono = sum(1 for _, _, mono, _, _ in fade_summary if mono)
    min_cos = min((mc for _, _, _, mc, _ in fade_summary), default=float("nan"))
    p6_fade = bool(fade_summary) and n_mono == len(fade_summary) and min_cos > 0.7
    verdicts["P6"] = (
        f"{'FADE' if p6_fade else 'MIXED/NONSENSE — inspect fade block'} — "
        f"cos(Δ_ROT,Δ_REC) median {np.median(colin):.3f}; fade curves "
        f"{n_mono}/{len(fade_summary)} monotone; min cos→ef.75 {min_cos:.3f}; "
        f"stability cos: "
        + ", ".join(f"{s:.3f}" for *_, s in fade_summary if s is not None)
    )

    verdicts["EQUIVALENCE"] = (
        f"{'ROT ≡ REC within margin' if report['decision']['tost_equivalent'] else 'NOT equivalent'} "
        f"— |d_ROT−d_REC|<2×floor(b) in {report['decision']['n_equiv']}/{n_total} cells "
        f"(need ≥{min(10, n_total)}), P5 all: {report['decision']['p5_all_cells']}"
        + ("; BORING NULL (both <3× floor) also holds"
           if report["decision"]["boring_null"] else "")
    )

    report["verdicts"] = verdicts
    for k in ("P1", "P2", "P3", "P4", "P5", "P6", "EQUIVALENCE"):
        print(f"  {k}: {verdicts[k]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
