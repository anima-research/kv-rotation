"""Unit tests for the pure statistics in experiments/exp11_family_decomp.py.

The experiment script is not a package module; it is loaded by file path. Tests cover
the mass-correction (RMS), the exact sign-flip permutation test, Holm adjustment, the
scale guard, and the grouping/LOFO index algebra (the pieces that are testable without
the banked run JSON or the anamnesis checkout — grouping tests skip if anamnesis-pl is
unavailable, per tests/conftest.py convention).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "exp11_family_decomp",
    Path(__file__).resolve().parent.parent / "experiments" / "exp11_family_decomp.py",
)
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod  # required for dataclass/annotation resolution
_SPEC.loader.exec_module(mod)


# ── rms (mass correction) ────────────────────────────────────────────────────────────


def test_rms_is_per_feature_mean_not_sum():
    v = np.array([3.0, 4.0])
    assert mod.rms(v) == pytest.approx(math.sqrt((9 + 16) / 2))
    # duplicating features must NOT change the statistic (mass invariance)
    assert mod.rms(np.tile(v, 100)) == pytest.approx(mod.rms(v))


def test_rms_empty_is_zero():
    assert mod.rms(np.array([])) == 0.0


# ── exact sign-flip permutation ──────────────────────────────────────────────────────


def test_signflip_all_positive_distinct():
    # all-positive g: only the identity assignment reaches the observed mean
    assert mod.signflip_p_one_sided([1.0, 2.0, 3.0]) == pytest.approx(1 / 8)


def test_signflip_symmetric_is_large():
    p = mod.signflip_p_one_sided([1.0, -1.0])
    assert p >= 0.5  # mean 0: at least half the assignments tie or exceed


def test_signflip_empty_and_cap():
    assert mod.signflip_p_one_sided([]) == 1.0
    with pytest.raises(ValueError):
        mod.signflip_p_one_sided(list(range(21)))


def test_signflip_matches_binomial_for_equal_magnitudes():
    # 4 equal positive values: assignments with mean >= obs are exactly the all-plus one
    assert mod.signflip_p_one_sided([2.0, 2.0, 2.0, 2.0]) == pytest.approx(1 / 16)


# ── Holm adjustment ──────────────────────────────────────────────────────────────────


def test_holm_monotone_and_capped():
    adj = mod.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.30})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["b"] == pytest.approx(0.08)
    assert adj["c"] == pytest.approx(0.30)
    assert adj["a"] <= adj["b"] <= adj["c"] <= 1.0


# ── scale guard / safe ratio ─────────────────────────────────────────────────────────


def test_guarded_scale_replaces_degenerate_only():
    s = mod.guarded_scale(np.array([0.0, 1e-12, 0.5, 2.0]))
    assert s[0] == 1.0 and s[1] == 1.0
    assert s[2] == 0.5 and s[3] == 2.0


def test_safe_ratio_degenerate_denominator_is_nan():
    assert math.isnan(mod.safe_ratio(1.0, 0.0))
    assert mod.safe_ratio(1.0, 2.0) == 0.5


# ── gap stats + LOFO algebra on synthetic cells ──────────────────────────────────────


def _synthetic_cells(n_cells: int = 6, n_feat: int = 10, seed: int = 0):
    """Cells where REC is exactly 2× ROT on features [0:5] and equal elsewhere."""
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(n_cells):
        rot = rng.normal(size=n_feat)
        rec = rot.copy()
        rec[:5] *= 2.0
        cells.append(
            mod.CellDeltas(
                cell_id=f"c{i}",
                regime="dialogue" if i % 2 == 0 else "doc",
                dz={"ROT": rot, "REC": rec, "NAIVE": rot * 1.1},
                dz_floor=np.full(n_feat, 0.1),
            )
        )
    return cells


def test_gap_stats_localizes_the_planted_effect():
    cells = _synthetic_cells()
    hot = mod.gap_stats(cells, np.arange(0, 5))
    cold = mod.gap_stats(cells, np.arange(5, 10))
    assert hot["all"]["median_log2"] == pytest.approx(1.0)  # exactly 2× => log2 = 1
    assert hot["all"]["n_rec_further"] == 6
    assert cold["all"]["median_log2"] == pytest.approx(0.0, abs=1e-12)
    # LOFO shape: removing the hot family kills the gap on the complement
    assert mod.gap_stats(cells, np.arange(5, 10))["all"]["n_rec_further"] == 0


def test_family_row_reports_size_and_floor_ratio():
    cells = _synthetic_cells()
    names = [f"f{i}" for i in range(10)]
    row = mod.family_row(cells, np.arange(0, 5), names)
    assert row["n"] == 5
    assert row["rms_ROT"]["xfloor_median"] > 0
    # NAIVE = 1.1×ROT: colinear within family
    assert row["cos_NAIVE_ROT"]["dlg"]["median"] == pytest.approx(1.0)


def test_top_gap_features_ranks_planted_features_first():
    cells = _synthetic_cells()
    names = [f"f{i}" for i in range(10)]
    top = mod.top_gap_features(cells, np.arange(10), names, k=5)
    assert {t["name"] for t in top} == {"f0", "f1", "f2", "f3", "f4"}


# ── grouping construction (needs anamnesis-pl) ───────────────────────────────────────


def test_build_groupings_partitions_features():
    pytest.importorskip("anamnesis.analysis.feature_map")
    names = (
        [f"pca_L14_t{t}_c{c}" for t in range(2) for c in range(3)]
        + ["activation_norm_mean_L0", "gate_sparsity_mean_L7", "cache_coverage"]
    )
    slices = {"tier3": [0, 6], "rest": [6, 9]}
    g = mod.build_groupings(names, slices)
    for gname in ("tier", "family", "source", "method"):
        total = sum(len(idx) for idx in g[gname].values())
        assert total == len(names), gname
    assert len(g["t3_layer"]["pca_L14"]) == 6
    assert set(g["t3_snapshot"]) == {"pca_t0", "pca_t1"}
