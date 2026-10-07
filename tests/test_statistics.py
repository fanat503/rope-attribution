"""Tests for the grid-independent attribution statistics.

The point of these tests is not to pin a number. It is to pin the *properties*
that distinguish a real measurement from a sampling artefact, so that the
artefact cannot be quietly reinstated.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rope_attribution import statistics as S

EXACT_TOL = 1e-12


# --------------------------------------------------------------------------
# configuration is fixed in advance, not tuned
# --------------------------------------------------------------------------


def test_the_distance_range_is_declared_and_fixed() -> None:
    assert S.DELTA_LO == 1
    assert S.DELTA_HI == 8192
    assert S.ENDPOINT_A == S.DELTA_LO and S.ENDPOINT_B == S.DELTA_HI
    # The endpoints must be inside the range, or "endpoint ratio" is meaningless.
    assert S.DELTA_LO < S.DELTA_HI


def test_the_grid_ladder_spans_enough_densities_to_expose_sampling_effects() -> None:
    assert len(S.GRID_DENSITIES) >= 5
    assert max(S.GRID_DENSITIES) / min(S.GRID_DENSITIES) >= 100, (
        "the density ladder must span at least two orders of magnitude, or the "
        "artefact cannot be demonstrated"
    )
    assert list(S.GRID_DENSITIES) == sorted(S.GRID_DENSITIES)


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_points", [5, 17, 54, 200])
def test_the_grid_is_increasing_within_range_and_has_distinct_points(n_points: int) -> None:
    grid = S._log_uniform_grid(n_points)
    assert grid.size >= 3
    assert np.all(np.diff(grid) > 0), "distances must be strictly increasing"
    assert grid.min() == S.DELTA_LO
    assert grid.max() == S.DELTA_HI
    assert grid.min() >= S.DELTA_LO and grid.max() <= S.DELTA_HI


# --------------------------------------------------------------------------
# the artefact
# --------------------------------------------------------------------------


def test_the_disputed_ratio_is_not_a_property_of_the_function() -> None:
    """The max/min ratio must vary wildly with density. That is the whole point.

    If this test ever starts failing because the ratio became stable, that would
    mean the quantity stopped being an artefact - and the paper's framing would
    need revisiting rather than this test.
    """
    rows = S.ratio_vs_density()
    ratios = [row["max_min_ratio"] for row in rows]
    assert all(r > 1.0 for r in ratios)

    independence = S.grid_independence()
    drift = independence["max_min_ratio_disputed"]["relative_drift"]
    assert drift > 1.0, (
        f"the max/min ratio drifted by only {drift:.3g} across grid densities; "
        "the critique of it depends on it being unstable"
    )

    # And it is not monotone in density, which is why quoting one grid is
    # meaningless rather than merely imprecise.
    increases = sum(
        1 for a, b in zip(ratios, ratios[1:], strict=False) if b > a
    )
    assert 0 < increases < len(ratios) - 1, (
        "expected the ratio to move non-monotonically with density; got "
        f"{ratios}"
    )


def test_ratio_vs_density_covers_the_declared_ladder() -> None:
    rows = S.ratio_vs_density()
    assert len(rows) == len(S.GRID_DENSITIES)
    assert [row["requested_n"] for row in rows] == list(S.GRID_DENSITIES)
    for row in rows:
        assert row["n_deltas"] >= 3
        assert row["log10_ratio"] == pytest.approx(math.log10(row["max_min_ratio"]))


# --------------------------------------------------------------------------
# the statistics that replace it
# --------------------------------------------------------------------------


def test_endpoint_statistic_is_exactly_grid_independent() -> None:
    """It evaluates at two fixed distances, so density cannot move it."""
    entry = S.grid_independence()["endpoint_log_ratio"]
    assert entry["relative_drift"] == 0.0
    assert len(set(entry["values_by_density"])) == 1


def test_sign_crossing_statistic_is_exactly_grid_independent() -> None:
    entry = S.grid_independence()["sign_crossing_fraction"]
    assert entry["relative_drift"] == 0.0


@pytest.mark.parametrize("name", ["cv_magnitude", "log_dynamic_range", "endpoint_log_ratio"])
def test_every_statistic_is_finite(name: str) -> None:
    deltas, contrib, _ = S.contribution_grid(n_points=54)
    values = S._all_statistics(deltas, contrib)[name]
    assert np.all(np.isfinite(values)), f"{name} produced non-finite values"


def test_cv_magnitude_is_a_coefficient_of_variation() -> None:
    """Checked against the definition, not against a stored number."""
    deltas, contrib, _ = S.contribution_grid(n_points=54)
    values = S._all_statistics(deltas, contrib)["cv_magnitude"]
    magnitude = np.abs(contrib)
    for i, reported in enumerate(values):
        expected = magnitude[i].std() / magnitude[i].mean()
        assert reported == pytest.approx(expected, rel=1e-12)


def test_sign_crossing_fraction_detects_a_constructed_sign_change() -> None:
    """The statistic must respond to a real sign change and ignore a flat one."""
    deltas = S._log_uniform_grid(54)
    changing = np.array([[1.0, -1.0] * (deltas.size // 2 + 1)])[:, : deltas.size]
    flat = np.ones((1, deltas.size))
    assert S._sign_crossing_fraction(changing) == 1.0
    assert S._sign_crossing_fraction(flat) == 0.0


def test_sign_crossing_fraction_is_a_proportion() -> None:
    deltas, contrib, _ = S.contribution_grid(n_points=54)
    value = S._sign_crossing_fraction(contrib)
    assert 0.0 <= value <= 1.0
    assert value * contrib.shape[0] == pytest.approx(round(value * contrib.shape[0]))


# --------------------------------------------------------------------------
# seed variance: the error bars the repository did not have
# --------------------------------------------------------------------------


def test_the_disputed_ratio_has_no_usable_seed_precision() -> None:
    """Across seeds its spread exceeds its own mean. This is why it is withdrawn.

    If a future change made this ratio precise, the paper's original framing
    would become defensible again and this test should fail loudly.
    """
    stats = S.seed_variance(n_seeds=8)["statistics"]
    entry = stats["max_min_ratio"]
    assert entry["relative_std"] > 0.5, (
        f"the max/min ratio now has {entry['relative_std']:.2%} seed variance; "
        "the paper's withdrawal of it may no longer be warranted"
    )


def test_a_stable_statistic_really_is_stable_across_seeds() -> None:
    """The replacement statistic must be tight, or it is no better than the ratio."""
    stats = S.seed_variance(n_seeds=8)["statistics"]
    assert stats["cv_magnitude_mean"]["relative_std"] < 0.25
    assert stats["sign_crossing_fraction_mean"]["relative_std"] == 0.0


def test_seed_variance_reports_ordering_within_every_row() -> None:
    report = S.seed_variance(n_seeds=4)
    assert len(report["seeds"]) == 4
    seeds = [row["seed"] for row in report["seeds"]]
    assert len(set(seeds)) == 4, "seeds must be distinct"
    for row in report["seeds"]:
        assert row["additivity_residual_max"] >= 0.0
    for entry in report["statistics"].values():
        assert entry["min"] <= entry["mean"] <= entry["max"]
        assert entry["std"] >= 0.0


def test_seed_variance_std_is_zero_for_a_single_seed() -> None:
    report = S.seed_variance(n_seeds=1)
    for entry in report["statistics"].values():
        assert entry["std"] == 0.0
        assert entry["relative_std"] == 0.0


def test_different_seeds_give_different_measurements() -> None:
    a = S.contribution_grid(seed=11)[1]
    b = S.contribution_grid(seed=12)[1]
    assert not np.allclose(a, b)


def test_the_same_seed_is_reproducible() -> None:
    a = S.contribution_grid(seed=11)[1]
    b = S.contribution_grid(seed=11)[1]
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# additivity is preserved on every grid, since the attribution must stay exact
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_points", [5, 54, 200])
def test_additivity_holds_on_every_grid_density(n_points: int) -> None:
    _, _, residuals = S.contribution_grid(n_points=n_points)
    assert residuals.max() < 1e-10, (
        f"the feature attribution stopped being additive at n_points={n_points}: "
        f"residual {residuals.max():.3e}"
    )


# --------------------------------------------------------------------------
# the written report
# --------------------------------------------------------------------------


def test_run_all_is_json_serialisable_and_complete() -> None:
    result = S.run_all(n_seeds=3)
    assert set(result) == {
        "configuration",
        "ratio_vs_density",
        "grid_independence",
        "seed_variance",
    }
    json.dumps(result)  # must not raise
    cfg = result["configuration"]
    assert cfg["head_dim"] == S.HEAD_DIM
    assert cfg["delta_range"] == [S.DELTA_LO, S.DELTA_HI]
    assert cfg["n_seeds"] == 3
    assert set(result["grid_independence"]) == set(S.STATISTICS) | {"max_min_ratio_disputed"}


def test_main_writes_the_report_it_prints() -> None:

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "projects"), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-m", "rope_attribution.statistics"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = root / "results" / "statistics.json"
    assert out.exists(), "the module must write results/statistics.json when run"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["configuration"]["head_dim"] == S.HEAD_DIM
    assert len(data["ratio_vs_density"]) == len(S.GRID_DENSITIES)
    # stdout is informational; the JSON file is the contract this test checks.