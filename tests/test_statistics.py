"""Tests for the grid-independent attribution statistics.

The point of these tests is not to pin a number. It is to pin the *properties*
that distinguish a real measurement from a sampling artefact, so that the
artefact cannot be quietly reinstated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from rope_attribution import statistics as S

EXACT_TOL = 1e-12
#: Absolute ceiling for a value that *is* the float64 roundoff of subtracting
#: two analytically equal expressions. Its digits are whatever order the host's
#: summation picked, so a relative test on it is meaningless; the claim the
#: paper makes is "below the noise floor", not "equal to the digits I got".
#: The ceiling is five orders above the observed residuals (3.5e-14 to
#: 7.1e-14) and five below a real conservation failure (1e-9).
NOISE_FLOOR = 1e-12
#: Relative tolerance for a quotient whose divisor is a cancellation residue.
#: ``max_min_ratio`` divides the largest contribution by the smallest cell of
#: that same cancellation-accumulated sum, so BLAS order moves it by ~2e-11
#: relative -- far above the results window, far below any convention change
#: (the paper quotes the ratio to four digits).  This is the same three-class
#: policy as ``tests/test_paper_claims.py::_leaves``, applied to the statistics
#: artifact rather than to measurements.json.
AMPLIFIED_TOL = 1e-9
#: Absolute tolerance for a base-10 logarithm of a ratio whose denominator is a
#: cancellation residue.  ``position_free_error_min`` can be ~3e-3 while the
#: numerator is O(1), so a 1e-15 change in the denominator moves the logarithm
#: by O(1e-13); the observed Windows/Linux gap on the same seeded run was 0.079
#: (15.068 vs 15.147), and py3.11 on e4f9c37 moved it to 15.033, a 0.114 gap.
#: The logarithm is a diagnostic, not a quoted quantity.  A 0.2 window admits
#: both and still rejects a 0.3 change (a 2x change in the ratio).
LOG_RATIO_ABS = 0.2
#: Absolute tolerance for ``...log10_ratio_max.relative_drift``, the
#: ``(max - min) / min`` of two near-equal cancellation-dominated logarithms.
#: The quantity is a difference of noise, so its own noise is the order of its
#: value: py3.10 on 4bd653f returned 0.009742 against the committed 0.014393, a
#: 32% relative gap and 4.7e-3 in absolute units. A relative window is
#: meaningless there, so the leaf is compared absolutely, and the window is
#: capped by the claim it protects rather than by the observed noise:
#: 0.014393 + 0.03 = 0.044 < 0.05, so every value this window admits still
#: satisfies the paper's ``relative_drift < 0.05``. The fallback
#: (``LOG_RATIO_ABS`` = 0.2) would admit 0.064, which does not.
DRIFT_ABS = 0.03


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


def test_the_withdrawn_ladder_matches_the_figure_it_came_from() -> None:
    """The 920x and its density ladder are the same vector set.

    An earlier draft quoted the *pooled* spread while naming the per-feature
    ratio, and the test compared the paper against the pooled number. This pins
    the per-feature value to the committed fig09 sidecar and keeps the two
    statistics apart.
    """
    import csv
    from collections import defaultdict
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "fig09_position_conditional_attribution.csv"
    )
    curves: dict[int, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            curves[int(row["feature_index"])].append(abs(float(row["contribution"])))
    ratios = [max(values) / min(values) for values in curves.values() if min(values) > 0]
    assert max(ratios) == pytest.approx(920.044450, rel=1e-6)

    ladder = S.withdrawn_ratio_ladder()
    assert ladder["seed"] == S.FIG09_SEED
    assert ladder["rows"][-1]["worst_feature_ratio"] > 10 * ladder["rows"][0]["worst_feature_ratio"]
    assert ladder["spread_worst_feature"] != pytest.approx(ladder["spread_pooled"])
    assert ladder["seed_sweep"]["worst_feature_relative_std"] < 1.0
    assert ladder["seed_sweep"]["pooled_relative_std"] > 1.0


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
        "withdrawn_ratio_ladder",
        "grid_independence",
        "position_free_error",
        "seed_variance",
    }
    json.dumps(result)  # must not raise
    cfg = result["configuration"]
    assert cfg["head_dim"] == S.HEAD_DIM
    assert cfg["delta_range"] == [S.DELTA_LO, S.DELTA_HI]
    assert cfg["n_seeds"] == 3
    assert set(result["grid_independence"]) == set(S.STATISTICS) | {"max_min_ratio_disputed"}


def test_the_position_free_residual_is_fifteen_orders_of_magnitude() -> None:
    """The replacement headline: how badly does position-free approximation fail?

    This is the number that took over from the withdrawn ratio, so its magnitude
    and its sign of reasoning both matter. It must be enormous, and it must be
    enormous for every feature rather than for one lucky one.
    """
    result = S.position_free_error()
    for row in result["by_density"]:
        assert row["log10_ratio_max"] > 14.0, row
        assert row["log10_ratio_min"] > 10.0, row
        assert row["n_features_reversing"] == S.N_FEATURES, row


def test_the_position_free_residual_does_not_degenerate_with_grid_density() -> None:
    """The property the withdrawn ratio lacked, and the reason it replaced it.

    ``max/min`` grows without bound because its denominator is a minimum of a
    continuous function. This statistic's denominator is an additive residual and
    its numerator is bounded by the feature's own magnitude, so the quotient has
    to stay put. If this ever starts drifting, the paper's replacement claim is
    no better than the artefact it replaced, and this test is what will say so.
    """
    result = S.position_free_error()
    band = result["log10_ratio_max"]
    assert band["relative_drift"] < 0.05, band
    # Compare with the artefact on the same footing.
    ratios = [row["max_min_ratio"] for row in S.ratio_vs_density()]
    artefact_drift = (max(ratios) - min(ratios)) / min(ratios)
    assert band["relative_drift"] < 0.1 * artefact_drift, (
        band["relative_drift"],
        artefact_drift,
    )


def test_the_position_free_residual_is_what_the_mean_summary_actually_incurs() -> None:
    """The claim is about the mean summary specifically, and the mean is L2-optimal.

    The paper does not claim the mean is the best summary in every norm, only
    that summarising by it costs what is measured. This test pins the part that
    can be pinned: the per-feature mean is the least-squares optimum among all
    position-independent constants, so the reported squared error is a lower
    bound on what *any* position-free choice incurs in that norm. The reported
    max-norm error is simply what the mean incurs, and is large.
    """
    _deltas, contrib, _residuals = S.contribution_grid(n_points=54)
    mean = contrib.mean(axis=1, keepdims=True)
    residual = float(_residuals.max())

    def squared_error(summary: np.ndarray) -> float:
        return float(((contrib - summary) ** 2).sum())

    baseline = squared_error(mean)
    for scale in (0.5, 0.9, 1.1, 2.0):
        assert squared_error(scale * mean) >= baseline - 1e-9
    # A shifted per-feature summary is worse still.
    assert squared_error(mean + 1.0) > baseline

    # The measured quantity is the max-norm error, and it is not roundoff.
    assert np.abs(contrib - mean).max() / residual > 1e14
    # It is also large in the norm where the mean is optimal, so no norm choice
    # rescues a position-free summary.
    assert np.sqrt(baseline / contrib.size) / residual > 1e13


def _tolerances(where: str) -> tuple[float, float]:
    """``(relative, absolute)`` for the leaf at ``where``; seven classes.

    Ordinary results keep the strict window. Error floors and amplified
    quotients get the tolerances the module docstring of ``_leaves`` in
    ``tests/test_paper_claims.py`` explains; both are properties of float64
    summation order, not of the measurement. The fourth class is the base-10
    logarithm of a ratio with a near-zero denominator: it is compared on the
    log scale because a relative window there is a different quantity. The
    fifth is a relative standard deviation across seeds, a second moment of
    ratios whose individual values inherit the same ordering noise. The sixth
    is the withdrawn ratio ladder as a block: every leaf in it is a quotient of
    the same near-zero sums, so the block as a whole moves with the platform.
    The seventh is the relative drift of the log-ratio summary, a difference of
    two noise-dominated logarithms; it is compared in absolute units, with the
    window capped by the claim it protects (``DRIFT_ABS``).
    """
    if where.endswith(".additivity_residual_max"):
        return 0.0, NOISE_FLOOR
    if where.startswith("root.withdrawn_ratio_ladder") and "relative_std" not in where:
        # Every number in this block is a quotient of the same sweeps, and the
        # block is the statistic the paper *withdrew*: it measures the grid, not
        # the model. The noise is 1.43e-12 relative on Linux (4937.569559608477
        # against 4937.569559615532), which the ordinary window rejects by a
        # hair; nothing here is quoted, and no value is used by a claim. The
        # ``relative_std`` leaves keep their own clause below, so that dropping
        # that clause is still caught by the CI pair rather than absorbed here.
        return AMPLIFIED_TOL, 1e-15
    if where.endswith(".log10_ratio"):
        # ``log10(max_min_ratio)``: the same amplified quotient, compressed.
        return AMPLIFIED_TOL, 1e-15
    if "log_dynamic_range" in where or "endpoint_log_ratio" in where:
        # Two more base-10 logarithms of min-sensitive ratios (``log10(max/min)``
        # and ``log10(|c(hi)|/|c(lo)|)``). CI py3.11 moved one per-density
        # window from 0.9692215094603904 to 0.9692215094615737, 1.18e-12
        # relative, just outside the ordinary window; the quantity has the same
        # cancellation-residue denominator as ``log10_ratio`` above.
        return AMPLIFIED_TOL, 1e-15
    if where.endswith(".relative_drift"):
        # ``(max - min) / min`` of two near-equal logs: a difference of noise.
        # Compared in absolute units; see DRIFT_ABS for why the cap is 0.03.
        return 0.0, DRIFT_ABS
    if "log10_ratio_max" in where or "log10_ratio_min" in where:
        # A ratio against a near-zero cancellation residue; see LOG_RATIO_ABS.
        # The block has two shapes: a float per density (``...by_density[3]
        # .log10_ratio_max``) and a summary dict at the top level
        # (``...log10_ratio_max.min``, ``.max``). The relative-drift leaf of
        # that summary has its own clause above, because a relative window on a
        # difference of two logs is a different quantity.
        return 0.0, LOG_RATIO_ABS
    if where.endswith("relative_std"):
        # Std of per-seed ratios (paper: 0.93x and 1.13x their means). The
        # ratios are quotients of near-zero sums, so their second moment moves
        # by ~3e-12 between BLAS builds. Quoted to two digits; a real change is
        # orders above this window. Every relative std in the artifact is this
        # object, including the four in ``seed_variance.statistics`` (the
        # endpoint mean crosses zero, so its relative std is 2.8).
        return AMPLIFIED_TOL, 1e-15
    if where.endswith(".position_free_error_min") or where.endswith(
        ".position_free_error_max"
    ):
        # Order statistics over the same per-cell errors. The minimum is a
        # near-zero cell (0.0143), so its relative spread is larger than the
        # ordinary window: CI py3.10 moved it by 3.55e-14 absolute, 2.5e-12
        # relative. Neither leaf is a quoted literal; the paper quotes the band
        # and the max/min stability ratio instead.
        return AMPLIFIED_TOL, 1e-15
    if "max_min_ratio" in where:
        return AMPLIFIED_TOL, 1e-15
    return EXACT_TOL, 1e-15


def _assert_json_close(computed, committed, where: str = "root") -> None:
    """Recursively compare a computed result with the committed JSON.

    Floats must agree to the leaf's tolerance (see ``_tolerances``); a changed
    committed value, a renamed field, or an extra row is a failure. That is the
    point: the test exists to detect drift, not to re-write the artifact and
    agree with itself.
    """
    if isinstance(committed, dict):
        assert isinstance(computed, dict), where
        assert computed.keys() == committed.keys(), f"{where}: keys differ"
        for key in committed:
            _assert_json_close(computed[key], committed[key], f"{where}.{key}")
    elif isinstance(committed, list):
        assert isinstance(computed, list), where
        assert len(computed) == len(committed), f"{where}: length differs"
        for index, (got, want) in enumerate(zip(computed, committed, strict=True)):
            _assert_json_close(got, want, f"{where}[{index}]")
    elif isinstance(committed, bool) or not isinstance(committed, (int, float)):
        assert computed == committed, where
    else:
        rel, abs_ = _tolerances(where)
        assert computed == pytest.approx(committed, rel=rel, abs=abs_), where


def test_the_committed_statistics_json_is_current(tmp_path: Path) -> None:
    """Recompute, compare value by value, and leave the artifact untouched.

    The previous version ran the module as a subprocess, which *overwrote*
    ``results/statistics.json`` and then read the file it had just written: it
    could never detect a stale artifact. ``main`` now accepts an explicit
    output path, so this test writes to ``tmp_path`` and compares against the
    committed bytes.
    """
    root = Path(__file__).resolve().parents[1]
    committed_path = root / "results" / "statistics.json"
    before = committed_path.read_bytes()
    committed = json.loads(before)
    out = tmp_path / "statistics.json"
    S.main(out=out)
    _assert_json_close(json.loads(out.read_text(encoding="utf-8")), committed)
    assert committed_path.read_bytes() == before, "the test must not rewrite the artifact"


def test_the_tolerances_admit_platform_noise_and_still_reject_staleness() -> None:
    """The three classes are not a blanket exemption for the whole file.

    The pair on the left is the one CI actually produced: the same seeded
    Windows and Linux runs returned ``...5544146`` and ``...579107`` for the
    amplified quotient, and residuals of 3.55e-14 and 6.22e-14 for the same
    analytically-zero cancellation. Both must pass; a 1% change in the
    quotient, or a residual that stopped cancelling at 1e-9, must not.

    Mutation verdict: dropping the ``max_min_ratio`` clause fails the first
    assertion; widening the residual ceiling to 1e-8 fails the third; dropping
    the two log-ratio clauses fails the fourth; widening ``LOG_RATIO_ABS`` to
    1.0 fails the fifth; dropping the relative-std clause fails the sixth
    (the 3.3e-12 CI pair no longer passes); matching the log-ratio block by
    ``endswith`` again (the shape that let CI's 1.2e-2 gap through) fails the
    seventh; dropping the drift window fails the eighth (the log-ratio fallback
    is absolute 0.2 and would admit a drift of 0.064, past the paper's 0.05);
    dropping the
    withdrawn-ladder block clause fails the ninth (its 1.43e-12 CI pair no
    longer passes); dropping the log-dynamic-range clause fails the tenth (its
    1.18e-12 py3.11 pair no longer passes); restricting the relative-std clause
    to the two ladder leaf names fails the eleventh (a ``seed_variance``
    relative std falls through to the exact window); dropping the
    position-free-order-statistic clause fails the twelfth (its 2.5e-12 py3.10
    pair no longer passes).
    """
    committed = {
        "ratio_vs_density": [{"max_min_ratio": 1111.7298865544146}],
        "position_free_error": {
            "by_density": [{"additivity_residual_max": 3.552713678800501e-14}]
        },
    }
    runner = {
        "ratio_vs_density": [{"max_min_ratio": 1111.729886579107}],
        "position_free_error": {
            "by_density": [{"additivity_residual_max": 6.217248937900877e-14}]
        },
    }
    _assert_json_close(runner, committed)

    stale_ratio = {
        "ratio_vs_density": [{"max_min_ratio": 1111.7298865544146 * 1.01}],
        "position_free_error": committed["position_free_error"],
    }
    with pytest.raises(AssertionError):
        _assert_json_close(stale_ratio, committed)

    stale_residual = {
        "ratio_vs_density": committed["ratio_vs_density"],
        "position_free_error": {
            "by_density": [{"additivity_residual_max": 1e-9}]
        },
    }
    with pytest.raises(AssertionError):
        _assert_json_close(stale_residual, committed)

    # The two leaves CI actually moved on the same seeded run: log10 of a
    # quotient (3.045999280925157 vs 3.045999280934803) and log10 of a ratio
    # whose denominator is a cancellation residue (15.147 vs 15.033 on py3.11).
    # Both are diagnostics, not quoted claims; the windows are chosen from the
    # observed platform gap, and a 1% / 0.3 change must still be rejected.
    ci_logs = {
        "ratio_vs_density": [{"log10_ratio": 3.045999280925157}],
        "position_free_error": {
            "by_density": [{"log10_ratio_max": 15.147178543837066}]
        },
    }
    runner_logs = {
        "ratio_vs_density": [{"log10_ratio": 3.045999280934803}],
        "position_free_error": {
            "by_density": [{"log10_ratio_max": 15.033235191530233}]
        },
    }
    _assert_json_close(runner_logs, ci_logs)

    stale_log_ratio = {
        "ratio_vs_density": [{"log10_ratio": 3.045999280925157 + 0.01}],
        "position_free_error": ci_logs["position_free_error"],
    }
    with pytest.raises(AssertionError):
        _assert_json_close(stale_log_ratio, ci_logs)

    stale_log_ratio_max = {
        "ratio_vs_density": ci_logs["ratio_vs_density"],
        "position_free_error": {
            "by_density": [{"log10_ratio_max": 15.147178543837066 + 0.3}]
        },
    }
    with pytest.raises(AssertionError):
        _assert_json_close(stale_log_ratio_max, ci_logs)

    # The same ratio, in the *other* shape the artifact uses: a top-level
    # summary dict of ``min``/``max``/``relative_drift``. CI on 7f436ca moved
    # ``position_free_error.log10_ratio_max.min`` from the committed 15.019034
    # to 15.006799, a 1.2e-2 gap that the exact window rejected. On 4bd653f
    # py3.10 moved ``relative_drift`` from 0.014393 to 0.009742: 32% relative,
    # 4.7e-3 absolute, so the leaf is compared absolutely (DRIFT_ABS). The min
    # must still reject a 0.3 change, and a drift past the paper's 0.05 bound
    # must still be rejected even though the log-ratio fallback would take it.
    ci_summary = {
        "position_free_error": {
            "log10_ratio_max": {
                "min": 15.019034227272984,
                "max": 15.235210318507315,
                "relative_drift": 0.014393474837535034,
            }
        }
    }
    runner_summary = {
        "position_free_error": {
            "log10_ratio_max": {
                "min": 15.006799770855972,
                "max": 15.235210318507315,
                "relative_drift": 0.009742267533322128,
            }
        }
    }
    _assert_json_close(runner_summary, ci_summary)

    for stale in (
        {"min": 15.019034227272984 + 0.3},
        {"relative_drift": 0.014393474837535034 + 0.05},
    ):
        block = dict(ci_summary["position_free_error"]["log10_ratio_max"])
        block.update(stale)
        with pytest.raises(AssertionError):
            _assert_json_close(
                {"position_free_error": {"log10_ratio_max": block}}, ci_summary
            )

    # The third quantity CI moved on: a relative standard deviation across
    # seeds. Same seeded run, two builds: 0.9265496496875988 vs
    # 0.9265496496845552, a 3.3e-12 relative gap. Must pass; a 1e-4 change
    # (far below anything a reader would notice in the quoted 0.93) must not.
    seed_std = {
        "withdrawn_ratio_ladder": {
            "seed_sweep": {"worst_feature_relative_std": 0.9265496496875988}
        }
    }
    runner_seed_std = {
        "withdrawn_ratio_ladder": {
            "seed_sweep": {"worst_feature_relative_std": 0.9265496496845552}
        }
    }
    _assert_json_close(runner_seed_std, seed_std)

    stale_seed_std = {
        "withdrawn_ratio_ladder": {
            "seed_sweep": {"worst_feature_relative_std": 0.9265496496875988 * 1.0001}
        }
    }
    with pytest.raises(AssertionError):
        _assert_json_close(stale_seed_std, seed_std)

    # The CI failure that sent this block to its own class: the same seeded
    # sweep on Linux returned 4937.569559608477 for the pooled mean against the
    # committed 4937.569559615532, a 1.43e-12 relative gap. The ordinary window
    # rejected it by a hair; the block's window admits it, a 1e-4 change does
    # not.
    ladder = {
        "withdrawn_ratio_ladder": {
            "seed_sweep": {
                "worst_feature_mean": 1145.068905459464,
                "pooled_mean": 4937.569559615532,
            }
        }
    }
    runner_ladder = {
        "withdrawn_ratio_ladder": {
            "seed_sweep": {
                "worst_feature_mean": 1145.068905459464,
                "pooled_mean": 4937.569559608477,
            }
        }
    }
    _assert_json_close(runner_ladder, ladder)

    stale_ladder = {
        "withdrawn_ratio_ladder": {
            "seed_sweep": {"pooled_mean": 4937.569559615532 * 1.0001}
        }
    }
    with pytest.raises(AssertionError):
        _assert_json_close(stale_ladder, ladder)

    # py3.11 moved a per-density log dynamic range: 0.9692215094615737 against
    # the committed 0.9692215094603904, 1.18e-12 relative. It is another log10
    # of a min-sensitive ratio, so it gets the same window; a 1e-4 change must
    # still fail.
    grid = {
        "grid_independence": {
            "log_dynamic_range": {"values_by_density": [0.9692215094603904]},
            "endpoint_log_ratio": {"values_by_density": [-0.6182788131496617]},
        }
    }
    runner_grid = {
        "grid_independence": {
            "log_dynamic_range": {"values_by_density": [0.9692215094615737]},
            "endpoint_log_ratio": {"values_by_density": [-0.6182788131496617]},
        }
    }
    _assert_json_close(runner_grid, grid)

    stale_grid = {
        "grid_independence": {
            "log_dynamic_range": {"values_by_density": [0.9692215094603904 * 1.0001]},
            "endpoint_log_ratio": {"values_by_density": [-0.6182788131496617]},
        }
    }
    with pytest.raises(AssertionError):
        _assert_json_close(stale_grid, grid)

    # The relative-std class is not just the two ladder leaves. This one lives
    # in ``seed_variance.statistics`` and is 2.8 because the endpoint mean
    # changes sign across seeds; a 3e-12 relative gap must pass, 1e-4 must not.
    seed_block = {
        "seed_variance": {
            "statistics": {
                "endpoint_log_ratio_mean": {"relative_std": 2.8372034570410545}
            }
        }
    }
    runner_seed_block = {
        "seed_variance": {
            "statistics": {
                "endpoint_log_ratio_mean": {
                    "relative_std": 2.8372034570410545 * (1.0 + 3e-12)
                }
            }
        }
    }
    _assert_json_close(runner_seed_block, seed_block)

    stale_seed_block = {
        "seed_variance": {
            "statistics": {
                "endpoint_log_ratio_mean": {
                    "relative_std": 2.8372034570410545 * 1.0001
                }
            }
        }
    }
    with pytest.raises(AssertionError):
        _assert_json_close(stale_seed_block, seed_block)

    # A minimum over cells is an order statistic: py3.10 moved this one by
    # 3.55e-14 absolute (2.5e-12 relative), just outside the ordinary window.
    # It is not a quoted literal; the CI pair must pass, a 1e-4 change must
    # not.
    free_min = {
        "position_free_error": {
            "by_density": [{"position_free_error_min": 0.014299891029988743}]
        }
    }
    runner_free_min = {
        "position_free_error": {
            "by_density": [{"position_free_error_min": 0.014299891029953216}]
        }
    }
    _assert_json_close(runner_free_min, free_min)

    stale_free_min = {
        "position_free_error": {
            "by_density": [
                {"position_free_error_min": 0.014299891029988743 * 1.0001}
            ]
        }
    }
    with pytest.raises(AssertionError):
        _assert_json_close(stale_free_min, free_min)
