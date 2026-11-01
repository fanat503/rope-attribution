"""The threshold lemma, its vectorised implementation, and the dilution claim.

Every test here is exact: no tolerances beyond the float64 round-off of the
formulas themselves, and no dependence on a trained checkpoint. The expensive
checkpoint measurement is covered by ``tests/test_collapse_artifact.py``.

Mutation verdicts
-----------------
* ``test_error_is_exactly_one_past_threshold`` fails if the peak/trough
  detection in ``phase_extremes`` is removed (the endpoint maximum then misses
  ``+1``) or if the threshold comparison is ``>`` instead of ``>=``.
* ``test_centred_window_attains_the_bound`` fails if the sub-threshold bound is
  written as ``1 - cos(x)`` instead of ``1 - cos(x/2)`` --- the two agree only at
  ``x = 0``.
* ``test_no_placement_beats_the_bound`` fails if the bound is claimed for the
  wrong window geometry (e.g. ``span`` used where ``span + 1`` is the length).
* ``test_pair_table_matches_pair_profile`` fails if the vectorised path swaps
  ``min``/``max`` or drops the ``+ WINDOW_LO`` offset.
* ``test_score_dilution`` fails if the unrotated tail is omitted from the score
  denominator, which is the whole point of the partial-RoPE explanation.
"""

from __future__ import annotations

import numpy as np
import pytest

from rope_attribution import position_free_collapse as C


def test_phase_extremes_sees_a_full_period() -> None:
    low, high = C.phase_extremes(0.3, 0.3 + 2.0 * np.pi)
    assert high == pytest.approx(1.0, abs=1e-12)
    assert low == pytest.approx(-1.0, abs=1e-12)


def test_phase_extremes_uses_endpoints_below_a_period() -> None:
    low, high = C.phase_extremes(-0.5, 0.5)
    assert high == pytest.approx(1.0, abs=1e-12)  # contains the peak at 0
    assert low == pytest.approx(np.cos(0.5), abs=1e-12)


def test_threshold_span_inverts_the_criterion() -> None:
    theta = 0.037
    span = C.threshold_span(theta)
    assert theta * span == pytest.approx(C.TWO_PI, rel=1e-12)
    assert C.pair_profile(theta, 0.0, int(np.ceil(span)))["past_threshold"] is True
    assert C.pair_profile(theta, 0.0, int(np.floor(span)) - 1)["past_threshold"] is False


@pytest.mark.parametrize("x", [2.0 * np.pi, 7.0, 12.0, 40.0, 500.0])
@pytest.mark.parametrize("psi", [0.0, 0.7, 2.9, -1.3, 6.2])
def test_error_is_exactly_one_past_threshold(x: float, psi: float) -> None:
    """The window contains a full turn, so the best constant is zero and error is 1."""
    theta = 0.01
    span = int(np.ceil(x / theta))
    prof = C.pair_profile(theta, psi, span)
    assert prof["past_threshold"] is True
    assert prof["rel_minimax"] == pytest.approx(1.0, abs=1e-12)
    assert prof["minimax"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("x", [1.0, 2.0, 3.1, 5.0, 6.2])
def test_centred_window_attains_the_bound(x: float) -> None:
    """Centring on a peak gives exactly ``(1 - cos(x/2)) / 2``."""
    theta = 1.0
    span = int(round(x))
    # The window is [1, 1 + span]; its centre is 1 + span/2, so a peak there
    # means psi = theta * (1 + span / 2).
    psi = theta * (1.0 + span / 2.0)
    prof = C.pair_profile(theta, psi, span)
    real_x = theta * span
    assert prof["rel_minimax"] == pytest.approx((1.0 - np.cos(real_x / 2.0)) / 2.0, abs=1e-12)


def test_no_placement_beats_the_bound() -> None:
    """The centred window is optimal, so every phase offset is at least as bad."""
    rng = np.random.default_rng(11)
    theta = 0.031
    for span in (8, 40, 200, 700):
        bound = (1.0 - np.cos(theta * span / 2.0)) / 2.0
        for _ in range(50):
            psi = float(rng.uniform(-np.pi, np.pi))
            prof = C.pair_profile(theta, psi, span)
            if prof["past_threshold"]:
                assert prof["rel_minimax"] == pytest.approx(1.0, abs=1e-12)
            else:
                assert prof["rel_minimax"] >= bound - 1e-12


def test_upper_edge_of_the_achievable_band_is_attained() -> None:
    """``E_rel`` spans exactly ``[(1-cos(x/2))/2, sin(min(x,pi)/2)]``.

    The upper end is ``sin(x/2)`` below ``pi`` (a window centred on an inflection
    point) and ``1`` above it (a window containing a peak and a trough). Both are
    checked by constructing the placement that attains them, so this is a
    two-sided statement rather than a bound.
    """
    rng = np.random.default_rng(17)
    theta = 0.021
    for span in (3, 10, 40, 120, 200, 290):
        x = theta * span
        upper = np.sin(min(x, np.pi) / 2.0)
        if x <= np.pi:
            # Centre the window on an inflection point of cos: then the extremes
            # are the two endpoints and the range is 2*sin(x/2).
            psi = theta * (C.WINDOW_LO + span / 2.0) - np.pi / 2.0
        else:
            # Start the window at a peak: an interval of length at least pi then
            # contains both a peak and a trough, so the range is exactly 2.
            psi = theta * C.WINDOW_LO
        prof = C.pair_profile(theta, float(psi), span)
        assert prof["rel_minimax"] == pytest.approx(upper, abs=1e-9)
        # And no placement exceeds it.
        for _ in range(60):
            other = float(rng.uniform(-np.pi, np.pi))
            assert C.pair_profile(theta, other, span)["rel_minimax"] <= upper + 1e-12


def test_the_band_collapses_to_one_at_the_threshold() -> None:
    """At ``x = 2*pi`` the best and worst placements coincide and both equal 1."""
    theta = 0.5
    span = int(np.ceil(C.TWO_PI / theta))
    rng = np.random.default_rng(19)
    for _ in range(30):
        psi = float(rng.uniform(-np.pi, np.pi))
        prof = C.pair_profile(theta, psi, span)
        assert prof["past_threshold"] is True
        assert prof["rel_minimax"] == pytest.approx(1.0, abs=1e-12)
        assert prof["minimax"] == pytest.approx(0.0, abs=1e-12)


def test_least_squares_is_never_better_than_minimax() -> None:
    rng = np.random.default_rng(5)
    for _ in range(200):
        theta = float(rng.uniform(1e-3, 1.0))
        psi = float(rng.uniform(-np.pi, np.pi))
        span = int(rng.integers(1, 4096))
        prof = C.pair_profile(theta, psi, span)
        assert prof["rel_mean"] >= prof["rel_minimax"] - 1e-12


def test_pair_table_matches_pair_profile() -> None:
    rng = np.random.default_rng(3)
    freqs = np.sort(rng.uniform(1e-3, 1.0, size=6))[::-1]
    psi = rng.uniform(-np.pi, np.pi, size=freqs.size)
    spans = (8, 64, 512)
    table = C.pair_table(freqs, psi, spans)
    for i, theta in enumerate(freqs):
        for j, span in enumerate(spans):
            ref = C.pair_profile(float(theta), float(psi[i]), int(span))
            assert table["x"][i, j] == pytest.approx(ref["x"], rel=1e-12)
            assert table["rel_minimax"][i, j] == pytest.approx(ref["rel_minimax"], abs=1e-12)
            assert table["rel_mean"][i, j] == pytest.approx(ref["rel_mean"], abs=1e-12)
            assert bool(table["past_threshold"][i, j]) is ref["past_threshold"]


def test_threshold_table_counts_the_fast_pairs() -> None:
    """A steeper ladder has more pairs past the threshold at the same window."""
    fast = np.array([1.0, 0.5, 0.25, 0.125])
    slow = fast * 1e-3
    table_fast = C._threshold_table(fast, (64, 4096))
    table_slow = C._threshold_table(slow, (64, 4096))
    assert table_fast[0]["fraction_past_threshold"] == 1.0
    assert table_slow[0]["fraction_past_threshold"] == 0.0
    assert table_slow[1]["fraction_past_threshold"] == 0.0


def test_score_dilution_is_the_unrotated_tail() -> None:
    """A large unrotated tail hides the per-pair error from the score-level rate."""
    rng = np.random.default_rng(7)
    rope_dim = 8
    head_dim = 32
    q = np.zeros(head_dim)
    k = np.zeros(head_dim)
    q[:rope_dim] = rng.normal(size=rope_dim)
    k[:rope_dim] = rng.normal(size=rope_dim)
    # Same rotary content, but the unrotated tail is scaled up 100x. The scale
    # is set explicitly rather than drawn, because a random tail of the same
    # order as the rotary block would not be a dilution test at all.
    q_tail = q.copy()
    k_tail = k.copy()
    q_tail[rope_dim:] = 100.0 * rng.normal(size=head_dim - rope_dim)
    k_tail[rope_dim:] = 100.0 * rng.normal(size=head_dim - rope_dim)

    freqs = np.array([1.0, 0.5, 0.25, 0.125])
    from rope_attribution.experiments import pair_terms

    def profile(qq: np.ndarray, kk: np.ndarray) -> dict:
        terms = pair_terms(qq[:rope_dim], kk[:rope_dim], freqs, 1)
        return C.score_profile(
            qq, kk, terms.aligned, terms.crossed, freqs, rope_dim, 64
        )

    small_tail = profile(q, k)
    big_tail = profile(q_tail, k_tail)
    # The pair-level error is a property of the rotary block and does not move.
    assert big_tail["pair_rel_mean"] == pytest.approx(small_tail["pair_rel_mean"], rel=1e-12)
    # The score-level error collapses because the denominator grew.
    assert big_tail["score_rel"] < 0.05 * small_tail["score_rel"]
    assert big_tail["tail_share"] > 0.9


def test_bins_saturate_past_the_threshold() -> None:
    x = np.array([1e3, 1e4, 1e3])
    minimax = np.array([1.0, 1.0, 1.0])
    mean = np.array([1.2, 1.2, 1.2])
    past = np.array([1, 1, 1])
    bins = C._bin_table(x, minimax, mean, past)
    assert bins and all(b["rel_minimax_median"] == 1.0 for b in bins)
    assert all(b["past_threshold_fraction"] == 1.0 for b in bins)
