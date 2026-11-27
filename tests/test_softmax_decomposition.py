"""Tests for the exact softmax decomposition.

Each test names the mutation that it detects.  The point of the suite is that
the decomposition is *exact*, so the tolerances are at the floating-point floor
for the algebraic identities and loose only where a finite difference is used.
"""

from __future__ import annotations

import numpy as np
import pytest
from rope_attribution import softmax_decomposition as S
from rope_attribution.real_model import HeadVectors
from rope_attribution.rope import inv_freq


def _random_pair(rng, n_pairs=6, head_dim=12):
    a = rng.standard_normal(n_pairs)
    b = rng.standard_normal(n_pairs)
    theta = inv_freq(head_dim, 10_000.0)
    return a, b, theta


def test_the_contrast_closed_form_equals_the_direct_difference() -> None:
    """``-2 rho sin(mid - psi) sin(half)`` is the same number as ``c(j)-c(k)``.

    Mutation verdict: dropping the factor 2, swapping ``mid`` and ``half``, or
    using ``psi`` with the wrong sign makes the two expressions disagree by
    O(1) and this test fails.
    """
    rng = np.random.default_rng(0)
    a, b, theta = _random_pair(rng)
    for dj, dk in [(0.0, 1.0), (1.0, 64.0), (7.0, 3.0), (0.5, 0.5)]:
        np.testing.assert_allclose(
            S.contrast_closed_form(a, b, theta, dj, dk),
            S.contrast_direct(a, b, theta, dj, dk),
            rtol=0,
            atol=1e-12,
        )


def test_a_pair_is_blind_at_one_full_period() -> None:
    """Separating the same content by ``2*pi/theta`` gives zero contrast.

    Mutation verdict: using ``sin(theta * separation)`` instead of
    ``sin(theta * separation / 2)`` moves the zero to twice the period and this
    test fails on every pair.
    """
    rng = np.random.default_rng(1)
    a, b, theta = _random_pair(rng)
    period = 2.0 * np.pi / theta
    np.testing.assert_allclose(
        S.contrast_closed_form(a, b, theta, 0.0, period), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        S.contrast_closed_form(a, b, theta, 3.0, 3.0 + period), 0.0, atol=1e-12
    )


def test_the_contrast_is_maximal_at_half_a_period() -> None:
    """The absolute contrast peaks at ``theta * separation = pi``.

    The midpoint is held at zero and the content phase at ``pi/2``, so the
    midpoint factor is exactly one and the remaining dependence is
    ``|sin(theta * separation / 2)|``.  Mutation verdict: a sign error inside
    the sine, or a missing absolute value, moves the sampled maximum away from
    half a period.
    """
    theta = np.array([1.0, 0.3, 0.07])
    a = np.zeros(3)
    b = np.ones(3)
    half = np.pi / theta
    for pair in range(3):
        grid = np.linspace(0.05 * half[pair], 2.5 * half[pair], 600)
        values = np.abs(
            S.contrast_closed_form(
                a[pair : pair + 1],
                b[pair : pair + 1],
                theta[pair : pair + 1],
                -grid / 2.0,
                grid / 2.0,
            )
        )
        assert abs(grid[int(np.argmax(values))] - half[pair]) < 0.02 * half[pair]


def test_log_odds_reconstruct_the_score_difference() -> None:
    """``sum_p (c_jp - c_0p) + tail_j - tail_0 == s_j - s_0`` exactly.

    Mutation verdict: dropping the tail, anchoring at the wrong key, or
    summing the un-centred contributions leaves an O(1) residual.
    """
    rng = np.random.default_rng(3)
    head_dim, n_keys = 12, 9
    q = rng.standard_normal(head_dim)
    k = rng.standard_normal((n_keys, head_dim))
    freqs = inv_freq(8, 10_000.0)
    table = S.score_matrix(q, k, freqs, 8, n_keys - 1)
    delta = S.log_odds_matrix(table["pair_scores"], anchor=0)
    rebuilt = delta.sum(axis=0) + (table["tail"] - table["tail"][0])
    np.testing.assert_allclose(
        rebuilt, table["scores"] - table["scores"][0], rtol=0, atol=1e-12
    )


def test_the_odds_ratio_factorises_over_pairs() -> None:
    """``p_j/p_0 == exp(sum_p Delta_p + tail diff)`` exactly, in log space.

    Mutation verdict: applying the softmax before the decomposition, or
    normalising the per-pair factors separately, breaks the identity by the
    log-normaliser.
    """
    rng = np.random.default_rng(4)
    head_dim, n_keys = 12, 10
    q = rng.standard_normal(head_dim)
    k = rng.standard_normal((n_keys, head_dim))
    freqs = inv_freq(8, 10_000.0)
    table = S.score_matrix(q, k, freqs, 8, n_keys - 1)
    p = S.softmax_weights(table["scores"], 0.37)
    delta = S.log_odds_matrix(table["pair_scores"], anchor=0)
    log_ratio = np.log(p) - np.log(p[0])
    rebuilt = 0.37 * (
        delta.sum(axis=0) + (table["tail"] - table["tail"][0])
    )
    np.testing.assert_allclose(log_ratio, rebuilt, rtol=0, atol=1e-12)


def test_the_centered_gradient_is_the_softmax_derivative() -> None:
    """``d p_j / d w = p_j (c_j - E_p[c])`` for a weight on one pair.

    Mutation verdict: omitting the centring term leaves a residual proportional
    to ``p_j c_j``; centring by the unweighted mean instead of the
    attention-weighted mean leaves a residual of order the score spread.
    """
    rng = np.random.default_rng(5)
    n_keys, n_pairs = 11, 6
    scores = rng.standard_normal(n_keys)
    pair_scores = rng.standard_normal((n_pairs, n_keys))
    p = S.softmax_weights(scores)
    pair = 2
    eps = 1e-6
    finite = (S.softmax_weights(scores + eps * pair_scores[pair]) - p) / eps
    exact = p * S.centered_profile(pair_scores[pair : pair + 1], p)[0]
    np.testing.assert_allclose(finite, exact, rtol=0, atol=1e-6)


def test_probabilities_have_no_additive_per_pair_decomposition() -> None:
    """The mixed partial of ``sigmoid(a+b)`` is nonzero away from symmetry.

    Mutation verdict: replacing the mixed partial by its value at ``a=b=0``
    (which is zero by symmetry) would make the no-go argument vacuous; this
    test checks a point where it is genuinely nonzero and matches a finite
    difference of the sigmoid.
    """
    a = b = 0.5
    h = 1e-4

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    finite = (
        sigmoid(a + h + b + h)
        - sigmoid(a + h + b)
        - sigmoid(a + b + h)
        + sigmoid(a + b)
    ) / (h * h)
    closed = S.softmax_mixed_partial(a, b)
    assert abs(closed) > 1e-3
    np.testing.assert_allclose(closed, finite, rtol=1e-3, atol=1e-5)


def test_measure_head_reports_floor_residuals_on_random_vectors() -> None:
    """A correct implementation has nothing to hide behind.

    Mutation verdict: any of the identities above being wrong makes the
    corresponding residual O(1); the test checks each one separately so the
    failure names the broken identity.
    """
    rng = np.random.default_rng(6)
    head_dim = 16
    hv = HeadVectors(
        q_hat=rng.standard_normal((1, 1, 24, head_dim)),
        k_hat=rng.standard_normal((1, 1, 24, head_dim)),
        head_dim=head_dim,
        n_layers=1,
        n_heads=1,
    )
    row = S.measure_head(
        hv, inv_freq(head_dim, 10_000.0), head_dim, 0, 0, [23]
    )
    assert 0.0 < row["log_odds_residual"] < 1e-12
    assert 0.0 < row["odds_residual"] < 1e-12
    assert 0.0 < row["contrast_residual"] < 1e-12
    assert 0.0 < row["blind_at_period_residual"] < 1e-12
    assert 0.0 < row["centered_gradient_residual"] < 1e-6
    assert row["softcap_log_odds_residual"] is None


def test_the_softcap_residual_is_reported_not_asserted_away() -> None:
    """Gemma 2's softcap breaks additive log-odds; the module says so.

    Mutation verdict: returning zero for the softcap residual would claim an
    identity that does not hold.  With a tight cap the residual must be O(1).
    """
    rng = np.random.default_rng(7)
    head_dim = 16
    hv = HeadVectors(
        q_hat=rng.standard_normal((1, 1, 24, head_dim)),
        k_hat=rng.standard_normal((1, 1, 24, head_dim)),
        head_dim=head_dim,
        n_layers=1,
        n_heads=1,
    )
    row = S.measure_head(
        hv,
        inv_freq(head_dim, 10_000.0),
        head_dim,
        0,
        0,
        [23],
        score_scale=0.25,
        softcap=5.0,
    )
    assert row["softcap_log_odds_residual"] is not None
    assert row["softcap_log_odds_residual"] > 1e-3


def test_measure_head_rejects_an_empty_query_list() -> None:
    rng = np.random.default_rng(8)
    hv = HeadVectors(
        q_hat=rng.standard_normal((1, 1, 4, 8)),
        k_hat=rng.standard_normal((1, 1, 4, 8)),
        head_dim=8,
        n_layers=1,
        n_heads=1,
    )
    row = S.measure_head(hv, inv_freq(8), 8, 0, 0, [])
    assert row["log_odds_residual"] == 0.0


def test_softmax_weights_are_invariant_to_a_constant_shift() -> None:
    rng = np.random.default_rng(9)
    scores = rng.standard_normal(7)
    np.testing.assert_allclose(
        S.softmax_weights(scores),
        S.softmax_weights(scores + 123.0),
        rtol=1e-14,
    )


def test_softmax_weights_refuse_an_empty_score_vector() -> None:
    with pytest.raises(ValueError, match="at least one score"):
        S.softmax_weights(np.array([]))


def test_an_anchor_outside_the_key_range_is_refused() -> None:
    with pytest.raises(ValueError, match="anchor is outside"):
        S.log_odds_matrix(np.zeros((2, 3)), anchor=3)
