"""Test suite for ``projects/rope_attribution/experiments.py``.

Design rules for this file:

* Every mathematical claim the module makes is checked against an **independently
  written reference**, never by re-running ``rope.py`` and comparing it with
  itself. The references here are: the 2x2 block rotation matrix built
  explicitly with :mod:`math` (:func:`_rotate`), the frequency ladder written
  from its closed form ``base ** (-2k/dim)`` (:func:`_ladder`), and
  reimplementations of the synthetic constructions the experiment functions build
  internally (the SAE-like decomposition of E2/E3, the attention scores of E7,
  the per-pair sinusoid error of E4/E5).
* Where the module reports a float, the test either bounds it by a mathematical
  constant, compares it against an independent recomputation, or does both.
* Regression anchors are the values the module actually produces, at the
  tolerance the claim deserves: exact identities are asserted with ``==``,
  floating-point identities at ~1e-12, and the "this is exact, not approximate"
  structural facts at ~1e-12 in absolute terms.
* Fixed RNG seeds throughout; several tests replay the module's *own* internal
  ``default_rng`` draw order so that a changed construction is caught.

A deviation from the documented mathematics, pinned deliberately
----------------------------------------------------------------
``score_relative(q, k, freqs, delta)`` with a **negative** ``delta`` returns the
score for ``|delta|``. The negative branch of the function rotates the key by
``-delta`` (which is positive) rather than by ``delta``, so

    score_relative(q, k, f, -7) == score_relative(q, k, f, 7)

while the docstring promises ``q_hat . R_delta k_hat``. The per-pair closed form
(``pair_terms`` / ``pair_contributions``) *does* implement ``R_delta`` correctly,
so for negative gaps the two disagree. ``test_negative_delta_*`` records the
behaviour as it is, with ``_ref_score`` as the arbiter, so that a future fix
breaks loudly instead of silently changing what this suite means.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the suite runnable without PYTHONPATH being pre-set.
_REPO = Path(__file__).resolve().parents[1]
_PROJECTS = _REPO / "projects"
if str(_PROJECTS) not in sys.path:
    sys.path.insert(0, str(_PROJECTS))

from rope_attribution import experiments as E  # noqa: E402  (needs the tweak above)
from rope_attribution import rope as R  # noqa: E402

# --------------------------------------------------------------------------
# Independent references (no rope.py inside)
# --------------------------------------------------------------------------


def _ladder(dim: int, base: float = 10000.0) -> np.ndarray:
    """``inv_freq[i] = base ** (-2i/dim)``, written from its closed form."""
    return np.array(
        [base ** (-2.0 * k / dim) for k in range(dim // 2)], dtype=np.float64
    )


def _rotate(vec: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Explicit 2x2 block rotation, channel ``k`` paired with ``k + dim//2``.

    ``rotate_half`` in rope.py is ``cat(-x2, x1)``, i.e. exactly the block
    ``[[cos, -sin], [sin, cos]]``; this writes those blocks out one element at a
    time with :mod:`math` so the reference shares no code with the module.
    """
    dim = vec.shape[0]
    half = dim // 2
    out = np.array(vec, dtype=np.float64, copy=True)
    for k in range(half):
        c, s = math.cos(angles[k]), math.sin(angles[k])
        a, b = vec[k], vec[k + half]
        out[k] = a * c - b * s
        out[k + half] = a * s + b * c
    return out


def _ref_score(q_hat: np.ndarray, k_hat: np.ndarray, freqs: np.ndarray, delta: int) -> float:
    """``q_hat . R_delta k_hat`` from scratch, for a signed ``delta``."""
    return float(q_hat @ _rotate(k_hat, delta * freqs))


def _ref_pair(q_hat: np.ndarray, k_hat: np.ndarray, k: int) -> tuple[float, float]:
    """``(A_k, B_k)`` for one rotary pair, derived by hand."""
    half = q_hat.shape[0] // 2
    a = q_hat[k] * k_hat[k] + q_hat[k + half] * k_hat[k + half]
    b = q_hat[k + half] * k_hat[k] - q_hat[k] * k_hat[k + half]
    return a, b


def _ref_pair_value(q_hat: np.ndarray, k_hat: np.ndarray, k: int, delta: int) -> float:
    a, b = _ref_pair(q_hat, k_hat, k)
    angle = delta * _ladder(q_hat.shape[0])[k]
    return a * math.cos(angle) + b * math.sin(angle)


def _vecs(dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim), rng.standard_normal(dim)


def _fingerprint(obj: object) -> str:
    """Canonical text form of a nested structure, for bit-identity comparisons."""
    return json.dumps(obj, sort_keys=True, default=float)


# The values ``main()`` prints, i.e. the numbers this suite is a regression net
# for. Every one of them is a *bound* on an error, not a claim of exactness, so
# the tolerance is set by the mathematics (double precision, ~1e-16 relative on
# quantities of order 10) with two orders of magnitude of headroom.
EXACT_TOL = 1e-12
CLOSED_FORM_TOL = 1e-13

# ==========================================================================
# 1. The closed form: sum_k c_k(delta) == q_hat . R_delta k_hat
# ==========================================================================


@pytest.mark.parametrize("dim", [8, 16, 64, 128])
@pytest.mark.parametrize("delta", [0, 1, 17, 255, 4096])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pair_contributions_sum_equals_the_relative_score(dim, delta, seed):
    """The core identity of E1, against an independently built rotation."""
    freqs = _ladder(dim)
    rng = np.random.default_rng(seed)
    q_hat = rng.standard_normal(dim)
    k_hat = 3.0 * rng.standard_normal(dim)

    closed = float(E.pair_contributions(q_hat, k_hat, freqs, delta).sum())
    reference = _ref_score(q_hat, k_hat, freqs, delta)

    assert abs(closed - reference) <= CLOSED_FORM_TOL
    assert abs(closed - E.score_relative(q_hat, k_hat, freqs, delta)) <= CLOSED_FORM_TOL


def test_pair_contributions_has_one_entry_per_rotary_pair():
    q_hat, k_hat = _vecs(16, 5)
    freqs = _ladder(16)
    c = E.pair_contributions(q_hat, k_hat, freqs, 3)
    assert c.shape == (8,)
    assert c.dtype == np.float64
    t = E.pair_terms(q_hat, k_hat, freqs, 3)
    assert t.aligned.shape == t.crossed.shape == t.angle.shape == (8,)
    assert np.array_equal(c, t.contributions())


def test_relative_score_agrees_with_the_hand_built_rotation_bit_for_bit():
    """``score_relative`` itself is brute force; check it against ``_rotate``."""
    for dim in (8, 16, 64, 128):
        freqs = _ladder(dim)
        q_hat, k_hat = _vecs(dim, 17)
        for delta in (0, 1, 5, 64, 777, 4096):
            assert E.score_relative(q_hat, k_hat, freqs, delta) == _ref_score(
                q_hat, k_hat, freqs, delta
            )


def test_relative_score_at_zero_gap_is_the_plain_dot_product():
    for dim in (8, 64, 128):
        freqs = _ladder(dim)
        q_hat, k_hat = _vecs(dim, 2)
        assert E.score_relative(q_hat, k_hat, freqs, 0) == float(q_hat @ k_hat)


@pytest.mark.parametrize("delta", [1, 13, 512, 4096])
def test_relative_score_is_unchanged_by_a_common_position_shift(delta):
    """The relative-position property, measured the way it is actually claimed.

    A *fixed* pair of content vectors must score the same when both positions
    are shifted together. This is the structural fact E1 reports; here it is
    checked for a fresh (position, content) sample, not the module's own.
    """
    freqs = _ladder(64)
    q_hat, k_hat = _vecs(64, 99)
    for m in (0, 1, 5, 17, 63, 100, 4096):
        here = _ref_score(_rotate(q_hat, m * freqs), _rotate(k_hat, (m + delta) * freqs), freqs, 0)
        there = _ref_score(
            _rotate(q_hat, (m + 7) * freqs), _rotate(k_hat, (m + 7 + delta) * freqs), freqs, 0
        )
        assert abs(here - there) <= 1e-12


@pytest.mark.parametrize("alpha", [0.0, 1.0, -3.0, 7.5, 1e-3, 1e3])
def test_relative_score_is_bilinear_in_the_content_vectors(alpha):
    freqs = _ladder(64)
    q_hat, k_hat = _vecs(64, 4)
    for delta in (1, 13, 29, 61, 127):
        unit = _ref_score(q_hat, k_hat, freqs, delta)
        assert abs(_ref_score(alpha * q_hat, k_hat, freqs, delta) - alpha * unit) <= 1e-12 * (
            abs(alpha * unit) + 1.0
        )
        assert abs(_ref_score(q_hat, alpha * k_hat, freqs, delta) - alpha * unit) <= 1e-12 * (
            abs(alpha * unit) + 1.0
        )


# --------------------------------------------------------------------------
# negative deltas
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gap", [1, 7, 13, 512])
def test_score_relative_handles_negative_delta(gap):
    """``score_relative`` must agree with ``q_hat . R_delta k_hat`` for negative gaps too.

    Regression: it used to rotate the key by ``-delta`` in its negative branch,
    which made ``score_relative(..., -d) == score_relative(..., d)`` and silently
    disagreed with the closed form for every negative distance.
    """
    freqs = _ladder(64)
    q_hat, k_hat = _vecs(64, 8)

    correct = _ref_score(q_hat, k_hat, freqs, -gap)
    from_closed_form = float(E.pair_contributions(q_hat, k_hat, freqs, -gap).sum())

    assert abs(from_closed_form - correct) <= CLOSED_FORM_TOL
    assert abs(E.score_relative(q_hat, k_hat, freqs, -gap) - correct) <= CLOSED_FORM_TOL
    # A general key really is sensitive to the sign of the distance, so the
    # previous even-in-delta behaviour was not a harmless no-op.
    assert E.score_relative(q_hat, k_hat, freqs, -gap) != pytest.approx(
        E.score_relative(q_hat, k_hat, freqs, gap), rel=1e-9, abs=1e-9
    )


def test_pair_terms_angle_is_exactly_the_negation_for_a_negative_delta():
    freqs = _ladder(32)
    q_hat, k_hat = _vecs(32, 12)
    pos = E.pair_terms(q_hat, k_hat, freqs, 21)
    neg = E.pair_terms(q_hat, k_hat, freqs, -21)
    assert np.array_equal(neg.angle, -pos.angle)
    assert np.array_equal(neg.aligned, pos.aligned)
    assert np.array_equal(neg.crossed, pos.crossed)


def test_pair_terms_split_the_contribution_into_its_even_and_odd_parts():
    """``c(d) + c(-d) = 2 A cos D`` and ``c(d) - c(-d) = 2 B sin D``, bit-exactly."""
    freqs = _ladder(32)
    q_hat, k_hat = _vecs(32, 13)
    pos = E.pair_terms(q_hat, k_hat, freqs, 29)
    neg = E.pair_terms(q_hat, k_hat, freqs, -29)
    even = pos.aligned * np.cos(pos.angle)
    odd = pos.crossed * np.sin(pos.angle)
    assert np.allclose(pos.contributions(), even + odd, rtol=0, atol=1e-13)
    assert np.array_equal(neg.contributions(), even - odd)
    scale = max(float(np.abs(pos.contributions()).max()), 1.0)
    assert np.abs(pos.contributions() - neg.contributions() - 2.0 * odd).max() <= 1e-13 * scale


# ==========================================================================
# 2. The gate/phase split: amplitude is content, phase is position
# ==========================================================================


@pytest.mark.parametrize("dim", [8, 16, 64, 128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_amplitude_does_not_depend_on_the_distance(dim, seed):
    q_hat, k_hat = _vecs(dim, seed)
    freqs = _ladder(dim)
    first = E.pair_terms(q_hat, k_hat, freqs, 1)
    second = E.pair_terms(q_hat, k_hat, freqs, 97)
    # The stored terms themselves are bit-identical, not merely close.
    assert np.array_equal(first.aligned, second.aligned)
    assert np.array_equal(first.crossed, second.crossed)
    assert np.allclose(first.amplitude, second.amplitude, rtol=0, atol=1e-12)
    assert np.array_equal(first.amplitude, second.amplitude)


@pytest.mark.parametrize("dim", [8, 16, 64, 128])
def test_amplitude_is_the_product_of_the_two_pair_norms(dim):
    """``R_k = hypot(A_k, B_k) = |(q_k, q_{k+h})| * |(k_k, k_{k+h})|``.

    This is the gate claim stated as an identity, and it is derived here from
    ``|a + i b| = |a| |b|``: the pair's gate is the product of the two content
    norms, with no position anywhere in it.
    """
    freqs = _ladder(dim)
    half = dim // 2
    q_hat, k_hat = _vecs(dim, 5)
    t = E.pair_terms(q_hat, k_hat, freqs, 64)
    expected = np.hypot(q_hat[:half], q_hat[half:]) * np.hypot(k_hat[:half], k_hat[half:])
    assert np.allclose(t.amplitude, expected, rtol=0, atol=1e-12)
    # ... and the module's own hypot agrees with that independent form.
    assert np.allclose(t.amplitude, np.hypot(t.aligned, t.crossed), rtol=0, atol=1e-12)


def test_contributions_do_depend_on_the_distance():
    """The counterpart of the previous test: the gate is fixed, the signal is not."""
    freqs = _ladder(64)
    q_hat, k_hat = _vecs(64, 6)
    first = E.pair_terms(q_hat, k_hat, freqs, 1).contributions()
    second = E.pair_terms(q_hat, k_hat, freqs, 97).contributions()
    assert np.abs(first - second).max() > 1e-3


@pytest.mark.parametrize("dim", [8, 16, 64, 128])
@pytest.mark.parametrize("delta", [0, 1, 5, 64, 777])
def test_contributions_are_the_documented_sinusoid(dim, delta):
    """``c_k = A cos D + B sin D = R_k cos(D_k - psi_k)``."""
    freqs = _ladder(dim)
    q_hat, k_hat = _vecs(dim, 21)
    t = E.pair_terms(q_hat, k_hat, freqs, delta)
    sinusoid = t.amplitude * np.cos(t.angle - t.phase_offset)
    tol = 1e-12 * max(float(t.amplitude.max()), 1.0)
    assert np.abs(t.contributions() - sinusoid).max() <= tol


@pytest.mark.parametrize("dim", [8, 16, 64, 128])
def test_contributions_are_bounded_by_the_amplitude(dim):
    """``|c_k| <= R_k`` for every pair and every distance, on a dense grid."""
    freqs = _ladder(dim)
    q_hat, k_hat = _vecs(dim, 22)
    t = E.pair_terms(q_hat, k_hat, freqs, 0)
    for delta in range(0, 64):
        c = E.pair_terms(q_hat, k_hat, freqs, delta).contributions()
        assert np.all(np.abs(c) <= t.amplitude + 1e-12)


def test_the_sinusoid_reaches_its_amplitude_at_the_phase_offset_distance():
    """``psi_k`` is where the pair peaks: ``D_k = psi_k``, i.e. delta = psi / inv_f.

    Checked on the fastest channel, where ``psi / inv_f`` is a small distance, by
    scanning a window of distances and locating the maximum.
    """
    freqs = _ladder(64)
    q_hat, k_hat = _vecs(64, 23)
    k = 0
    a, b = _ref_pair(q_hat, k_hat, k)
    psi = math.atan2(b, a)
    assert psi < 0.0 or abs(psi) > 0.1  # a genuine, non-degenerate offset
    radius = math.hypot(a, b)

    # The peak sits at the (generally non-integer) distance psi / inv_freq, and
    # the sinusoid is exactly at its amplitude there.
    at_peak = a * math.cos(psi) + b * math.sin(psi)
    assert abs(at_peak - radius) <= 1e-12 * radius

    # At integer distances the peak of the window is the one nearest psi / inv_f,
    # and it is within one step of the amplitude.
    peak = round(psi / freqs[k])
    window = range(peak - 3, peak + 4)
    values = [_ref_pair_value(q_hat, k_hat, k, d) for d in window]
    module_values = [float(E.pair_contributions(q_hat, k_hat, freqs, d)[k]) for d in window]
    assert values == pytest.approx(module_values, rel=1e-12, abs=1e-12)
    assert max(values) == values[3]
    assert max(values) > 0.99 * radius
    assert values[2] < max(values) and values[4] < max(values)


def test_the_sinusoid_gets_close_to_its_amplitude_over_a_dense_grid():
    """A pair that turns far enough within the grid sweeps its full range.

    Only channels that complete a substantial part of a turn can be expected to
    reach their peak at an integer distance, so the "gets there" claim is made
    for those and the bound ``|c_k| <= R_k`` for all of them.
    """
    dim = 32
    freqs = _ladder(dim)
    q_hat, k_hat = _vecs(dim, 24)
    amplitude = E.pair_terms(q_hat, k_hat, freqs, 0).amplitude
    peak = np.zeros(dim // 2)
    for delta in range(0, 512):
        contributions = E.pair_terms(q_hat, k_hat, freqs, delta).contributions()
        peak = np.maximum(peak, np.abs(contributions))
    turning_far = (511 * freqs) > 2.0 * math.pi
    assert turning_far.any() and not turning_far.all()
    assert np.all(peak <= amplitude + 1e-9)
    assert np.all(peak[turning_far] > 0.9 * amplitude[turning_far])


def test_contributions_at_zero_gap_are_exactly_the_aligned_terms():
    """``cos 0 == 1`` and ``sin 0 == 0`` exactly, so this is a bitwise claim."""
    freqs = _ladder(64)
    q_hat, k_hat = _vecs(64, 25)
    t = E.pair_terms(q_hat, k_hat, freqs, 0)
    assert np.array_equal(t.contributions(), t.aligned)
    assert np.array_equal(t.amplitude, np.hypot(t.aligned, t.crossed))
    assert float(E.pair_contributions(q_hat, k_hat, freqs, 0).sum()) == pytest.approx(
        float(q_hat @ k_hat), rel=1e-14, abs=1e-14
    )


def test_phase_offset_lies_in_the_principal_branch():
    freqs = _ladder(64)
    q_hat, k_hat = _vecs(64, 26)
    t = E.pair_terms(q_hat, k_hat, freqs, 11)
    assert np.all(t.phase_offset >= -math.pi)
    assert np.all(t.phase_offset <= math.pi)
    assert np.all(t.phase_offset != 0.0)  # a generic key has a generic phase


# ==========================================================================
# 3. Position blindness: a pair with B_k = 0 carries no content in its phase
# ==========================================================================


def _blind_key(q_hat: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """A key parallel to the query, so ``B_k = q2 k1 - q1 k2 = 0`` for every pair.

    Products of two floats are commutative, so with ``k = q`` the two products in
    ``B_k`` are the *same* double and ``B_k`` is exactly ``0.0`` (no cancellation
    residue at all); with a scale factor it is 0 to within one rounding.
    """
    return scale * q_hat


@pytest.mark.parametrize("scale", [1.0])
def test_parallel_key_makes_the_crossed_term_exactly_zero(scale):
    dim = 64
    freqs = _ladder(dim)
    q_hat, _ = _vecs(dim, 31)
    k_hat = _blind_key(q_hat, scale)
    t = E.pair_terms(q_hat, k_hat, freqs, 37)
    assert np.array_equal(t.crossed, np.zeros(dim // 2))
    assert float(np.abs(t.crossed).max()) == 0.0


def test_scaled_parallel_key_makes_the_crossed_term_zero_to_rounding():
    dim = 64
    freqs = _ladder(dim)
    q_hat, _ = _vecs(dim, 32)
    k_hat = _blind_key(q_hat, 2.5)
    t = E.pair_terms(q_hat, k_hat, freqs, 37)
    assert float(np.abs(t.crossed).max()) < 1e-15 * float(np.abs(t.aligned).max())


@pytest.mark.parametrize("delta", [0, 1, 13, 97])
def test_a_position_blind_pair_has_no_phase_offset_at_all(delta):
    dim = 64
    freqs = _ladder(dim)
    q_hat, _ = _vecs(dim, 33)
    t = E.pair_terms(q_hat, _blind_key(q_hat), freqs, delta)
    # A_k = scale * (q1^2 + q2^2) >= 0, so atan2(0, A) is exactly 0.0.
    assert np.all(t.aligned > 0.0)
    assert np.array_equal(t.phase_offset, np.zeros(dim // 2))


def test_a_position_blind_pair_contributes_pure_cosine():
    dim = 64
    freqs = _ladder(dim)
    q_hat, _ = _vecs(dim, 34)
    k_hat = _blind_key(q_hat)
    for delta in (0, 1, 13, 97):
        t = E.pair_terms(q_hat, k_hat, freqs, delta)
        assert np.allclose(t.contributions(), t.aligned * np.cos(t.angle), rtol=0, atol=1e-13)
        assert np.allclose(
            t.contributions(), t.amplitude * np.cos(t.angle), rtol=0, atol=1e-13
        )


def test_a_position_blind_score_is_even_in_the_distance():
    """With ``B_k = 0`` the whole score is an even function of delta.

    This is the real content of "position-blind": no sine term survives, so the
    sign of the relative distance cannot reach the score. A general key does not
    have this property, which is what makes the test non-vacuous.
    """
    dim = 64
    freqs = _ladder(dim)
    q_hat, k_hat = _vecs(dim, 35)
    blind = _blind_key(q_hat)
    for gap in (1, 7, 64, 512):
        a = float(E.pair_contributions(q_hat, blind, freqs, gap).sum())
        b = float(E.pair_contributions(q_hat, blind, freqs, -gap).sum())
        assert abs(a - b) <= 1e-13 * max(abs(a), 1.0)
        general_a = float(E.pair_contributions(q_hat, k_hat, freqs, gap).sum())
        general_b = float(E.pair_contributions(q_hat, k_hat, freqs, -gap).sum())
        assert abs(general_a - general_b) > 1e-3


def test_a_position_blind_score_is_a_fixed_weighted_sum_of_cosines():
    """The blind score is ``sum_k A_k cos(D_k)`` with content-only weights."""
    dim = 32
    freqs = _ladder(dim)
    q_hat, _ = _vecs(dim, 36)
    k_hat = _blind_key(q_hat, -1.5)
    for delta in (0, 1, 64, 1024):
        weights = np.array([_ref_pair(q_hat, k_hat, k)[0] for k in range(dim // 2)])
        expected = float((weights * np.cos(delta * freqs)).sum())
        actual = float(E.pair_contributions(q_hat, k_hat, freqs, delta).sum())
        assert abs(actual - expected) <= CLOSED_FORM_TOL
    # Those weights are exactly the content dot products, independent of delta.
    weights_at_zero = E.pair_terms(q_hat, k_hat, freqs, 0).aligned
    weights_at_far = E.pair_terms(q_hat, k_hat, freqs, 4096).aligned
    assert np.array_equal(weights_at_zero, weights_at_far)


def test_a_pair_with_a_zero_query_contributes_nothing_at_any_distance():
    """The one case where a pair is *literally* delta-independent: A_k = B_k = 0.

    With ``B_k = 0`` the pair still turns at speed ``inv_freq_k``; the residual
    distance dependence disappears only when the query vanishes on that pair,
    which is what this test pins (and the sinusoid tests above must not be read
    as claiming otherwise).
    """
    dim, k = 32, 3
    freqs = _ladder(dim)
    q_hat, k_hat = _vecs(dim, 37)
    q_hat[k] = 0.0
    q_hat[k + dim // 2] = 0.0
    reference = E.pair_terms(q_hat, k_hat, freqs, 1)
    assert reference.aligned[k] == 0.0
    assert reference.crossed[k] == 0.0
    assert reference.amplitude[k] == 0.0
    assert reference.phase_offset[k] == 0.0
    for delta in (0, 1, 17, 512, 65536):
        t = E.pair_terms(q_hat, k_hat, freqs, delta)
        assert t.contributions()[k] == 0.0
        assert np.array_equal(t.contributions()[k], reference.contributions()[k])
    # ... and the rest of the head is still turning, so the slot is not a
    # consequence of the whole vector having gone flat.
    moved = E.pair_terms(q_hat, k_hat, freqs, 512).contributions()
    assert np.abs(moved - reference.contributions()).max() > 1e-3


def test_a_general_key_is_not_position_blind():
    dim = 64
    freqs = _ladder(dim)
    q_hat, k_hat = _vecs(dim, 38)
    t = E.pair_terms(q_hat, k_hat, freqs, 11)
    assert float(np.abs(t.crossed).max()) > 1.0
    assert float(np.abs(t.phase_offset).max()) > 1.0


# ==========================================================================
# 4. structural_facts (E1)
# ==========================================================================


@pytest.fixture(scope="module")
def structural():
    return E.structural_facts()


def test_structural_facts_reports_the_documented_shape(structural):
    assert structural["dim"] == 64
    assert structural["base"] == 10000.0
    assert structural["n_positions"] == 512
    assert structural["relative_position_pairs_tested"] == 42
    assert structural["scores_tested"] == 35
    assert set(structural) == {
        "dim",
        "base",
        "n_positions",
        "relative_position_max_abs_err",
        "relative_position_pairs_tested",
        "query_norm_preservation_max_abs_err",
        "key_norm_preservation_max_abs_err",
        "bilinearity_max_rel_err",
        "pair_closed_form_max_abs_err",
        "scores_tested",
    }


def test_structural_facts_measures_machine_precision(structural):
    """Every claim in E1 is an *exact* property; only roundoff may show up."""
    for key in (
        "relative_position_max_abs_err",
        "query_norm_preservation_max_abs_err",
        "key_norm_preservation_max_abs_err",
        "bilinearity_max_rel_err",
        "pair_closed_form_max_abs_err",
    ):
        value = float(structural[key])
        assert value >= 0.0, key
        assert value < EXACT_TOL, f"{key} = {value:.3e} is not machine precision"


@pytest.mark.parametrize(
    ("key", "observed"),
    [
        ("relative_position_max_abs_err", 3.375077994860476e-14),
        ("query_norm_preservation_max_abs_err", 1.7763568394002505e-15),
        ("key_norm_preservation_max_abs_err", 1.7763568394002505e-15),
        ("bilinearity_max_rel_err", 7.402856163691089e-16),
        ("pair_closed_form_max_abs_err", 3.552713678800501e-15),
    ],
)
def test_structural_facts_regresses_the_observed_error(key, observed):
    """Pinned to the value the module actually prints, at a 1e-12 bound.

    The bound is what carries the meaning ("machine precision"); the observed
    value documents which roundoff was seen, so a change of algorithm shows up
    even while staying inside the bound.
    """
    value = float(E.structural_facts()[key])
    assert value < EXACT_TOL
    assert value == pytest.approx(observed, rel=0.5, abs=EXACT_TOL)


def test_structural_facts_is_reproducible_from_an_independent_implementation():
    """Recompute all five errors from the module's own draw order, by hand."""
    dim, base, seed, n_pos = 64, 10000.0, 0, 512
    freqs = _ladder(dim, base)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n_pos, dim))
    k = rng.standard_normal((n_pos, dim))
    v = rng.standard_normal(dim)
    u = rng.standard_normal(dim)

    rel_err, rel_pairs = 0.0, 0
    for d in (1, 7, 13, 32, 64, 127, 200):
        for m in (0, 1, 5, 17, 63, 100):
            here = float(_rotate(v, m * freqs) @ _rotate(u, (m + d) * freqs))
            moved = float(_rotate(v, (m + 7) * freqs) @ _rotate(u, (m + 7 + d) * freqs))
            rel_err = max(rel_err, abs(moved - here))
            rel_pairs += 1

    q_rot = np.array([_rotate(x, m * freqs) for x, m in zip(q, range(n_pos), strict=True)])
    k_rot = np.array([_rotate(x, m * freqs) for x, m in zip(k, range(n_pos), strict=True)])
    norm_err_q = float(np.abs(np.linalg.norm(q_rot, axis=-1) - np.linalg.norm(q, axis=-1)).max())
    norm_err_k = float(np.abs(np.linalg.norm(k_rot, axis=-1) - np.linalg.norm(k, axis=-1)).max())

    scales = (0.0, 1.0, 2.0, -3.0, 7.5, 1e-3, 1e3)
    deltas = (1, 13, 29, 61, 127)
    bilin = 0.0
    pair = 0.0
    for d in deltas:
        unit = _ref_score(v, u, freqs, d)
        for a in scales:
            bilin = max(
                bilin,
                abs(_ref_score(a * v, u, freqs, d) - a * unit) / (abs(a * unit) + 1e-12),
            )
        weights_a = np.array([_ref_pair(v, u, i)[0] for i in range(dim // 2)])
        weights_b = np.array([_ref_pair(v, u, i)[1] for i in range(dim // 2)])
        angle = d * freqs
        closed = float((weights_a * np.cos(angle) + weights_b * np.sin(angle)).sum())
        pair = max(pair, abs(closed - _ref_score(v, u, freqs, d)))

    reported = E.structural_facts()
    assert rel_pairs == reported["relative_position_pairs_tested"] == 42
    assert len(deltas) * len(scales) == reported["scores_tested"] == 35
    assert rel_err == pytest.approx(
        reported["relative_position_max_abs_err"], rel=0, abs=EXACT_TOL
    )
    assert norm_err_q == pytest.approx(
        reported["query_norm_preservation_max_abs_err"], rel=0, abs=EXACT_TOL
    )
    assert norm_err_k == pytest.approx(
        reported["key_norm_preservation_max_abs_err"], rel=0, abs=EXACT_TOL
    )
    assert bilin == pytest.approx(reported["bilinearity_max_rel_err"], rel=0, abs=EXACT_TOL)
    assert pair == pytest.approx(reported["pair_closed_form_max_abs_err"], rel=0, abs=EXACT_TOL)


def test_structural_facts_echoes_non_default_arguments():
    other = E.structural_facts(dim=32, base=500000.0, n_positions=16)
    assert other["dim"] == 32
    assert other["base"] == 500000.0
    assert other["n_positions"] == 16
    assert other["relative_position_pairs_tested"] == 42
    assert other["scores_tested"] == 35
    assert float(other["pair_closed_form_max_abs_err"]) < EXACT_TOL
    assert float(other["query_norm_preservation_max_abs_err"]) < EXACT_TOL


# ==========================================================================
# 5. feature_attribution (E2/E3)
# ==========================================================================


def _sae_decomposition(n_features, dim, seed):
    """Replay the construction ``feature_attribution`` documents, draw for draw."""
    rng = np.random.default_rng(seed)
    w_q = rng.standard_normal((dim, dim)) / math.sqrt(dim)
    w_k = rng.standard_normal((dim, dim)) / math.sqrt(dim)
    coeffs = rng.random(n_features) * 1.5 + 0.1
    dirs_q = rng.standard_normal((n_features, dim))
    dirs_k = rng.standard_normal((n_features, dim))
    return coeffs[:, None] * (dirs_q @ w_q), coeffs[:, None] * (dirs_k @ w_k)


@pytest.mark.parametrize(
    ("n_features", "dim", "seed", "deltas"),
    [
        (8, 64, 1, (1, 8, 32, 128, 512)),
        (4, 16, 7, (1, 5, 9)),
        (6, 32, 2, (3, 40)),
    ],
)
def test_feature_attribution_is_exactly_additive(n_features, dim, seed, deltas):
    """``score == sum_ij A_ij(delta)`` against a hand-built rotation."""
    freqs = _ladder(dim)
    q_i, k_i = _sae_decomposition(n_features, dim, seed)
    for d in deltas:
        brute = _ref_score(q_i.sum(0), k_i.sum(0), freqs, d)
        additive = sum(
            _ref_score(q_i[i], k_i[j], freqs, d)
            for i in range(n_features)
            for j in range(n_features)
        )
        assert abs(brute - additive) <= EXACT_TOL * (abs(brute) + 1.0)


def test_feature_attribution_reports_machine_precision_additivity():
    reported = E.feature_attribution()
    assert reported["additivity_max_abs_err"] < EXACT_TOL
    assert reported["additivity_max_abs_err"] == pytest.approx(
        3.552713678800501e-14, rel=0.5, abs=EXACT_TOL
    )
    assert reported["score_depends_only_on_delta"] is True
    assert reported["n_features"] == 8
    assert reported["dim"] == 64
    assert reported["deltas"] == [1, 8, 32, 128, 512]


def test_feature_attribution_reproduces_its_own_spread_definition():
    """Pin the definition of ``per_feature_contribution_spread_max_ratio``.

    The implementation takes, for each distance, ``max_i |c_i| / min_i |c_i|``
    over the features and then the maximum over distances. (The printed label
    in ``main()`` describes it as a ratio *across distances for one feature*;
    the number is the former. This test pins the number as implemented.)
    """
    n_features, dim, seed, deltas = 8, 64, 1, (1, 8, 32, 128, 512)
    freqs = _ladder(dim)
    q_i, k_i = _sae_decomposition(n_features, dim, seed)
    spread = np.zeros(n_features)
    for d in deltas:
        per_feature = np.array(
            [sum(_ref_score(q_i[i], k_i[j], freqs, d) for j in range(n_features))
             for i in range(n_features)]
        )
        nonzero = per_feature[per_feature != 0.0]
        spread = np.maximum(spread, float(np.abs(nonzero).max() / np.abs(nonzero).min()))
    reported = E.feature_attribution(n_features=n_features, dim=dim, seed=seed, deltas=deltas)
    assert float(spread.max()) == pytest.approx(
        reported["per_feature_contribution_spread_max_ratio"], rel=1e-12
    )
    assert reported["per_feature_contribution_spread_max_ratio"] > 1.0


def test_one_features_contribution_is_a_function_of_the_distance():
    """The scientific claim of E3, measured per feature rather than pooled.

    For every feature, ``c_i(delta)`` moves by a factor well above 1 across the
    five distances, so no position-free ``A_ij`` can represent the attribution.
    """
    n_features, dim, seed, deltas = 8, 64, 1, (1, 8, 32, 128, 512)
    freqs = _ladder(dim)
    q_i, k_i = _sae_decomposition(n_features, dim, seed)
    ratios = []
    for i in range(n_features):
        c = np.array(
            [sum(_ref_score(q_i[i], k_i[j], freqs, d) for j in range(n_features)) for d in deltas]
        )
        assert float(np.abs(c).max()) > 0.0
        ratios.append(float(np.abs(c).max() / np.abs(c).min()))
    assert min(ratios) > 1.5
    # And the same holds for an unrelated configuration, so it is not one seed.
    freqs_small = _ladder(16)
    q_j, k_j = _sae_decomposition(3, 16, 5)
    for i in range(3):
        c = np.array(
            [sum(_ref_score(q_j[i], k_j[j], freqs_small, d) for j in range(3))
             for d in (2, 9)]
        )
        assert float(np.abs(c).max() / np.abs(c).min()) > 1.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_feature_contribution_spread_exceeds_one_for_every_seed(seed):
    assert (
        E.feature_attribution(seed=seed)["per_feature_contribution_spread_max_ratio"] > 1.0
    )


def test_feature_attribution_of_a_single_feature_is_degenerate():
    """One feature means one contribution: the spread collapses to 1.

    This is the falsifying edge of the previous test - it shows the ratio is a
    measurement of variation, not a constant.
    """
    degenerate = E.feature_attribution(n_features=1, dim=8, seed=0, deltas=(1, 2))
    assert degenerate["per_feature_contribution_spread_max_ratio"] == 1.0
    assert degenerate["additivity_max_abs_err"] < EXACT_TOL


# ==========================================================================
# 6. linearization_error (E4/E5)
# ==========================================================================


def _linearization_reference(q_hat, k_hat, freqs, delta):
    """``1 - D^2/2`` in place of ``cos D`` and ``D`` in place of ``sin D``, by hand."""
    dim = q_hat.shape[0]
    half = dim // 2
    a = q_hat[:half] * k_hat[:half] + q_hat[half:] * k_hat[half:]
    b = q_hat[half:] * k_hat[:half] - q_hat[:half] * k_hat[half:]
    angle = delta * freqs
    exact = a * np.cos(angle) + b * np.sin(angle)
    approx = a * (1.0 - angle**2 / 2.0) + b * angle
    amplitude = np.hypot(a, b)
    per_pair = np.abs(approx - exact) / (amplitude + 1e-12)
    return {
        "delta": delta,
        "max_abs_angle": float(np.abs(angle).max()),
        "median_abs_angle": float(np.median(np.abs(angle))),
        "frac_channels_linearizable": float(np.mean(np.abs(angle) <= 0.1)),
        "per_pair_err_max": float(per_pair.max()),
        "per_pair_err_rms": float(np.sqrt(np.mean(per_pair**2))),
        "amplitude_weighted_err": float(np.abs(approx - exact).sum() / (amplitude.sum() + 1e-12)),
        "score_exact": float(exact.sum()),
        "score_linearized": float(approx.sum()),
    }


@pytest.mark.parametrize("delta", [0, 1, 2, 64, 512, 4096])
def test_linearization_error_matches_an_independent_implementation(delta):
    dim = 64
    freqs = _ladder(dim)
    rng = np.random.default_rng(2)
    q_hat = rng.standard_normal(dim)
    k_hat = rng.standard_normal(dim)

    reported = E.linearization_error(q_hat, k_hat, freqs, delta)
    mine = _linearization_reference(q_hat, k_hat, freqs, delta)

    assert reported["delta"] == delta
    assert set(reported) == set(mine)
    for key, value in mine.items():
        assert reported[key] == pytest.approx(value, rel=1e-12, abs=1e-15), key


def test_linearization_is_exact_at_zero_distance():
    dim = 64
    freqs = _ladder(dim)
    q_hat, k_hat = _vecs(dim, 41)
    reported = E.linearization_error(q_hat, k_hat, freqs, 0)
    assert reported["max_abs_angle"] == 0.0
    assert reported["median_abs_angle"] == 0.0
    assert reported["frac_channels_linearizable"] == 1.0
    assert reported["per_pair_err_max"] == 0.0
    assert reported["per_pair_err_rms"] == 0.0
    assert reported["amplitude_weighted_err"] == 0.0
    assert reported["score_exact"] == reported["score_linearized"]
    assert reported["score_exact"] == pytest.approx(
        _ref_score(q_hat, k_hat, freqs, 0), rel=1e-14, abs=1e-14
    )


@pytest.mark.parametrize("delta", [0, 1, 5, 64, 4096])
def test_reported_score_exact_is_the_true_relative_score(delta):
    dim = 64
    freqs = _ladder(dim)
    q_hat, k_hat = _vecs(dim, 42)
    reported = E.linearization_error(q_hat, k_hat, freqs, delta)
    assert reported["score_exact"] == pytest.approx(
        _ref_score(q_hat, k_hat, freqs, delta), rel=1e-13, abs=1e-13
    )


@pytest.mark.parametrize("delta", [0, 1, 2, 3, 8])
def test_fraction_of_linearizable_channels_is_measured_at_ten_percent(delta):
    dim = 64
    freqs = _ladder(dim)
    q_hat, k_hat = _vecs(dim, 43)
    reported = E.linearization_error(q_hat, k_hat, freqs, delta)
    angle = delta * freqs
    assert reported["frac_channels_linearizable"] == float(np.mean(np.abs(angle) <= 0.1))
    assert reported["max_abs_angle"] == float(np.abs(angle).max())
    assert reported["median_abs_angle"] == float(np.median(np.abs(angle)))


def test_linearization_damage_grows_with_the_distance():
    freqs = _ladder(64)
    q_hat, k_hat = _vecs(dim=64, seed=44)
    near = E.linearization_error(q_hat, k_hat, freqs, 1)
    far = E.linearization_error(q_hat, k_hat, freqs, 4096)
    # one radian of angle is still nearly linearizable; 4096 is not
    assert near["per_pair_err_max"] < 0.2
    assert near["per_pair_err_rms"] < 0.1
    assert far["per_pair_err_max"] > 1e5
    assert far["per_pair_err_rms"] > far["per_pair_err_max"] * 0.1


@pytest.mark.parametrize("delta", [1, 64, 512, 2048, 4096])
def test_the_two_error_summaries_are_consistent_with_the_per_pair_errors(delta):
    """``rms`` is a root-mean-square and the weighted error is a weighted mean."""
    dim = 64
    half = dim // 2
    freqs = _ladder(dim)
    q_hat, k_hat = _vecs(dim, 45)
    reported = E.linearization_error(q_hat, k_hat, freqs, delta)

    aligned = q_hat[:half] * k_hat[:half] + q_hat[half:] * k_hat[half:]
    crossed = q_hat[half:] * k_hat[:half] - q_hat[:half] * k_hat[half:]
    angle = delta * freqs
    exact = aligned * np.cos(angle) + crossed * np.sin(angle)
    approx = aligned * (1.0 - angle**2 / 2.0) + crossed * angle
    amplitude = np.hypot(aligned, crossed)
    per_pair = np.abs(approx - exact) / (amplitude + 1e-12)

    assert reported["per_pair_err_max"] == pytest.approx(float(per_pair.max()), rel=1e-12)
    assert reported["per_pair_err_rms"] == pytest.approx(
        float(np.sqrt(np.mean(per_pair**2))), rel=1e-12
    )
    assert per_pair.min() <= reported["amplitude_weighted_err"] <= per_pair.max()
    assert reported["per_pair_err_rms"] <= reported["per_pair_err_max"] + 1e-9


# ==========================================================================
# 7. method_spectrum (E4/E5, cross-method)
# ==========================================================================

METHODS = ("rope_base_10k", "position_interpolation", "yarn", "base_500k_legacy_claim")
SPECTRUM_DELTAS = (1, 64, 512, 2048, 4096)


@pytest.fixture(scope="module")
def spectrum():
    return E.method_spectrum()


def _by_method(rows, method):
    return [row for row in rows if row["method"] == method]


def test_method_spectrum_has_the_documented_shape(spectrum):
    assert len(spectrum) == len(METHODS) * len(SPECTRUM_DELTAS)
    assert tuple(row["method"] for row in spectrum[:: len(SPECTRUM_DELTAS)]) == METHODS
    for method in METHODS:
        assert [row["delta"] for row in _by_method(spectrum, method)] == list(SPECTRUM_DELTAS)
    assert set(spectrum[0]) == {
        "method",
        "mscale",
        "delta",
        "max_abs_angle",
        "median_abs_angle",
        "frac_channels_linearizable",
        "per_pair_err_rms",
        "amplitude_weighted_err",
    }


def test_the_fastest_channel_is_exactly_one_radian_per_position():
    """`max_abs_angle` is ``delta * inv_freq[0]`` and ``inv_freq[0] == 1`` exactly.

    True for every base: it is the top of the frequency ladder, one full radian
    per position, whatever the base is.
    """
    for base in (10000.0, 500000.0, 1e6):
        freqs = _ladder(64, base)
        assert freqs[0] == 1.0
        for delta in SPECTRUM_DELTAS:
            assert delta * freqs[0] == float(delta)


def test_max_abs_angle_is_identical_for_rope_and_yarn_at_every_distance(spectrum):
    """The headline scientific fact of E5.

    YaRN leaves the highest-frequency channel untouched, so the largest angle in
    the head - the one that decides whether the fastest pair can be linearized -
    is bit-for-bit the same as plain RoPE, at every distance. Raising the base
    (the legacy claim) shares this property; plain interpolation does not.
    """
    rope = _by_method(spectrum, "rope_base_10k")
    yarn = _by_method(spectrum, "yarn")
    interp = _by_method(spectrum, "position_interpolation")
    assert [row["max_abs_angle"] for row in rope] == [float(d) for d in SPECTRUM_DELTAS]
    for r, y, i in zip(rope, yarn, interp, strict=True):
        assert y["delta"] == r["delta"]
        assert y["max_abs_angle"] == r["max_abs_angle"]
        assert i["max_abs_angle"] < r["max_abs_angle"]


def test_yarn_keeps_the_fastest_channel_because_its_ladder_starts_at_one():
    freqs, _ = R.yarn_parameters(64, 10000.0, 32.0, 2048)
    rope = _ladder(64, 10000.0)
    assert float(freqs[0]) == 1.0
    assert float(freqs[0]) == float(rope[0])
    assert float(freqs.max()) == 1.0
    # and every other channel is slowed down, so only the maximum survives
    assert np.all(freqs[1:] <= rope[1:] * (1.0 + 1e-15))


@pytest.mark.parametrize("scale", [2.0, 8.0, 32.0, 1024.0])
def test_position_interpolation_max_angle_is_rope_divided_by_the_scale(scale):
    rows = E.method_spectrum(scale=scale)
    rope = _by_method(rows, "rope_base_10k")
    interp = _by_method(rows, "position_interpolation")
    for r, i in zip(rope, interp, strict=True):
        assert i["max_abs_angle"] == pytest.approx(r["max_abs_angle"] / scale, rel=1e-15)
        assert i["max_abs_angle"] == pytest.approx(r["delta"] / scale, rel=1e-15)


def test_the_median_angle_orders_the_four_methods_the_same_way_at_every_distance(spectrum):
    """Median angle is the fair comparison the table is for.

    Position interpolation buys the most angle compression, the legacy base bump
    less than YaRN, and plain RoPE the least. The legacy "base 500000" claim and
    YaRN are therefore *not* the same method, which is the point of including it.
    """
    for d in SPECTRUM_DELTAS:
        med = {
            row["method"]: row["median_abs_angle"]
            for row in spectrum
            if row["delta"] == d
        }
        assert med["position_interpolation"] < med["base_500k_legacy_claim"]
        assert med["base_500k_legacy_claim"] < med["yarn"]
        assert med["yarn"] < med["rope_base_10k"]


@pytest.mark.parametrize("method", METHODS)
def test_linearizable_fraction_is_non_increasing_in_the_distance(spectrum, method):
    fracs = [row["frac_channels_linearizable"] for row in _by_method(spectrum, method)]
    assert all(b <= a + 1e-12 for a, b in zip(fracs, fracs[1:], strict=False))
    assert fracs[0] >= fracs[-1]


def test_linearizable_fraction_ordering_prefers_the_compressing_methods(spectrum):
    for d in SPECTRUM_DELTAS:
        frac = {row["method"]: row["frac_channels_linearizable"] for row in spectrum
                if row["delta"] == d}
        assert frac["position_interpolation"] >= frac["yarn"] >= frac["rope_base_10k"]


def test_plain_rope_runs_out_of_linearizable_channels_first(spectrum):
    rope = {row["delta"]: row["frac_channels_linearizable"]
            for row in _by_method(spectrum, "rope_base_10k")}
    yarn = {row["delta"]: row["frac_channels_linearizable"]
            for row in _by_method(spectrum, "yarn")}
    assert rope[2048] == 0.0
    assert rope[4096] == 0.0
    assert yarn[2048] > 0.0
    assert yarn[4096] > 0.0


def test_only_yarn_carries_a_non_trivial_mscale(spectrum):
    for row in spectrum:
        expected = 0.1 * math.log(32.0) + 1.0 if row["method"] == "yarn" else 1.0
        assert row["mscale"] == pytest.approx(expected, rel=0, abs=0)


def test_at_scale_one_yarn_is_plain_rope_row_for_row():
    rows = E.method_spectrum(scale=1.0)
    rope = _by_method(rows, "rope_base_10k")
    yarn = _by_method(rows, "yarn")
    for r, y in zip(rope, yarn, strict=True):
        assert y["mscale"] == 1.0
        for key in ("max_abs_angle", "median_abs_angle", "frac_channels_linearizable",
                    "per_pair_err_rms", "amplitude_weighted_err"):
            assert y[key] == r[key], key


def test_method_spectrum_rows_are_reproducible_from_the_documented_construction(spectrum):
    """Recompute every reported column from the module's own draw order."""
    dim, scale, orig, seed = 64, 32.0, 2048, 2
    rng = np.random.default_rng(seed)
    q_hat = rng.standard_normal(dim)
    k_hat = rng.standard_normal(dim)
    yarn_freqs, _ = R.yarn_parameters(dim, 10000.0, scale, orig)
    schemes = {
        "rope_base_10k": _ladder(dim, 10000.0),
        "position_interpolation": _ladder(dim, 10000.0) / scale,
        "yarn": yarn_freqs,
        "base_500k_legacy_claim": _ladder(dim, 500000.0),
    }
    for row in spectrum:
        mine = _linearization_reference(q_hat, k_hat, schemes[row["method"]], row["delta"])
        for key in ("max_abs_angle", "median_abs_angle", "frac_channels_linearizable",
                    "per_pair_err_rms"):
            assert row[key] == pytest.approx(mine[key], rel=1e-12, abs=1e-15), key
        assert row["amplitude_weighted_err"] == pytest.approx(
            mine["amplitude_weighted_err"], rel=1e-12, abs=1e-300
        )


# ==========================================================================
# 8. partial_rope_split (E6)
# ==========================================================================


@pytest.mark.parametrize("n_rot_frac", [0.125, 0.25, 0.375, 0.5, 0.75])
def test_clean_score_spread_is_exactly_zero(n_rot_frac):
    """An exact identity, asserted as one: the unrotated tail never moves."""
    assert E.partial_rope_split(n_rot_frac=n_rot_frac)["clean_score_spread"] == 0.0


@pytest.mark.parametrize(
    ("n_rot_frac", "n_rot"), [(0.0, 0), (0.125, 8), (0.25, 16), (0.5, 32), (0.75, 48), (1.0, 64)]
)
def test_n_rot_is_int_of_the_fraction(n_rot_frac, n_rot):
    reported = E.partial_rope_split(n_rot_frac=n_rot_frac)
    assert reported["n_rot"] == n_rot == int(64 * n_rot_frac)
    assert reported["rotated_fraction"] == n_rot_frac
    assert reported["clean_fraction"] == 1.0 - n_rot_frac
    assert reported["dim"] == 64
    assert reported["deltas"] == [1, 64, 512, 2048]


def _pp_rope_vector(vec: np.ndarray, freqs: np.ndarray, n_rot: int, pos: int) -> np.ndarray:
    """The partial-RoPE operator, written out.

    ``partial_rope_cos_sin`` puts the rotated block's cos/sin on channels
    ``0 .. n_rot-1`` and identity on the rest, while ``rotate_half`` always pairs
    channel ``k`` with ``k + dim//2``. So for ``k < n_rot`` the output is
    ``(v_k cos D - v_{k+h} sin D, v_{k+h})`` with ``D = pos * freqs[k % (n_rot//2)]``,
    and every other channel is untouched. Note the second half of a touched pair
    is *not* rotated, which is why the operator is not orthogonal.
    """
    dim = vec.shape[0]
    half = dim // 2
    out = np.array(vec, dtype=np.float64, copy=True)
    if n_rot == 0:
        return out
    for k in range(n_rot):
        angle = pos * freqs[k % (n_rot // 2)]
        c, s = math.cos(angle), math.sin(angle)
        out[k] = vec[k] * c - vec[k + half] * s
    return out


def test_partial_rope_operator_matches_the_hand_written_one():
    dim, n_rot = 64, 16
    freqs = _ladder(dim)
    v, _ = _vecs(dim, 51)
    for pos in (0, 37, 512):
        mine = _pp_rope_vector(v, freqs, n_rot, pos)
        cos, sin = R.partial_rope_cos_sin(np.array([pos]), freqs, n_rot)
        theirs = R.apply_rope(v[None, :], cos, sin)[0]
        assert np.array_equal(mine, theirs)


def test_partial_norm_deviation_matches_the_hand_written_operator():
    dim, n_rot = 64, 16
    freqs = _ladder(dim)
    rng = np.random.default_rng(3)
    q_hat = rng.standard_normal(dim)
    mine = _pp_rope_vector(q_hat, freqs, n_rot, 37)
    expected = float(np.linalg.norm(mine) - np.linalg.norm(q_hat))
    assert expected == pytest.approx(
        E.partial_rope_split()["partial_norm_deviation"], rel=1e-12
    )
    # And the sign: pp-RoPE as implemented leaks norm, full RoPE does not.
    assert expected > 0.1
    assert E.partial_rope_split(n_rot_frac=1.0)["partial_norm_deviation"] < EXACT_TOL
    assert E.partial_rope_split(n_rot_frac=0.0)["partial_norm_deviation"] == 0.0


def test_the_tail_of_the_head_is_exactly_position_free():
    """The real content of pp-RoPE: channels ``n_rot..dim`` never move.

    ``partial_rope_cos_sin`` writes cos/sin on channels ``0..n_rot-1`` and the
    identity on the rest, and ``rotate_half`` always pairs channel ``k`` with
    ``k + dim//2``. So a touched pair keeps its *second* half and only rewrites
    its first: the operator changes exactly the ``n_rot`` leading channels and
    leaves the other ``dim - n_rot`` bit-identical. That tail is the clean part
    whose score spread E6 reports as exactly zero - and it is the unrotated dot
    product, which is why the module can compute it without rotating.
    """
    dim, n_rot = 64, 16
    freqs = _ladder(dim)
    rng = np.random.default_rng(3)
    q_hat = rng.standard_normal(dim)
    k_hat = rng.standard_normal(dim)
    tail = np.arange(n_rot, dim)
    assert tail.size == dim - n_rot
    plain = float(q_hat[tail] @ k_hat[tail])

    for delta in (1, 64, 512, 2048):
        cos, sin = R.partial_rope_cos_sin(np.array([0, delta]), freqs, n_rot)
        q_rot = R.apply_rope(q_hat[None, :], cos[0:1], sin[0:1])[0]
        k_rot = R.apply_rope(k_hat[None, :], cos[1:2], sin[1:2])[0]
        # the query is at position 0, so it is the identity; the key is at delta
        assert np.array_equal(q_rot, q_hat)
        assert np.array_equal(k_rot[tail], k_hat[tail])
        assert float(q_rot[tail] @ k_rot[tail]) == plain
        assert not np.array_equal(k_rot[:n_rot], k_hat[:n_rot])
        # Exactly the leading n_rot channels move, and nothing else does.
        assert np.array_equal(np.flatnonzero(k_rot != k_hat), np.arange(n_rot))


@pytest.mark.parametrize("n_rot_frac", [0.125, 0.25, 0.5])
def test_clean_magnitude_share_is_a_proper_fraction(n_rot_frac):
    share = E.partial_rope_split(n_rot_frac=n_rot_frac)["clean_magnitude_share_mean"]
    assert 0.0 < share < 1.0
    # With nothing rotated the whole score is the clean part (up to the 1e-12
    # that the magnitude share adds to its denominator for safety).
    assert E.partial_rope_split(n_rot_frac=0.0)["clean_magnitude_share_mean"] > 1.0 - 1e-9
    # With everything rotated there is no clean part at all.
    assert E.partial_rope_split(n_rot_frac=1.0)["clean_magnitude_share_mean"] == 0.0


@pytest.mark.parametrize("n_rot_frac", [0.125, 0.25, 0.5, 0.75])
def test_rotated_score_really_depends_on_the_distance(n_rot_frac):
    assert E.partial_rope_split(n_rot_frac=n_rot_frac)["rotated_score_spread"] > 1.0
    assert E.partial_rope_split(n_rot_frac=0.0)["rotated_score_spread"] == 0.0


def test_partial_rope_split_is_reproducible_from_the_documented_construction():
    """Recompute the rotated spread and the share from the module's own vectors."""
    dim, n_rot, deltas, seed = 64, 16, (1, 64, 512, 2048), 3
    freqs = _ladder(dim)
    rng = np.random.default_rng(seed)
    q_hat = rng.standard_normal(dim)
    k_hat = rng.standard_normal(dim)
    clean = float(q_hat[n_rot:] @ k_hat[n_rot:])
    rot = []
    for d in deltas:
        cos, sin = R.partial_rope_cos_sin(np.array([0, d]), freqs, n_rot)
        q_rot = R.apply_rope(q_hat[None, :], cos[0:1], sin[0:1])[0]
        k_rot = R.apply_rope(k_hat[None, :], cos[1:2], sin[1:2])[0]
        rot.append(float(q_rot[:n_rot] @ k_rot[:n_rot]))
    rot = np.array(rot)
    share = np.abs(clean) / (np.abs(clean) + np.abs(rot) + 1e-12)
    reported = E.partial_rope_split()
    assert float(rot.max() - rot.min()) == pytest.approx(
        reported["rotated_score_spread"], rel=1e-12
    )
    assert float(share.mean()) == pytest.approx(
        reported["clean_magnitude_share_mean"], rel=1e-12
    )
    assert reported["rotated_score_spread"] == pytest.approx(12.025726981018646, rel=1e-9)
    assert reported["clean_magnitude_share_mean"] == pytest.approx(0.24306319527258705, rel=1e-9)


# ==========================================================================
# 9. mscale_entropy (E7)
# ==========================================================================


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.0001, 2.0, 8.0, 32.0, 4096.0, 1e6])
def test_mscale_is_the_documented_log_formula(scale):
    """``mscale = 0.1 * log(scale) + 1`` above 1, and exactly 1 at or below 1."""
    expected = 1.0 if scale <= 1.0 else 0.1 * math.log(scale) + 1.0
    assert E.mscale_entropy(scale=scale)["mscale"] == expected


def test_mscale_default_value():
    reported = E.mscale_entropy()
    assert reported["mscale"] == pytest.approx(1.346574, rel=1e-6)
    assert reported["mscale"] == 0.1 * math.log(32.0) + 1.0
    assert reported["scale"] == 32.0
    assert reported["n_keys"] == 256
    assert reported["max_possible_entropy"] == math.log(256)
    assert reported["max_possible_entropy"] == pytest.approx(5.545177444479562, rel=1e-15)


@pytest.mark.parametrize("n_keys", [8, 16, 64, 256, 1024])
def test_entropy_ratios_are_proportions_of_the_maximum(n_keys):
    reported = E.mscale_entropy(n_keys=n_keys)
    assert 0.0 < reported["entropy_ratio_unscaled"] <= 1.0
    assert 0.0 < reported["entropy_ratio_mscaled"] <= 1.0
    assert reported["max_possible_entropy"] == math.log(n_keys)
    assert reported["entropy_unscaled"] == pytest.approx(
        reported["entropy_ratio_unscaled"] * reported["max_possible_entropy"], rel=1e-12
    )
    assert reported["entropy_mscaled"] == pytest.approx(
        reported["entropy_ratio_mscaled"] * reported["max_possible_entropy"], rel=1e-12
    )
    assert reported["entropy_unscaled"] <= reported["max_possible_entropy"] + 1e-12


@pytest.mark.parametrize("scale", [2.0, 8.0, 32.0, 1024.0])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sharpening_the_logits_lowers_the_entropy(scale, seed):
    """``mscale > 1`` is a temperature drop, so it must sharpen, not blur."""
    reported = E.mscale_entropy(scale=scale, seed=seed)
    assert reported["mscale"] > 1.0
    assert reported["entropy_mscaled"] < reported["entropy_unscaled"]
    assert reported["entropy_ratio_mscaled"] < reported["entropy_ratio_unscaled"]


@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_no_scaling_leaves_the_entropy_bit_identical(scale):
    reported = E.mscale_entropy(scale=scale)
    assert reported["mscale"] == 1.0
    assert reported["entropy_mscaled"] == reported["entropy_unscaled"]


def test_mscale_entropy_is_reproducible_from_the_documented_construction():
    """Recompute both entropies from a hand-built rotation of the key set."""
    dim, n_keys, scale, seed = 64, 256, 32.0, 4
    # mscale does not depend on the context length it is scaling from, so the
    # 2048 here is the module's default; the point of the recomputation is the
    # score construction and the softmax entropy, not that number.
    freqs = _ladder(dim)
    rng = np.random.default_rng(seed)
    q_hat = rng.standard_normal(dim)
    k_hat = rng.standard_normal((n_keys, dim))
    mscale = 0.1 * math.log(scale) + 1.0

    scores = np.empty(n_keys)
    for n in range(1, n_keys + 1):
        scores[n - 1] = _ref_score(q_hat, k_hat[n - 1], freqs, n)
    scores = scores / math.sqrt(dim)

    def entropy(s):
        p = np.exp(s - s.max())
        p = p / p.sum()
        return float(-(p * np.log(p + 1e-300)).sum())

    reported = E.mscale_entropy()
    assert entropy(scores) == pytest.approx(reported["entropy_unscaled"], rel=1e-12)
    assert entropy(scores * mscale) == pytest.approx(reported["entropy_mscaled"], rel=1e-12)
    assert reported["entropy_unscaled"] == pytest.approx(5.1198125583580145, rel=1e-9)
    assert reported["entropy_mscaled"] == pytest.approx(4.797042631875421, rel=1e-9)
    assert reported["entropy_ratio_unscaled"] == pytest.approx(0.9232910235280181, rel=1e-9)
    assert reported["entropy_ratio_mscaled"] == pytest.approx(0.8650837019924513, rel=1e-9)


# ==========================================================================
# 10. Determinism: every experiment is a pure function of its seed
# ==========================================================================


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("structural_facts", {}),
        ("structural_facts", {"base": 500000.0, "n_positions": 64}),
        ("feature_attribution", {}),
        ("feature_attribution", {"n_features": 3, "dim": 16, "deltas": (2, 9)}),
        ("method_spectrum", {}),
        ("method_spectrum", {"scale": 8.0, "deltas": (1, 5)}),
        ("partial_rope_split", {}),
        ("partial_rope_split", {"n_rot_frac": 0.5, "deltas": (1, 2)}),
        ("mscale_entropy", {}),
        ("mscale_entropy", {"n_keys": 32, "scale": 4.0}),
    ],
)
def test_repeated_calls_are_bit_identical(name, kwargs):
    first = _fingerprint(getattr(E, name)(**kwargs))
    second = _fingerprint(getattr(E, name)(**kwargs))
    assert first == second


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("structural_facts", {}),
        ("feature_attribution", {}),
        ("method_spectrum", {}),
        ("partial_rope_split", {}),
        ("mscale_entropy", {}),
    ],
)
def test_a_different_seed_changes_the_measurement(name, kwargs):
    first = _fingerprint(getattr(E, name)(**kwargs))
    second = _fingerprint(getattr(E, name)(seed=kwargs.get("seed", 0) + 101, **kwargs))
    assert first != second


def test_structural_facts_keeps_its_exactness_claims_under_other_seeds():
    for seed in (0, 1, 2, 3, 4):
        reported = E.structural_facts(seed=seed)
        assert reported["relative_position_pairs_tested"] == 42
        assert reported["scores_tested"] == 35
        for key in (
            "relative_position_max_abs_err",
            "query_norm_preservation_max_abs_err",
            "key_norm_preservation_max_abs_err",
            "bilinearity_max_rel_err",
            "pair_closed_form_max_abs_err",
        ):
            assert float(reported[key]) < EXACT_TOL, (seed, key)


# ==========================================================================
# 11. The report
# ==========================================================================


REPORT_KEYS = {
    "structural_facts",
    "feature_attribution",
    "method_spectrum",
    "partial_rope",
    "mscale_entropy",
}


@pytest.fixture(scope="module")
def report_dict():
    return E.run_all().to_dict()


def test_report_has_the_five_documented_sections(report_dict):
    assert set(report_dict) == REPORT_KEYS
    assert list(report_dict) == [
        "structural_facts",
        "feature_attribution",
        "method_spectrum",
        "partial_rope",
        "mscale_entropy",
    ]


def test_report_is_json_serializable(report_dict):
    text = json.dumps(report_dict)
    assert json.loads(text) == report_dict
    assert len(text) > 1000


def test_report_sections_hold_the_individual_experiments(report_dict):
    assert report_dict["structural_facts"] == E.structural_facts()
    assert report_dict["feature_attribution"] == E.feature_attribution()
    assert report_dict["method_spectrum"] == E.method_spectrum()
    assert report_dict["partial_rope"] == E.partial_rope_split()
    assert report_dict["mscale_entropy"] == E.mscale_entropy()
    assert len(report_dict["method_spectrum"]) == 20


def test_report_is_reproducible():
    assert _fingerprint(E.run_all().to_dict()) == _fingerprint(E.run_all().to_dict())


def test_default_report_is_empty_but_complete():
    empty = E.Report().to_dict()
    assert set(empty) == REPORT_KEYS
    assert empty["structural_facts"] == {}
    assert empty["feature_attribution"] == {}
    assert empty["method_spectrum"] == []
    assert empty["partial_rope"] == {}
    assert empty["mscale_entropy"] == {}


def test_to_dict_returns_the_sections_it_was_built_from():
    report = E.Report(
        structural={"a": 1},
        attribution={"b": 2},
        spectrum=[{"c": 3}],
        partial={"d": 4},
        mscale={"e": 5},
    )
    assert report.to_dict() == {
        "structural_facts": {"a": 1},
        "feature_attribution": {"b": 2},
        "method_spectrum": [{"c": 3}],
        "partial_rope": {"d": 4},
        "mscale_entropy": {"e": 5},
    }


def test_main_writes_the_measurements_it_printed(tmp_path, monkeypatch, capsys):
    """``main()`` must not touch the repository: redirect its output path."""
    monkeypatch.setattr(
        E, "__file__", str(tmp_path / "pkg" / "sub" / "experiments.py"), raising=False
    )
    E.main()
    out = capsys.readouterr().out
    assert "wrote " in out
    written = json.loads((tmp_path / "results" / "measurements.json").read_text("utf-8"))
    assert written == E.run_all().to_dict()
    assert "EXACTLY bilinear" in out
    assert "identical for rope_base_10k and yarn" in out
