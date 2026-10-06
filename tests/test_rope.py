"""Test suite for ``projects/rope_attribution/rope.py``.

Design rules for this file:

* Every claim is checked against an **independently written formula** or against a
  **hard mathematical identity** (orthogonality of the rotation, the
  relative-position property, ``cos^2 + sin^2 = 1``, "this many full turns inside
  the original context window", the geometric structure of the frequency ladder).
  Nothing is asserted merely because it restates the body of the function.
* Exactness claims use :func:`numpy.testing.assert_array_equal`; numerical claims
  use :func:`numpy.testing.assert_allclose` with an explicit tolerance.
* Fixed RNG seeds, so the suite is deterministic.
"""

from __future__ import annotations

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

from rope_attribution import rope as R  # noqa: E402  (needs the sys.path tweak above)

# --------------------------------------------------------------------------
# Parametrization
# --------------------------------------------------------------------------

DIMS = (8, 16, 64, 128)
BASES = (10000.0, 500000.0, 1000000.0)
SEED = 20260929


def _positions(n: int = 64) -> np.ndarray:
    return np.arange(n, dtype=np.float64)


def _vecs(dim: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim), rng.standard_normal(dim)


def _score(v: np.ndarray, u: np.ndarray, m: float, n: float, inv_f: np.ndarray) -> float:
    """``<R_m v, R_n u>`` computed through the public API only."""
    cos, sin = R.rope_cos_sin(np.array([float(m), float(n)], dtype=np.float64), inv_f)
    vm = R.apply_rope(v[None, :], cos[:1], sin[:1])[0]
    un = R.apply_rope(u[None, :], cos[1:], sin[1:])
    return float(vm @ un[0])


def _rotation_matrices(angles: np.ndarray, dim: int) -> np.ndarray:
    """Independent 2x2 block rotations: channel ``k`` paired with ``k + dim // 2``."""
    half = dim // 2
    mats = np.zeros((angles.shape[0], dim, dim), dtype=np.float64)
    idx = np.arange(half)
    c, s = np.cos(angles), np.sin(angles)
    mats[:, idx, idx] = c
    mats[:, idx, idx + half] = -s
    mats[:, idx + half, idx] = s
    mats[:, idx + half, idx + half] = c
    return mats


# ==========================================================================
# inv_freq
# ==========================================================================


@pytest.mark.parametrize("dim", DIMS)
def test_inv_freq_shape_and_dtype(dim):
    f = R.inv_freq(dim)
    assert f.shape == (dim // 2,)
    assert f.dtype == np.float64
    assert np.all(f > 0.0)
    assert np.all(np.isfinite(f))


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("base", BASES)
def test_inv_freq_matches_closed_form(dim, base):
    """``inv_freq[i] == base ** (-2 * i / dim)``."""
    expected = base ** (-2.0 * np.arange(dim // 2, dtype=np.float64) / dim)
    np.testing.assert_allclose(R.inv_freq(dim, base), expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("base", BASES)
def test_inv_freq_matches_transformers_reciprocal_form(dim, base):
    """HuggingFace writes ``1.0 / (base ** (arange(0, dim, 2) / dim))`` -- same numbers."""
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    np.testing.assert_array_equal(R.inv_freq(dim, base), 1.0 / pos_freqs)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("base", BASES)
def test_inv_freq_is_a_geometric_ladder(dim, base):
    """A hard identity: consecutive channels differ by a constant ratio ``base**(-2/dim)``."""
    f = R.inv_freq(dim, base)
    ratio = f[1:] / f[:-1]
    expected = base ** (-2.0 / dim)
    np.testing.assert_allclose(ratio, np.full(ratio.shape, expected), rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("base", BASES)
def test_inv_freq_is_monotone_non_increasing(dim, base):
    f = R.inv_freq(dim, base)
    assert np.all(np.diff(f) <= 0.0)
    assert np.all(np.diff(f) < 0.0)  # strictly decreasing, base > 1


@pytest.mark.parametrize("dim", DIMS)
def test_inv_freq_endpoints(dim):
    """``i = 0`` is exactly 1.0; the last channel is the slowest."""
    f = R.inv_freq(dim)
    assert f[0] == 1.0  # exact
    assert f[0] == pytest.approx(1.0, abs=0.0)
    # independent closed form for the final index
    assert f[-1] == pytest.approx(10000.0 ** (-2.0 * (dim // 2 - 1) / dim), rel=1e-15)


def test_inv_freq_known_endpoint_values():
    """Numbers quoted in the module docstring / by the orchestrator."""
    f = R.inv_freq(64, 10000.0)
    assert f[0] == 1.0
    assert f[-1] == pytest.approx(1.334e-04, rel=1e-3)
    assert f[1] == pytest.approx(10000.0 ** (-2.0 / 64.0), rel=1e-15)
    # 10000 ** (-62/64)
    assert f[-1] == pytest.approx(10000.0 ** (-62.0 / 64.0), rel=1e-15)


@pytest.mark.parametrize("dim", [1, 3, 5, 7, 9, 63, 65, 127])
def test_inv_freq_rejects_odd_dim(dim):
    with pytest.raises(ValueError, match="even"):
        R.inv_freq(dim)


def test_inv_freq_rejects_odd_dim_mentions_the_value():
    with pytest.raises(ValueError, match="13"):
        R.inv_freq(13)


# ==========================================================================
# rotary_angles / rope_cos_sin
# ==========================================================================


@pytest.mark.parametrize("dim", DIMS)
def test_rotary_angles_shape_is_outer_product(dim):
    inv_f = R.inv_freq(dim)
    pos = np.array([0.0, 1.0, 7.5, -3.0, 4096.0])
    angles = R.rotary_angles(pos, inv_f)
    assert angles.shape == (pos.size, dim // 2)
    np.testing.assert_array_equal(angles, pos[:, None] * inv_f[None, :])


@pytest.mark.parametrize("dim", DIMS)
def test_rotary_angles_cast_integer_positions_to_float64(dim):
    inv_f = R.inv_freq(dim)
    angles = R.rotary_angles(np.arange(5), inv_f)
    assert angles.dtype == np.float64
    np.testing.assert_array_equal(angles, np.arange(5.0)[:, None] * inv_f[None, :])


@pytest.mark.parametrize("dim", DIMS)
def test_rope_cos_sin_shapes_and_dtype(dim):
    inv_f = R.inv_freq(dim)
    pos = _positions(9)
    cos, sin = R.rope_cos_sin(pos, inv_f)
    assert cos.shape == (9, dim)
    assert sin.shape == (9, dim)
    assert cos.dtype == np.float64 and sin.dtype == np.float64


@pytest.mark.parametrize("dim", DIMS)
def test_rope_cos_sin_duplicates_the_frequency_vector(dim):
    """The split-half convention: ``emb = concat([angles, angles])``."""
    half = dim // 2
    inv_f = R.inv_freq(dim)
    pos = _positions(33)
    cos, sin = R.rope_cos_sin(pos, inv_f)
    np.testing.assert_array_equal(cos[:, :half], cos[:, half:])
    np.testing.assert_array_equal(sin[:, :half], sin[:, half:])


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("mscale", [1.0, 0.5, 2.0, 1.3465735902799727])
def test_rope_cos_sin_matches_independently_computed_trig(dim, mscale):
    inv_f = R.inv_freq(dim)
    pos = _positions(17)
    angles = np.concatenate([pos[:, None] * inv_f[None, :]] * 2, axis=-1)
    cos, sin = R.rope_cos_sin(pos, inv_f, mscale)
    np.testing.assert_allclose(cos, np.cos(angles) * mscale, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(sin, np.sin(angles) * mscale, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("mscale", [1.0, 0.5, 1.3465735902799727, 3.0])
def test_rope_cos_sin_obeys_pythagorean_identity(dim, mscale):
    """Hard identity: cos and sin are both scaled, so cos^2 + sin^2 == mscale^2."""
    cos, sin = R.rope_cos_sin(_positions(257), R.inv_freq(dim), mscale)
    np.testing.assert_allclose(cos**2 + sin**2, mscale**2, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("dim", DIMS)
def test_rope_cos_sin_at_position_zero_is_one_and_zero(dim):
    cos, sin = R.rope_cos_sin(np.zeros(1), R.inv_freq(dim))
    np.testing.assert_array_equal(cos, np.ones((1, dim)))
    np.testing.assert_array_equal(sin, np.zeros((1, dim)))


def test_rope_cos_sin_accepts_a_plain_list():
    cos, sin = R.rope_cos_sin([0.0, 1.0], R.inv_freq(16))
    assert cos.shape == (2, 16) and sin.shape == (2, 16)


# ==========================================================================
# rotate_half
# ==========================================================================


@pytest.mark.parametrize("last", [2, 8, 16, 64])
@pytest.mark.parametrize("prefix", [(), (3,), (2, 5)])
def test_rotate_half_matches_the_negated_swap_definition(last, prefix):
    rng = np.random.default_rng(SEED + 1)
    x = rng.standard_normal(prefix + (last,))
    half = last // 2
    expected = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    np.testing.assert_array_equal(R.rotate_half(x), expected)


def test_rotate_half_is_a_signed_permutation_matrix():
    """As a linear operator, ``rotate_half`` is ``x @ P`` with ``P @ P == -I``."""
    dim = 16
    x = np.random.default_rng(SEED + 2).standard_normal((5, dim))
    p = np.zeros((dim, dim))
    got = R.rotate_half(x)
    for j in range(dim):
        basis = np.zeros(dim)
        basis[j] = 1.0
        p[:, j] = R.rotate_half(basis[None, :])[0]
    # every column is a signed one-hot, at the half-swap offset
    expected = np.zeros((dim, dim))
    for i in range(dim):
        expected[i, (i + dim // 2) % dim] = 1.0
    np.testing.assert_array_equal(np.abs(p), expected)
    assert set(np.unique(p)) <= {-1.0, 0.0, 1.0}
    np.testing.assert_array_equal(x @ p.T, got)
    np.testing.assert_array_equal(p @ p, -np.eye(dim))


@pytest.mark.parametrize("last", [2, 8, 64])
def test_rotate_half_applied_twice_is_negation(last):
    x = np.random.default_rng(SEED + 3).standard_normal((3, last))
    np.testing.assert_array_equal(R.rotate_half(R.rotate_half(x)), -x)


def test_rotate_half_preserves_norm_and_is_odd():
    x = np.random.default_rng(SEED + 4).standard_normal((4, 6, 32))
    np.testing.assert_array_equal(R.rotate_half(-x), -R.rotate_half(x))
    np.testing.assert_allclose(
        np.linalg.norm(R.rotate_half(x), axis=-1), np.linalg.norm(x, axis=-1), rtol=0.0, atol=1e-14
    )


# ==========================================================================
# apply_rope
# ==========================================================================


@pytest.mark.parametrize("dim", DIMS)
def test_apply_rope_at_position_zero_is_exactly_the_identity(dim):
    cos, sin = R.rope_cos_sin(np.zeros(1), R.inv_freq(dim))
    x = np.random.default_rng(SEED + 5).standard_normal((3, 5, dim))
    np.testing.assert_array_equal(R.apply_rope(x, cos, sin), x)


@pytest.mark.parametrize("dim", DIMS)
def test_apply_rope_equals_an_explicit_block_rotation_matrix(dim):
    """Channel ``k`` is rotated against channel ``k + dim // 2`` by ``cos``/``sin``."""
    inv_f = R.inv_freq(dim)
    pos = _positions(48)
    cos, sin = R.rope_cos_sin(pos, inv_f)
    mats = _rotation_matrices(pos[:, None] * inv_f[None, :], dim)
    x = np.random.default_rng(SEED + 6).standard_normal((pos.size, dim))
    np.testing.assert_allclose(
        R.apply_rope(x, cos, sin), (mats @ x[..., None])[..., 0], rtol=0.0, atol=1e-14
    )


@pytest.mark.parametrize("dim", DIMS)
def test_apply_rope_operator_is_orthogonal(dim):
    """Recover the linear operator from its action on the basis; it must be orthogonal."""
    cos, sin = R.rope_cos_sin(np.array([1234.0]), R.inv_freq(dim))
    mat = np.zeros((dim, dim))
    for j in range(dim):
        e = np.zeros(dim)
        e[j] = 1.0
        mat[:, j] = R.apply_rope(e[None, :], cos, sin)[0]
    np.testing.assert_allclose(mat.T @ mat, np.eye(dim), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(mat), 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("dim", DIMS)
def test_apply_rope_at_position_zero_recovers_identity_from_the_operator(dim):
    cos, sin = R.rope_cos_sin(np.array([0.0]), R.inv_freq(dim))
    mat = np.zeros((dim, dim))
    for j in range(dim):
        e = np.zeros(dim)
        e[j] = 1.0
        mat[:, j] = R.apply_rope(e[None, :], cos, sin)[0]
    np.testing.assert_array_equal(mat, np.eye(dim))


# ==========================================================================
# The three headline properties
# ==========================================================================


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("delta", [0, 1, 2, 7, 100, 1000])
@pytest.mark.parametrize("m", [0, 1, 5, 37, 500, 2048])
def test_attention_score_depends_only_on_the_position_gap(dim, delta, m):
    """RoPE's defining property: ``score(m, m + delta) == score(0, delta)``."""
    inv_f = R.inv_freq(dim)
    v, u = _vecs(dim)
    assert _score(v, u, m, m + delta, inv_f) == pytest.approx(
        _score(v, u, 0, delta, inv_f), rel=1e-11, abs=1e-13
    )


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("pos", [0, 1, 4096, 131072])
def test_attention_score_at_equal_positions_is_the_plain_dot_product(dim, pos):
    inv_f = R.inv_freq(dim)
    v, u = _vecs(dim, seed=SEED + 7)
    assert _score(v, u, pos, pos, inv_f) == pytest.approx(float(v @ u), rel=1e-13, abs=1e-13)


@pytest.mark.parametrize("dim", DIMS)
def test_full_rope_preserves_vector_norm(dim):
    inv_f = R.inv_freq(dim)
    pos = np.arange(0, 4096, dtype=np.float64)
    cos, sin = R.rope_cos_sin(pos, inv_f)
    x = np.random.default_rng(SEED + 8).standard_normal((pos.size, dim))
    np.testing.assert_allclose(
        np.linalg.norm(R.apply_rope(x, cos, sin), axis=-1),
        np.linalg.norm(x, axis=-1),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("mscale", [0.5, 1.0, 1.3465735902799727, 2.0])
def test_rope_scales_norm_by_exactly_mscale(dim, mscale):
    """cos and sin are both multiplied by ``mscale``, so the norm is multiplied by it."""
    inv_f = R.inv_freq(dim)
    pos = np.arange(0, 512, dtype=np.float64)
    cos, sin = R.rope_cos_sin(pos, inv_f, mscale)
    x = np.random.default_rng(SEED + 9).standard_normal((pos.size, dim))
    np.testing.assert_allclose(
        np.linalg.norm(R.apply_rope(x, cos, sin), axis=-1) / mscale,
        np.linalg.norm(x, axis=-1),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("alpha", [2.0, 0.25, 4.0])
def test_score_scales_exactly_with_the_query_vector(dim, alpha):
    """Bit-exact homogeneity for dyadic alphas (fact 3)."""
    inv_f = R.inv_freq(dim)
    v, u = _vecs(dim, seed=SEED + 10)
    base = _score(v, u, 3, 29, inv_f)
    assert _score(alpha * v, u, 3, 29, inv_f) == alpha * base


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("alpha", [-3.0, 7.5, 1e-8, 1234.5678, -0.125])
def test_score_scales_linearly_with_the_query_vector(dim, alpha):
    inv_f = R.inv_freq(dim)
    v, u = _vecs(dim, seed=SEED + 11)
    base = _score(v, u, 3, 29, inv_f)
    assert _score(alpha * v, u, 3, 29, inv_f) == pytest.approx(alpha * base, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("alpha,beta", [(2.0, 0.5), (-3.0, 4.0), (1.0, 1.0)])
def test_score_is_bilinear_in_both_query_and_key(dim, alpha, beta):
    inv_f = R.inv_freq(dim)
    v, u = _vecs(dim, seed=SEED + 12)
    base = _score(v, u, 11, 400, inv_f)
    assert _score(alpha * v, beta * u, 11, 400, inv_f) == pytest.approx(
        alpha * beta * base, rel=1e-13, abs=1e-13
    )


def _closed_form_score(v, u, m, n, inv_f):
    """Independent transcription of the split-half closed form (fact 4)."""
    half = inv_f.size
    theta = (float(n) - float(m)) * inv_f
    v1, v2 = v[:half], v[half:]
    u1, u2 = u[:half], u[half:]
    c, s = np.cos(theta), np.sin(theta)
    return float(np.dot(v1 * u1 + v2 * u2, c) + np.dot(v2 * u1 - v1 * u2, s))


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize(
    "m,n", [(0, 0), (0, 1), (0, 2), (0, 37), (0, 500), (11, 4000), (7, 7), (300, 301), (1000, 9000)]
)
def test_score_matches_the_closed_form(dim, m, n):
    inv_f = R.inv_freq(dim)
    v, u = _vecs(dim, seed=SEED + 13)
    assert _score(v, u, m, n, inv_f) == pytest.approx(
        _closed_form_score(v, u, m, n, inv_f), rel=1e-11, abs=1e-12
    )


# ==========================================================================
# partial_rope_cos_sin
# ==========================================================================


def _n_rot_cases(dim):
    return tuple(nr for nr in (0, 2, dim // 4, dim // 2, dim - 2, dim) if 0 <= nr <= dim and nr % 2 == 0)


@pytest.mark.parametrize("dim", DIMS)
def test_partial_cos_sin_shapes(dim):
    pos = _positions(11)
    inv_f = R.inv_freq(dim)
    for n_rot in _n_rot_cases(dim):
        cos, sin = R.partial_rope_cos_sin(pos, inv_f, n_rot)
        assert cos.shape == (11, dim)
        assert sin.shape == (11, dim)
        assert cos.dtype == np.float64 and sin.dtype == np.float64


@pytest.mark.parametrize("dim", DIMS)
def test_partial_tail_of_cos_sin_is_exactly_one_and_zero(dim):
    """The un-rotated channels are not merely *close* to identity -- they are exact."""
    pos = _positions(64)
    inv_f = R.inv_freq(dim)
    for n_rot in _n_rot_cases(dim):
        cos, sin = R.partial_rope_cos_sin(pos, inv_f, n_rot)
        np.testing.assert_array_equal(cos[:, n_rot:], np.ones((pos.size, dim - n_rot)))
        np.testing.assert_array_equal(sin[:, n_rot:], np.zeros((pos.size, dim - n_rot)))


@pytest.mark.parametrize("dim", DIMS)
def test_partial_apply_rope_leaves_the_tail_bit_identical(dim):
    """Fact 8: with ``n_rot=16, dim=64`` the tail survives bit-for-bit."""
    pos = np.arange(1, 65, dtype=np.float64)
    inv_f = R.inv_freq(dim)
    n_rot = min(16, dim)
    cos, sin = R.partial_rope_cos_sin(pos, inv_f, n_rot)
    x = np.random.default_rng(SEED + 14).standard_normal((pos.size, dim))
    out = R.apply_rope(x, cos, sin)
    np.testing.assert_array_equal(out[:, n_rot:], x[:, n_rot:])
    assert not np.array_equal(out[:, :n_rot], x[:, :n_rot])
    assert np.max(np.abs(out[:, :n_rot] - x[:, :n_rot])) > 1e-3


@pytest.mark.parametrize("dim", DIMS)
def test_partial_rope_repacks_the_lowest_frequencies_into_the_rotated_block(dim):
    """The rotated block re-packs the ``n_rot // 2`` fastest channels twice.

    Consequence: for ``0 < n_rot < dim`` the partial layout is *not* the leading slice
    of the full layout (the full layout would repeat frequencies ``n_rot//2 ..
    n_rot-1`` there instead).
    """
    pos = _positions(9)
    inv_f = R.inv_freq(dim)
    full_cos, full_sin = R.rope_cos_sin(pos, inv_f)
    for n_rot in _n_rot_cases(dim):
        if n_rot == 0:
            continue
        k = n_rot // 2
        cos, sin = R.partial_rope_cos_sin(pos, inv_f, n_rot)
        np.testing.assert_array_equal(cos[:, :k], full_cos[:, :k])
        np.testing.assert_array_equal(cos[:, k:n_rot], full_cos[:, :k])
        np.testing.assert_array_equal(sin[:, :k], full_sin[:, :k])
        np.testing.assert_array_equal(sin[:, k:n_rot], full_sin[:, :k])
        if n_rot < dim:
            assert not np.array_equal(cos[:, :n_rot], full_cos[:, :n_rot])


@pytest.mark.parametrize("dim", DIMS)
def test_partial_with_full_n_rot_equals_full_rope(dim):
    pos = _positions(33)
    inv_f = R.inv_freq(dim)
    x = np.random.default_rng(SEED + 15).standard_normal((pos.size, dim))
    full = R.rope_cos_sin(pos, inv_f)
    part = R.partial_rope_cos_sin(pos, inv_f, dim)
    np.testing.assert_array_equal(part[0], full[0])
    np.testing.assert_array_equal(part[1], full[1])
    np.testing.assert_array_equal(R.apply_rope(x, *part), R.apply_rope(x, *full))


@pytest.mark.parametrize("dim", DIMS)
def test_partial_n_rot_zero_is_exactly_no_rotation(dim):
    pos = _positions(7)
    inv_f = R.inv_freq(dim)
    cos, sin = R.partial_rope_cos_sin(pos, inv_f, 0)
    np.testing.assert_array_equal(cos, np.ones((7, dim)))
    np.testing.assert_array_equal(sin, np.zeros((7, dim)))
    x = np.random.default_rng(SEED + 16).standard_normal((3, 7, dim))
    np.testing.assert_array_equal(R.apply_rope(x, cos, sin), x)


@pytest.mark.parametrize("dim", DIMS)
def test_partial_at_position_zero_is_exactly_the_identity(dim):
    inv_f = R.inv_freq(dim)
    n_rot = max(2, dim // 4)
    cos, sin = R.partial_rope_cos_sin(np.zeros(1), inv_f, n_rot)
    np.testing.assert_array_equal(cos[0, :n_rot], np.ones(n_rot))
    np.testing.assert_array_equal(sin[0, :n_rot], np.zeros(n_rot))
    x = np.random.default_rng(SEED + 17).standard_normal((4, dim))
    np.testing.assert_array_equal(R.apply_rope(x, cos, sin), x)


@pytest.mark.parametrize("dim", DIMS)
def test_partial_propagates_mscale_to_the_rotated_block_only(dim):
    pos = _positions(5)
    inv_f = R.inv_freq(dim)
    n_rot = min(16, dim)
    mscale = 1.3465735902799727
    cos, sin = R.partial_rope_cos_sin(pos, inv_f, n_rot, mscale)
    ref_cos, ref_sin = R.rope_cos_sin(pos, inv_f[: n_rot // 2], mscale)
    np.testing.assert_array_equal(cos[:, :n_rot], ref_cos)
    np.testing.assert_array_equal(sin[:, :n_rot], ref_sin)
    np.testing.assert_allclose(cos[:, :n_rot] ** 2 + sin[:, :n_rot] ** 2, mscale**2, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    "dim, n_rot",
    [(8, 1), (8, 3), (8, 5), (8, 7), (16, 1), (16, 7), (16, 15), (64, 1), (64, 31), (64, 63), (128, 1), (128, 127)],
)
def test_partial_rejects_odd_n_rot(dim, n_rot):
    with pytest.raises(ValueError, match="even"):
        R.partial_rope_cos_sin(_positions(3), R.inv_freq(dim), n_rot)


@pytest.mark.parametrize("n_rot", [1, 3, 63, 65, -1, -3])
def test_partial_rejects_odd_n_rot_even_above_dim(n_rot):
    """An odd ``n_rot`` is rejected on parity, whatever its size."""
    with pytest.raises(ValueError, match="even"):
        R.partial_rope_cos_sin(_positions(3), R.inv_freq(64), n_rot)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("offset", [2, 4, 6])
def test_partial_rejects_n_rot_above_head_dim(dim, offset):
    with pytest.raises(ValueError, match=rf"\[0, {dim}\]"):
        R.partial_rope_cos_sin(_positions(3), R.inv_freq(dim), dim + offset)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("n_rot", [-2, -4, -18])
def test_partial_rejects_negative_n_rot(dim, n_rot):
    with pytest.raises(ValueError, match=rf"\[0, {dim}\]"):
        R.partial_rope_cos_sin(_positions(3), R.inv_freq(dim), n_rot)


@pytest.mark.parametrize("dim, n_rot", [(8, 2), (16, 8), (64, 16), (128, 32)])
def test_partial_rope_is_not_norm_preserving(dim, n_rot):
    """Contrast with the full-RoPE test above: pp-RoPE mixes channels across the split.

    Only asserted loosely -- this documents a *failure* of a property, so the
    threshold just has to be unmistakably above floating-point noise.
    """
    inv_f = R.inv_freq(dim)
    pos = np.arange(0, 257, dtype=np.float64)
    x = np.random.default_rng(SEED + 18).standard_normal((pos.size, dim))
    base_norm = np.linalg.norm(x, axis=-1)

    full_dev = np.max(np.abs(np.linalg.norm(R.apply_rope(x, *R.rope_cos_sin(pos, inv_f)), axis=-1) - base_norm))
    part_dev = np.max(np.abs(np.linalg.norm(R.apply_rope(x, *R.partial_rope_cos_sin(pos, inv_f, n_rot)), axis=-1) - base_norm))

    assert full_dev <= 1e-12
    assert part_dev > 1e-2


# ==========================================================================
# get_mscale
# ==========================================================================


@pytest.mark.parametrize("scale", [0.0, 0.25, 0.5, 1.0, 1.0 - 1e-12])
def test_get_mscale_is_exactly_one_for_scale_at_most_one(scale):
    assert R.get_mscale(scale) == 1.0
    assert R.get_mscale(scale, 99.0) == 1.0  # the mscale argument is ignored here


@pytest.mark.parametrize("scale", [1.0000001, 2.0, 4.0, 8.0, 32.0, 4096.0, 1e6])
def test_get_mscale_matches_its_log_formula_exactly(scale):
    assert R.get_mscale(scale) == 0.1 * math.log(scale) + 1.0


def test_get_mscale_known_value():
    assert R.get_mscale(32.0) == pytest.approx(1.346574, abs=5e-7)
    assert R.get_mscale(32.0) == pytest.approx(1.3465735902799727, rel=1e-15)


@pytest.mark.parametrize("scale", [2.0, 8.0, 32.0, 1000.0])
@pytest.mark.parametrize("mscale", [0.5, 1.0, 2.0, 4.0, 0.1])
def test_get_mscale_argument_scales_the_log_term(scale, mscale):
    assert R.get_mscale(scale, mscale) == pytest.approx(
        0.1 * mscale * math.log(scale) + 1.0, rel=1e-15, abs=0.0
    )


@pytest.mark.parametrize("scale", [2.0, 8.0, 32.0, 1000.0])
def test_get_mscale_argument_doubles_the_excess_over_one_exactly(scale):
    base = R.get_mscale(scale, 1.0)
    assert R.get_mscale(scale, 2.0) == 2.0 * (base - 1.0) + 1.0


def test_get_mscale_is_strictly_increasing_above_one():
    scales = [1.0] + [1.0 + k * 1e-6 for k in range(1, 40)] + [2.0, 8.0, 32.0, 1024.0, 1e5]
    values = [R.get_mscale(s) for s in scales]
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))
    assert values[0] == 1.0
    assert values[1] == pytest.approx(1.0 + 1e-7, rel=1e-9)
    # it is a logarithm: doubling the scale adds 0.1 * ln 2
    step = 0.1 * math.log(2.0)
    assert R.get_mscale(64.0) - R.get_mscale(32.0) == pytest.approx(step, rel=1e-14)


# ==========================================================================
# linear_ramp_mask
# ==========================================================================


@pytest.mark.parametrize("dim", [4, 8, 32, 64])
@pytest.mark.parametrize("lo, hi", [(0.0, 1.0), (1.0, 3.0), (3.5, 11.25), (8.0, 21.0), (0.0, 40.0)])
def test_linear_ramp_mask_matches_closed_form(dim, lo, hi):
    expected = np.clip((np.arange(dim, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    np.testing.assert_array_equal(R.linear_ramp_mask(lo, hi, dim), expected)


@pytest.mark.parametrize("dim", [4, 8, 32, 64])
def test_linear_ramp_mask_is_clipped_and_monotone(dim):
    for lo, hi in ((0.0, 1.0), (2.0, 9.0), (-5.0, 3.0), (7.5, 7.75)):
        m = R.linear_ramp_mask(lo, hi, dim)
        assert m.shape == (dim,)
        assert np.all(m >= 0.0) and np.all(m <= 1.0)
        assert np.all(np.diff(m) >= 0.0)


@pytest.mark.parametrize("lo, hi", [(1.0, 3.0), (2.0, 9.0), (5.0, 5.5)])
def test_linear_ramp_mask_saturates_at_both_ends(lo, hi):
    dim = 32
    m = R.linear_ramp_mask(lo, hi, dim)
    np.testing.assert_array_equal(m[: int(math.floor(lo)) + 1], np.zeros(int(math.floor(lo)) + 1))
    np.testing.assert_array_equal(m[int(math.ceil(hi)) :], np.ones(dim - int(math.ceil(hi))))


def test_linear_ramp_mask_is_strictly_between_the_endpoints_inside_the_ramp():
    lo, hi, dim = 2.0, 6.0, 9
    m = R.linear_ramp_mask(lo, hi, dim)
    assert m[0] == 0.0 and m[-1] == 1.0
    for k in (3, 4, 5):
        assert 0.0 < m[k] < 1.0
    np.testing.assert_allclose(m[4], 0.5, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dim", [1, 2, 4, 5, 6, 32])
def test_linear_ramp_mask_survives_the_min_equals_max_singularity(dim):
    """``maximum`` is bumped by 0.001, which turns the ramp into a step at ``minimum``."""
    m = R.linear_ramp_mask(4.0, 4.0, dim)
    assert m.shape == (dim,)
    assert np.all(np.isfinite(m))
    assert np.all(m >= 0.0) and np.all(m <= 1.0)
    np.testing.assert_array_equal(
        m, np.clip((np.arange(dim, dtype=np.float64) - 4.0) / 0.001, 0.0, 1.0)
    )
    # everything up to and including index `minimum` is exactly 0, everything above is 1
    np.testing.assert_array_equal(m[: min(dim, 5)], np.zeros(min(dim, 5)))
    if dim > 5:
        np.testing.assert_array_equal(m[5:], np.ones(dim - 5))


def test_linear_ramp_mask_accepts_integer_bounds():
    np.testing.assert_array_equal(
        R.linear_ramp_mask(1, 3, 6), R.linear_ramp_mask(1.0, 3.0, 6)
    )


# ==========================================================================
# find_correction_dim / find_correction_range
# ==========================================================================


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("rotations", [0.5, 1.0, 2.0, 8.0, 32.0])
def test_find_correction_dim_identifies_the_channel_with_that_many_turns(dim, base, rotations):
    """Hard meaning of the index: the channel it names makes exactly ``rotations``
    full turns of ``2 * pi`` radians across ``max_position_embeddings`` positions."""
    index = R.find_correction_dim(rotations, dim, base, 2048)
    inv_f_of_channel = base ** (-2.0 * index / dim)
    turns = 2048 * inv_f_of_channel / (2.0 * math.pi)
    assert turns == pytest.approx(rotations, rel=1e-13)


@pytest.mark.parametrize("dim", DIMS)
def test_find_correction_dim_decreases_with_the_rotation_count(dim):
    idx = [R.find_correction_dim(r, dim, 10000.0, 2048) for r in (0.25, 1.0, 4.0, 16.0, 64.0, 256.0)]
    assert all(idx[i] > idx[i + 1] for i in range(len(idx) - 1))
    # doubling the requested rotation count shifts the index down by dim * ln2 / (2 ln base)
    one = R.find_correction_dim(1.0, dim, 10000.0, 2048)
    two = R.find_correction_dim(2.0, dim, 10000.0, 2048)
    assert one - two == pytest.approx(dim * math.log(2.0) / (2.0 * math.log(10000.0)), rel=1e-12)


@pytest.mark.parametrize("dim", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("beta_fast, beta_slow", [(32.0, 1.0), (8.0, 2.0), (128.0, 0.5), (2.0, 1.0)])
def test_find_correction_range_is_ordered_and_inside_the_index_range(dim, beta_fast, beta_slow):
    lo, hi = R.find_correction_range(beta_fast, beta_slow, dim, 10000.0, 2048)
    assert 0.0 <= lo <= hi <= dim - 1.0


@pytest.mark.parametrize("dim", [8, 16, 32, 64, 128, 256])
def test_find_correction_range_is_untruncated_inside_the_index_range(dim):
    lo, hi = R.find_correction_range(32.0, 1.0, dim, 10000.0, 2048, truncate=False)
    assert 0.0 <= lo <= hi <= dim // 2


@pytest.mark.parametrize("dim", DIMS)
def test_find_correction_range_clamps_low_to_zero(dim):
    """Asking for a huge rotation count pushes the index below 0; it is clamped.

    The clamp is unconditional -- it applies with ``truncate=False`` as well.
    """
    raw = R.find_correction_dim(1e6, dim, 10000.0, 2048)
    assert raw < 0.0
    assert R.find_correction_range(1e6, 1e5, dim, 10000.0, 2048)[0] == 0.0
    assert R.find_correction_range(1e6, 1e5, dim, 10000.0, 2048, truncate=False)[0] == 0.0
    assert R.find_correction_range(1e9, 1e8, dim, 10000.0, 2048)[0] == 0.0


@pytest.mark.parametrize("dim", DIMS)
def test_find_correction_range_clamps_high_to_head_dim_minus_one(dim):
    """Asking for a tiny rotation count pushes the index above ``dim - 1``; it is clamped."""
    raw = R.find_correction_dim(1e-14, dim, 10000.0, 2048)
    assert raw > dim - 1.0
    assert R.find_correction_range(1e-12, 1e-14, dim, 10000.0, 2048)[1] == float(dim - 1)
    assert R.find_correction_range(1e-12, 1e-14, dim, 10000.0, 2048, truncate=False)[1] == float(dim - 1)
    assert R.find_correction_range(1e-30, 1e-31, dim, 10000.0, 2048)[1] == float(dim - 1)


@pytest.mark.parametrize("dim", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("beta_fast, beta_slow", [(32.0, 1.0), (8.0, 2.0), (128.0, 0.5)])
def test_find_correction_range_truncated_brackets_the_raw_values(dim, beta_fast, beta_slow):
    raw_lo, raw_hi = R.find_correction_range(beta_fast, beta_slow, dim, 10000.0, 2048, truncate=False)
    lo, hi = R.find_correction_range(beta_fast, beta_slow, dim, 10000.0, 2048)
    assert lo == math.floor(raw_lo) or lo == 0.0
    assert hi == math.ceil(raw_hi) or hi == dim - 1.0
    assert lo <= raw_lo <= raw_hi <= hi


@pytest.mark.parametrize("dim", DIMS)
def test_find_correction_range_is_untouched_by_clamping_for_sane_arguments(dim):
    lo, hi = R.find_correction_range(32.0, 1.0, dim, 10000.0, 2048)
    raw_lo, raw_hi = R.find_correction_range(32.0, 1.0, dim, 10000.0, 2048, truncate=False)
    assert lo > 0.0 and hi < dim - 1.0
    assert lo == math.floor(raw_lo) and hi == math.ceil(raw_hi)


def test_find_correction_range_known_values():
    assert R.find_correction_range(32.0, 1.0, 64, 10000.0, 2048) == (8, 21)
    assert R.find_correction_range(32.0, 1.0, 128, 10000.0, 2048) == (16, 41)
    assert R.find_correction_range(32.0, 1.0, 8, 10000.0, 2048) == (1, 3)


# ==========================================================================
# yarn_parameters
# ==========================================================================


def _yarn_range(dim, base=10000.0, mpe=2048, beta_fast=32.0, beta_slow=1.0):
    lo, hi = R.find_correction_range(beta_fast, beta_slow, dim, base, mpe)
    return int(lo), int(hi)


@pytest.mark.parametrize("dim", DIMS)
def test_yarn_scale_one_is_exactly_plain_rope(dim):
    """Fact 6."""
    f, mscale = R.yarn_parameters(dim, 10000.0, 1.0, 2048)
    plain = R.inv_freq(dim, 10000.0)
    np.testing.assert_allclose(f, plain, rtol=1e-15, atol=0.0)
    assert mscale == 1.0


def test_yarn_scale_one_is_bit_exactly_plain_rope_for_the_reference_config():
    f, mscale = R.yarn_parameters(64, 10000.0, 1.0, 2048)
    np.testing.assert_array_equal(f, R.inv_freq(64, 10000.0))
    assert mscale == 1.0


@pytest.mark.parametrize("dim", DIMS)
def test_yarn_scale_below_one_shrinks_nothing_into_plain_rope(dim):
    """A sub-unit ``scale`` is not the identity -- it can only lengthen every channel."""
    f, mscale = R.yarn_parameters(dim, 10000.0, 0.5, 2048)
    assert mscale == 1.0
    assert np.all(f >= R.inv_freq(dim, 10000.0))
    assert not np.array_equal(f, R.inv_freq(dim, 10000.0))


@pytest.mark.parametrize("scale", [1.5, 2.0, 8.0, 32.0, 4096.0])
def test_yarn_mscale_matches_get_mscale(scale):
    _f, mscale = R.yarn_parameters(64, 10000.0, scale, 2048)
    assert mscale == R.get_mscale(scale)
    assert mscale == pytest.approx(0.1 * math.log(scale) + 1.0, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("scale", [0.5, 1.0, 32.0])
def test_yarn_attn_factor_scales_the_mscale(scale):
    _f, plain = R.yarn_parameters(64, 10000.0, scale, 2048)
    for factor in (0.5, 2.0, 3.7):
        _f2, scaled = R.yarn_parameters(64, 10000.0, scale, 2048, attn_factor=factor)
        assert scaled == plain * factor


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("scale", [2.0, 8.0, 32.0, 100.0])
def test_yarn_preserves_the_highest_frequency_channels_exactly(dim, scale):
    """Channels with index ``<= low`` sit on a zero ramp, so they are untouched."""
    f, _ = R.yarn_parameters(dim, 10000.0, scale, 2048)
    plain = R.inv_freq(dim, 10000.0)
    lo, _hi = _yarn_range(dim)
    np.testing.assert_array_equal(f[: lo + 1], plain[: lo + 1])


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("scale", [2.0, 8.0, 32.0, 100.0])
def test_yarn_divides_the_lowest_frequency_channels_by_scale(dim, scale):
    """Channels with index ``>= high`` sit on a saturated ramp, so they are scaled by 1/s."""
    f, _ = R.yarn_parameters(dim, 10000.0, scale, 2048)
    plain = R.inv_freq(dim, 10000.0)
    _lo, hi = _yarn_range(dim)
    np.testing.assert_allclose(f[hi:], plain[hi:] / scale, rtol=1e-15, atol=0.0)
    assert np.all(f[hi:] < plain[hi:])


def test_yarn_low_frequency_channels_are_exactly_one_over_scale_for_the_reference_config():
    """Fact 7, tight form: the last six channels of the 64-dim case."""
    f, mscale = R.yarn_parameters(64, 10000.0, 32.0, 2048)
    plain = R.inv_freq(64, 10000.0)
    np.testing.assert_array_equal(f[:6], plain[:6])
    np.testing.assert_array_equal(f[-6:] / plain[-6:], np.full(6, 1.0 / 32.0))
    assert mscale == pytest.approx(0.1 * math.log(32) + 1.0, rel=1e-15)
    assert mscale == pytest.approx(1.346574, abs=5e-7)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("scale", [2.0, 8.0, 32.0, 128.0])
def test_yarn_blended_channels_lie_strictly_between_extrapolation_and_interpolation(dim, scale):
    f, _ = R.yarn_parameters(dim, 10000.0, scale, 2048)
    plain = R.inv_freq(dim, 10000.0)
    lo, hi = _yarn_range(dim)
    mid = f[lo + 1 : hi]
    if mid.size == 0:
        pytest.skip("no blended channels for this configuration")
    assert np.all(mid < plain[lo + 1 : hi])
    assert np.all(mid > plain[lo + 1 : hi] / scale)
    # the blend is monotone along the channel index
    assert np.all(np.diff(f) <= 0.0)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("scale", [2.0, 8.0, 32.0, 128.0])
def test_yarn_is_not_uniform_position_interpolation(dim, scale):
    """Fact 7: exactly the ``dim//2 - high`` slowest channels coincide with a plain
    ``inv_freq / scale``; every other channel is *faster* than that."""
    f, _ = R.yarn_parameters(dim, 10000.0, scale, 2048)
    uniform = R.inv_freq(dim, 10000.0) / scale
    lo, hi = _yarn_range(dim)
    np.testing.assert_array_equal(f[hi:], uniform[hi:])
    assert np.all(f[:hi] > uniform[:hi])
    n_differing = int(np.sum(f != uniform))
    assert n_differing == hi
    assert int(np.sum(f == uniform)) == dim // 2 - hi
    assert n_differing >= 1


def test_yarn_differs_from_uniform_interpolation_in_21_of_32_channels():
    """The exact count quoted in fact 7."""
    f, _ = R.yarn_parameters(64, 10000.0, 32.0, 2048)
    uniform = R.inv_freq(64, 10000.0) / 32.0
    assert int(np.sum(f != uniform)) == 21
    assert int(np.sum(f == uniform)) == 11


@pytest.mark.parametrize("dim", DIMS)
def test_yarn_never_increases_any_inverse_frequency_as_scale_grows(dim):
    plain = R.inv_freq(dim, 10000.0)
    previous = plain
    for scale in (1.0001, 1.5, 2.0, 4.0, 16.0, 32.0, 256.0, 4096.0, 1e6):
        f, mscale = R.yarn_parameters(dim, 10000.0, scale, 2048)
        assert np.all(f <= previous), f"channel frequency rose at scale={scale}"
        assert mscale >= 1.0
        previous = f
    lo, hi = _yarn_range(dim)
    # the fastest ``lo + 1`` channels are never touched, every slower one is lengthened
    np.testing.assert_array_equal(f[: lo + 1], plain[: lo + 1])
    assert f[0] == plain[0] == 1.0
    assert np.all(f[lo + 1 :] < plain[lo + 1 :])
    assert hi > lo


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("scale", [2.0, 32.0, 100.0])
def test_yarn_with_zero_extrapolation_reduces_to_uniform_interpolation(dim, scale):
    """``extrapolation_factor=0`` disables the extrapolation branch everywhere, which
    is exactly plain position interpolation -- the thing YaRN is *not*."""
    f, mscale = R.yarn_parameters(dim, 10000.0, scale, 2048, extrapolation_factor=0.0)
    np.testing.assert_allclose(f, R.inv_freq(dim, 10000.0) / scale, rtol=1e-15, atol=0.0)
    assert mscale == R.get_mscale(scale)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("scale", [2.0, 32.0])
def test_yarn_with_zero_extrapolation_is_bit_exactly_uniform_for_power_of_two_scales(dim, scale):
    """``1 / (scale * f)`` and ``(1 / f) / scale`` agree bit-for-bit only for dyadic
    ``scale``; this pins the claim exactly where it is exact."""
    f, _ = R.yarn_parameters(dim, 10000.0, scale, 2048, extrapolation_factor=0.0)
    np.testing.assert_array_equal(f, R.inv_freq(dim, 10000.0) / scale)


@pytest.mark.parametrize("dim", DIMS)
def test_yarn_does_not_change_the_base(dim):
    """The module's "common misstatement" note: ``base`` is not retuned to 500000."""
    f, _ = R.yarn_parameters(dim, 10000.0, 32.0, 2048)
    lo, _hi = _yarn_range(dim)
    np.testing.assert_array_equal(f[: lo + 1], R.inv_freq(dim, 10000.0)[: lo + 1])
    assert not np.allclose(f, R.inv_freq(dim, 500000.0), rtol=1e-3, atol=0.0)
    # the high-frequency channels *are* the base-10000 ladder, channel by channel
    assert f[0] == 1.0


@pytest.mark.parametrize("dim", DIMS)
def test_yarn_result_keeps_the_ladder_of_the_given_base(dim):
    for base in (10000.0, 500000.0):
        f, _ = R.yarn_parameters(dim, base, 32.0, 2048)
        plain = R.inv_freq(dim, base)
        lo, _hi = _yarn_range(dim, base=base)
        np.testing.assert_array_equal(f[: lo + 1], plain[: lo + 1])
        assert f[0] == 1.0


@pytest.mark.parametrize("dim", DIMS)
def test_yarn_beta_fast_controls_how_much_is_interpolated(dim):
    _f_default, _ = R.yarn_parameters(dim, 10000.0, 32.0, 2048)
    plain = R.inv_freq(dim, 10000.0)
    counts = []
    for beta_fast in (4.0, 8.0, 32.0, 64.0, 128.0):
        f, _ = R.yarn_parameters(dim, 10000.0, 32.0, 2048, beta_fast=beta_fast)
        lo, _hi = _yarn_range(dim, beta_fast=beta_fast)
        counts.append(int(np.sum(f == plain)))
        assert np.all(f[: lo + 1] == plain[: lo + 1])
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1]


def test_yarn_beta_fast_default_matches_the_documented_32_1():
    f, _ = R.yarn_parameters(64, 10000.0, 32.0, 2048, beta_fast=32.0, beta_slow=1.0)
    g, _ = R.yarn_parameters(64, 10000.0, 32.0, 2048)
    np.testing.assert_array_equal(f, g)


@pytest.mark.parametrize("dim", DIMS)
def test_yarn_mscale_is_independent_of_dim_and_base(dim):
    _f, mscale = R.yarn_parameters(dim, 10000.0, 32.0, 2048)
    assert mscale == 1.3465735902799727
    for base in (500000.0, 1e6):
        _f2, m2 = R.yarn_parameters(dim, base, 32.0, 2048)
        assert m2 == mscale


def test_yarn_output_shape_and_dtype():
    f, mscale = R.yarn_parameters(64, 10000.0, 32.0, 2048)
    assert f.shape == (32,)
    assert f.dtype == np.float64
    assert isinstance(mscale, float)


# ==========================================================================
# per_dim_angle
# ==========================================================================


@pytest.mark.parametrize("dim", DIMS)
def test_per_dim_angle_is_the_outer_product_of_delta_and_inv_freq(dim):
    inv_f = R.inv_freq(dim)
    delta = np.array([0.0, 1.0, -4.0, 3.5, 1000.0, 131072.0])
    angles = R.per_dim_angle(delta, inv_f)
    assert angles.shape == (delta.size, dim // 2)
    np.testing.assert_array_equal(angles, delta[:, None] * inv_f[None, :])


@pytest.mark.parametrize("dim", DIMS)
def test_per_dim_angle_agrees_with_rotary_angles(dim):
    """Both are the same quantity under two names; the module must not disagree."""
    inv_f = R.inv_freq(dim)
    delta = np.array([0.0, 2.0, -17.25, 4096.0])
    np.testing.assert_array_equal(R.per_dim_angle(delta, inv_f), R.rotary_angles(delta, inv_f))


@pytest.mark.parametrize("dim", DIMS)
def test_per_dim_angle_is_exactly_zero_for_a_zero_gap(dim):
    inv_f = R.inv_freq(dim)
    np.testing.assert_array_equal(R.per_dim_angle(np.zeros(4), inv_f), np.zeros((4, dim // 2)))


def test_per_dim_angle_casts_to_float64():
    inv_f = R.inv_freq(64, 10000.0)
    for delta in (np.arange(4), np.arange(4, dtype=np.float32), np.array([1, 2], dtype=np.int32)):
        angles = R.per_dim_angle(delta, inv_f)
        assert angles.dtype == np.float64
        np.testing.assert_array_equal(angles, np.asarray(delta, dtype=np.float64)[:, None] * inv_f[None, :])


def test_per_dim_angle_is_additive_over_the_gap():
    """``theta(a + b) == theta(a) + theta(b)`` -- the reason RoPE is relative."""
    inv_f = R.inv_freq(64, 10000.0)
    a, b = 7.0, 130.0
    lhs = R.per_dim_angle(np.array([a + b]), inv_f)
    rhs = R.per_dim_angle(np.array([a]), inv_f) + R.per_dim_angle(np.array([b]), inv_f)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-15, atol=1e-14)
