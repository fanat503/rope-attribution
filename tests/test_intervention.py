"""Pure-function tests for the intervention harness.

These run without torch: they pin the indexing, the sign, and the closed-form
attention, which are the three places a score-patching harness goes wrong
silently. The end-to-end hook is covered by ``tests/test_intervention_artifact.py``
when a checkpoint is available.

Mutation verdicts
-----------------
* ``test_pair_channels_fused_layout`` fails if the GPT-NeoX per-head interleave is
  replaced by a contiguous-thirds split --- the bug documented in
  ``qk_attribution.head_projections``, which is silent because shapes still fit.
* ``test_score_matrix_matches_rotated_dot`` fails if the causal angle is fed as
  ``+delta`` instead of ``-delta``; that mirrors every sine term and is the
  erratum corrected in the QK artefacts.
* ``test_attention_probs_is_a_distribution_and_masks_future`` fails if the masked
  upper triangle is zeroed rather than set to ``-inf``, which would put attention
  mass on future tokens.
* ``test_select_pairs_is_positive_only`` fails if the rule is changed to rank by
  magnitude, because suppressing a negative contribution raises the score.
* ``test_pair_contribution_uses_both_tokens`` fails if a single ``(A_j, B_j)`` is
  used for the whole matrix, which is the mistake the first draft made.
"""

from __future__ import annotations

import numpy as np
import pytest
from rope_attribution import qk_intervention as I
from rope_attribution.experiments import pair_terms
from rope_attribution.qk_attribution import rotated_dot
from rope_attribution.rope import inv_freq


def test_pair_channels_fused_layout() -> None:
    """GPT-NeoX interleaves (q, k, v) per head, so q starts at 3*h*head_dim."""
    channels = I.pair_channels(
        head=2, pairs=[0, 3], head_dim=8, rotary_dim=8, n_heads=4, layout="fused"
    )
    base = 2 * 3 * 8
    assert channels == [base, base + 4, base + 3, base + 7]


def test_pair_channels_split_layout() -> None:
    channels = I.pair_channels(
        head=3, pairs=[1], head_dim=16, rotary_dim=8, n_heads=4, layout="split"
    )
    base = 3 * 16
    assert channels == [base + 1, base + 1 + 4]


def test_pair_channels_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        I.pair_channels(0, [2], head_dim=8, rotary_dim=4, n_heads=1, layout="split")


def test_score_matrix_matches_rotated_dot() -> None:
    """Two independent constructions of the same score must agree everywhere."""
    rng = np.random.default_rng(4)
    rotary_dim = 4
    n_tokens = 7
    head_dim = 8
    freqs = inv_freq(rotary_dim, 10000.0)
    q = rng.normal(size=(n_tokens, head_dim))
    k = rng.normal(size=(n_tokens, head_dim))
    scores = I.score_matrix(q, k, freqs, rotary_dim)
    assert np.all(np.isneginf(scores[np.triu_indices(n_tokens, k=1)]))
    for t in range(n_tokens):
        for s in range(t + 1):
            expected = rotated_dot(
                q[t : t + 1], k[s : s + 1], -(t - s), freqs, rotary_dim
            )[0, 0]
            assert scores[t, s] == pytest.approx(expected, abs=1e-12)


def test_pair_contribution_uses_both_tokens() -> None:
    """A_j and B_j depend on the token pair, so the matrix cannot be rank one."""
    rng = np.random.default_rng(12)
    rotary_dim = 8
    n_tokens = 6
    freqs = inv_freq(rotary_dim, 10000.0)
    q = rng.normal(size=(n_tokens, rotary_dim))
    k = rng.normal(size=(n_tokens, rotary_dim))
    matrix = I.pair_contribution_matrix(q, k, 0, float(freqs[0]), rotary_dim)
    assert np.all(matrix[np.triu_indices(n_tokens, k=1)] == 0.0)
    # Row 2's entries at distances 0, 1, 2 must match the per-pair terms of the
    # (query 2, key s) pair, computed token by token.
    for s in (2, 1, 0):
        terms = pair_terms(q[2], k[s], freqs, -(2 - s))
        expected = terms.aligned[0] * np.cos(terms.angle[0]) + terms.crossed[0] * np.sin(
            terms.angle[0]
        )
        assert matrix[2, s] == pytest.approx(float(expected), abs=1e-12)
    assert not np.allclose(matrix, np.tril(np.outer(np.diag(matrix), np.ones(n_tokens))))


def test_attention_probs_is_a_distribution_and_masks_future() -> None:
    rng = np.random.default_rng(6)
    scores = I.score_matrix(
        rng.normal(size=(5, 4)), rng.normal(size=(5, 4)), inv_freq(4, 10000.0), 4
    )
    probs = I.attention_probs(scores, head_dim=4)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs[np.triu_indices(5, k=1)] == 0.0)


def test_select_pairs_is_positive_only() -> None:
    values = np.array([-5.0, 0.0, 2.0, 1.0, 3.0])
    assert I.select_pairs(values, 2) == [4, 2]
    assert I.select_pairs(values, 10) == [4, 2, 3]
    assert I.select_pairs(np.array([-1.0, -2.0]), 1) == []


def test_predict_attention_without_ablation_matches_the_score_matrix() -> None:
    rng = np.random.default_rng(8)
    rotary_dim = 4
    n_tokens = 6
    freqs = inv_freq(rotary_dim, 10000.0)
    q = rng.normal(size=(n_tokens, rotary_dim))
    k = rng.normal(size=(n_tokens, rotary_dim))
    probs = I.predict_attention(q, k, freqs, rotary_dim, head_dim=4)
    expected = I.attention_probs(I.score_matrix(q, k, freqs, rotary_dim), 4)
    assert np.allclose(probs, expected)


def test_ablation_removes_exactly_the_selected_pair() -> None:
    """Predicting with pair j removed must use the true contribution of pair j."""
    rng = np.random.default_rng(9)
    rotary_dim = 8
    n_tokens = 6
    freqs = inv_freq(rotary_dim, 10000.0)
    q = rng.normal(size=(n_tokens, rotary_dim))
    k = rng.normal(size=(n_tokens, rotary_dim))
    pair = 2
    ablated = I.predict_attention(q, k, freqs, rotary_dim, head_dim=8, ablate=(pair,))
    scores = I.score_matrix(q, k, freqs, rotary_dim)
    kept = scores - I.pair_contribution_matrix(q, k, pair, float(freqs[pair]), rotary_dim)
    assert np.allclose(ablated, I.attention_probs(kept, 8))


def test_surrogate_is_a_constant_over_distance() -> None:
    """The surrogate vector carries no delta: one number per pair, by construction."""
    freqs = inv_freq(8, 10000.0)
    aligned = np.array([1.0, 2.0, -0.5, 0.25])
    crossed = np.array([0.1, -0.3, 0.7, 0.0])
    first = I._surrogate_vector(aligned, crossed, freqs, 16)
    second = I._surrogate_vector(aligned, crossed, freqs, 16)
    assert np.array_equal(first, second)
    assert first.shape == freqs.shape


def test_surrogate_window_is_the_visible_one() -> None:
    """With one visible key the surrogate must be the exact contribution.

    The window of a query at position ``q`` is distances ``1 .. q``. At ``q = 1``
    there is exactly one distance, so a least-squares fit over the window has to
    reproduce the truth --- this is the degenerate case that pins the window
    boundaries. An off-by-one in either direction (``arange(1, q)`` or
    ``arange(0, q)``) fails here, and the same off-by-one is what makes the
    surrogate agree with the exact rule for the wrong reason everywhere else.
    """
    rng = np.random.default_rng(23)
    freqs = inv_freq(8, 10000.0)
    aligned = rng.normal(size=freqs.size)
    crossed = rng.normal(size=freqs.size)
    exact_at_one = aligned * np.cos(freqs * 1.0) + crossed * np.sin(freqs * 1.0)
    assert np.allclose(
        I._surrogate_vector(aligned, crossed, freqs, 1), exact_at_one, atol=1e-15
    )
    # And the fit must actually move with the window, or "the surrogate" is a
    # single number pretending to be a table.
    assert not np.allclose(
        I._surrogate_vector(aligned, crossed, freqs, 8),
        I._surrogate_vector(aligned, crossed, freqs, 256),
    )
