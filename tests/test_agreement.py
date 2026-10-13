"""Tests for ``projects/rope_attribution/agreement.py``.

The module answers one question: if you rank the rotary pairs by how much each
contributes and act on the top one, does the position-free surrogate put the same
pair first? That makes it the one place in the repository where a statistic can be
satisfied by the *absence* of signal - an all-zero row has argmax 0 in both
orderings and reports perfect agreement - so the degenerate case is tested
explicitly rather than assumed away.

Every test on known answers is checkable without a checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "projects") not in sys.path:
    sys.path.insert(0, str(_REPO / "projects"))

from rope_attribution import agreement as A  # noqa: E402


def test_average_ranks_breaks_ties_by_averaging() -> None:
    """Ties must share a rank, or the correlation depends on the sort order."""
    assert list(A.average_ranks([10.0, 20.0, 20.0, 30.0])) == [1.0, 2.5, 2.5, 4.0]


def test_average_ranks_handles_a_constant_row() -> None:
    assert list(A.average_ranks([5.0, 5.0, 5.0, 5.0])) == [2.5, 2.5, 2.5, 2.5]


def test_average_ranks_rejects_a_single_value() -> None:
    with pytest.raises(ValueError):
        A.average_ranks([1.0])


def test_spearman_of_a_row_with_itself_is_one() -> None:
    x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    assert A.spearman(x, x) == pytest.approx(1.0)


def test_spearman_of_a_reversed_row_is_minus_one() -> None:
    """A strictly decreasing transform is exactly anticorrelated.

    The row must have no ties: with a repeated value the reversed ordering is
    only approximately anticorrelated, and an approximate assertion would hide a
    real sign error in the rank machinery.
    """
    x = [3.0, 1.0, 4.0, 5.0, 9.0, 2.0, 6.0]
    assert len(set(x)) == len(x)
    # Negation, not reversal: reversing a list is not the rank-space negation
    # unless the permutation is an involution, and asserting on the reverse gave
    # +0.32 on a row that is exactly anticorrelated under negation.
    assert A.spearman(x, [-v for v in x]) == pytest.approx(-1.0)


def test_spearman_of_a_constant_row_is_zero_not_a_crash() -> None:
    """A constant ranking carries no information, and must not raise or NaN."""
    assert A.spearman([1.0, 2.0, 3.0], [7.0, 7.0, 7.0]) == 0.0
    assert np.isfinite(A.spearman([1.0, 2.0, 3.0], [7.0, 7.0, 7.0]))


def test_spearman_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        A.spearman([1.0, 2.0], [1.0, 2.0, 3.0])


def test_spearman_is_insensitive_to_a_positive_scale() -> None:
    assert A.spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)


def test_top1_agreement_is_one_for_identical_inputs() -> None:
    rng = np.random.default_rng(5)
    e = rng.normal(size=(10, 12))
    assert A.top1_agreement(e, e) == pytest.approx(1.0)


def test_top1_agreement_is_zero_for_an_opposite_ranking() -> None:
    e = np.tile(np.arange(12, dtype=float), (6, 1))
    assert A.top1_agreement(e, -e) == pytest.approx(0.0)


def test_top1_agreement_needs_two_dimensions() -> None:
    with pytest.raises(ValueError):
        A.top1_agreement(np.zeros(5), np.zeros(5))


def test_top1_agreement_rejects_a_shape_mismatch() -> None:
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        A.top1_agreement(rng.normal(size=(4, 3)), rng.normal(size=(4, 5)))


def test_the_conditioned_variants_are_not_masking_no_ops() -> None:
    """The bug this test exists to prevent.

    The first version computed the conditioned agreement by masking a row with
    ``np.where`` and taking the argmax. For the top decile that is a no-op, since
    a row's maximum is in its own top decile by construction, so all three
    conditioned statistics reported the unconditional number. A mildly perturbed
    surrogate is the case where they must differ.
    """
    rng = np.random.default_rng(9)
    e = rng.normal(size=(40, 20))
    s = e + rng.normal(scale=0.4, size=e.shape)
    prof = A.agreement_profile(e, s)
    assert prof["top1_agreement"] < prof["top1_agreement_top_decile"], prof


def test_the_decile_variant_approaches_chance_for_an_independent_surrogate() -> None:
    rng = np.random.default_rng(13)
    e = rng.normal(size=(400, 20))
    s = rng.normal(size=e.shape)
    prof = A.agreement_profile(e, s)
    assert abs(prof["top1_agreement_top_decile"] - A.TOP_DECILE) < 0.06, prof
    assert prof["top1_agreement"] < 0.10, prof


def test_material_restriction_reports_the_subset_it_covers() -> None:
    rng = np.random.default_rng(17)
    scale = np.array([1e-9] * 5 + [1.0] * 11)
    e = rng.normal(size=(20, 16)) * scale
    s = e + rng.normal(scale=0.1, size=e.shape)
    prof = A.agreement_profile(e, s, materiality=0.5)
    # the five attenuated columns make three of twenty rows fall below the cut
    assert 0 < prof["n_material_rows"] < 20, prof
    assert np.isfinite(prof["top1_agreement_material"])
    # and the cut is a real subset: raising it must shrink the counted rows
    stricter = A.agreement_profile(e, s, materiality=0.99)
    assert stricter["n_material_rows"] <= prof["n_material_rows"]


def test_an_all_zero_surrogate_is_visible_as_vacuous() -> None:
    """The degenerate case that produced a false result.

    The ranking experiment first measured a head whose contributions were all
    zero. argmax of an all-zero row is index 0 in both orderings, so it reported
    perfect agreement on the absence of signal. The caller has to be able to see
    that: zero correlation, and a decile figure of 1.0 which is vacuously true.
    """
    prof = A.agreement_profile(np.zeros((4, 8)), np.zeros((4, 8)))
    assert prof["rank_correlation"]["mean"] == 0.0
    assert prof["top1_agreement_top_decile"] == 1.0


def test_the_profile_reports_every_statistic_it_documents() -> None:
    rng = np.random.default_rng(23)
    prof = A.agreement_profile(rng.normal(size=(7, 9)), rng.normal(size=(7, 9)))
    expected = {
        "n_distances", "n_pairs", "top1_agreement",
        "top1_agreement_top_decile", "top1_agreement_material",
        "n_material_rows", "rank_correlation",
    }
    assert set(prof) == expected, sorted(set(prof) ^ expected)
    assert set(prof["rank_correlation"]) == {"per_row", "mean", "sd", "min", "max"}


def test_the_decile_is_fixed_in_advance() -> None:
    """A threshold chosen after seeing the result is not a threshold."""
    assert A.TOP_DECILE == 0.10
