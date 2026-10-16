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

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "projects") not in sys.path:
    sys.path.insert(0, str(_REPO / "projects"))

from rope_attribution import agreement as A  # noqa: E402
from rope_attribution import real_model as RM  # noqa: E402
from rope_attribution import usefulness as U  # noqa: E402


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


# ==========================================================================
# ranking_votes -- the vote counter the checkpoint measurement pools
# ==========================================================================


def test_ranking_votes_agree_when_both_rankings_name_the_same_pair() -> None:
    """One surrogate per content pair, so its argmax is fixed; every exact row
    that also peaks at that pair agrees. Building the rows that way is what the
    checkpoint data does with a dominant pair."""
    rng = np.random.default_rng(31)
    e = rng.normal(size=(13, 8))
    e[:, 2] += 100.0
    s = np.zeros(8)
    s[2] = 50.0
    votes = A.ranking_votes(e, s)
    assert votes["n_live"] == 13
    assert votes["n_agree"] == 13
    assert votes["n_decile"] == 13
    assert votes["n_excluded_exact_zero"] == 0
    assert votes["n_excluded_surrogate_zero"] == 0


def test_ranking_votes_never_agree_when_the_surrogate_names_a_different_pair() -> None:
    """The exact ranking always peaks at pair 2, the surrogate always at pair 6.

    This is the strongest disagreement the statistic can see, and on the 8-pair
    heads the decile is the top-1, so both counts must be zero rather than
    merely small.
    """
    rng = np.random.default_rng(32)
    e = rng.normal(size=(13, 8))
    e[:, 2] += 100.0
    s = np.zeros(8)
    s[6] = 50.0
    votes = A.ranking_votes(e, s)
    assert votes["n_agree"] == 0
    assert votes["n_decile"] == 0
    assert votes["n_live"] == 13


def test_ranking_votes_excludes_an_all_zero_exact_row_and_counts_it() -> None:
    """The degenerate case that produced a false 100%: argmax 0 vs argmax 0.

    The row must not vote, and the exclusion must be visible in the counts --
    an excluded row that vanished silently would be the same bug one level down.
    """
    e = np.zeros((13, 8))
    e[3] = [3.0, 1.0, 2.0, 0.5, 0.1, 0.2, 0.4, 2.5]
    s = e[3]
    votes = A.ranking_votes(e, s)
    assert votes["n_rows"] == 13
    assert votes["n_live"] == 1
    assert votes["n_agree"] == 1
    assert votes["n_excluded_exact_zero"] == 12


def test_ranking_votes_excludes_an_all_zero_surrogate_and_counts_every_row() -> None:
    e = np.random.default_rng(33).normal(size=(13, 8))
    votes = A.ranking_votes(e, np.zeros(8))
    assert votes["n_live"] == 0
    assert votes["n_agree"] == 0
    assert votes["n_excluded_surrogate_zero"] == 13
    assert votes["n_excluded_exact_zero"] == 0


def test_ranking_votes_top_decile_differs_from_top1_for_larger_rows() -> None:
    """With 32 pairs the decile holds 3: a second-best pick is decile but not top1.

    On the 8-pair heads the two statistics are the same number by construction;
    this is the case where they must come apart, so the pooling cannot hide a
    broken decile behind a correct top1.
    """
    exact = np.zeros((1, 32))
    exact[0, :5] = [10.0, 9.0, 8.0, 1.0, 0.5]
    surrogate = exact[0].copy()
    surrogate[0] = 9.4
    surrogate[1] = 9.5  # the surrogate's best pair is the one the exact row ranks second
    votes = A.ranking_votes(exact, surrogate)
    assert votes["n_agree"] == 0
    assert votes["n_decile"] == 1


def test_ranking_votes_rejects_shape_mismatches_and_a_single_pair() -> None:
    with pytest.raises(ValueError):
        A.ranking_votes(np.zeros((4, 8)), np.zeros(9))
    with pytest.raises(ValueError):
        A.ranking_votes(np.zeros(8), np.zeros(8))
    with pytest.raises(ValueError):
        A.ranking_votes(np.zeros((4, 1)), np.zeros(1))


# ==========================================================================
# The committed artifact, results/agreement.json
# ==========================================================================


_JSON = _REPO / "results" / "agreement.json"


def _committed() -> dict:
    if not _JSON.exists():
        pytest.skip("results/agreement.json is absent; run agreement.py")
    return json.loads(_JSON.read_text(encoding="utf-8"))


def test_the_committed_agreement_artifact_is_self_describing() -> None:
    data = _committed()
    assert data["schema"] == "rope_attribution/agreement/v1"
    assert data["distance_grid"] == list(U.DISTANCE_GRID)
    assert data["top_decile"] == A.TOP_DECILE
    assert data["seed"] == RM.DEFAULT_SEED
    assert {m["model_id"] for m in data["models"]} == set(RM.DEFAULT_MODEL_IDS)


def test_every_committed_model_record_is_internally_consistent() -> None:
    data = _committed()
    for m in data["models"]:
        assert m["n_rotary_pairs"] == m["rope_dim"] // 2, m["model_id"]
        assert m["chance_top1"] == pytest.approx(1.0 / m["n_rotary_pairs"]), m["model_id"]
        assert m["deltas"] == list(U.DISTANCE_GRID), m["model_id"]
        assert m["n_agree"] <= m["n_votes"], m["model_id"]
        assert m["n_decile"] <= m["n_votes"], m["model_id"]
        # Every agreement is also a decile hit: if the surrogate names the exact
        # top pair, its pick is trivially inside the exact top decile.
        assert m["n_agree"] <= m["n_decile"], m["model_id"]
        assert m["top1_agreement"] == pytest.approx(m["n_agree"] / m["n_votes"]), m
        assert (
            m["top1_agreement_top_decile"]
            == pytest.approx(m["n_decile"] / m["n_votes"])
        ), m
        # The scientific claim, checked where the numbers live: the surrogate
        # names the exact top pair more often than chance on every checkpoint.
        assert m["top1_agreement"] > m["chance_top1"], m["model_id"]


def test_the_pooled_agreement_is_the_vote_weighted_mean_of_the_models() -> None:
    data = _committed()
    pooled = data["pooled"]
    assert pooled["n_agree"] == sum(m["n_agree"] for m in data["models"])
    assert pooled["n_live"] == sum(m["n_votes"] for m in data["models"])
    assert pooled["top1_agreement"] == pytest.approx(
        pooled["n_agree"] / pooled["n_live"]
    )
    assert (
        pooled["top1_agreement_top_decile"]
        == pytest.approx(pooled["n_decile"] / pooled["n_live"])
    )
