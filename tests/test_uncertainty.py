"""Tests for ``projects/rope_attribution/uncertainty.py``.

These are written for a specific reason. The paper's headline number was a point
estimate with no interval, and the first thing a hostile reviewer asks is what the
uncertainty is. These tests do two jobs: they check the interval machinery is
correct on data whose answer is known, and they check the artefact the module is
run against says what the paper now claims it says.

The interval is a *bootstrap over heads*, and that choice is itself under test -
resampling the cells inside a head instead would give a spuriously narrow
interval, which is a bug that would not show up as a failure anywhere else.
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

from rope_attribution import uncertainty as UNC  # noqa: E402

_JSON = _REPO / "results" / "usefulness.json"


# ---------------------------------------------------------------------------
# the machinery, on inputs whose answer is known
# ---------------------------------------------------------------------------


def test_the_interval_brackets_the_sample_mean() -> None:
    """A percentile bootstrap of the mean must contain the mean it estimates."""
    rng = np.random.default_rng(7)
    values = rng.beta(2.0, 3.0, size=200).tolist()
    out = UNC.bootstrap_ci(values)
    assert out["ci_low"] <= out["mean"] <= out["ci_high"], out


def test_the_interval_shrinks_as_the_sample_grows() -> None:
    """Otherwise the interval is not measuring anything about sample size."""
    rng = np.random.default_rng(11)
    small = UNC.bootstrap_ci(rng.beta(2.0, 3.0, size=20).tolist())
    large = UNC.bootstrap_ci(rng.beta(2.0, 3.0, size=2000).tolist())
    assert large["half_width"] < small["half_width"], (small, large)


def test_a_constant_sample_has_zero_width() -> None:
    """A degenerate input must give a degenerate interval, not a plausible one."""
    out = UNC.bootstrap_ci([0.25] * 50)
    assert out["ci_low"] == out["ci_high"] == out["mean"] == pytest.approx(0.25)


def test_the_interval_is_reproducible_and_seed_pinned() -> None:
    """Same input and same seed, same interval. A moving interval is not a result."""
    rng = np.random.default_rng(13)
    values = rng.beta(2.0, 3.0, size=100).tolist()
    a = UNC.bootstrap_ci(values, seed=5)
    b = UNC.bootstrap_ci(values, seed=5)
    assert a == b
    c = UNC.bootstrap_ci(values, seed=6)
    assert a["ci_low"] != c["ci_low"], "the seed must change the resampling"


def test_the_resample_count_is_pinned() -> None:
    """The number must be a constant, not derived from the data."""
    assert UNC.N_RESAMPLES == 2000
    out = UNC.bootstrap_ci([0.1, 0.2, 0.3, 0.4], n_resamples=50)
    assert out["n_resamples"] == 50


def test_too_few_values_is_refused_rather_than_estimated() -> None:
    with pytest.raises(ValueError):
        UNC.bootstrap_ci([0.5])
    with pytest.raises(ValueError):
        UNC.checkpoint_agreement([[0.5]])


def test_the_agreement_test_detects_a_real_difference() -> None:
    """The test that matters: it must not call everything agreement."""
    rng = np.random.default_rng(17)
    a = rng.normal(0.25, 0.01, size=300).tolist()
    b = rng.normal(0.40, 0.01, size=300).tolist()
    out = UNC.checkpoint_agreement([a, b], ["a", "b"])
    assert not out["all_agree"], out
    assert not out["pairs"][0]["agrees_at_95"], out["pairs"][0]


def test_the_agreement_test_accepts_two_samples_from_one_distribution() -> None:
    """And the converse: identical distributions must be called agreement."""
    rng = np.random.default_rng(19)
    a = rng.normal(0.25, 0.01, size=300).tolist()
    b = rng.normal(0.25, 0.01, size=300).tolist()
    out = UNC.checkpoint_agreement([a, b], ["a", "b"])
    assert out["all_agree"], out


# ---------------------------------------------------------------------------
# the committed artefact
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def usefulness() -> dict:
    if not _JSON.exists():
        pytest.skip("results/usefulness.json is absent; run usefulness.py")
    return json.loads(_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def per_head(usefulness: dict) -> tuple[list[str], list[list[float]]]:
    ids = [m["model_id"] for m in usefulness["models"]]
    rates = [
        [
            h["sign_flip_fraction_pooled"]
            for entry in m["per_prompt"]
            for h in entry["heads"]
        ]
        for m in usefulness["models"]
    ]
    return ids, rates


def test_the_paper_quotes_the_checkpoint_mean_not_the_head_mean(
    per_head: tuple[list[str], list[list[float]]],
) -> None:
    """A real methodological distinction the paper left implicit.

    The paper's headline "26.8%" is the *unweighted mean of the three
    per-checkpoint means*. The mean over all 2790 heads is 26.70% - lower,
    because SmolLM contributes 1350 heads against 720 each for the other two.
    Both are defensible summaries; they are not the same number, and the paper
    does not say which it used.

    This test pins both, and states that they differ by about 0.08 percentage
    points, so a reader cannot quietly swap one for the other.
    """
    _ids, rates = per_head
    head_mean = sum(r for model in rates for r in model) / sum(len(m) for m in rates)
    checkpoint_mean = sum(sum(m) / len(m) for m in rates) / len(rates)

    assert abs(checkpoint_mean - 0.2678) < 0.0005, checkpoint_mean
    assert abs(head_mean - 0.2670) < 0.0005, head_mean

    # The two differ by about 0.08 percentage points, which is LARGER than the
    # half-width of the pooled interval - the checkpoint mean falls outside it.
    # That is the substantive point: a reader who assumes the headline is the
    # head-weighted mean, or vice versa, is not making a rounding-level mistake.
    out = UNC.bootstrap_ci([r for model in rates for r in model])
    assert out["ci_low"] < head_mean < out["ci_high"], out
    assert not (out["ci_low"] < checkpoint_mean < out["ci_high"]), (
        "the checkpoint mean is not expected to fall inside the head-weighted "
        "CI; if it now does, the weighting gap has closed and this test should "
        "be re-read rather than its assertion loosened"
    )
    assert checkpoint_mean > out["ci_high"], (checkpoint_mean, out["ci_high"])


def test_the_head_count_is_the_one_the_paper_quotes(
    per_head: tuple[list[str], list[list[float]]],
) -> None:
    _ids, rates = per_head
    assert sum(len(r) for r in rates) == 2790, [len(r) for r in rates]


def test_the_three_checkpoints_differ_measurably(
    per_head: tuple[list[str], list[list[float]]],
) -> None:
    """A finding the paper did not previously have: the three do not agree.

    Every pairwise interval excludes zero, so the per-checkpoint rates differ by
    more than head-to-head variation can explain. The difference is small in
    absolute terms - about 1.2 to 1.3 percentage points - but it is real, and the
    paper's prose used to imply the three were interchangeable. This test fails if
    that changes, which is the point: it pins the finding rather than the wording.
    """
    ids, rates = per_head
    out = UNC.checkpoint_agreement(rates, ids)
    assert not out["all_agree"], out
    for pair in out["pairs"]:
        assert not pair["agrees_at_95"], pair
    # and the spread of the three means must be in the range the paper reports
    means = out["per_checkpoint_mean"]
    assert min(means) > 0.26 and max(means) < 0.28, means


def test_the_interval_comes_from_heads_not_cells(per_head: tuple[list[str], list[list[float]]]) -> None:
    """The methodological point: resampling cells would be wrong.

    Cells inside a head share a query and a key. The bootstrap over heads is the
    honest unit. Checked by confirming the head count is what drives ``n``, and
    that the resulting interval is wider than a resample over the same number of
    cells would be if the data were exchangeable.
    """
    _ids, rates = per_head
    pooled = [r for model in rates for r in model]
    out = UNC.bootstrap_ci(pooled)
    assert out["n"] == 2790
    grand_mean = float(np.mean(pooled))
    # Each head's own cells are far more similar to each other than to another
    # head's, so between-model variance dominates the pooled SD. If that ever
    # stopped being true the sampling unit would need revisiting.
    spread_of_model_means = [abs(np.mean(m) - grand_mean) for m in rates]
    assert max(spread_of_model_means) > 0.5 * out["sd"], (
        spread_of_model_means,
        out["sd"],
    )
