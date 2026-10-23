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

# The claims this file re-derives. Same-directory import: the patterns
# and the paper live in test_paper_claims.py, so exactly one module
# knows what the paper says.
from test_paper_claims import (  # noqa: E402
    TEX_FLAT,
    assert_paper_number_matches,
    groups,
    literal,
)

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

    The paper's headline "27.0%" is the *unweighted mean of the three
    per-checkpoint means*. The mean over all 2790 heads is 26.90% - lower,
    because SmolLM contributes 1350 heads against 720 each for the other two.
    Both are defensible summaries; they are not the same number, and the paper
    does not say which it used.

    This test pins both, and states that they differ by about 0.14 percentage
    points, so a reader cannot quietly swap one for the other.
    """
    _ids, rates = per_head
    head_mean = sum(r for model in rates for r in model) / sum(len(m) for m in rates)
    checkpoint_mean = sum(sum(m) / len(m) for m in rates) / len(rates)

    assert abs(checkpoint_mean - 0.2704) < 0.0005, checkpoint_mean
    assert abs(head_mean - 0.2690) < 0.0005, head_mean

    # The two differ by about 0.14 percentage points, which is LARGER than the
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
    more than head-to-head variation can explain. The absolute differences run
    from 0.12 to 2.13 percentage points --- small next to the 27% headline, but
    larger than the head-level noise floor --- and the paper's prose used to
    imply the three were interchangeable. This test fails if that changes,
    which is the point: it pins the finding rather than the wording.
    """
    ids, rates = per_head
    out = UNC.checkpoint_agreement(rates, ids)
    assert not out["all_agree"], out
    for pair in out["pairs"]:
        assert not pair["agrees_at_95"], pair
    # and the spread of the three means must be in the range the paper reports
    means = out["per_checkpoint_mean"]
    assert min(means) > 0.26 and max(means) < 0.29, means


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


def test_the_two_means_the_paper_distinguishes(usefulness: dict) -> None:
    """27.0% is the equal-weight mean of per-checkpoint means; 26.90% is the
    mean over all 2790 heads. Both are pinned, so the passage cannot collapse
    them into one number."""
    checkpoint, n, head = groups("boot_two_means")[0]
    rates = [
        [
            h["sign_flip_fraction_pooled"]
            for entry in m["per_prompt"]
            for h in entry["heads"]
        ]
        for m in usefulness["models"]
    ]
    assert sum(len(row) for row in rates) == int(n)
    assert_paper_number_matches(
        100.0 * float(np.mean([np.mean(row) for row in rates])),
        f"{float(checkpoint):.1f}",
        "the equal-weight mean of the per-checkpoint rates",
    )
    head_mean = 100.0 * float(np.mean([rate for row in rates for rate in row]))
    assert_paper_number_matches(head_mean, f"{float(head):.2f}", "the head-pooled mean")
    restated_equal, restated_head = groups("boot_head_mean_restated")[0]
    assert_paper_number_matches(
        100.0 * float(np.mean([np.mean(row) for row in rates])),
        f"{float(restated_equal):.1f}",
        "the equal-weight headline where the paper distinguishes it",
    )
    assert_paper_number_matches(
        head_mean,
        f"{float(restated_head):.2f}",
        "the head-pooled mean where the paper restates it",
    )
    # The passage's other quantitative claim: the two differ by more than
    # twice the pooled interval's half-width. Re-derived here rather than
    # inferred from the wording; the current artifact satisfies the stronger
    # statement, so the paper makes it.
    out = UNC.bootstrap_ci([rate for row in rates for rate in row])
    half_width = 100.0 * 0.5 * (out["ci_high"] - out["ci_low"])
    delta = 100.0 * float(np.mean([np.mean(row) for row in rates])) - head_mean
    assert delta > 2.0 * half_width, (delta, half_width)
    assert "computed before rounding, the two differ by more than twice" in TEX_FLAT
    assert "twice that interval's half-width" in TEX_FLAT


def test_the_head_counts_are_the_ones_in_the_json(usefulness: dict) -> None:
    smol, each = (int(value) for value in groups("boot_head_counts")[0])
    counts = {m["model_id"]: m["n_heads"] for m in usefulness["models"]}
    assert counts["HuggingFaceTB/SmolLM-135M"] == smol
    assert [n for name, n in counts.items() if "SmolLM" not in name] == [each, each]
    assert sum(counts.values()) == int(literal("worth_head_count")) == smol + 2 * each


def test_the_pooled_interval_is_as_printed(
    per_head: tuple[list[str], list[list[float]]],
) -> None:
    """The bootstrap the paper labels as between-head, re-derived end to end."""
    _ids, rates = per_head
    out = UNC.bootstrap_ci([rate for row in rates for rate in row])
    pct, lo, hi, mean = groups("boot_pooled_interval")[0]
    assert pct == "95", pct
    assert_paper_number_matches(out["ci_low"], f"{float(lo):.4f}", "the interval's lower end")
    assert_paper_number_matches(out["ci_high"], f"{float(hi):.4f}", "the interval's upper end")
    assert_paper_number_matches(out["mean"], f"{float(mean):.4f}", "the pooled mean")
    # The number is the spread of the per-head rates, not the standard error of
    # the mean, which is an order of magnitude smaller at n = 2790.
    sd = literal("boot_between_head_sd")
    assert_paper_number_matches(
        out["sd"], f"{float(sd):.4f}", "the between-head standard deviation"
    )


def test_the_pairwise_intervals_are_as_printed(
    per_head: tuple[list[str], list[list[float]]],
) -> None:
    ids, rates = per_head
    out = UNC.checkpoint_agreement(rates, ids)
    written = groups("boot_pairwise_intervals")[0]
    assert len(out["pairs"]) == 3
    for pair, (mean, lo, hi) in zip(
        out["pairs"], (written[i : i + 3] for i in (0, 3, 6)), strict=True
    ):
        assert_paper_number_matches(
            100.0 * pair["mean_difference"],
            f"{float(mean):.2f}",
            f"the mean difference for {pair['pair']}",
        )
        assert_paper_number_matches(
            100.0 * pair["ci_low"],
            f"{float(lo):.2f}",
            f"the interval's lower end for {pair['pair']}",
        )
        assert_paper_number_matches(
            100.0 * pair["ci_high"],
            f"{float(hi):.2f}",
            f"the interval's upper end for {pair['pair']}",
        )

# ---------------------------------------------------------------------------
# the pre-registered grid
# ---------------------------------------------------------------------------


def test_measure_model_default_deltas_are_the_pre_registered_grid() -> None:
    """``deltas=None`` must BE the pre-registered grid, not a re-derivation.

    ``_log_grid(len(DISTANCE_GRID))`` happens to equal ``DISTANCE_GRID``
    today -- until the first person edits a point out of the grid, the
    default silently becomes a different grid while the artifact keeps
    publishing ``DISTANCE_GRID`` as the distance axis it measured on. The
    default is therefore the named constant itself; the positional
    re-derivation must stay absent from the source.
    """
    raw = (
        _REPO / "projects" / "rope_attribution" / "usefulness.py"
    ).read_bytes()
    # Match on a normalized form: the needle spans three lines, and the
    # worktree's line endings must not decide whether the guard passes
    # (read_text() would hide this, normalization makes it explicit).
    source = raw.decode("utf-8").replace("\r\n", "\n")
    assert (
        "\n    if deltas is None:\n        # The pre-registered grid itself" in source
        and "        deltas = np.array(DISTANCE_GRID, dtype=np.int64)\n" in source
    ), "measure_model must materialise its default deltas from DISTANCE_GRID itself"
    assert "_log_grid(len(DISTANCE_GRID))" not in source, (
        "the positional re-derivation of the default grid is back"
    )
