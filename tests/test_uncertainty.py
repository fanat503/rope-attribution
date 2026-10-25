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
    """The layer-head rates, not the head-prompt instances.

    v1 of the artifact had no ``head_rates`` field; this fixture would fail
    loudly rather than fall back to the five-fold over-count.
    """
    ids = [m["model_id"] for m in usefulness["models"]]
    rates = [
        [
            float(h["sign_flip_fraction"])
            for h in m["head_rates"]
            if h["sign_flip_fraction"] is not None
        ]
        for m in usefulness["models"]
    ]
    return ids, rates


def test_the_paper_quotes_the_checkpoint_mean_not_the_head_mean(
    per_head: tuple[list[str], list[list[float]]],
) -> None:
    """The headline is the equal-weight mean; the interval is head-pooled.

    The paper's headline "27.0%" is the *unweighted mean of the three
    per-checkpoint means*. The mean over all 558 layer-heads is 26.90% - lower,
    because SmolLM contributes 270 heads against 144 each for the other two.
    Both are defensible summaries; they are not the same number.

    The two differ by about 0.14 percentage points, which is *about one*
    half-width of the pooled interval, not more than two: the corrected
    interval includes the equal-weight headline. The earlier test claimed the
    opposite because it resampled 2790 head-prompt instances, which made the
    interval about sqrt(5) too narrow. This test now pins the corrected
    comparison rather than the old overstatement.
    """
    _ids, rates = per_head
    head_mean = sum(r for model in rates for r in model) / sum(len(m) for m in rates)
    checkpoint_mean = sum(sum(m) / len(m) for m in rates) / len(rates)

    assert abs(checkpoint_mean - 0.2704) < 0.0005, checkpoint_mean
    assert abs(head_mean - 0.2690) < 0.0005, head_mean

    out = UNC.bootstrap_ci([r for model in rates for r in model])
    assert out["ci_low"] < head_mean < out["ci_high"], out
    # The corrected interval includes the equal-weight headline, by a hair.
    assert out["ci_low"] < checkpoint_mean < out["ci_high"], (
        checkpoint_mean,
        out["ci_high"],
    )
    ratio = (checkpoint_mean - head_mean) / out["half_width"]
    assert 0.9 < ratio < 1.1, (ratio, checkpoint_mean, head_mean, out["half_width"])


def test_the_head_count_is_the_one_the_paper_quotes(
    per_head: tuple[list[str], list[list[float]]],
) -> None:
    """The replication unit. 2790 head-prompt instances are not 2790 heads.

    Mutation verdict: reverting ``_cluster_heads`` to pooling instances makes
    this assertion fail with 2790, which is exactly the defect the fix removes.
    """
    _ids, rates = per_head
    assert [len(r) for r in rates] == [144, 144, 270], [len(r) for r in rates]
    assert sum(len(r) for r in rates) == 558


def test_the_artifact_clusters_prompts_into_heads(usefulness: dict) -> None:
    """One record per layer-head, carrying how many prompts contributed.

    Mutation verdict: an unclustered artifact has 720 records per model where
    ``n_heads`` says 144, so this fails before the interval is ever computed.
    """
    for model in usefulness["models"]:
        assert model["n_heads"] == len(model["head_rates"]), model["model_id"]
        assert {h["n_prompts"] for h in model["head_rates"]} == {
            model["n_prompts"]
        }, model["model_id"]
        assert all(h["sign_flip_fraction"] is not None for h in model["head_rates"])


def test_a_v1_artifact_is_refused_rather_than_reinterpreted(
    usefulness: dict,
) -> None:
    """The schema gate that stops the old over-count from silently returning.

    Mutation verdict: deleting the schema check makes this return 2790 rates
    instead of raising, which is the exact regression the v2 bump prevents.
    """
    stale = dict(usefulness)
    stale["schema"] = "rope_attribution/usefulness/v1"
    with pytest.raises(ValueError, match="usefulness/v2"):
        UNC.head_rates_from_artifact(stale)


def test_the_three_checkpoints_differ_measurably(
    per_head: tuple[list[str], list[list[float]]],
) -> None:
    """Pythia differs from both; llama and SmolLM do not differ measurably.

    The earlier version of this test asserted that every pairwise interval
    excluded zero. That was an artefact of the same five-fold over-count: with
    the correct unit the llama--SmolLM interval straddles zero. A test that
    keeps the old assertion would force the paper to keep a false claim.
    """
    ids, rates = per_head
    out = UNC.checkpoint_agreement(rates, ids)
    assert not out["all_agree"], out
    agreeing = [p for p in out["pairs"] if p["agrees_at_95"]]
    assert len(agreeing) == 1, out["pairs"]
    assert agreeing[0]["pair"] == "JackFram/llama-160m vs HuggingFaceTB/SmolLM-135M"
    # and the spread of the three means must be in the range the paper reports
    means = out["per_checkpoint_mean"]
    assert min(means) > 0.26 and max(means) < 0.29, means


def test_the_interval_comes_from_heads_not_cells(per_head: tuple[list[str], list[list[float]]]) -> None:
    """The methodological point: resampling instances or cells would be wrong.

    Cells inside a head share a query and a key, and the five prompts are
    repeated looks at the same weights. The bootstrap over layer-heads is the
    honest unit. Checked by confirming the head count is what drives ``n``.
    """
    _ids, rates = per_head
    pooled = [r for model in rates for r in model]
    out = UNC.bootstrap_ci(pooled)
    assert out["n"] == 558
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
    mean over all 558 heads. Both are pinned, so the passage cannot collapse
    them into one number."""
    checkpoint, n, head = groups("boot_two_means")[0]
    rates = [
        [
            float(h["sign_flip_fraction"])
            for h in m["head_rates"]
            if h["sign_flip_fraction"] is not None
        ]
        for m in usefulness["models"]
    ]
    assert sum(len(row) for row in rates) == int(n) == 558
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
    # The passage's other quantitative claim: the two differ by about one
    # pooled interval half-width (0.142 vs 0.143 points), and the corrected
    # interval therefore includes the equal-weight headline. Re-derived here
    # rather than inferred from the wording.
    out = UNC.bootstrap_ci([rate for row in rates for rate in row])
    half_width = 100.0 * 0.5 * (out["ci_high"] - out["ci_low"])
    delta = 100.0 * float(np.mean([np.mean(row) for row in rates])) - head_mean
    assert 0.9 < delta / half_width < 1.1, (delta, half_width)
    gap_written = float(literal("boot_weighting_gap"))
    assert_paper_number_matches(delta, f"{gap_written:.2f}", "the weighting gap")
    unit_heads, unit_instances = (int(x) for x in groups("boot_interval_unit")[0])
    assert unit_heads == 558
    assert unit_instances == 2790
    assert "about one pooled interval" in TEX_FLAT
    assert "more than twice" not in TEX_FLAT


def test_the_head_counts_are_the_ones_in_the_json(usefulness: dict) -> None:
    smol, each = (int(value) for value in groups("boot_head_counts")[0])
    counts = {m["model_id"]: m["n_heads"] for m in usefulness["models"]}
    assert counts["HuggingFaceTB/SmolLM-135M"] == smol == 270
    assert [n for name, n in counts.items() if "SmolLM" not in name] == [each, each]
    assert each == 144
    assert sum(counts.values()) == int(literal("worth_head_count")) == smol + 2 * each
    instances = {
        m["model_id"]: m["n_head_prompt_instances"] for m in usefulness["models"]
    }
    assert sum(instances.values()) == 2790
    for model in usefulness["models"]:
        assert (
            model["n_head_prompt_instances"] == model["n_heads"] * model["n_prompts"]
        ), model["model_id"]


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
    # the mean, which is an order of magnitude smaller at n = 558.
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
