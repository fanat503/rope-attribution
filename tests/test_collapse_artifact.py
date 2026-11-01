"""The committed collapse artifact, checked against the theorem it is evidence for.

``tests/test_collapse.py`` pins the formulas. This file pins the *measurement*:
it reads ``results/collapse.json`` and asks whether the three checkpoints obey
Proposition~5 where the theorem says they must, and whether the dilution claim
holds where the architecture makes it testable.

The distinction matters. A formula test can pass while the artifact was produced
by a buggy loop, a stale grid, or a different model set. These tests read the
numbers the paper quotes and check them against the closed form, so a silent
change in either one fails the suite.

Mutation verdicts
-----------------
* ``test_bins_past_threshold_are_exactly_one`` fails if ``phase_extremes`` stops
  detecting the trough inside the window (the error then falls below 1) or if
  any pair is dropped from the bin aggregate. It is the theorem at the artifact
  level, so it cannot be satisfied by a wrong implementation.
* ``test_bins_below_threshold_respect_the_bound`` fails if the sub-threshold
  bound is computed with ``x`` instead of ``x/2``, and also if the window is
  recorded as ``span`` instead of ``span`` worth of phase at the recorded
  frequencies --- the two disagree by a factor of two, which no tolerance hides.
* ``test_least_squares_never_beats_minimax`` fails if the mean surrogate is
  accidentally fitted with knowledge of the phase (e.g. a midrange instead of a
  mean), which would make it look better than the bound allows.
* ``test_median_error_is_monotone_in_window`` fails if the window lengths are
  applied to the wrong axis of the broadcast table, which would shuffle the rows
  between windows.
* ``test_partial_rope_has_a_tail_and_full_rope_does_not`` fails if the unrotated
  tail is omitted from ``score_profile``'s denominator, because then a partial
  RoPE checkpoint would report the same tail share as a full-RoPE one.
* ``test_pooled_summary_matches_the_model_tables`` fails if the cross-model
  summary is filled in from a different run than the per-model tables.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "results" / "collapse.json"

TWO_PI = 2.0 * math.pi

# The three checkpoints the paper uses everywhere else. Restating them here means
# a run that silently drops one fails, rather than producing a smaller artifact
# that still validates.
EXPECTED_MODELS = (
    "EleutherAI/pythia-160m",
    "JackFram/llama-160m",
    "HuggingFaceTB/SmolLM-135M",
)

# Tolerances. The theorem is exact, so these only absorb float64 round-off in
# ``cos``/``sin`` and the JSON round-trip, which is ~1e-16 per value.
EXACT = 1e-12


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.fail(
            "results/collapse.json is absent; it is committed evidence, not a "
            "build product. Regenerate it with\n"
            "  python -m projects.rope_attribution.position_free_collapse "
            "--pairs 32\n"
            "and commit the result."
        )
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    if data.get("schema") != "rope-collapse/v1":
        pytest.fail(f"{ARTIFACT}: schema {data.get('schema')!r}, expected 'rope-collapse/v1'")
    return data


def _models(artifact: dict) -> list[dict]:
    return list(artifact["models"])


def test_artifact_covers_the_three_checkpoints(artifact: dict) -> None:
    ids = tuple(m["model_id"] for m in _models(artifact))
    assert ids == EXPECTED_MODELS
    assert artifact["pooled"]["n_models"] == len(EXPECTED_MODELS)


def test_windows_and_seed_are_the_ones_the_paper_quotes(artifact: dict) -> None:
    windows = tuple(int(w) for w in artifact["windows"])
    assert windows == (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    assert artifact["window_lo"] == 1
    assert isinstance(artifact["seed"], int)
    for model in _models(artifact):
        assert tuple(int(w) for w in model["windows"]) == windows
        assert len(model["by_window"]) == len(windows)


def test_theorem_block_is_the_statement_the_tests_check(artifact: dict) -> None:
    """Guard against the artifact drifting from the proposition it cites."""
    theorem = artifact["theorem"]
    assert "2*pi" in theorem["threshold"]
    assert "error == 1" in theorem["threshold"]
    assert "1 - cos(x/2))/2" in theorem["below_threshold"]
    assert "not a new result" in theorem["textbook"]


def test_bins_past_threshold_are_exactly_one(artifact: dict) -> None:
    """Every bin whose lower edge is past ``2*pi`` must sit at the ceiling."""
    checked = 0
    for model in _models(artifact):
        for entry in model["bins"]:
            if float(entry["x_lo"]) < TWO_PI:
                continue
            checked += 1
            assert entry["rel_minimax_median"] == pytest.approx(1.0, abs=EXACT), (
                f"{model['model_id']} bin [{entry['x_lo']}, {entry['x_hi']}): "
                f"median minimax error {entry['rel_minimax_median']}"
            )
            assert entry["rel_minimax_min"] == pytest.approx(1.0, abs=EXACT)
            assert entry["past_threshold_fraction"] == pytest.approx(1.0, abs=EXACT)
    assert checked >= 2, "no bin lies entirely past the threshold; the grid shrank"


def test_bins_below_threshold_respect_the_bound(artifact: dict) -> None:
    """``E_rel >= (1 - cos(x/2))/2`` for every measured pair below the threshold.

    The bound is increasing in ``x``, so the smallest bound inside a bin is at
    its lower edge. Asserting the per-bin minimum against that edge is therefore
    a valid check of every row in the bin, not just of the aggregate.
    """
    checked = 0
    for model in _models(artifact):
        for entry in model["bins"]:
            if float(entry["x_hi"]) > TWO_PI:
                continue
            checked += 1
            bound = (1.0 - math.cos(float(entry["x_lo"]) / 2.0)) / 2.0
            assert entry["rel_minimax_min"] >= bound - EXACT, (
                f"{model['model_id']} bin [{entry['x_lo']}, {entry['x_hi']}): "
                f"min {entry['rel_minimax_min']} < bound {bound}"
            )
    assert checked >= 3, "the sub-threshold region is empty; the grid changed"


def test_every_bin_lies_inside_the_achievable_band(artifact: dict) -> None:
    """The two-sided form of Proposition 5(3), on every measured bin.

    ``E_rel`` ranges over exactly ``[(1-cos(x/2))/2, sin(min(x,pi)/2)]`` as the
    window placement varies. A bin aggregates rows with different ``x``, so the
    check is against the bin's own edges, widened by nothing: the lower bound is
    evaluated at ``x_lo`` (it is increasing) and the upper at ``x_hi`` (also
    increasing). Exceeding the upper edge would mean a pair's minimax error was
    larger than the range of a cosine allows, which is impossible and therefore a
    real signal of a broken measurement.
    """
    checked = 0
    for model in _models(artifact):
        for entry in model["bins"]:
            x_lo = float(entry["x_lo"])
            x_hi = float(entry["x_hi"])
            lower = (1.0 - math.cos(x_lo / 2.0)) / 2.0
            upper = math.sin(min(x_hi, math.pi) / 2.0)
            checked += 1
            assert entry["rel_minimax_min"] >= lower - EXACT, (
                f"{model['model_id']} bin [{x_lo}, {x_hi}) violates the floor"
            )
            assert entry["rel_minimax_max"] <= upper + EXACT, (
                f"{model['model_id']} bin [{x_lo}, {x_hi}): max "
                f"{entry['rel_minimax_max']} exceeds the achievable ceiling {upper}"
            )
    assert checked == sum(len(m["bins"]) for m in _models(artifact))


def test_least_squares_never_beats_minimax(artifact: dict) -> None:
    for model in _models(artifact):
        for entry in model["bins"]:
            assert entry["rel_mean_median"] >= entry["rel_minimax_median"] - EXACT


def test_median_error_is_monotone_in_window(artifact: dict) -> None:
    """A longer window can only widen the range of a sinusoid.

    ``rel_minimax`` for one pair is the half-range of ``cos`` over the window,
    and enlarging the window cannot shrink a range. The median over a fixed set
    of pairs therefore cannot shrink either. This catches a table whose rows were
    assembled from mismatched windows --- the failure mode a vectorised broadcast
    actually has.
    """
    for model in _models(artifact):
        medians = [float(row["channel_rel_minimax_median"]) for row in model["by_window"]]
        assert medians == sorted(medians), model["model_id"]
        fractions = [float(row["past_threshold_fraction"]) for row in model["by_window"]]
        assert fractions == sorted(fractions), model["model_id"]


def test_threshold_table_matches_the_frequency_ladder(artifact: dict) -> None:
    """The count of pairs past threshold must be computable from the ladder alone."""
    for model in _models(artifact):
        spans = np.asarray(model["threshold_span_per_pair"], dtype=np.float64)
        freqs = np.asarray(model["freqs"], dtype=np.float64)
        assert spans.size == model["n_rotary_pairs"]
        assert freqs.size == model["n_rotary_pairs"]
        # theta * threshold_span == 2*pi by definition.
        assert np.allclose(freqs * spans, TWO_PI, rtol=1e-12)
        # The ladder is descending, so the thresholds ascend.
        assert np.all(np.diff(freqs) < 0.0), model["model_id"]
        assert np.all(np.diff(spans) > 0.0), model["model_id"]
        for entry in model["threshold"]:
            expected = int(np.sum(spans <= float(entry["span"])))
            assert entry["pairs_past_threshold"] == expected
            assert entry["fraction_past_threshold"] == pytest.approx(
                expected / spans.size, abs=1e-12
            )


def test_partial_rope_has_a_tail_and_full_rope_does_not(artifact: dict) -> None:
    """The dilution claim is architecture-specific, and the artifact must show it.

    ``pythia`` rotates ``rotary_pct = 0.25`` of its channels, so a real tail
    exists and the score-level rate must fall well below the channel-level one.
    The two Llama-family checkpoints rotate every channel, so their tail is
    exactly zero and no dilution is available.
    """
    by_id = {m["model_id"]: m for m in _models(artifact)}
    pythia = by_id["EleutherAI/pythia-160m"]
    last = pythia["by_window"][-1]
    assert last["tail_share_median"] > 0.5, (
        "pythia rotates a quarter of its channels, so most of its score is the "
        f"unrotated tail; measured {last['tail_share_median']}"
    )
    assert last["score_rel_median"] < 0.5 * last["channel_rel_minimax_median"]

    for model_id in ("JackFram/llama-160m", "HuggingFaceTB/SmolLM-135M"):
        model = by_id[model_id]
        for row in model["by_window"]:
            assert row["tail_share_median"] == pytest.approx(0.0, abs=EXACT), (
                f"{model_id} rotates every channel, so tail_share must be 0; "
                f"got {row['tail_share_median']} at span {row['span']}"
            )


def test_pooled_summary_matches_the_model_tables(artifact: dict) -> None:
    pooled = {entry["model_id"]: entry["fraction"] for entry in artifact["pooled"]["past_threshold_at_4096"]}
    for model in _models(artifact):
        entry = next(t for t in model["threshold"] if t["span"] == 4096)
        assert pooled[model["model_id"]] == entry["fraction_past_threshold"]


def test_the_headline_window_is_the_last_one(artifact: dict) -> None:
    """The paper quotes ``L = 4096``; that must be the top of the measured grid."""
    for model in _models(artifact):
        assert model["by_window"][-1]["span"] == 4096
        assert model["threshold"][-1]["span"] == 4096


def test_row_counts_are_consistent_with_the_nesting(artifact: dict) -> None:
    """The pair table must be the score table times the number of rotary pairs.

    Both are appended in the same inner loop, so the ratio is exact. It fails if
    the pair rows are built over the wrong axis, or if one of the two tables
    silently drops a span.
    """
    by_id = {m["model_id"]: m for m in _models(artifact)}
    for model in _models(artifact):
        assert model["n_pair_rows"] == model["n_score_rows"] * model["n_rotary_pairs"]
        # ``n_score_rows`` is layer-heads x prompts x 32 content pairs x windows;
        # the 32 is the documented ``--pairs 32`` of the committed run. A run at a
        # different budget must update this test too, which is the point.
        groups = model["n_heads"] * model["n_windows"] * 32
        assert model["n_score_rows"] % groups == 0, model["model_id"]

    # pythia-160m and llama-160m share a layer count and a head count, so a
    # correct loop yields the same score-row total for both. It is also an
    # independent check that neither run silently lost a prompt or a layer.
    assert (
        by_id["EleutherAI/pythia-160m"]["n_score_rows"]
        == by_id["JackFram/llama-160m"]["n_score_rows"]
    )
