"""The committed campaign artifact, checked against its pre-registration.

``tests/test_law_campaign.py`` pins the runner's behaviour. This file pins the
*measurement*: it reads ``results/law_campaign.json`` and asks whether the three
baselines obey the window law, reproduce the committed rate, and keep the
softmax identities at their floors. A runner test can pass while the artifact
was produced by a stale grid or a different model set; these tests read the
numbers the paper quotes.

Mutation verdicts
-----------------
* ``test_the_campaign_has_three_clean_baseline_rows`` fails if a checkpoint is
  dropped, a row errors, or the artifact is written from a dirty tree.
* ``test_every_row_is_pinned_to_its_registry_revision`` fails if the loader
  ignores the registry revision, because ``revision_resolved`` then differs.
* ``test_bins_past_threshold_are_exactly_one`` fails if the phase detector
  misses the trough inside a window; the minimax error then falls below 1.
* ``test_bins_below_threshold_stay_inside_the_band`` fails if the band is
  evaluated with ``x`` instead of ``x/2``, or if the window is applied to the
  wrong axis.
* ``test_the_pooled_rate_reproduces_the_committed_baseline`` fails if the
  distance grid or the prompt set changes, because the rate moves by more than
  the pre-registered point.
* ``test_the_softmax_identities_stay_under_their_ceilings`` fails if the tail is
  dropped, the anchor is wrong, or the capture is taken before QK-norm.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "results" / "law_campaign.json"
REGISTRY = REPO / "configs" / "models.json"
BASELINE_USEFULNESS = REPO / "results" / "usefulness.json"

SCHEMA = "rope_attribution/law_campaign/v1"
TWO_PI = 2.0 * math.pi

BASELINES = (
    "EleutherAI/pythia-160m",
    "JackFram/llama-160m",
    "HuggingFaceTB/SmolLM-135M",
)

# The theorem is exact; these absorb the JSON round-trip and float64 trig.
EXACT = 1e-12
# The pre-registered H2 tolerance: one percentage point.
RATE_TOLERANCE = 0.01
# H5 ceilings.  The log-odds identities sit near 1e-13; the centred gradient is
# a finite difference with eps=1e-6, so its ceiling is four orders looser.
ALGEBRA_CEILING = 1e-9
GRADIENT_CEILING = 1e-4


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.fail(
            "results/law_campaign.json is absent; it is committed evidence. "
            "Regenerate it with\n"
            "  python -m rope_attribution.law_campaign --out results/law_campaign.json"
        )
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert data["schema"] == SCHEMA
    return data


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _rows(artifact: dict) -> list[dict]:
    return artifact["models"]


def _by_id(artifact: dict) -> dict[str, dict]:
    return {row["provenance"]["model_id"]: row for row in _rows(artifact)}


def test_the_campaign_has_three_clean_baseline_rows(artifact: dict) -> None:
    rows = _rows(artifact)
    assert [row["provenance"]["model_id"] for row in rows] == list(BASELINES)
    assert artifact["n_models"] == 3
    assert artifact["n_errors"] == 0
    for row in rows:
        assert row["status"] == "ok"
        assert row["provenance"]["n_texts"] == 5
        assert len(row["provenance"]["text_sha256"]) == 5
        commit = row["provenance"]["git"]["commit"]
        assert isinstance(commit, str) and len(commit) == 40
        assert row["provenance"]["git"]["dirty"] is False


def test_every_row_is_pinned_to_its_registry_revision(
    artifact: dict, registry: dict
) -> None:
    entries = {entry["id"]: entry for entry in registry["models"]}
    for row in _rows(artifact):
        mid = row["provenance"]["model_id"]
        entry = entries[mid]
        assert row["provenance"]["revision_requested"] == entry["revision"]
        assert row["provenance"]["revision_resolved"] == entry["revision"]
        assert all(check["ok"] for check in row["registry_checks"])
        checked = {check["field"] for check in row["registry_checks"]}
        assert set(entry["expected"]) <= checked


def test_bins_past_threshold_are_exactly_one(artifact: dict) -> None:
    n_bins = 0
    for row in _rows(artifact):
        for b in row["collapse"]["bins"]:
            n_bins += 1
            if b["x_lo"] >= TWO_PI:
                assert b["rel_minimax_median"] == pytest.approx(1.0, abs=EXACT)
                assert b["rel_minimax_min"] == pytest.approx(1.0, abs=EXACT)
    assert n_bins == 44


def test_bins_below_threshold_stay_inside_the_band(artifact: dict) -> None:
    checked = 0
    for row in _rows(artifact):
        for b in row["collapse"]["bins"]:
            if b["x_hi"] > TWO_PI:
                continue
            checked += 1
            upper = math.sin(min(b["x_hi"], math.pi) / 2.0)
            lower = (1.0 - math.cos(b["x_lo"] / 2.0)) / 2.0
            assert b["rel_minimax_max"] <= upper + EXACT
            assert b["rel_minimax_min"] >= lower - EXACT
    assert checked > 0


def test_the_pooled_rate_reproduces_the_committed_baseline(
    artifact: dict,
) -> None:
    rows = _by_id(artifact)
    pooled = sum(
        rows[mid]["usefulness"]["sign_flip_fraction_pooled"] for mid in BASELINES
    ) / len(BASELINES)
    baseline = json.loads(BASELINE_USEFULNESS.read_text(encoding="utf-8"))
    assert abs(pooled - baseline["pooled"]["sign_flip_fraction"]) <= RATE_TOLERANCE

    # The ordering is part of the baseline: the partial-RoPE checkpoint sits
    # above the two full-RoPE ones, and the two full-RoPE ones are a tenth of a
    # point apart.  A grid or prompt change large enough to reorder them is a
    # finding, not noise.
    pythia = rows["EleutherAI/pythia-160m"]["usefulness"][
        "sign_flip_fraction_pooled"
    ]
    llama = rows["JackFram/llama-160m"]["usefulness"]["sign_flip_fraction_pooled"]
    smol = rows["HuggingFaceTB/SmolLM-135M"]["usefulness"][
        "sign_flip_fraction_pooled"
    ]
    assert pythia > smol > llama


def test_the_softmax_identities_stay_under_their_ceilings(artifact: dict) -> None:
    for row in _rows(artifact):
        sm = row["softmax_decomposition"]
        assert sm["schema"] == "rope_attribution/softmax_decomposition/v1"
        assert sm["n_heads_measured"] > 0
        assert sm["log_odds_residual_max"] <= ALGEBRA_CEILING
        assert sm["odds_residual_max"] <= ALGEBRA_CEILING
        assert sm["contrast_residual_max"] <= ALGEBRA_CEILING
        assert sm["blind_at_period_residual_max"] <= ALGEBRA_CEILING
        assert sm["centered_gradient_residual_max"] <= GRADIENT_CEILING


def test_the_headline_is_copied_from_the_payloads(artifact: dict) -> None:
    for row in _rows(artifact):
        window = next(
            w for w in row["collapse"]["by_window"] if int(w["span"]) == 4096
        )
        headline = row["headline"]
        assert (
            headline["channel_rel_minimax_median_4096"]
            == window["channel_rel_minimax_median"]
        )
        assert (
            headline["sign_flip_fraction_pooled"]
            == row["usefulness"]["sign_flip_fraction_pooled"]
        )
        assert (
            headline["softmax_log_odds_residual_max"]
            == row["softmax_decomposition"]["log_odds_residual_max"]
        )
