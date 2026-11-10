"""Tests for ``usefulness_extension``: the measured cost under extension geometry.

The artifact is a *counterfactual geometry* measurement: the checkpoint's own
unrotated query/key vectors are held fixed and only the frequency ladder
changes.  Two things therefore have to be true before a number from it may be
quoted:

1. the plain-RoPE row must reproduce the headline ``usefulness.json`` exactly,
   because it is the same vectors, grid, seed and aggregator; and
2. the measured rate must be distinguished from the minimax bound, because the
   whole point of the artifact is that they are different functionals.

Mutation verdict.  The load-bearing detectors are the cross-artifact equality
(kills a changed seed, grid, pair count or aggregator in the new module), the
position-interpolation ordering (kills a uniform-rescaling implementation of
YaRN, or a PI ladder that does not divide the fast channels), and the
"never zero" assertion (kills a claim that extension makes a position-free
summary valid).  A test that would survive those edits is not a detector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_PROJECTS = _REPO / "projects"
if str(_PROJECTS) not in sys.path:
    sys.path.insert(0, str(_PROJECTS))

from rope_attribution import extension_law as E  # noqa: E402
from rope_attribution import usefulness as U  # noqa: E402
from rope_attribution import usefulness_extension as X  # noqa: E402

_ARTIFACT = _REPO / "results" / "usefulness_extension.json"
_HEADLINE = _REPO / "results" / "usefulness.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(_ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def headline() -> dict:
    return json.loads(_HEADLINE.read_text(encoding="utf-8"))


def _row(model: dict, scheme: str, scale: float) -> dict:
    matches = [
        row
        for row in model["rows"]
        if row["scheme"] == scheme and row["scale"] == scale
    ]
    assert len(matches) == 1, (model["model_id"], scheme, scale, len(matches))
    return matches[0]


def _pooled(artifact: dict, scheme: str, scale: float) -> dict:
    matches = [
        row
        for row in artifact["pooled"]
        if row["scheme"] == scheme and row["scale"] == scale
    ]
    assert len(matches) == 1
    return matches[0]


# --------------------------------------------------------------------------
# the control row
# --------------------------------------------------------------------------


def test_the_plain_rope_row_reproduces_the_headline_artifact(
    artifact: dict, headline: dict
) -> None:
    """Mutation: a different seed, grid, pair count or aggregator in this module
    changes at least one of these rates.  The control must be exact, not close,
    because the two modules are supposed to run the same measurement."""
    canonical = {m["model_id"]: m for m in headline["models"]}
    assert set(canonical) == {m["model_id"] for m in artifact["models"]}
    for model in artifact["models"]:
        row = _row(model, "rope", 1.0)
        ref = canonical[model["model_id"]]
        assert row["sign_flip_fraction_pooled"] == pytest.approx(
            ref["sign_flip_fraction_pooled"], abs=0.0, rel=1e-12
        )
        assert row["rel_err_median"] == pytest.approx(
            ref["rel_err_median"], abs=0.0, rel=1e-12
        )
        assert row["n_heads"] == ref["n_heads"]


def test_the_plain_rope_pool_matches_the_headline_pool(
    artifact: dict, headline: dict
) -> None:
    """The pooled control is the same equal-weight mean as the paper's 27.0%."""
    row = _pooled(artifact, "rope", 1.0)
    assert row["sign_flip_fraction_mean"] == pytest.approx(
        headline["pooled"]["sign_flip_fraction"], abs=0.0, rel=1e-12
    )


# --------------------------------------------------------------------------
# the geometry prediction
# --------------------------------------------------------------------------


def test_the_channel_counts_are_the_scheme_ladders(artifact: dict) -> None:
    """Mutation: a hard-coded count, a wrong head width or a wrong base makes at
    least one model x scheme x length disagree with the ladder itself."""
    for model in artifact["models"]:
        for row in model["rows"]:
            freqs = X._frequencies(
                row["scheme"],
                row["scale"],
                head_dim=model["rope_dim"],
                base=model["base"],
                original_context=model["original_context"],
            )
            assert row["n_channels"] == freqs.size
            for length in artifact["context_lengths"]:
                expected = int(np.sum(freqs * float(length) >= E.THRESHOLD))
                got = row["channels"][str(length)]["n_past_threshold"]
                assert got == expected, (model["model_id"], row["scheme"], length)


def test_position_interpolation_leaves_no_more_past_threshold_than_yarn(
    artifact: dict,
) -> None:
    """The structural prediction behind the design statement.

    Mutation: treating YaRN as a uniform rescaling makes the two fractions equal
    at every scale; a PI ladder that fails to divide the fast channels leaves
    *more* past threshold than YaRN.  The inequality is strict from scale 8 on;
    at scale 4 both schemes still leave the same coarse channel past threshold
    on the narrow 16-wide rotary ladder, so equality there is expected.
    """
    for model in artifact["models"]:
        for scale in (2.0, 4.0, 8.0, 16.0, 32.0):
            yarn = _row(model, "yarn", scale)["channels"]["4096"][
                "fraction_past_threshold"
            ]
            pi = _row(model, "position_interpolation", scale)["channels"]["4096"][
                "fraction_past_threshold"
            ]
            assert pi <= yarn + 1e-12
            if scale >= 8.0:
                assert pi < yarn


def test_the_measured_rate_falls_under_extension_but_never_reaches_zero(
    artifact: dict,
) -> None:
    """The paper's new empirical claim.

    Mutation: claiming extension makes a position-free summary valid would make
    the scale-32 row zero; claiming it does nothing would make it equal to the
    control.  Both are killed here, for both families.
    """
    rope = _pooled(artifact, "rope", 1.0)["sign_flip_fraction_mean"]
    yarn32 = _pooled(artifact, "yarn", 32.0)["sign_flip_fraction_mean"]
    pi32 = _pooled(artifact, "position_interpolation", 32.0)[
        "sign_flip_fraction_mean"
    ]
    assert rope == pytest.approx(0.27042014920964974, abs=1e-12)
    assert yarn32 == pytest.approx(0.2147, abs=5e-5)
    assert pi32 == pytest.approx(0.1193, abs=5e-5)
    assert 0.0 < pi32 < yarn32 < rope


def test_position_interpolation_is_monotone_in_scale(artifact: dict) -> None:
    """Mutation: a ladder that stops scaling, or a summary that ignores the
    scale, produces a flat or non-monotone PI curve."""
    rates = [
        _pooled(artifact, "position_interpolation", scale)["sign_flip_fraction_mean"]
        for scale in (2.0, 4.0, 8.0, 16.0, 32.0)
    ]
    assert all(a > b for a, b in zip(rates, rates[1:]))


def test_yarn_plateaus_above_position_interpolation(artifact: dict) -> None:
    """YaRN's untouched fast channels are the floor; its curve must flatten
    above the PI curve rather than converge to it at scale 32."""
    yarn = [
        _pooled(artifact, "yarn", scale)["sign_flip_fraction_mean"]
        for scale in (2.0, 4.0, 8.0, 16.0, 32.0)
    ]
    pi = [
        _pooled(artifact, "position_interpolation", scale)["sign_flip_fraction_mean"]
        for scale in (2.0, 4.0, 8.0, 16.0, 32.0)
    ]
    assert all(p < y for p, y in zip(pi, yarn))
    # The last two YaRN points are within 0.003 of one another: the ramp has
    # saturated, while PI is still falling by more than 0.02 per step.
    assert abs(yarn[-1] - yarn[-2]) < 0.003
    assert pi[-2] - pi[-1] > 0.02


# --------------------------------------------------------------------------
# artifact integrity
# --------------------------------------------------------------------------


def test_every_model_has_every_scheme_and_scale(artifact: dict) -> None:
    assert artifact["schema"] == X.SCHEMA
    assert artifact["context_lengths"] == list(X.CONTEXT_LENGTHS)
    assert len(artifact["models"]) == len(U.DEFAULT_MODEL_IDS)
    assert len(artifact["pooled"]) == len(X.DEFAULT_SCHEMES)
    for model in artifact["models"]:
        assert len(model["rows"]) == len(X.DEFAULT_SCHEMES)
        assert {row["scheme"] for row in model["rows"]} == {
            scheme for scheme, _ in X.DEFAULT_SCHEMES
        }


def test_the_pool_is_the_equal_weight_mean_of_the_models(artifact: dict) -> None:
    """Mutation: pooling head instances instead of models changes the mean
    whenever the models have different head counts."""
    for pooled in artifact["pooled"]:
        values = [
            _row(model, pooled["scheme"], pooled["scale"])[
                "sign_flip_fraction_pooled"
            ]
            for model in artifact["models"]
        ]
        assert pooled["sign_flip_fraction_mean"] == pytest.approx(
            float(np.mean(values)), abs=1e-15
        )
        assert pooled["sign_flip_fraction_min"] == pytest.approx(min(values))
        assert pooled["sign_flip_fraction_max"] == pytest.approx(max(values))


def test_the_head_rates_are_a_distribution(artifact: dict) -> None:
    for model in artifact["models"]:
        for row in model["rows"]:
            rates = [h["sign_flip_fraction"] for h in row["head_rates"]]
            assert len(rates) == row["n_heads"]
            assert all(0.0 <= rate <= 1.0 for rate in rates)
            assert min(rates) == pytest.approx(row["sign_flip_fraction_min_head"])
            assert max(rates) == pytest.approx(row["sign_flip_fraction_max_head"])


def test_the_measured_rate_and_the_minimax_bound_are_different_objects(
    artifact: dict,
) -> None:
    """The paper's central distinction, pinned as a number.

    The minimax reference at L=4096 is a lower bound on a *different*
    functional.  If a future refactor accidentally replaces the measured rate
    with the reference, this test fails because the two differ by a wide margin.
    """
    rope = _row(artifact["models"][0], "rope", 1.0)
    reference = rope["channels"]["4096"]["minimax_mean"]
    assert reference > 0.8
    assert abs(rope["sign_flip_fraction_pooled"] - reference) > 0.4
