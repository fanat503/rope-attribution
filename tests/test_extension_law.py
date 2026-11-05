"""Tests for the threshold law under context extension.

The module makes a negative design claim that the rest of the paper relies on:
no practical extension scheme makes a position-free attribution globally valid.
These tests pin the two exact facts behind it (the YaRN fast-channel floor and
the position-interpolation critical scale) and the arithmetic of the artifact,
so a changed ramp or a changed threshold cannot pass silently.

Mutation verdict.  Each test names the edit it is meant to kill.  The three
load-bearing ones are: lowering the threshold from ``2*pi`` (kills the headline
counts), treating the YaRN ramp as a uniform rescaling (kills the nine-channel
floor), and reporting a finite critical scale for YaRN (kills the design
statement).  A test that would still pass after one of those edits is not a
detector and is not written.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from rope_attribution import extension_law as E
from rope_attribution import rope as R

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "results" / "extension_law.json"
TOL = 1e-12


# --------------------------------------------------------------------------
# the law itself
# --------------------------------------------------------------------------


def test_minimax_error_is_exactly_one_at_and_above_the_threshold() -> None:
    """Mutation: lowering the threshold, or using the below-threshold formula
    above it, makes at least one of these values fall below 1."""
    x = np.array([E.THRESHOLD, E.THRESHOLD + 1e-9, 10.0, 100.0])
    assert np.all(E.minimax_error(x) == 1.0)


def test_below_threshold_error_matches_the_band_endpoints() -> None:
    """Mutation: swapping the lower and upper ends, or dropping ``min(x, pi)``,
    breaks the equality or the ordering."""
    x = np.array([0.1, 0.5, 1.0, math.pi, E.THRESHOLD - 1e-6])
    lower = E.minimax_error(x)
    upper = E.worst_case_error(x)
    expected_lower = 0.5 * (1.0 - np.cos(0.5 * x))
    expected_upper = np.sin(0.5 * np.minimum(x, math.pi))
    assert np.allclose(lower, expected_lower, rtol=0, atol=TOL)
    assert np.allclose(upper, expected_upper, rtol=0, atol=TOL)
    assert np.all(lower <= upper + TOL)
    # At x = pi the band is [0.5, 1]; the best placement is twice as good as
    # the worst.  This is the point a reader is most likely to misread as
    # "always 1 below the threshold".
    assert abs(float(lower[3]) - 0.5) < TOL
    assert abs(float(upper[3]) - 1.0) < TOL


def test_the_band_collapses_to_one_at_the_threshold() -> None:
    """Mutation: any formula that is continuous but not exactly 1 at 2*pi
    leaves a gap between the two ends, which the law says cannot exist."""
    x = E.THRESHOLD
    assert abs(float(E.minimax_error(x)) - 1.0) < TOL
    assert abs(float(E.worst_case_error(x)) - 1.0) < TOL


# --------------------------------------------------------------------------
# the four schemes
# --------------------------------------------------------------------------


def test_scheme_frequencies_match_the_reference_ladders() -> None:
    """Mutation: using the base for position interpolation, or forgetting the
    scale, makes the PI row disagree with ``inv_freq / s``."""
    head_dim, base = E.HEAD_DIM, E.BASE
    rope = R.inv_freq(head_dim, base)
    yarn, _ = R.yarn_parameters(
        head_dim,
        base,
        32.0,
        E.ORIGINAL_CONTEXT,
        beta_fast=E.BETA_FAST,
        beta_slow=E.BETA_SLOW,
    )
    assert np.array_equal(E.scheme_frequencies("rope"), rope)
    assert np.array_equal(E.scheme_frequencies("yarn", 32.0), yarn)
    assert np.array_equal(
        E.scheme_frequencies("position_interpolation", 32.0), rope / 32.0
    )
    assert np.array_equal(
        E.scheme_frequencies("base_500k"), R.inv_freq(head_dim, 500000.0)
    )


def test_yarn_leaves_exactly_nine_fast_channels_untouched_at_scale_32() -> None:
    """The floor behind the YaRN design statement.

    Mutation: a uniform rescaling (the common misconception) gives zero
    untouched channels; a different ramp gives a different count.  The test
    also pins *which* channels: the first nine, not an arbitrary nine, because
    the ramp is monotone in the channel index.
    """
    rope = R.inv_freq(E.HEAD_DIM, E.BASE)
    yarn = E.scheme_frequencies("yarn", 32.0)
    equal = yarn == rope
    assert int(equal.sum()) == 9
    assert bool(equal[:9].all())
    assert not bool(equal[9])


def test_the_headline_counts_at_4096_are_pinned() -> None:
    """Mutation: changing the threshold, the ramp, or a scheme's ladder moves
    at least one count.  These are the numbers the paper quotes."""
    expected = {
        "rope": 23,
        "yarn": 18,
        "position_interpolation": 11,
        "base_500k": 16,
    }
    for scheme, count in expected.items():
        row = E.summarize(scheme, 32.0, 4096)
        assert row["n_past_threshold"] == count, (scheme, row)
        assert row["n_channels"] == 32


def test_critical_scale_is_finite_only_for_position_interpolation() -> None:
    """Mutation: reporting ``L/(2*pi)`` for YaRN or the raised base claims a
    fix that the untouched fastest channel makes impossible."""
    assert E.critical_scale("position_interpolation", 4096) == pytest.approx(
        4096.0 / E.THRESHOLD, rel=0, abs=TOL
    )
    assert E.critical_scale("yarn", 4096) is None
    assert E.critical_scale("base_500k", 4096) is None
    assert E.critical_scale("rope", 4096) is None
    # Below the threshold even plain RoPE is globally attributable.
    assert E.critical_scale("rope", 4) == 1.0


def test_the_untouched_floor_is_a_real_floor_at_every_length() -> None:
    """Mutation: a scheme with fewer untouched fast channels would have a lower
    floor, but the pinned 9 at L=4096 still catches the specific YaRN claim."""
    floor = E.summarize("yarn", 32.0, 4096)["untouched_past_threshold"]
    assert floor == 9
    for context in E.CONTEXT_LENGTHS:
        row = E.summarize("yarn", 32.0, context)
        assert row["n_past_threshold"] >= row["untouched_past_threshold"]
        assert row["n_past_threshold"] >= floor if context >= 4096 else True


# --------------------------------------------------------------------------
# the artifact
# --------------------------------------------------------------------------


def test_the_artifact_is_self_consistent_and_current() -> None:
    """Mutation: hand-editing a count without regenerating, or changing the
    grid in the module without rewriting the artifact, fails here."""
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert data["schema"] == "rope-extension-law/v1"
    cfg = data["configuration"]
    assert cfg["head_dim"] == E.HEAD_DIM
    assert cfg["base"] == E.BASE
    assert cfg["threshold_radians"] == E.THRESHOLD
    assert cfg["context_lengths"] == list(E.CONTEXT_LENGTHS)
    assert cfg["scales"] == list(E.SCALES)
    assert cfg["schemes"] == list(E.SCHEMES)
    n_rows = len(E.SCHEMES) * len(E.SCALES) * len(E.CONTEXT_LENGTHS)
    assert len(data["rows"]) == n_rows
    for row in data["rows"]:
        assert 0 <= row["n_past_threshold"] <= row["n_channels"] == 32
        assert row["fraction_past_threshold"] == pytest.approx(
            row["n_past_threshold"] / row["n_channels"], rel=0, abs=TOL
        )
        assert 0.0 <= row["mean_minimax_error"] <= 1.0
        assert 0.0 <= row["mean_worst_case_error"] <= 1.0
        assert row["mean_minimax_error"] <= row["mean_worst_case_error"] + TOL
    canonical = data["canonical_channels_4096"]
    assert set(canonical) == set(E.SCHEMES)
    for scheme, channels in canonical.items():
        assert len(channels) == 32
        assert [c["channel"] for c in channels] == list(range(32))
        assert sum(1 for c in channels if c["past_threshold"]) == data[
            "headline"
        ]["context_4096"][scheme]["n_past_threshold"]
