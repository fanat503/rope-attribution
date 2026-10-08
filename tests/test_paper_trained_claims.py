"""Check Section~\\ref{sec:trained} of the paper against the trained checkpoints.

Why this is a separate module
-----------------------------
``results/real_model.json`` is the only source in this repository whose numbers
come from *pretrained weights* rather than from synthetic draws. It was, until
now, referenced by nothing outside ``real_model.py`` and ``README.md`` --- the
paper did not report a single number from it, while asserting that it contained
no measurements of trained models. That is the gap this file closes.

Each test below re-derives a printed number from the JSON. None of them asserts
the paper is internally consistent; they all compare against the measurement, so
editing the paper without re-running the measurement fails here.

The claim patterns live in ``test_paper_claims.CLAIMS`` (extended there with
``CLAIMS +=``) so that the exhaustiveness guard in that module treats these
numbers like every other one in the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_paper_claims import (
    MEASUREMENTS,
    TEX_FLAT,
    assert_paper_number_matches,
    assert_paper_sci_matches,
    groups,
    literal,
)

_REPO = Path(__file__).resolve().parents[1]
_JSON = _REPO / "results" / "real_model.json"

PYTHIA = "EleutherAI/pythia-160m"
LLAMA = "JackFram/llama-160m"
SMOLLM = "HuggingFaceTB/SmolLM-135M"

# float32 has a 24-bit significand; this is exactly its eps.
FLOAT32_EPS = 1.1920928955078125e-07


@pytest.fixture(scope="module")
def trained() -> dict:
    if not _JSON.exists():
        pytest.skip("results/real_model.json is absent; run real_model.py")
    return json.loads(_JSON.read_text(encoding="utf-8"))


def _ident(trained: dict, model_id: str, name: str) -> dict:
    return trained["aggregates"][model_id]["identities"][name]


def _shares(trained: dict, model_id: str) -> dict:
    return trained["aggregates"][model_id]["share_quantiles_across_prompts"]


def _blind(trained: dict, model_id: str) -> float:
    return trained["aggregates"][model_id]["frac_position_blind_across_heads"]["mean"]


# --------------------------------------------------------------------------
# the section must be about the checkpoints that were actually measured
# --------------------------------------------------------------------------


def test_the_section_names_the_three_checkpoints_it_measured(trained: dict) -> None:
    assert set(trained["aggregates"]) == {PYTHIA, LLAMA, SMOLLM}
    for short in ("pythia-160m", "llama-160m", "SmolLM-135M"):
        assert f"\\texttt{{{short}}}" in TEX_FLAT, short


def test_the_swept_distance_range_is_the_one_that_was_run(trained: dict) -> None:
    deltas = trained["deltas"]
    assert len(deltas) == int(literal("trained_delta_range")), deltas
    # The range has to actually be a range, or "across N distances" is vacuous.
    assert len(set(deltas)) == len(deltas) and min(deltas) >= 1
    assert max(deltas) > min(deltas) * 100, deltas


def test_the_model_size_range_matches_the_checkpoint_names(trained: dict) -> None:
    """The paper prints a parameter-size range; the model ids must sit inside it.

    This checks the identifiers, not the parameter counts. ``real_model.json``
    records no parameter counts --- they live in each checkpoint's Hugging Face
    config --- so asserting against the JSON would mean asserting nothing.
    """
    lo, hi = (int(x) for x in groups("trained_model_sizes")[0])
    assert lo < hi
    sizes = set()
    for model_id in trained["aggregates"]:
        for tag in model_id.split("/")[-1].split("-"):
            if tag.endswith("m") and tag[:-1].isdigit():
                sizes.add(int(tag[:-1]))
    assert sizes, sorted(trained["aggregates"])
    assert min(sizes) >= lo and max(sizes) <= hi, (sorted(sizes), lo, hi)


def test_delta_512_is_in_the_grid_the_paper_sweeps(trained: dict) -> None:
    """delta = 512 is quoted as a comparison point; it must be one we measured."""
    delta = int(literal("delta_512"))
    assert delta in trained["deltas"], (delta, trained["deltas"])


def test_gpt2_is_reported_as_a_control_and_not_as_a_measurement(trained: dict) -> None:
    """The negative control has to actually be a control.

    GPT-2 has no rotary embedding: position enters through a learned absolute
    table. Reporting it as one of the three measured models would be wrong, and
    omitting the reason would leave a reader wondering why the most obvious
    checkpoint is missing.
    """
    gpt2 = trained["gpt2_rope_check"]
    assert gpt2["uses_rope"] is False
    assert gpt2["rotary_module_found"] is False
    assert gpt2["learned_absolute_position_embedding"] is True
    assert gpt2["learned_position_table_size"] == int(literal("gpt2_no_rope"))
    assert "no rotary embedding at all" in TEX_FLAT
    assert "gpt2" not in trained["aggregates"]


# --------------------------------------------------------------------------
# the identities are exact; the measurement shows nothing corrects them
# --------------------------------------------------------------------------


def test_the_relative_position_identity_is_exact_on_trained_weights(trained: dict) -> None:
    worst = max(
        _ident(trained, m, "relative_position_max_abs_err")["max"]
        for m in (PYTHIA, LLAMA, SMOLLM)
    )
    assert worst < 1e-10, f"the identity is not exact on trained weights: {worst}"
    # The literal *string*, not a float: the accepted interval is derived from the
    # decimal the paper wrote, and float("1.14") would hand it exponent -52.
    assert_paper_sci_matches(
        worst, literal("trained_relpos_worst"), -12, "the trained relative-position error"
    )


def test_the_dtype_floor_the_paper_names_is_the_real_one(trained: dict) -> None:
    assert_paper_sci_matches(
        FLOAT32_EPS, literal("trained_float32_eps"), -7, "the float32 epsilon"
    )
    assert trained["models"], "no per-model records to check the dtype against"


def test_the_closed_form_and_bilinearity_sit_at_that_floor(trained: dict) -> None:
    """Exact in principle; what matters is that no checkpoint corrects them.

    The paper's stronger reading is "these are the numbers you get when the
    implementation is correct and the arithmetic runs out of precision". That
    only holds if each measured error is at or below float32 eps, so that is
    what is asserted --- a bare "is small" test would let a 1e-5 slip through.
    """
    for model, index in ((PYTHIA, 1), (LLAMA, 2), (SMOLLM, 3)):
        closed_form = _ident(trained, model, "pair_closed_form_max_rel_err")["median"]
        assert_paper_sci_matches(
            closed_form,
            literal("trained_closed_form", index),
            -8,
            f"the per-pair closed form on {model}",
        )
        assert closed_form < FLOAT32_EPS, (model, closed_form)

        bilinear = _ident(trained, model, "bilinearity_max_scaled_err")["median"]
        assert_paper_sci_matches(
            bilinear,
            literal("trained_bilinearity", index),
            -8,
            f"bilinearity on {model}",
        )
        assert bilinear < FLOAT32_EPS, (model, bilinear)


# --------------------------------------------------------------------------
# the empirical payoff: per-feature share is not a stable quantity
# --------------------------------------------------------------------------


def test_the_per_feature_share_spread_is_as_quoted(trained: dict) -> None:
    for model, name in (
        (PYTHIA, "trained_share_pythia"),
        (LLAMA, "trained_share_llama"),
        (SMOLLM, "trained_share_smollm"),
    ):
        q = _shares(trained, model)
        lo, hi, factor = groups(name)[0]
        assert_paper_number_matches(q["min"], f"{float(lo):.4f}", f"the share minimum on {model}")
        assert_paper_number_matches(q["max"], f"{float(hi):.4f}", f"the share maximum on {model}")
        assert_paper_number_matches(
            q["max"] / q["min"], f"{float(factor):.1f}", f"the share spread on {model}"
        )
        # Not a rounding artefact: the range really is most of the unit interval.
        assert q["min"] < 0.05, (model, q["min"])
        assert q["max"] > 0.99, (model, q["max"])
        assert q["std"] > 0.3, (model, q["std"])


def test_the_quoted_interquartile_range_is_the_pooled_one(trained: dict) -> None:
    lo, hi = groups("trained_share_iqr")[0]
    p25 = [float(_shares(trained, m)["p25"]) for m in (PYTHIA, LLAMA, SMOLLM)]
    p75 = [float(_shares(trained, m)["p75"]) for m in (PYTHIA, LLAMA, SMOLLM)]
    assert_paper_number_matches(min(p25), f"{float(lo):.3f}", "the pooled lower quartile")
    assert_paper_number_matches(max(p75), f"{float(hi):.3f}", "the pooled upper quartile")
    assert max(p75) > 0.95 and min(p25) < 0.2, (p25, p75)


def test_almost_every_channel_is_position_carrying(trained: dict) -> None:
    """The claim that makes the gate/phase split useless on real weights."""
    for model, index in ((PYTHIA, 1), (LLAMA, 2), (SMOLLM, 3)):
        written = float(literal("trained_blind_fraction", index))
        assert_paper_number_matches(
            _blind(trained, model), f"{written:.4f}", f"the blind fraction on {model}"
        )
    fractions = [_blind(trained, m) for m in (PYTHIA, LLAMA, SMOLLM)]
    assert all(0.001 < f < 0.02 for f in fractions), fractions
    # "remarkably consistent across three unrelated architectures"
    assert max(fractions) / min(fractions) < 1.5, fractions
    assert trained["position_blind_ratio"] == 0.01


def test_the_stated_band_brackets_every_checkpoint(trained: dict) -> None:
    lo = float(literal("trained_blind_band", 1))
    hi = float(literal("trained_blind_band", 2))
    for model in (PYTHIA, LLAMA, SMOLLM):
        pct = 100.0 * _blind(trained, model)
        assert lo <= pct <= hi, (model, pct, lo, hi)


def test_the_section_does_not_claim_a_downstream_result(trained: dict) -> None:
    """The limit of the evidence is part of the claim, so it is tested too."""
    for phrase in (
        "cannot tell you whether per-distance attribution identifies anything",
        "no downstream task",
    ):
        # The flattened source has no newlines, so match on the flat text rather
        # than on a line-wrap the next reformat would move.
        assert phrase in TEX_FLAT, phrase
    # And the synthetic measurements the characterisation rests on are still
    # present, so the two halves cannot be confused.
    assert MEASUREMENTS["feature_attribution"]["n_features"] == 8