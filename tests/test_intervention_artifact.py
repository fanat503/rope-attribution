"""The committed intervention artifact, checked against what it claims to measure.

``tests/test_intervention.py`` pins the harness functions on synthetic input.
This file reads ``results/intervention.json`` --- the numbers the paper quotes
--- and checks the properties that make the experiment meaningful rather than
merely large. The distinction is the same as in ``test_collapse_artifact.py``: a
unit test can pass while the committed run used the wrong cells, edited the
wrong head, or summarised a different run.

Mutation verdicts
-----------------
* ``test_exact_rule_is_an_oracle`` fails if the exact selection stops ranking by
  the same quantity it removes. It is the algebraic identity
  ``achieved == intended``, so no tolerance is involved and a wrong coefficient
  cannot satisfy it.
* ``test_edit_adds_no_prediction_error`` fails if the query hook edits a
  different head or the closed form mis-indexes channels: the predicted
  post-softmax attention then diverges from the measured one, and the median
  error rises above the *unedited* forward pass, which is the control.
* ``test_sign_reversals_are_the_differing_cells`` fails if a reversal is counted
  for a cell where the two rules chose the same pair --- impossible, since the
  two numbers are then identical --- or if the pooled fraction is filled in from
  a different cell list.
* ``test_cells_are_unsaturated_and_long_enough`` fails if the peak filter is
  removed (peaks reach 1.0, where ablation cannot move attention) or if the
  visibility filter is removed (a query at position 3 has a one-period window).
* ``test_pooled_summary_matches_the_cells`` fails if the summary is computed
  before a cell is dropped, which is exactly the drift a rerun introduces.
* ``test_extreme_ratio_is_a_small_denominator`` fails if the heavy tail is
  produced by a numerical blow-up instead of a genuinely small intended
  suppression; it pins the explanation rather than hiding the value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "results" / "intervention.json"

EXPECTED_MODELS = (
    "EleutherAI/pythia-160m",
    "JackFram/llama-160m",
    "HuggingFaceTB/SmolLM-135M",
)

# The committed run: one 320-token passage, 32 cells per checkpoint, one pair
# removed (budget 1). A run at another budget must update these constants, which
# is the point: the paper's numbers are tied to this configuration.
CELLS = 32
BUDGET = 1
MIN_VISIBLE_KEYS = 32
MAX_PEAK = 0.9

EXACT = 1e-12


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.fail(
            "results/intervention.json is absent; it is committed evidence, not "
            "a build product. Regenerate it with\n"
            "  python -m projects.rope_attribution.qk_intervention\n"
            "and commit the result."
        )
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    if data.get("schema") != "rope-intervention/v1":
        pytest.fail(
            f"{ARTIFACT}: schema {data.get('schema')!r}, expected 'rope-intervention/v1'"
        )
    return data


def _cells(artifact: dict) -> list[dict]:
    return [cell for model in artifact["models"] for cell in model["cells"]]


def test_artifact_covers_the_three_checkpoints(artifact: dict) -> None:
    assert tuple(m["model_id"] for m in artifact["models"]) == EXPECTED_MODELS
    assert artifact["budgets"] == [BUDGET]
    assert artifact["cells_per_prompt"] == CELLS
    for model in artifact["models"]:
        assert model["n_cells"] == CELLS, model["model_id"]
        assert len(model["cells"]) == CELLS
        assert len({(c["layer"], c["head"]) for c in model["cells"]}) == CELLS, (
            f"{model['model_id']}: the run must spread its cells over distinct "
            "heads, otherwise it is many queries of one head"
        )


def test_exact_rule_is_an_oracle(artifact: dict) -> None:
    """``achieved / intended`` is exactly 1 for the rule that ranks by ``c(delta*)``.

    This is not a tolerance: the exact selection maximises ``c_exact`` and then
    removes ``c_exact``, so the two numbers are the same float. A test that only
    checked ``ratio == 1`` with ``approx`` would accept a rule that removed a
    different pair and got lucky; here the stored numerator and denominator must
    be bit-identical.
    """
    checked = 0
    for model in artifact["models"]:
        for cell in model["cells"]:
            exact = cell["exact"]
            assert exact["achieved_suppression"] == exact["intended_suppression"], (
                f"{model['model_id']} L{cell['layer']}H{cell['head']}q{cell['query']}: "
                f"exact achieved {exact['achieved_suppression']} != intended "
                f"{exact['intended_suppression']}"
            )
            assert exact["suppression_ratio"] == pytest.approx(1.0, abs=EXACT)
            checked += 1
    assert checked == 3 * CELLS


def test_edit_adds_no_prediction_error(artifact: dict) -> None:
    """The closed form predicts the *intervened* pass as well as the baseline.

    The baseline median is the control: it is the same closed form and softmax
    against an unedited forward pass, so it carries the float32 floor of the
    model. If the hook edits the wrong head, or the channel indexing is off, the
    intervened prediction is wrong while the baseline prediction stays right,
    and the comparison below fails.
    """
    for model in artifact["models"]:
        pooled = model["pooled"]
        baseline = float(pooled["baseline_prediction_error_median"])
        for rule in ("exact", "surrogate"):
            overview = float(pooled[f"{rule}_prediction_error_median"])
            assert overview <= 10.0 * baseline, (
                f"{model['model_id']}: {rule} median error {overview:.3e} exceeds "
                f"10x the unedited control {baseline:.3e}"
            )
        # The worst cell is a genuine tail, not a blow-up: 1e-3 is three orders
        # above the float32 floor of a 320-token, 12-layer pass.
        assert float(pooled["exact_prediction_error_max"]) < 1e-3


def test_sign_reversals_are_the_differing_cells(artifact: dict) -> None:
    """A reversed suppression can only happen where the two selections differ.

    If the rules pick the same pair the two ratios are the same number, so a
    reversal outside the differing set is arithmetically impossible and signals
    that the two fields were assembled from different runs.
    """
    for model in artifact["models"]:
        cells = model["cells"]
        differing = [c for c in cells if not c["same_pair"]]
        reversed_cells = [
            c
            for c in cells
            if c["surrogate"]["suppression_ratio"] is not None
            and c["surrogate"]["suppression_ratio"] < 0.0
        ]
        assert {id(c) for c in reversed_cells} <= {id(c) for c in differing}
        assert model["n_selection_differs"] == len(differing)
        assert model["pooled"]["fraction_selection_differs"] == pytest.approx(
            len(differing) / len(cells), abs=EXACT
        )
        assert model["pooled"]["surrogate_sign_reversal_fraction"] == pytest.approx(
            len(reversed_cells) / len(cells), abs=EXACT
        )


def test_the_paper_headline_is_the_measured_totals(artifact: dict) -> None:
    """Pin the three totals the paper quotes, so a rerun cannot move them silently."""
    cells = _cells(artifact)
    differing = sum(1 for c in cells if not c["same_pair"])
    reversed_cells = sum(
        1 for c in cells
        if c["surrogate"]["suppression_ratio"] is not None
        and c["surrogate"]["suppression_ratio"] < 0.0
    )
    assert len(cells) == 96
    assert differing == 25
    assert reversed_cells == 7
    # Per-model counts, because the paper reports the architecture split: the
    # partial-RoPE checkpoint carries most of the cost.
    counts = {
        m["model_id"]: (
            m["n_selection_differs"],
            sum(
                1 for c in m["cells"]
                if c["surrogate"]["suppression_ratio"] is not None
                and c["surrogate"]["suppression_ratio"] < 0.0
            ),
        )
        for m in artifact["models"]
    }
    assert counts["EleutherAI/pythia-160m"] == (13, 5)
    assert counts["JackFram/llama-160m"] == (9, 2)
    assert counts["HuggingFaceTB/SmolLM-135M"] == (3, 0)


def test_cells_are_unsaturated_and_long_enough(artifact: dict) -> None:
    """Both filters are what make the intervention measurable at all.

    A saturated row (peak 1.0) cannot respond to a score edit; a query at
    position 5 has a window too short for the surrogate to be a table. The
    artifact stores the bounds it used, and every cell must respect them.
    """
    for model in artifact["models"]:
        selection = model["cell_selection"]
        assert selection["max_peak"] == MAX_PEAK
        assert selection["min_visible_keys"] == MIN_VISIBLE_KEYS
        for cell in model["cells"]:
            assert cell["baseline_attention"] <= MAX_PEAK + EXACT
            assert cell["baseline_attention"] >= selection["min_peak"]
            assert cell["query"] >= MIN_VISIBLE_KEYS
            assert cell["n_visible"] == cell["query"]
            assert cell["target"] < cell["query"]
            assert cell["delta"] == cell["query"] - cell["target"]


def test_pooled_summary_matches_the_cells(artifact: dict) -> None:
    """Every pooled number is recomputed from the cell list it summarises."""
    for model in artifact["models"]:
        pooled = model["pooled"]
        cells = model["cells"]
        exact_err = np.asarray([c["exact"]["prediction_error"] for c in cells])
        surrogate_err = np.asarray([c["surrogate"]["prediction_error"] for c in cells])
        assert pooled["exact_prediction_error_median"] == pytest.approx(
            float(np.median(exact_err)), rel=1e-12
        )
        assert pooled["exact_prediction_error_max"] == pytest.approx(
            float(exact_err.max()), rel=1e-12
        )
        assert pooled["surrogate_prediction_error_median"] == pytest.approx(
            float(np.median(surrogate_err)), rel=1e-12
        )
        ratios = np.asarray(
            [
                c["surrogate"]["suppression_ratio"]
                for c in cells
                if c["surrogate"]["suppression_ratio"] is not None
            ]
        )
        assert pooled["surrogate_suppression_ratio_median"] == pytest.approx(
            float(np.median(ratios)), rel=1e-12
        )
        assert pooled["surrogate_suppression_ratio_min"] == pytest.approx(
            float(ratios.min()), rel=1e-12
        )
        deltas = np.abs(np.asarray([c["exact"]["delta_attention"] for c in cells]))
        assert pooled["exact_abs_delta_attention_median"] == pytest.approx(
            float(np.median(deltas)), rel=1e-12
        )


def test_extreme_ratio_is_a_small_denominator(artifact: dict) -> None:
    """The −92.9 tail is a tiny intended suppression, not a numerical failure.

    A ratio of achieved to intended is unbounded when the intended term is near
    zero, and that is a real property of the estimand rather than a bug. This
    test pins the explanation: the outlier's denominator is a small fraction of
    the run's typical denominator, so the paper can report it as heavy-tailed and
    quote the median and the sign-reversal fraction instead of the extreme.
    """
    cells = _cells(artifact)
    ratios = np.asarray(
        [c["surrogate"]["suppression_ratio"] for c in cells
         if c["surrogate"]["suppression_ratio"] is not None]
    )
    intended = np.abs(
        np.asarray(
            [c["surrogate"]["intended_suppression"] for c in cells
             if c["surrogate"]["suppression_ratio"] is not None]
        )
    )
    assert ratios.min() < -50.0, "the heavy tail the paper describes is absent"
    outlier = int(np.argmin(ratios))
    assert intended[outlier] < 0.01 * float(np.median(intended)), (
        "the extreme ratio is not explained by a small denominator; it would be "
        "a numerical artifact and must not be reported as a measured cost"
    )
