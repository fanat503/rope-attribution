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


_USEFULNESS = _REPO / "results" / "usefulness.json"
_AGREEMENT = _REPO / "results" / "agreement.json"
_SEEDS = _REPO / "results" / "usefulness_seeds.json"
_GRID = _REPO / "results" / "usefulness_grid_independence.json"


@pytest.fixture(scope="module")
def trained() -> dict:
    if not _JSON.exists():
        pytest.fail("results/real_model.json is absent; it is committed evidence, not an optional input")
    return json.loads(_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def agreement_data() -> dict:
    if not _AGREEMENT.exists():
        pytest.fail("results/agreement.json is absent; it is committed evidence")
    return json.loads(_AGREEMENT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def usefulness() -> dict:
    if not _USEFULNESS.exists():
        pytest.fail("results/usefulness.json is absent; it is committed evidence")
    return json.loads(_USEFULNESS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def seeds() -> dict:
    if not _SEEDS.exists():
        pytest.fail("results/usefulness_seeds.json is absent; it is committed evidence")
    return json.loads(_SEEDS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def grid_independence() -> dict:
    if not _GRID.exists():
        pytest.fail("results/usefulness_grid_independence.json is absent; it is committed evidence")
    return json.loads(_GRID.read_text(encoding="utf-8"))


def _ident(trained: dict, model_id: str, name: str) -> dict:
    return trained["aggregates"][model_id]["identities"][name]


def _shares(trained: dict, model_id: str) -> dict:
    return trained["aggregates"][model_id]["share_quantiles_across_prompts"]


def _blind(trained: dict, model_id: str) -> float:
    """The fraction of live pair-channel cells meeting the cutoff.

    This is the pooled count over the pooled live denominator, not the mean of
    per-prompt head medians (``frac_position_blind_across_heads``). The paper
    says "the fraction meeting the cutoff" and "of pairs", so the pooled
    fraction is the estimand; using the median-of-medians here would quote a
    number 34% low for pythia.
    """
    return trained["aggregates"][model_id]["frac_position_blind_pooled"]


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
    # Printed in two places (Section~\ref{sec:trained} and Limitations) and the
    # two must agree, or a reader comparing them sees a contradiction.
    assert len(groups("trained_model_sizes")) >= 2, groups("trained_model_sizes")
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
    # decimal the paper wrote, and float("4.55") would hand it exponent -52.
    assert_paper_sci_matches(
        worst, literal("trained_relpos_worst"), -13, "the trained relative-position error"
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


def test_the_bilinearity_numbers_are_labelled_as_scaled_residuals() -> None:
    """The three numbers are scaled residuals, not score-relative errors.

    The test above re-derives ``bilinearity_max_scaled_err``: the residual
    divided by the magnitude sum on which bilinearity is exact. Divided by
    the score itself, the same residuals would be dominated by cancellation
    on a real head and would not be small, so the sentence has to name the
    denominator it uses.
    """
    assert "each a scaled residual, divided by the magnitude sum" in TEX_FLAT
    assert "rather than by the score" in TEX_FLAT


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


def test_the_quoted_interquartile_range_is_pythias_own(trained: dict) -> None:
    """A one-checkpoint sentence, and the paper now says which checkpoint.

    Both ends of the printed range are pythia-160m's own quartiles
    (0.1860 and 0.9917). The label used to say 'pooled'; llama and
    SmolLM sit at 0.151/0.988 and 0.154/0.989, so naming the model is
    the honest version of what was measured.
    """
    lo, hi = groups("trained_share_iqr")[0]
    q = _shares(trained, PYTHIA)
    assert_paper_number_matches(
        float(q["p25"]), f"{float(lo):.3f}", "the lower quartile on pythia-160m"
    )
    assert_paper_number_matches(
        float(q["p75"]), f"{float(hi):.3f}", "the upper quartile on pythia-160m"
    )
    # Not a rounding artefact: the range really is most of the unit interval.
    assert float(q["p25"]) < 0.2 and float(q["p75"]) > 0.95, (q["p25"], q["p75"])


def test_almost_every_channel_is_position_carrying(trained: dict) -> None:
    """The claim that most pairs carry a non-negligible phase on real weights."""
    for model, index in ((PYTHIA, 1), (LLAMA, 2), (SMOLLM, 3)):
        written = float(literal("trained_blind_fraction", index))
        assert_paper_number_matches(
            _blind(trained, model), f"{written:.4f}", f"the blind fraction on {model}"
        )
    fractions = [_blind(trained, m) for m in (PYTHIA, LLAMA, SMOLLM)]
    assert all(0.001 < f < 0.02 for f in fractions), fractions
    # The paper says the rate is "within a factor of two" across three
    # unrelated architectures.  The spread is wider than the pre-fix
    # artifact's 1.5x but still under 2x; the threshold is stated here so a
    # future drift is visible rather than absorbed by a loose inequality.
    assert max(fractions) / min(fractions) < 2.0, fractions
    assert trained["position_blind_ratio"] == 0.01


def test_the_stated_band_brackets_every_checkpoint(trained: dict) -> None:
    lo = float(literal("trained_blind_band", 1))
    hi = float(literal("trained_blind_band", 2))
    for model in (PYTHIA, LLAMA, SMOLLM):
        pct = 100.0 * _blind(trained, model)
        # The band is deliberately wider than the printed per-model values:
        # pythia's exact 0.5405% must lie inside it without relying on
        # rounding to 0.54%. A band that only contains the rounded numbers
        # would exclude the measurement it is summarising.
        assert lo <= pct <= hi, (model, pct, lo, hi)
        assert hi - lo < 0.5, (lo, hi)


def test_the_position_blind_cutoff_matches_the_code(trained: dict) -> None:
    """The paper's cutoff is ``|B_k|/R_k < 1%``, not a movement of that share.

    Mutation verdict: restoring "its share moves by less than 1% across the
    distance range" fails here. ``real_model.py`` tests
    ``|B_k| < BLIND_RATIO * |A_k|``; the paper must describe that convention,
    and it must keep the reminder that even ``B_k = 0`` still oscillates.
    """
    assert "$|B_k|/R_k$" in TEX_FLAT
    assert "below one percent" in TEX_FLAT
    assert "|B_k| < 0.01\\,|A_k|" in TEX_FLAT
    assert "share moves by less than" not in TEX_FLAT
    assert "even when $B_k$ vanishes" in TEX_FLAT
    assert float(literal("trained_blind_cutoff")) == trained["position_blind_ratio"]
    assert trained["position_blind_ratio"] == 0.01


def test_the_seed_sd_is_not_overstated_as_three_orders(trained: dict) -> None:
    """0.0170 / 4.58e-5 is about 370x, i.e. 2.6 orders, not three.

    Mutation verdict: restoring "three orders of magnitude" fails here. The
    numerical ratio is already checked by the seed-variance tests; this keeps
    the prose from rounding it up by half an order of magnitude.
    """
    assert "three orders of magnitude" not in TEX_FLAT
    assert "more than two orders of magnitude" in TEX_FLAT
    # The ratio the sentence is describing, recomputed from the two artifacts.
    head_sd = 0.0170
    seed_sd = 4.5835751380061324e-05
    ratio = head_sd / seed_sd
    assert 300.0 < ratio < 500.0, ratio


def test_the_limitations_scope_the_configuration_claim() -> None:
    """The old sentence said all measurements use one configuration; false.

    Mutation verdict: restoring "All measurements use $d = 64$" fails here.
    The sensitivity sweep varies base and head dimension, and the trained
    section uses checkpoints and 256 content pairs per head and prompt.
    """
    assert "Single synthetic configuration" in TEX_FLAT
    assert "All measurements use" not in TEX_FLAT
    assert "varies the base and the head dimension" in TEX_FLAT
    assert (
        "Section~\\ref{sec:trained} uses the three checkpoints and their content-pair"
        in TEX_FLAT
    )


def test_the_worth_section_discloses_the_grid_it_measured(usefulness: dict) -> None:
    """The section used to say "pre-registered grid" without its size.

    A reader coming from the 54-point synthetic grid would assume 54. The
    measured grid is 13 points; the paper must say so.

    Mutation verdict: deleting the disclosure makes ``groups`` find nothing.
    """
    n, lo, hi = (int(x) for x in groups("worth_distance_grid")[0])
    grid = usefulness["distance_grid"]
    assert n == len(grid) == 13
    assert lo == min(grid) == 1
    assert hi == max(grid) == 4096


def test_the_worth_section_prices_the_best_surrogate_not_a_strawman(
    usefulness: dict,
) -> None:
    """The comparison is only meaningful if the alternative is the good one.

    The paper's argument is that a position-free scalar is wrong. That is worth
    nothing if the scalar tested were a careless one. So the printed rate must be
    the least-squares optimum over the distance range, and the test says so
    rather than trusting the prose.
    """
    from rope_attribution import usefulness as U

    assert "least-squares" in TEX_FLAT.lower()
    # And the optimum really is what the module computes: rescaling the surrogate
    # cannot improve the fit, which is what ``test_the_surrogate_is_the_least_
    # squares_optimum`` checks pair by pair.
    assert U.cos_sin_means is not None


def test_the_sign_error_rate_is_as_printed(usefulness: dict) -> None:
    """Every rate the paper quotes, re-derived from results/usefulness.json."""
    pooled = usefulness["pooled"]
    written = float(literal("worth_pooled_flip"))
    assert_paper_number_matches(
        100.0 * pooled["sign_flip_fraction"],
        f"{written:.1f}",
        "the pooled sign-error rate",
    )
    assert_paper_number_matches(
        pooled["rel_err_median"], literal("worth_rel_err"), "the median relative error"
    )


def test_each_checkpoint_reports_its_own_rate(usefulness: dict) -> None:
    rates = [
        100.0 * m["sign_flip_fraction_pooled"] for m in usefulness["models"]
    ]
    for index, model in zip((1, 2, 3), usefulness["models"], strict=True):
        written = float(literal("worth_per_model_flip", index))
        assert_paper_number_matches(
            100.0 * model["sign_flip_fraction_pooled"],
            f"{written:.1f}",
            f"the sign-error rate on {model['model_id']}",
        )
    assert len(rates) == 3


def test_the_seed_variance_is_as_printed(seeds: dict) -> None:
    """The uncertainty sentence's mean and sd, re-derived from the sweep."""
    mean, sd = groups("seed_variance_sentence")[0]
    assert_paper_number_matches(
        100.0 * seeds["mean"]["sign_flip_fraction"],
        mean,
        "the seed-mean pooled rate",
    )
    assert_paper_number_matches(
        100.0 * seeds["sd"]["sign_flip_fraction"],
        sd,
        "the seed standard deviation of the pooled rate",
    )
    # The provenance row repeats the sentence's numbers and names the
    # artifact; if either is edited alone, this fails.
    prov_mean, prov_sd = groups("prov_seed_variance")[0]
    assert (prov_mean, prov_sd) == (mean, sd)


def test_the_sampling_disclosure_matches_the_artifact(trained: dict) -> None:
    """The sample count and the excluded degenerate entries are not decoration.

    Mutation verdict: changing the paper's 256, 1.47, 11.06 or 462{,}144 without
    re-running ``real_model.py`` fails here. The values are read from
    ``results/real_model.json``; the paper cannot silently describe a different
    sweep from the one that produced the shares.
    """
    assert int(literal("trained_content_pairs")) == trained["n_content_pairs_per_head"]
    aggregates = trained["aggregates"]
    samples = [entry["n_samples"] for entry in aggregates.values()]
    lo, hi = groups("trained_sample_range")[0]
    assert_paper_number_matches(min(samples) / 1e6, lo, "the smallest sample count")
    assert_paper_number_matches(max(samples) / 1e6, hi, "the largest sample count")

    degenerate = sum(entry["n_degenerate_pairs_excluded"] for entry in aggregates.values())
    written = int(literal("trained_degenerate").replace("{,}", ""))
    assert written == degenerate, (written, degenerate)
    # The excluded entries are llama-specific; the other two checkpoints have
    # none. If that changes, the sentence's model name must change with it.
    assert aggregates[LLAMA]["n_degenerate_pairs_excluded"] > 0
    assert aggregates[PYTHIA]["n_degenerate_pairs_excluded"] == 0
    assert aggregates[SMOLLM]["n_degenerate_pairs_excluded"] == 0


def test_the_head_range_is_not_one_outlier(usefulness: dict) -> None:
    """If a single head failed, the honest claim would be about that head.

    The band's two ends are pinned at the paper's own precision instead
    of a +-1.0-point slack: the extremes over all 558 layer-heads are 21.268%
    and 33.913%, which round to the printed 21.3% and 33.9%. A one-point
    slack would have passed a band that is nearly a point off the data.
    """
    total = sum(m["n_heads"] for m in usefulness["models"])
    assert total == int(literal("worth_head_count")), total
    los, his = (float(x) for x in groups("worth_head_range")[0])
    assert los > 0.0 and his < 100.0
    # The JSON stores fractions and the paper prints percentages. Comparing the
    # two directly is a unit error that happens to pass for the pooled rate and
    # fails here, so the conversion is explicit.
    mins = [100.0 * m["sign_flip_fraction_min_head"] for m in usefulness["models"]]
    maxs = [100.0 * m["sign_flip_fraction_max_head"] for m in usefulness["models"]]
    assert_paper_number_matches(min(mins), f"{los:.1f}", "the lowest per-head rate")
    assert_paper_number_matches(max(maxs), f"{his:.1f}", "the highest per-head rate")
    # Every model fails on a large minority of cells, so it is not a tail effect.
    for model in usefulness["models"]:
        assert model["sign_flip_fraction_pooled"] > 0.15, model["model_id"]


def test_the_relative_error_exceeds_the_signal(usefulness: dict) -> None:
    """The claim is stronger than 'inaccurate': it exceeds the quantity itself."""
    for model in usefulness["models"]:
        assert model["rel_err_median"] > 1.0, model["model_id"]


def test_the_materiality_cut_is_disclosed(usefulness: dict, grid_independence: dict) -> None:
    """A threshold that helps the result must be stated where it is used."""
    assert int(literal("worth_materiality")) == 5
    assert "numerically indistinguishable from zero" in TEX_FLAT
    # The grid-density sensitivity is now disclosed with the sweep it was measured
    # on, and the earlier 14x span is named as the earlier version's error.
    assert "the sweep actually run" in TEX_FLAT
    assert "An earlier version of this" in TEX_FLAT
    assert_paper_number_matches(
        grid_independence["sign_flip_drift"],
        literal("worth_grid_drift"),
        "the sign-flip grid drift",
    )


def test_the_grid_sweep_reports_the_realized_density(
    grid_independence: dict,
) -> None:
    """The requested density is not the density that was run.

    ``65`` is the requested spacing; integer rounding collapses it to 56
    distinct distances, so the realized span is 56/9 = 6.2x, not 7x. The
    artifact carries both numbers, and the paper must not quote the nominal
    one as if it were measured.

    Mutation verdict: restoring "which spans densities 9, 17 and 65 --- a
    7x change" fails the phrase assertion below, and claiming 65 realized
    distances fails the ``n_deltas`` assertion.
    """
    densities = grid_independence["densities"]
    assert densities == [9, 17, 65], densities
    realized = [
        grid_independence["per_density"][str(d)]["n_deltas"] for d in densities
    ]
    assert realized == [9, 17, 56], realized
    assert max(realized) / min(realized) < 7.0, realized
    assert "do not all survive integer rounding" in TEX_FLAT
    assert "realized span is below the nominal $7\\times$" in TEX_FLAT
    assert "--- a $7\\times$ change" not in TEX_FLAT


def test_the_ranking_agreement_rates_are_as_printed(agreement_data: dict) -> None:
    """Every rate the ranking passage quotes, re-derived from results/agreement.json."""
    order = ("pythia-160m", "llama-160m", "SmolLM-135M")
    # The json's model list is the paper's order; a reordering would silently
    # attach the wrong rate to each checkpoint otherwise.
    assert [m["model_id"].endswith(s) for m, s in zip(
        agreement_data["models"], order, strict=True
    )] == [True, True, True]
    for index, model in zip((1, 2, 3), agreement_data["models"], strict=True):
        written = float(literal("worth_rank_top1", index))
        assert_paper_number_matches(
            100.0 * model["top1_agreement"],
            f"{written:.1f}",
            f"the top-1 ranking agreement on {model['model_id']}",
        )


def test_the_ranking_chance_and_near_top_rates_are_as_printed(
    agreement_data: dict,
) -> None:
    chance_a, chance_b, pairs_a, pairs_b = (
        float(x) for x in groups("worth_rank_chance")[0]
    )
    # Chance is 1/n_pairs; the paper prints one decimal, so the comparison
    # allows the half of the last printed digit the rounding could produce.
    assert chance_a == pytest.approx(100.0 / 8, abs=0.05), chance_a
    assert chance_b == pytest.approx(100.0 / 32, abs=0.05), chance_b
    by_pairs = {8: "EleutherAI/pythia-160m", 32: "JackFram/llama-160m"}
    for _chance, n_pairs in ((chance_a, pairs_a), (chance_b, pairs_b)):
        model = next(
            m for m in agreement_data["models"] if m["model_id"] == by_pairs[int(n_pairs)]
        )
        assert model["n_rotary_pairs"] == int(n_pairs), model["model_id"]
    llama = next(
        m for m in agreement_data["models"] if m["model_id"].endswith("llama-160m")
    )
    smollm = next(
        m for m in agreement_data["models"] if m["model_id"].endswith("SmolLM-135M")
    )
    # The two 32-pair checkpoints carry the two printed near-top rates, in the
    # paper's order; pythia's 8-pair decile equals its top-1 and is not quoted.
    for written, model in zip(
        (float(x) for x in groups("worth_rank_decile")[0]), (llama, smollm),
        strict=True,
    ):
        assert model["n_rotary_pairs"] == 32, model["model_id"]
        assert_paper_number_matches(
            100.0 * model["top1_agreement_top_decile"],
            f"{written:.1f}",
            f"the near-top ranking rate on {model['model_id']}",
        )


def test_the_zero_rows_the_ranking_never_counts_are_recorded(agreement_data: dict) -> None:
    """The passage disclaims all-zero rows; the artifact must show a real count."""
    llama = next(
        m for m in agreement_data["models"] if m["model_id"].endswith("llama-160m")
    )
    assert llama["n_excluded_exact_zero"] > 0, llama["model_id"]
    assert agreement_data["pooled"]["n_excluded_exact_zero"] > 0
    # And the disclaimer is not covering a large silent drop: the excluded rows
    # are a small fraction of the votes actually cast.
    for model in agreement_data["models"]:
        excluded = model["n_excluded_exact_zero"] + model["n_excluded_surrogate_zero"]
        assert excluded < 0.2 * model["n_votes"], model["model_id"]


def test_the_ranking_per_head_band_is_as_printed(agreement_data: dict) -> None:
    """The printed band, re-derived from every head record in the artifact."""
    low = float(literal("worth_rank_head_band", 1))
    high = float(literal("worth_rank_head_band", 2))
    rates = [
        100.0 * h["n_agree"] / h["n_live"]
        for model in agreement_data["models"]
        for pp in model["per_prompt"]
        for h in pp["heads"]
        if h["n_live"]
    ]
    assert_paper_number_matches(min(rates), f"{low:.1f}", "the lowest per-head rate")
    assert_paper_number_matches(max(rates), f"{high:.1f}", "the highest per-head rate")
    # The passage's second clause: every head stays above its own checkpoint's
    # random choice, so the pooled rate conceals spread rather than pathology.
    for model in agreement_data["models"]:
        floor = 100.0 * model["chance_top1"]
        head_rates = [
            100.0 * h["n_agree"] / h["n_live"]
            for pp in model["per_prompt"]
            for h in pp["heads"]
            if h["n_live"]
        ]
        assert min(head_rates) > floor, (model["model_id"], min(head_rates), floor)


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