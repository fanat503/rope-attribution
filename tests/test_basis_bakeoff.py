"""Tests for the basis bake-off.

The module's whole claim is that changing the basis changes nothing about
exactness and everything about the decomposition, so the tests pin three things:

* the seam: ``Basis`` is duck-typed on ``W_dec`` and must reproduce
  ``qk_attribution.token_buckets`` exactly for every arm;
* the folding rule: a direction's norm moves into its activation, so
  ``f W_dec`` is unchanged and ``top-k`` selects by contribution;
* the selection rule: magnitude, not signed value, because a signed basis would
  otherwise spend its budget on the largest positives.

The neuron arm gets two extra tests because its most plausible implementation
is wrong: the residual is the block's *output*, so applying the block's MLP to
it uses the wrong vector and skips the block's own LayerNorm. The arm is bound
to the captured MLP input instead, and the tests pin that.

The fidelity probe is checked for identification, the pooled slope and the
undefined ceiling of a complete basis; its numbers are a measurement, not an
identity, so pinning them would pin a fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rope_attribution import basis_bakeoff as B
from rope_attribution import qk_attribution as Q
from rope_attribution import rope as R

TOL = 1e-10


def toy_setup(
    seed: int = 0,
    d_model: int = 16,
    head_dim: int = 8,
    n_heads: int = 2,
    rotary_dim: int = 8,
) -> tuple[np.ndarray, B.Geometry]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((12, d_model))
    geometry = B.Geometry(
        ln_weight=rng.uniform(0.5, 1.5, d_model),
        ln_bias=rng.standard_normal(d_model) * 0.1,
        ln_eps=1e-5,
        w_q=rng.standard_normal((n_heads, head_dim, d_model)) / np.sqrt(d_model),
        b_q=rng.standard_normal((n_heads, head_dim)) * 0.1,
        w_k=rng.standard_normal((n_heads, head_dim, d_model)) / np.sqrt(d_model),
        b_k=rng.standard_normal((n_heads, head_dim)) * 0.1,
        freqs=R.inv_freq(rotary_dim, 10000.0),
        rotary_dim=rotary_dim,
    )
    return x, geometry


def neuron_blocks(
    d_model: int = 16,
    d_ff: int = 32,
    n_tokens: int = 12,
    n_blocks: int = 2,
    seed: int = 0,
) -> list[B.NeuronBlock]:
    """Fake GPT-NeoX MLP writes with captured inputs, without torch."""
    rng = np.random.default_rng(seed)
    blocks = []
    for layer in range(n_blocks):
        blocks.append(
            B.NeuronBlock(
                block=layer,
                inputs=rng.standard_normal((n_tokens, d_model)),
                w_in=rng.standard_normal((d_ff, d_model)) / np.sqrt(d_model),
                b_in=rng.standard_normal(d_ff) * 0.1,
                w_out=rng.standard_normal((d_model, d_ff)) / np.sqrt(d_ff),
                b_out=rng.standard_normal(d_model) * 0.1,
            )
        )
    return blocks


def all_bases(d_model: int = 16, seed: int = 0) -> list[B.Basis]:
    dictionary = Q.SparseDictionary.random(d_model, 24, seed=seed + 1)
    return [
        B.sae_basis(dictionary),
        B.neuron_basis(neuron_blocks(d_model=d_model, seed=seed + 2)),
        B.random_basis(d_model, 24, seed=seed + 3),
        B.coordinate_basis(d_model),
    ]


# --------------------------------------------------------------------------
# The seam: every arm must plug into token_buckets unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(4))
def test_every_basis_decomposes_the_query_exactly(index: int) -> None:
    """``base_q + sum(feature_q) == q`` for every arm, not just the SAE."""
    x, geometry = toy_setup()
    basis = all_bases()[index]
    activations = B._mask_top_k(basis.encode(x)[0], 8)
    buckets = Q.token_buckets(
        x[0],
        activations,
        basis,
        geometry.ln_weight,
        geometry.ln_bias,
        geometry.ln_eps,
        geometry.w_q[0],
        geometry.b_q[0],
        geometry.w_k[0],
        geometry.b_k[0],
        8,
    )
    q = Q.layernorm(x[0], geometry.ln_weight, geometry.ln_bias, geometry.ln_eps)
    expected = q @ geometry.w_q[0].T + geometry.b_q[0]
    assert np.abs(buckets.base_q + buckets.feature_q.sum(axis=0) - expected).max() < TOL


@pytest.mark.parametrize("index", range(4))
def test_measure_basis_is_exact_for_every_arm(index: int) -> None:
    """The score-level identity, through the full measurement path."""
    x, geometry = toy_setup()
    rows = B.measure_basis(
        all_bases()[index],
        x,
        geometry,
        deltas=(1, 2, 4),
        n_pairs=4,
        seed=0,
        top_k=8,
        heads=(0,),
        sweep_deltas=None,
    )
    assert len(rows) == 1
    assert rows[0]["max_relative_exactness_error"] < 1e-9
    assert rows[0]["fidelity"]["n_pairs"] > 0


# --------------------------------------------------------------------------
# The folding rule: norms move into the activations, not into the directions
# --------------------------------------------------------------------------


def test_the_sae_basis_preserves_the_reconstruction() -> None:
    rng = np.random.default_rng(5)
    dictionary = Q.SparseDictionary.random(16, 24, seed=6)
    basis = B.sae_basis(dictionary)
    x = rng.standard_normal(16)
    activations = basis.encode(x)
    decoded = dictionary.decode(dictionary.encode(x))
    assert np.abs(activations @ basis.W_dec - (decoded - dictionary.b_dec)).max() < TOL


def test_the_neuron_basis_reconstructs_the_mlp_writes() -> None:
    blocks = neuron_blocks(seed=8)
    basis = B.neuron_basis(blocks)
    expected = sum(
        B._gelu(block.inputs @ block.w_in.T + block.b_in) @ block.w_out.T
        for block in blocks
    )
    actual = basis.encode(blocks[0].inputs) @ basis.W_dec
    assert np.abs(actual - expected).max() < TOL


def test_the_neuron_basis_is_bound_to_the_captured_input() -> None:
    """The residual is the block's output; applying the MLP to it is wrong.

    The encoder must return the activations computed from the captured MLP
    input, not from whatever residual matrix it is handed. Passing zeros would
    silently produce a different basis if this regressed.
    """
    blocks = neuron_blocks(seed=9)
    basis = B.neuron_basis(blocks)
    other = np.zeros_like(blocks[0].inputs)
    assert np.array_equal(basis.encode(other), basis.encode(blocks[0].inputs))
    with pytest.raises(ValueError, match="bound to the forward pass"):
        basis.encode(blocks[0].inputs[:-1])


def test_the_neuron_basis_rejects_a_wrong_width_matrix() -> None:
    """Mutation: dropping the width check accepts a matrix the encoder ignores."""
    blocks = neuron_blocks(seed=13)
    basis = B.neuron_basis(blocks)
    wrong = np.zeros((blocks[0].inputs.shape[0], blocks[0].inputs.shape[1] + 1))
    with pytest.raises(ValueError, match="bound to the forward pass"):
        basis.encode(wrong)


def test_the_neuron_basis_covers_every_captured_block() -> None:
    blocks = neuron_blocks(n_blocks=3, seed=10)
    basis = B.neuron_basis(blocks)
    assert basis.provenance["blocks"] == [0, 1, 2]
    assert basis.n_features == 3 * 32


def test_the_neuron_basis_leaves_the_output_bias_in_the_base() -> None:
    blocks = neuron_blocks(seed=11)
    basis = B.neuron_basis(blocks)
    write = basis.encode(blocks[0].inputs) @ basis.W_dec
    full = sum(
        B._gelu(block.inputs @ block.w_in.T + block.b_in) @ block.w_out.T
        + block.b_out
        for block in blocks
    )
    expected_bias = sum(block.b_out for block in blocks)
    assert np.abs((full - write) - expected_bias).max() < TOL


def test_the_neuron_basis_requires_contiguous_blocks() -> None:
    blocks = neuron_blocks(n_blocks=2, seed=12)
    with pytest.raises(ValueError, match="contiguous"):
        B.neuron_basis([blocks[1]])


def test_every_basis_has_unit_directions() -> None:
    for basis in all_bases():
        norms = np.linalg.norm(basis.W_dec, axis=1)
        assert np.abs(norms - 1.0).max() < 1e-9, basis.name


def test_the_random_basis_is_deterministic() -> None:
    first = B.random_basis(16, 24, seed=3)
    second = B.random_basis(16, 24, seed=3)
    other = B.random_basis(16, 24, seed=4)
    assert np.array_equal(first.W_dec, second.W_dec)
    assert not np.allclose(first.W_dec, other.W_dec)


def test_the_coordinate_basis_is_the_identity() -> None:
    basis = B.coordinate_basis(16)
    assert np.array_equal(basis.W_dec, np.eye(16))
    x = np.arange(16, dtype=np.float64)
    assert np.array_equal(basis.encode(x), x)


def test_the_coordinate_basis_at_full_width_puts_the_score_in_the_base() -> None:
    """LayerNorm is scale-invariant, so J x = 0 and the features sum to zero."""
    x, geometry = toy_setup()
    basis = B.coordinate_basis(16)
    activations = B._mask_top_k(basis.encode(x)[0], 16)
    buckets = Q.token_buckets(
        x[0],
        activations,
        basis,
        geometry.ln_weight,
        geometry.ln_bias,
        geometry.ln_eps,
        geometry.w_q[0],
        geometry.b_q[0],
        geometry.w_k[0],
        geometry.b_k[0],
        16,
    )
    terms = Q.attribution_terms(
        buckets, buckets, 1, geometry.freqs, geometry.rotary_dim
    )
    # LayerNorm is scale-invariant only up to ``eps``: J x is not exactly zero,
    # it is the small epsilon term. The feature sum and the base gap must be
    # small relative to the score, not zero to machine precision.
    tolerance = 1e-4 * abs(terms.score)
    assert abs(float(terms.query_features.sum())) < tolerance
    assert abs(terms.base - terms.score) < tolerance


# --------------------------------------------------------------------------
# The selection rule: magnitude, not signed value
# --------------------------------------------------------------------------


def test_the_mask_keeps_the_largest_magnitudes() -> None:
    """A signed basis must not spend its budget on the largest positives."""
    values = np.array([0.1, -5.0, 0.2, 3.0])
    masked = B._mask_top_k(values, 2)
    assert masked[1] == -5.0
    assert masked[3] == 3.0
    assert np.isneginf(masked[0])
    assert np.isneginf(masked[2])


def test_the_mask_keeps_exactly_k_finite_entries() -> None:
    values = np.linspace(-1.0, 1.0, 11)
    masked = B._mask_top_k(values, 4)
    assert np.isfinite(masked).sum() == 4


def test_the_mask_refuses_an_empty_selection() -> None:
    with pytest.raises(ValueError):
        B._mask_top_k(np.zeros(4), 0)


# --------------------------------------------------------------------------
# Small arithmetic helpers
# --------------------------------------------------------------------------


def test_gelu_matches_the_exact_definition() -> None:
    values = np.array([0.0, 1.0, -1.0, 3.0])
    expected = np.array([0.0, 0.8413447460685429, -0.15865525393145707, 2.99595071])
    assert np.abs(B._gelu(values) - expected).max() < 1e-6


def test_the_ratio_and_correlation_are_scale_free() -> None:
    # The scale guard is the detector for a zero score magnitude: without it
    # the ratio returns 1.0 and reports agreement where there is no score.
    assert B._ratio([2.0, 4.0], [1.0, 2.0]) == pytest.approx(2.0)
    assert B._ratio([1.0], [0.0]) is None
    assert B._ratio([1.0, -1.0], [1.0, -1.0]) is None
    assert B._ratio([1.0], [1.0], scale=1e6) is None
    assert B._ratio([1.0], [1.0], scale=0.0) is None
    # Non-finite values are not a measurement; without the guard they reach the
    # strict JSON writer and crash the run after the expensive part.
    assert B._ratio([float("nan")], [1.0]) is None
    assert B._ratio([1.0], [float("inf")]) is None
    assert B._correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert B._correlation([1.0], [2.0]) is None
    assert B._correlation([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0]) is None


def test_optional_float_turns_non_finite_into_none() -> None:
    assert B._optional_float(None) is None
    assert B._optional_float(float("nan")) is None
    assert B._optional_float(float("inf")) is None
    assert B._optional_float(1.5) == 1.5


def test_the_identification_floor_rejects_zero_predictions() -> None:
    """Mutation: dropping the floor makes a zero prediction look identified."""
    assert B._identified(0.0, 4, 100.0) is False
    assert B._identified(1e-12, 4, 100.0) is False
    assert B._identified(1.0, 4, 100.0) is True
    assert B._identified(1.0, 0, 100.0) is False
    assert B._identified(1.0, 4, 0.0) is False
    assert B._identified(float("inf"), 4, 100.0) is False
    assert B._identified(1.0, 4, float("nan")) is False


def _fidelity_row(
    predicted: list[float],
    actual: list[float],
    *,
    ceiling_predicted: list[float] | None = None,
    ceiling_actual: list[float] | None = None,
    score_scale: float = 100.0,
) -> dict:
    cp = predicted if ceiling_predicted is None else ceiling_predicted
    ca = actual if ceiling_actual is None else ceiling_actual
    return {
        "head": 0,
        "base_share_mean": 1.0,
        "top_k_share_mean": 1.0,
        "max_relative_exactness_error": 0.0,
        "sweep_feature_sign_flip_fraction": None,
        "sweep_pair_sign_flip_fraction": None,
        "fidelity": {
            "n_pairs": len(predicted),
            "graded_features": 2,
            "ceiling_features": 2,
            "score_scale": score_scale,
            "predicted_sum": float(sum(predicted)),
            "actual_sum": float(sum(actual)),
            "predicted_sq_sum": float(sum(x * x for x in predicted)),
            "actual_sq_sum": float(sum(x * x for x in actual)),
            "predicted_actual_sum": float(
                sum(p * a for p, a in zip(predicted, actual, strict=True))
            ),
            "ratio": None,
            "slope": None,
            "correlation": None,
            "ceiling_predicted_sum": float(sum(cp)),
            "ceiling_actual_sum": float(sum(ca)),
            "ceiling_predicted_sq_sum": float(sum(x * x for x in cp)),
            "ceiling_actual_sq_sum": float(sum(x * x for x in ca)),
            "ceiling_predicted_actual_sum": float(
                sum(p * a for p, a in zip(cp, ca, strict=True))
            ),
            "ceiling_ratio": None,
            "ceiling_slope": None,
            "ceiling_correlation": None,
        },
    }


def test_the_aggregate_pools_the_slope_over_pairs() -> None:
    """A mean of per-head slopes would report 0.75; the pooled slope is 0.512.

    Mutation: replacing the pooled slope with a mean of per-head slopes fails
    the first assertion, which is the detector for the exact bug the pooling
    rule exists to prevent.
    """
    first = _fidelity_row([1.0, 2.0], [1.0, 2.0], score_scale=10.0)
    second = _fidelity_row([10.0, 10.0], [5.0, 5.0], score_scale=100.0)
    pooled = B._aggregate([first, second])
    expected = (1 * 1 + 2 * 2 + 10 * 5 + 10 * 5) / (1 + 4 + 100 + 100)
    assert pooled["fidelity_slope"] == pytest.approx(expected)
    assert pooled["fidelity_ratio"] == pytest.approx((1 + 2 + 5 + 5) / (1 + 2 + 10 + 10))
    assert pooled["n_pairs"] == 4
    assert pooled["n_heads"] == 2


def test_the_ceiling_is_undefined_for_a_complete_basis() -> None:
    """A full basis predicts a near-zero full-ablation change; do not divide by it.

    Mutation: dropping the ceiling identification guard publishes a huge
    ``ceiling_slope`` for a denominator that is numerical residue, and the
    ``is None`` assertions fail.
    """
    x, geometry = toy_setup()
    rows = B.measure_basis(
        B.coordinate_basis(16),
        x,
        geometry,
        deltas=(1, 2, 4),
        n_pairs=3,
        seed=0,
        top_k=16,
        heads=(0,),
        sweep_deltas=None,
        ablation=4,
    )
    fidelity = rows[0]["fidelity"]
    assert fidelity["ceiling_features"] == 16
    assert fidelity["ceiling_slope"] is None
    assert fidelity["ceiling_ratio"] is None


def test_measure_basis_clips_unrealizable_distances() -> None:
    """A 12-token toy cannot realize delta 100; the row must say what it measured.

    Mutation: dropping ``distance_grid`` records delta 100 as measured, and the
    ``[1, 2]`` assertion fails.
    """
    x, geometry = toy_setup()
    rows = B.measure_basis(
        B.coordinate_basis(16),
        x,
        geometry,
        deltas=(1, 2, 100),
        n_pairs=2,
        seed=0,
        top_k=8,
        heads=(0,),
        sweep_deltas=None,
    )
    assert rows[0]["deltas"] == [1, 2]


def test_measure_basis_rejects_duplicate_heads() -> None:
    """Mutation: dropping the uniqueness check measures head 0 twice."""
    x, geometry = toy_setup()
    with pytest.raises(ValueError, match="unique"):
        B.measure_basis(
            B.coordinate_basis(16),
            x,
            geometry,
            deltas=(1,),
            n_pairs=1,
            seed=0,
            top_k=8,
            heads=(0, 0),
            sweep_deltas=None,
        )


def test_the_neuron_arm_refuses_a_non_gelu_checkpoint() -> None:
    """Mutation: dropping the guard accepts a tanh MLP and models the wrong write."""
    B._require_exact_gelu(SimpleNamespace(config=SimpleNamespace(hidden_act="gelu")))
    with pytest.raises(NotImplementedError, match="exact GELU"):
        B._require_exact_gelu(
            SimpleNamespace(config=SimpleNamespace(hidden_act="gelu_new"))
        )


# --------------------------------------------------------------------------
# The harness refuses what it cannot measure, before loading a checkpoint
# --------------------------------------------------------------------------


def test_the_sae_arm_requires_a_dictionary() -> None:
    with pytest.raises(ValueError, match="needs --sae"):
        B.bakeoff(bases=("sae",))


def test_a_missing_dictionary_is_rejected_before_the_checkpoint() -> None:
    """Mutation: dropping the pre-load check pays the model load for a typo."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        B.bakeoff(bases=("sae",), sae_path="does-not-exist.safetensors")


def test_the_bakeoff_rejects_an_empty_or_duplicated_arm_list() -> None:
    """Mutation: dropping the check loads a checkpoint for an empty or repeated arm."""
    with pytest.raises(ValueError, match="at least one basis"):
        B.bakeoff(bases=())
    with pytest.raises(ValueError, match="unique"):
        B.bakeoff(bases=("sae", "sae"), sae_path="unused.safetensors")


def test_an_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown bases"):
        B.bakeoff(bases=("sae", "nope"))


def test_write_results_round_trips(tmp_path: Path) -> None:
    path = B.write_results({"schema": B.SCHEMA}, tmp_path / "basis.json")
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == B.SCHEMA


def test_write_results_refuses_non_finite_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        B.write_results(
            {"schema": B.SCHEMA, "x": float("nan")}, tmp_path / "bad.json"
        )


# --------------------------------------------------------------------------
# The grid and the decision rule are pre-registered, not chosen after the run
# --------------------------------------------------------------------------


def test_the_sweep_grid_is_the_paper_grid() -> None:
    assert len(B.SWEEP_DELTAS) == 13
    assert B.SWEEP_DELTAS[0] == 1
    assert B.SWEEP_DELTAS[-1] == 4096


def test_the_decision_rule_is_fixed() -> None:
    assert "0.5" in B.KILL
    assert "> 0" in B.DECISION
    assert "0.8" in B.DECISION
    assert "0.10" in B.DECISION
    assert "fidelity_slope" in B.DECISION
    assert "neurons" in B.DECISION


def _arm(name: str, slope: float | None, base: float | None) -> dict:
    return {
        "name": name,
        "aggregate": {"fidelity_slope": slope, "base_share_mean": base},
    }


def test_the_verdict_requires_both_arms() -> None:
    result = B.verdict([_arm("sae", 1.0, 1.0)])
    assert result["applicable"] is False


def test_the_verdict_applies_the_pre_registered_rule() -> None:
    assert (
        B.verdict([_arm("sae", 1.0, 1.0), _arm("neurons", 0.9, 1.05)])["neurons_win"]
        is True
    )
    assert (
        B.verdict([_arm("sae", 1.0, 1.0), _arm("neurons", 0.7, 1.05)])["neurons_win"]
        is False
    )
    assert (
        B.verdict([_arm("sae", 1.0, 1.0), _arm("neurons", 0.9, 1.2)])["neurons_win"]
        is False
    )
    assert (
        B.verdict([_arm("sae", 1.0, 1.0), _arm("neurons", None, 1.0)])["applicable"]
        is False
    )
    # The rule uses >= and <=, so both boundaries must pass and a hair beyond
    # either must fail; an off-by-one-epsilon rewrite is the mutation.
    assert (
        B.verdict([_arm("sae", 1.0, 1.0), _arm("neurons", 0.8, 1.1)])["neurons_win"]
        is True
    )
    assert (
        B.verdict([_arm("sae", 1.0, 1.0), _arm("neurons", 0.799, 1.1)])["neurons_win"]
        is False
    )
    assert (
        B.verdict([_arm("sae", 1.0, 1.0), _arm("neurons", 0.9, 1.101)])["neurons_win"]
        is False
    )


def test_the_verdict_refuses_a_non_finite_metric() -> None:
    """Mutation: a NaN is not ``None`` and would otherwise decide the rule."""
    result = B.verdict(
        [_arm("sae", float("nan"), 1.0), _arm("neurons", 0.9, 1.0)]
    )
    assert result["applicable"] is False
    assert "undefined" in result["reason"]


def test_the_verdict_refuses_a_non_positive_sae_slope() -> None:
    """With a negative denominator the ratio rule is meaningless, not a pass.

    Mutation: dropping the guard turns a negative SAE slope into ``applicable``
    and lets the 0.8 comparison run with the wrong sign.
    """
    result = B.verdict([_arm("sae", -0.5, 1.0), _arm("neurons", 0.9, 1.0)])
    assert result["applicable"] is False
    assert "not positive" in result["reason"]


def test_a_negative_neuron_slope_is_a_stage_one_kill() -> None:
    """Mutation: dropping the kill leaves an anti-correlated arm eligible."""
    result = B.verdict([_arm("sae", 1.0, 1.0), _arm("neurons", -0.1, 1.0)])
    assert result["applicable"] is True
    assert result["stage1_kill"] is True
    assert result["neurons_win"] is False


def test_the_verdict_reports_the_stage_one_kill() -> None:
    result = B.verdict([_arm("sae", 1.0, 1.0), _arm("neurons", 0.4, 1.0)])
    assert result["stage1_kill"] is True
    assert result["neurons_win"] is False
    assert "stage-1 kill" in result["reason"]
