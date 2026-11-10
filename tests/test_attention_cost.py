"""Tests for the attention-level price of a position-free attribution.

Three things must hold before any number from ``attention_cost.json`` is quoted.

1. The *exact* score the module computes is the model's score. It is checked
   against ``experiments.score_relative``, which reaches the same quantity by a
   different route (rotate both vectors, then dot), so a sign or index error in
   the pair algebra cannot survive.
2. The *surrogate* is the best position-free constant, not a strawman. It is
   checked against an explicit least-squares fit over the distances that query
   actually sees.
3. The headline numbers in the committed artifact are re-derived here from the
   per-head rows, so the JSON cannot drift away from the aggregate it reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rope_attribution import attention_cost as A
from rope_attribution.rope import apply_rope, inv_freq, rope_cos_sin

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "results" / "attention_cost.json"

EXACT_TOL = 1e-12


def random_head(seed: int, n_tokens: int, head_dim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_tokens, head_dim)),
        rng.standard_normal((n_tokens, head_dim)),
    )


def reference_score(
    q_m: np.ndarray, k_n: np.ndarray, freqs: np.ndarray, rope_dim: int, delta: int
) -> float:
    """``q_m . R_delta k_n`` for partial RoPE, rotating the vectors explicitly.

    Deliberately a different route from the module's aligned/crossed algebra: if
    the two disagree, one of them is wrong and the paper cannot say "exact".
    """
    cos, sin = rope_cos_sin(np.array([0, delta]), freqs)
    q_rot = apply_rope(q_m[None, :rope_dim], cos[0:1], sin[0:1])[0]
    k_rot = apply_rope(k_n[None, :rope_dim], cos[1:2], sin[1:2])[0]
    return float(q_rot @ k_rot + q_m[rope_dim:] @ k_n[rope_dim:])


# --------------------------------------------------------------------------
# geometry: which distance is which
# --------------------------------------------------------------------------


def test_the_distance_matrix_is_key_minus_query() -> None:
    delta = A.distance_matrix(4)
    assert delta.shape == (4, 4)
    # RoPE rotates the key forward by ``n - m``; the query is the reference.
    assert delta[0, 3] == 3.0
    assert delta[3, 0] == -3.0
    assert np.all(np.diag(delta) == 0.0)
    # Causal pairs are exactly the non-positive entries.
    mask = A.causal_mask(4)
    assert np.array_equal(mask, delta <= 0)
    assert mask[0].tolist() == [True, False, False, False]


# --------------------------------------------------------------------------
# the exact score, against an independent implementation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rope_dim", [8, 4])
def test_exact_scores_match_the_rotate_then_dot_route(rope_dim: int) -> None:
    """``exact[m, n]`` is ``q_m . R_{n-m} k_n`` for every pair, not just the band.

    ``reference_score`` builds the rotation explicitly and dots the two rotated
    vectors. The module here uses the aligned/crossed pair algebra. Agreement to
    machine precision is what makes the word "exact" honest, and the ``4`` case
    covers partial RoPE, where the unrotated tail must survive untouched.
    """
    q, k = random_head(11, 7, head_dim=8)
    freqs = inv_freq(rope_dim, 10000.0)
    exact, _ = A.exact_and_surrogate_scores(q, k, freqs, rope_dim)
    worst = 0.0
    for m in range(q.shape[0]):
        for n in range(q.shape[0]):
            worst = max(worst, abs(exact[m, n] - reference_score(q[m], k[n], freqs, rope_dim, n - m)))
    assert worst < EXACT_TOL, worst


def test_the_unrotated_tail_is_carried_not_dropped() -> None:
    """Partial RoPE leaves channels unrotated; they must still be in the score.

    If the tail were dropped, the exact score would equal the pure-rotary score
    and this test would fail on a head whose tail dominates.
    """
    head_dim, rope_dim = 8, 4
    rng = np.random.default_rng(3)
    q = rng.standard_normal((5, head_dim))
    k = rng.standard_normal((5, head_dim))
    # Concentrate the signal in the unrotated channels.
    q[:, rope_dim:] *= 50.0
    k[:, rope_dim:] *= 50.0
    freqs = inv_freq(rope_dim, 10000.0)
    exact, _ = A.exact_and_surrogate_scores(q, k, freqs, rope_dim)
    reference = np.array(
        [
            [reference_score(q[m], k[n], freqs, rope_dim, n - m) for n in range(5)]
            for m in range(5)
        ]
    )
    assert np.abs(exact - reference).max() < 1e-10


# --------------------------------------------------------------------------
# the surrogate is the optimum, not a strawman
# --------------------------------------------------------------------------


def test_the_surrogate_is_the_least_squares_constant_per_query() -> None:
    """Compare against an explicit average, not against the formula that produced it.

    For a fixed query ``m`` the admissible distances are ``0, -1, ..., -m``, so
    the best position-free constant applied at ``(m, n)`` is the *mean of the
    exact score over that window*. ``reference_score`` computes the score at
    every distance by rotating the vectors explicitly, so this does not reuse
    the aligned/crossed algebra whose mean it is checking.
    """
    q, k = random_head(5, 6, head_dim=8)
    rope_dim = 8
    freqs = inv_freq(rope_dim, 10000.0)
    _, surrogate = A.exact_and_surrogate_scores(q, k, freqs, rope_dim)

    for m in range(1, q.shape[0]):
        window = -np.arange(0, m + 1, dtype=np.int64)
        for n in range(m + 1):
            reference = float(
                np.mean(
                    [
                        reference_score(q[m], k[n], freqs, rope_dim, int(d))
                        for d in window
                    ]
                )
            )
            assert surrogate[m, n] == pytest.approx(reference, abs=1e-10)


def test_a_symmetric_distance_window_would_cancel_the_surrogate() -> None:
    """The regression that motivated ``_per_query_means``.

    Sine is odd, so averaging ``sin`` over a window centred on zero gives zero
    and drops the position-carrying part of the surrogate entirely. The means
    must be taken over the causal distances the query can actually see.
    """
    freqs = inv_freq(8, 10000.0)
    n_tokens = 12
    cos_mean, sin_mean = A._per_query_means(n_tokens, freqs)

    # Row ``m`` averages delta = 0, -1, ..., -m.
    for m in (1, 5, n_tokens - 1):
        deltas = -np.arange(0, m + 1, dtype=np.float64)
        angles = np.outer(deltas, freqs)
        assert np.allclose(cos_mean[m], np.cos(angles).mean(axis=0), atol=1e-12)
        assert np.allclose(sin_mean[m], np.sin(angles).mean(axis=0), atol=1e-12)
        # And a symmetric window really is degenerate: the test would be vacuous
        # if sin did not cancel there.
        symmetric = np.arange(-m, m + 1, dtype=np.float64)
        assert np.abs(np.sin(np.outer(symmetric, freqs)).mean(axis=0)).max() < 1e-12


# --------------------------------------------------------------------------
# attention metrics
# --------------------------------------------------------------------------


def test_identical_scores_produce_no_disagreement() -> None:
    scores = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.4]])
    metrics = A.attention_metrics(scores, scores.copy(), scale=1.0)
    assert metrics["top1_flip_rate"] == 0.0
    assert metrics["total_variation_mean"] == 0.0
    assert metrics["kl_exact_surrogate_mean"] == 0.0
    assert metrics["top5_overlap_mean"] == 1.0
    assert metrics["n_queries"] == 3
    assert metrics["n_queries_scored"] == 2


def test_a_swapped_argmax_is_counted_and_position_zero_is_excluded() -> None:
    """Position 0 sees one key, so it cannot disagree and must not vote."""
    n = 4
    exact = np.zeros((n, n))
    surrogate = np.zeros((n, n))
    # Only row 1 is decided by the data: exact prefers key 0, surrogate key 1.
    # The margin is wide enough that the softmax is effectively one-hot.
    exact[1] = [10.0, -10.0, 0.0, 0.0]
    surrogate[1] = [-10.0, 10.0, 0.0, 0.0]
    metrics = A.attention_metrics(exact, surrogate, scale=1.0)
    # Rows 2 and 3 are ties (all-zero logits -> uniform), so they agree.
    assert metrics["n_queries_scored"] == 3
    assert metrics["top1_flip_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["total_variation_mean"] > 0.3
    # One flipped row contributes KL ~ 20 nats; the mean is over three queries.
    assert metrics["kl_exact_surrogate_mean"] > 5.0


def test_the_softmax_respects_the_causal_mask() -> None:
    scores = np.array([[0.0, 5.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
    weights = A._softmax_rows(scores, A.causal_mask(3), scale=1.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights[0, 1:] == 0.0)
    assert np.all(weights[:2, 2] == 0.0)


# --------------------------------------------------------------------------
# the committed artifact
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.fail(
            "results/attention_cost.json is absent; it is committed evidence, "
            "not an optional input"
        )
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_prompts_are_long_enough_for_distance_to_matter() -> None:
    """The first version of this experiment used the shared short prompts.

    ``real_model.DEFAULT_PROMPTS[:3]`` are 24-67 characters, i.e. 5-22 tokens.
    At that length the surrogate has almost nothing to be wrong about and every
    number comes out small for the wrong reason. Length is pinned here so the
    experiment cannot silently regress to them.
    """
    assert len(A.DEFAULT_PROMPTS) == 3
    assert min(len(p) for p in A.DEFAULT_PROMPTS) > 300


def test_the_artifact_covers_every_head_of_every_prompt(artifact: dict) -> None:
    assert artifact["schema"] == "attention-cost/v1"
    assert len(artifact["models"]) == 3
    for model in artifact["models"]:
        prompts = model["per_prompt"]
        assert len(prompts) == 3
        for prompt in prompts:
            assert prompt["n_tokens"] >= 60
            layers = {head["layer"] for head in prompt["heads"]}
            heads_per_layer = len(prompt["heads"]) // len(layers)
            assert len(layers) * heads_per_layer == len(prompt["heads"])
        by_layer = model["by_layer"]
        assert len(by_layer) == len(layers)
        assert [row["layer"] for row in by_layer] == sorted(layers)


def test_the_aggregate_is_re_derived_from_the_per_head_rows(artifact: dict) -> None:
    """The reported mean must be the mean of the rows beside it."""
    for model in artifact["models"]:
        heads = [head for prompt in model["per_prompt"] for head in prompt["heads"]]
        assert model["aggregate"]["n_heads"] == len(heads)
        for key in (
            "top1_flip_rate",
            "total_variation_mean",
            "kl_exact_surrogate_mean",
            "top5_overlap_mean",
        ):
            expected = float(np.mean([head[key] for head in heads]))
            assert model["aggregate"][key] == pytest.approx(expected, abs=1e-12), key


def test_the_measured_effect_is_large_and_the_reconstruction_is_exact(artifact: dict) -> None:
    """Two claims share this test, and they must not be confused.

    ``max_attention_residual`` is the distance between the module's *exact*
    softmax and the model's own attention. It must be at float32 scale: a large
    value would mean the reconstruction is wrong, not that the surrogate is bad.
    The surrogate's effect on top-1 attention is a separate, much larger number.
    """
    for model in artifact["models"]:
        for prompt in model["per_prompt"]:
            # float32 checkpoints accumulate ~1e-2 at worst in a sharp softmax.
            assert prompt["max_attention_residual"] < 0.02, (
                f"{model['model_id']}: the exact reconstruction is off by "
                f"{prompt['max_attention_residual']:.4g}; the surrogate's numbers "
                "below are not interpretable"
            )
            assert prompt["max_top1_disagreement_with_model"] <= 0.02
        aggregate = model["aggregate"]
        assert aggregate["top1_flip_rate"] > 0.15, aggregate["top1_flip_rate"]
        assert aggregate["total_variation_mean"] > 0.15
        assert aggregate["top5_overlap_mean"] < 0.8


def test_no_single_prompt_drives_the_effect(artifact: dict) -> None:
    """The effect must be visible in every prompt, not averaged in from one.

    Per-prompt flip rates vary with content --- a prompt with many near-ties
    flips more easily --- so the guard is a floor on each prompt, not a
    monotonicity claim about length, which the data does not support.
    """
    for model in artifact["models"]:
        for prompt in model["per_prompt"]:
            flip = float(np.mean([h["top1_flip_rate"] for h in prompt["heads"]]))
            assert flip > 0.15, (model["model_id"], prompt["prompt_index"], flip)
