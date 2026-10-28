"""Tests for ``qk_trajectory``.

The module makes one exact claim -- a distance trajectory of QK attribution
matrices has at most ``1 + rotary_dim`` degrees of freedom -- and three
descriptive ones -- stable rank, top-k mass, and the cost of flattening the
trajectory to its mean.  The tests pin the exact claim with a constructed
trajectory and pin the summaries with matrices whose answers are known.
"""

from __future__ import annotations

import numpy as np
import pytest
from rope_attribution import qk_attribution as QK
from rope_attribution import qk_trajectory as QT


def _buckets(rng: np.random.Generator, n_features: int, head_dim: int) -> QK.TokenBuckets:
    return QK.TokenBuckets(
        indices=np.arange(n_features, dtype=np.int64),
        activations=rng.random(n_features),
        feature_q=rng.standard_normal((n_features, head_dim)),
        feature_k=rng.standard_normal((n_features, head_dim)),
        base_q=rng.standard_normal(head_dim),
        base_k=rng.standard_normal(head_dim),
    )


def test_stable_rank_counts_equal_directions() -> None:
    assert QT.stable_rank(np.eye(5)) == pytest.approx(5.0)
    assert QT.stable_rank(np.ones((4, 4))) == pytest.approx(1.0)
    assert QT.stable_rank(np.zeros((4, 4))) == 0.0


def test_participation_ratio_counts_equal_directions() -> None:
    assert QT.participation_ratio(np.eye(5)) == pytest.approx(5.0)
    assert QT.participation_ratio(np.ones((4, 4))) == pytest.approx(1.0)
    # Two equal singular values and nothing else: two effective directions.
    two = np.diag([2.0, 2.0, 0.0])
    assert QT.participation_ratio(two) == pytest.approx(2.0)


def test_top_k_mass_is_a_fraction_of_absolute_mass() -> None:
    matrix = np.array([[3.0, -1.0], [0.0, 0.0]])
    assert QT.top_k_mass(matrix, 1) == pytest.approx(3.0 / 4.0)
    assert QT.top_k_mass(matrix, 2) == pytest.approx(1.0)
    assert QT.top_k_mass(np.zeros((3, 3)), 2) == 0.0


def test_flattening_a_constant_trajectory_is_free() -> None:
    stack = np.repeat(np.eye(3)[None, :, :], 7, axis=0)
    assert QT.flatten_relative_error(stack) == pytest.approx(0.0)
    # A trajectory that alternates sign is entirely lost by the mean.
    alternating = np.stack([np.eye(2), -np.eye(2)] * 3)
    assert QT.flatten_relative_error(alternating) == pytest.approx(1.0)


def test_the_rank_bound_is_exact_and_not_vacuous() -> None:
    """A trajectory built from ``P`` frequency pairs has rank at most ``1+2P``.

    Mutation verdict: dropping the sine term lowers the constructed rank below
    the bound but does not fail this test; replacing the bound with ``rotary_dim``
    (instead of ``1 + rotary_dim``) fails the equality; using random matrices
    instead of the trigonometric family makes the numerical rank exceed the
    bound, so the second half is what proves the bound is not vacuous.
    """
    rng = np.random.default_rng(0)
    n_features, rotary_dim, n_deltas = 3, 4, 16
    freqs = np.linspace(0.1, 0.7, rotary_dim // 2)
    deltas = np.arange(1, n_deltas + 1, dtype=np.float64)
    constant = rng.standard_normal((n_features, n_features))
    cosine = rng.standard_normal((freqs.size, n_features, n_features))
    sine = rng.standard_normal((freqs.size, n_features, n_features))
    trajectory = np.stack(
        [
            constant
            + sum(
                cosine[k] * np.cos(delta * freqs[k])
                + sine[k] * np.sin(delta * freqs[k])
                for k in range(freqs.size)
            )
            for delta in deltas
        ]
    )
    flat = trajectory.reshape(n_deltas, -1)
    bound = QT.trajectory_rank_bound(rotary_dim)
    assert bound == 1 + 2 * freqs.size == 1 + rotary_dim
    assert np.linalg.matrix_rank(flat) <= bound

    random_trajectory = rng.standard_normal((n_deltas, n_features, n_features))
    assert np.linalg.matrix_rank(random_trajectory.reshape(n_deltas, -1)) > bound


def test_the_rank_bound_rejects_an_odd_rotary_width() -> None:
    with pytest.raises(ValueError, match="even"):
        QT.trajectory_rank_bound(7)


def test_analyse_pair_reports_the_bound_and_one_row_per_distance() -> None:
    rng = np.random.default_rng(1)
    head_dim, rotary_dim = 8, 4
    buckets_m = _buckets(rng, 4, head_dim)
    buckets_n = _buckets(rng, 4, head_dim)
    deltas = np.array([1, 2, 4, 8, 16, 32])
    out = QT.analyse_pair(
        buckets_m,
        buckets_n,
        deltas,
        freqs=np.linspace(0.1, 0.7, rotary_dim // 2),
        rotary_dim=rotary_dim,
    )
    assert out["rank_bound"] == 1 + rotary_dim
    assert len(out["per_delta"]) == deltas.size
    assert 0.0 <= out["flatten_relative_error"] <= 1.0
    assert len(out["trajectory_singular_values"]) <= min(deltas.size, 4 * 4)
    for row in out["per_delta"]:
        assert 0.0 <= row["top_k_mass"]["4"] <= 1.0
        assert 0.0 <= row["negative_fraction"] <= 1.0
    for k in ("1", "2", "4", "8"):
        assert 0.0 <= out["trajectory_energy"][k] <= 1.0


def test_trajectory_matrices_are_the_decomposition_it_names() -> None:
    """The stack's first matrix is exactly the one-distance pair block."""
    rng = np.random.default_rng(2)
    head_dim, rotary_dim = 8, 4
    buckets_m = _buckets(rng, 3, head_dim)
    buckets_n = _buckets(rng, 3, head_dim)
    freqs = np.linspace(0.1, 0.7, rotary_dim // 2)
    deltas = [1, 5, 9]
    stack = QT.trajectory_matrices(
        buckets_m, buckets_n, deltas, freqs, rotary_dim
    )
    # ``deltas`` are distances and the key sits behind the query, so the
    # model's angle is the negative of the distance.
    direct = QK.attribution_terms(buckets_m, buckets_n, -5, freqs, rotary_dim).pairs
    assert stack.shape == (3, 3, 3)
    np.testing.assert_allclose(stack[1], direct)
    wrong = QK.attribution_terms(buckets_m, buckets_n, +5, freqs, rotary_dim).pairs
    assert np.abs(stack[1] - wrong).max() > 1e-3


def _fake_dataset(layer: int, value: float, *, n_tokens: int = 32) -> dict:
    mean = {
        "rank_bound": {"mean": 17.0, "sd": 0.0, "min": 17.0, "max": 17.0},
        "trajectory_participation_ratio": {
            "mean": value,
            "sd": 0.0,
            "min": value,
            "max": value,
        },
        "flatten_relative_error": {
            "mean": value / 10,
            "sd": 0.0,
            "min": value / 10,
            "max": value / 10,
        },
        "trajectory_stable_rank": {
            "mean": value,
            "sd": 0.0,
            "min": value,
            "max": value,
        },
        "trajectory_rank_numerical": {
            "mean": 3.0,
            "sd": 0.0,
            "min": 3.0,
            "max": 3.0,
        },
        "trajectory_energy_top1": {
            "mean": 0.8,
            "sd": 0.0,
            "min": 0.8,
            "max": 0.8,
        },
        "trajectory_energy_top2": {
            "mean": 0.9,
            "sd": 0.0,
            "min": 0.9,
            "max": 0.9,
        },
        "trajectory_energy_top4": {
            "mean": 0.95,
            "sd": 0.0,
            "min": 0.95,
            "max": 0.95,
        },
        "trajectory_energy_top8": {
            "mean": 0.99,
            "sd": 0.0,
            "min": 0.99,
            "max": 0.99,
        },
        "per_delta": [
            {
                "index": 0,
                "stable_rank": value,
                "participation_ratio": value,
                "negative_fraction": 0.0,
                "frobenius_norm": 1.0,
                "top_1_mass": 0.5,
                "top_2_mass": 0.8,
                "top_4_mass": 1.0,
                "top_8_mass": 1.0,
                "top_16_mass": 1.0,
            }
        ],
    }
    return {
        "schema": QT.SCHEMA,
        "model_id": "tiny",
        "text": "a passage",
        "seed": 0,
        "pairs_per_head": 2,
        "sae_layer": layer,
        "attention_layer": layer + 1,
        "n_tokens": n_tokens,
        "top_k": 32,
        "deltas": [1, 2, 4],
        "rotary_dim": 16,
        "head_dim": 64,
        "rank_bound": 17,
        "heads": [
            {
                "head": 0,
                "layer": layer + 1,
                "n_content_pairs": 2,
                "mean": mean,
            }
        ],
    }


def test_combining_layers_pools_heads_and_keeps_the_layer_curve() -> None:
    data = QT.combine_layers([_fake_dataset(0, 1.0), _fake_dataset(1, 3.0)])
    assert data["n_layers"] == 2
    assert data["n_heads"] == 2
    assert data["pairs_per_head"] == 2
    assert data["text"] == "a passage"
    assert data["layers"][0]["heads"][0]["n_content_pairs"] == 2
    assert data["pooled"]["trajectory_participation_ratio"]["mean"] == pytest.approx(2.0)
    assert data["layers"][0]["trajectory_participation_ratio"]["min"] == pytest.approx(1.0)
    assert data["layers"][1]["per_delta"][0]["top_1_mass"] == pytest.approx(0.5)


def test_combining_layers_refuses_a_mismatched_passage() -> None:
    with pytest.raises(ValueError, match="different n_tokens"):
        QT.combine_layers(
            [_fake_dataset(0, 1.0), _fake_dataset(1, 1.0, n_tokens=64)]
        )


def test_combining_layers_refuses_different_content_pair_counts() -> None:
    other = _fake_dataset(1, 1.0)
    other["pairs_per_head"] = 4
    with pytest.raises(ValueError, match="different pairs_per_head"):
        QT.combine_layers([_fake_dataset(0, 1.0), other])


def test_measure_layers_requires_a_template_placeholder() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        QT.measure_layers("tiny", "one.safetensors", [0, 1])


def test_the_summary_is_the_mean_of_its_pairs() -> None:
    summaries = [
        {
            "flatten_relative_error": value,
            "trajectory_stable_rank": value,
            "trajectory_participation_ratio": value,
            "trajectory_rank_numerical": value,
            "rank_bound": 5.0,
            "trajectory_energy": {"1": 0.8, "2": 0.9, "4": 0.95, "8": 0.99},
            "per_delta": [
                {
                    "stable_rank": value,
                    "participation_ratio": value,
                    "negative_fraction": 0.0,
                    "frobenius_norm": 1.0,
                    "top_k_mass": {"1": 0.5, "4": 1.0},
                }
            ],
        }
        for value in (1.0, 3.0)
    ]
    out = QT._mean_summary(summaries, ks=(1, 4))
    assert out["flatten_relative_error"]["mean"] == pytest.approx(2.0)
    assert out["flatten_relative_error"]["sd"] == pytest.approx(np.sqrt(2.0))
    assert out["per_delta"][0]["top_4_mass"] == pytest.approx(1.0)
