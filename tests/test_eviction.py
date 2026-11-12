"""Tests for the attribution-guided KV-eviction ranking core.

The harness is pure NumPy, so these tests run on every CPU install.  They pin
the four places a cache-eviction implementation goes silently wrong: the sign
of the rotary angle, the grouped-query reduction, the forced prefix/current
keys, and the mask's effect on the softmax.  The docstring of
``rope_attribution.eviction`` records the mutation verdict for each.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from rope_attribution import eviction as E
from rope_attribution.rope import (
    apply_partial_rope,
    apply_rope,
    inv_freq,
    partial_rope_cos_sin,
    rope_cos_sin,
)


def _qk(rng, n_query=2, n_kv=1, n_keys=6, dim=8):
    q = rng.standard_normal((n_query, dim))
    k = rng.standard_normal((n_kv, n_keys, dim))
    v = rng.standard_normal((n_kv, n_keys, dim))
    return q, k, v


def _rotated_dot(q, k, query_position, key_position, freqs, rotary_dim, full_dim):
    """Independent reference: apply the published rotation, then dot."""
    if rotary_dim == full_dim:
        cos_q, sin_q = rope_cos_sin(np.array([query_position]), freqs)
        cos_k, sin_k = rope_cos_sin(np.array([key_position]), freqs)
        q_rot = apply_rope(q[None, None, :], cos_q, sin_q)[0, 0]
        k_rot = apply_rope(k[None, None, :], cos_k, sin_k)[0, 0]
    else:
        cos_q, sin_q = partial_rope_cos_sin(
            np.array([query_position]), inv_freq(full_dim), rotary_dim
        )
        cos_k, sin_k = partial_rope_cos_sin(
            np.array([key_position]), inv_freq(full_dim), rotary_dim
        )
        q_rot = apply_partial_rope(q[None, None, :], cos_q, sin_q, rotary_dim)[0, 0]
        k_rot = apply_partial_rope(k[None, None, :], cos_k, sin_k, rotary_dim)[0, 0]
    return float(q_rot @ k_rot)


def test_exact_scores_match_rope_dot() -> None:
    rng = np.random.default_rng(0)
    dim, n_keys, t = 8, 6, 5
    q, k, _v = _qk(rng, n_query=2, n_kv=1, n_keys=n_keys, dim=dim)
    positions = np.arange(n_keys)
    freqs = inv_freq(dim)
    scores = E.exact_scores(q, k, t, positions, freqs, dim)
    assert scores.shape == (2, n_keys)
    for head in range(2):
        for j in range(n_keys):
            expected = _rotated_dot(
                q[head], k[0, j], t, positions[j], freqs, dim, dim
            )
            assert scores[head, j] == pytest.approx(expected, abs=1e-12)


def test_exact_scores_match_rope_dot_under_partial_rotation() -> None:
    rng = np.random.default_rng(1)
    dim, rotary_dim, n_keys, t = 8, 4, 6, 5
    q, k, _v = _qk(rng, n_query=2, n_kv=1, n_keys=n_keys, dim=dim)
    positions = np.arange(n_keys)
    freqs = inv_freq(rotary_dim)
    scores = E.exact_scores(q, k, t, positions, freqs, rotary_dim)
    for head in range(2):
        for j in range(n_keys):
            expected = _rotated_dot(
                q[head], k[0, j], t, positions[j], freqs, rotary_dim, dim
            )
            assert scores[head, j] == pytest.approx(expected, abs=1e-12)


def test_partial_rope_keeps_the_tail() -> None:
    """Dropping the unrotated channels changes the score by their dot product."""
    rng = np.random.default_rng(2)
    dim, rotary_dim, n_keys = 8, 4, 5
    q, k, _v = _qk(rng, n_query=1, n_kv=1, n_keys=n_keys, dim=dim)
    positions = np.arange(n_keys)
    scores = E.exact_scores(q, k, 4, positions, inv_freq(rotary_dim), rotary_dim)
    tail = q[0, rotary_dim:] @ k[0, :, rotary_dim:].T
    rotary_only = E.exact_scores(
        q[:, :rotary_dim], k[:, :, :rotary_dim], 4, positions,
        inv_freq(rotary_dim), rotary_dim,
    )
    np.testing.assert_allclose(scores[0] - rotary_only[0], tail, atol=1e-12)


def test_mean_scores_equal_exact_on_a_single_distance_window() -> None:
    rng = np.random.default_rng(3)
    dim, n_keys = 8, 1
    q, k, _v = _qk(rng, n_query=2, n_kv=1, n_keys=n_keys, dim=dim)
    positions = np.array([4])
    freqs = inv_freq(dim)
    exact = E.exact_scores(q, k, 4, positions, freqs, dim)
    mean = E.mean_scores(q, k, 4, positions, freqs, dim)
    np.testing.assert_allclose(mean, exact, atol=1e-12)


def test_mean_scores_use_the_observed_window_not_the_whole_sequence() -> None:
    """A short window must not be summarised by distances it cannot see."""
    rng = np.random.default_rng(4)
    dim, n_keys = 8, 6
    q, k, _v = _qk(rng, n_query=1, n_kv=1, n_keys=n_keys, dim=dim)
    positions = np.arange(n_keys)
    freqs = inv_freq(dim)
    short = E.mean_scores(q, k, 2, positions[:3], freqs, dim)
    long = E.mean_scores(q, k, 2, positions, freqs, dim)
    assert not np.allclose(short, long)


def test_identity_scores_are_the_unrotated_dot() -> None:
    rng = np.random.default_rng(5)
    dim, n_keys = 8, 6
    q, k, _v = _qk(rng, n_query=2, n_kv=1, n_keys=n_keys, dim=dim)
    scores = E.identity_scores(q, k, dim)
    np.testing.assert_allclose(scores, q @ k[0].T, atol=1e-12)


def test_group_scores_sum_abs_over_query_heads() -> None:
    per_query = np.array([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0], [7.0, -8.0]])
    grouped = E.group_scores(per_query, n_kv_heads=2)
    np.testing.assert_array_equal(grouped, np.array([[4.0, 6.0], [12.0, 14.0]]))


def test_group_scores_refuse_an_indivisible_head_count() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        E.group_scores(np.zeros((3, 4)), n_kv_heads=2)


def test_select_keep_always_keeps_prefix_and_current() -> None:
    scores = np.arange(12, dtype=float)
    keep = E.select_keep(scores, retention=0.25, prefix_tokens=4, forced=(11,))
    assert set(range(4)).issubset(keep)
    assert 11 in keep
    assert len(keep) >= 5


def test_select_keep_retains_the_top_fraction() -> None:
    scores = np.array([0.0, 10.0, 1.0, 9.0, 2.0, 8.0])
    keep = E.select_keep(scores, retention=0.5, prefix_tokens=0)
    assert set(keep) == {1, 3, 5}


def test_select_keep_is_deterministic_on_ties() -> None:
    scores = np.ones(6)
    assert np.array_equal(
        E.select_keep(scores, retention=0.5, prefix_tokens=0),
        np.array([0, 1, 2]),
    )


def test_masked_attention_identity_when_all_kept() -> None:
    rng = np.random.default_rng(6)
    q, k, v = _qk(rng, n_query=4, n_kv=2, n_keys=6, dim=8)
    keep = np.ones((2, 6), dtype=bool)
    out, weights = E.masked_attention(q, k, v, keep)
    k_exp = np.repeat(k, 2, axis=0)
    v_exp = np.repeat(v, 2, axis=0)
    logits = (q[:, None, :] * k_exp).sum(-1) / np.sqrt(8.0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    expected = np.exp(shifted)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(weights, expected, atol=1e-12)
    np.testing.assert_allclose(out, expected @ v_exp, atol=1e-12)


def test_masked_attention_puts_zero_weight_on_evicted_keys() -> None:
    rng = np.random.default_rng(7)
    q, k, v = _qk(rng, n_query=1, n_kv=1, n_keys=5, dim=8)
    keep = np.array([True, False, False, False, True])
    _out, weights = E.masked_attention(q, k, v, keep)
    assert weights[0, 1] == 0.0 and weights[0, 2] == 0.0 and weights[0, 3] == 0.0
    assert weights.sum() == pytest.approx(1.0)


def test_masked_attention_refuses_an_empty_keep_set() -> None:
    rng = np.random.default_rng(8)
    q, k, v = _qk(rng, n_query=1, n_kv=1, n_keys=3, dim=8)
    with pytest.raises(ValueError, match="at least one key"):
        E.masked_attention(q, k, v, np.zeros(3, dtype=bool))


def test_recency_scores_prefer_the_nearest_keys() -> None:
    scores = E.recency_scores(np.arange(5), 4, n_query_heads=2)
    np.testing.assert_array_equal(scores[0], np.array([-4.0, -3.0, -2.0, -1.0, 0.0]))


def test_random_scores_are_seeded_and_reproducible() -> None:
    first = E.random_scores(2, 6, seed=11)
    second = E.random_scores(2, 6, seed=11)
    other = E.random_scores(2, 6, seed=12)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_an_unknown_arm_is_refused() -> None:
    rng = np.random.default_rng(9)
    q, k, _v = _qk(rng, n_query=1, n_kv=1, n_keys=3, dim=8)
    with pytest.raises(ValueError, match="unknown eviction arm"):
        E.arm_scores("magic", q, k, 2, np.arange(3))


def test_evict_attention_keeps_the_current_key_and_masks_the_rest() -> None:
    rng = np.random.default_rng(10)
    q, k, v = _qk(rng, n_query=2, n_kv=1, n_keys=12, dim=8)
    positions = np.arange(12)
    result = E.evict_attention(
        q, k, v, 11, positions,
        arm="recency", retention=0.25, prefix_tokens=4,
    )
    keep = result["keep"][0]
    assert keep[11] and keep[:4].all()
    assert result["weights"].shape == (2, 12)
    assert np.all(result["weights"][:, ~keep] == 0.0)
    # The current key is retained even though the retention budget is small.
    assert result["n_kept"] >= 5


def test_evict_attention_exact_needs_rotary_parameters() -> None:
    rng = np.random.default_rng(11)
    q, k, v = _qk(rng, n_query=1, n_kv=1, n_keys=4, dim=8)
    with pytest.raises(ValueError, match="needs freqs and rotary_dim"):
        E.evict_attention(
            q, k, v, 3, np.arange(4), arm="exact", retention=0.5
        )


def test_dry_run_artifact_has_the_protocol_schema_and_no_numbers() -> None:
    artifact = E.dry_run_artifact(seed=7)
    assert artifact["schema"] == E.SCHEMA
    assert artifact["status"] == "dry-run"
    assert artifact["seed"] == 7
    assert artifact["models"] == [] and artifact["cells"] == [] and artifact["ceiling"] == []
    assert "no task number" in artifact["note"].lower()


def test_the_cli_dry_run_writes_valid_json(tmp_path) -> None:
    target = tmp_path / "eviction.json"
    E.main(["--dry-run", "--out", str(target)])
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["schema"] == E.SCHEMA
    assert data["status"] == "dry-run"


def test_the_cli_refuses_a_real_run_without_a_model_adapter() -> None:
    with pytest.raises(SystemExit, match="not the 1B--8B model adapter"):
        E.main([])
