"""Tests for the exact per-distance QK attribution.

The module makes one very strong claim --- that the decomposition is an identity,
not an approximation --- so most of these tests are exactness tests. The rest
pin the conventions that a plausible-looking implementation gets silently wrong:

* the fused GPT-NeoX projection stores ``(q, k, v)`` *per head*, so slicing the
  flat rows into three contiguous thirds corrupts every head but the first;
* the sparse dictionary subtracts ``b_dec`` before the encoder, not after;
* a partial-RoPE head has an unrotated tail whose contribution cannot depend on
  distance, which is what makes the rotary-only profile a different object from
  the total rather than a rescaling of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rope_attribution import qk_attribution as Q
from rope_attribution import rope as R

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "results" / "qk_attribution.json"

TOL = 1e-12


def random_setup(seed: int = 0, d_model: int = 16, head_dim: int = 8, rotary_dim: int = 8):
    rng = np.random.default_rng(seed)
    dictionary = Q.SparseDictionary.random(d_model, 24, seed=seed + 1)
    ln_weight = rng.uniform(0.5, 1.5, d_model)
    ln_bias = rng.standard_normal(d_model) * 0.1
    w_q = rng.standard_normal((head_dim, d_model)) / np.sqrt(d_model)
    w_k = rng.standard_normal((head_dim, d_model)) / np.sqrt(d_model)
    b_q = rng.standard_normal(head_dim) * 0.1
    b_k = rng.standard_normal(head_dim) * 0.1
    freqs = R.inv_freq(rotary_dim, 10000.0)
    return dictionary, ln_weight, ln_bias, w_q, b_q, w_k, b_k, freqs


# --------------------------------------------------------------------------
# LayerNorm: the Jacobian is derived by hand, so check it independently
# --------------------------------------------------------------------------


def test_the_layernorm_jvp_matches_a_central_difference() -> None:
    """A differentiator-independent check of the hand-derived Jacobian."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(32)
    v = rng.standard_normal(32)
    weight = rng.uniform(0.5, 1.5, 32)
    bias = rng.standard_normal(32) * 0.1
    eps = 1e-5

    analytic = Q.layernorm_jvp(x, v, weight, bias, eps)
    step = 1e-6
    numeric = (
        Q.layernorm(x + step * v, weight, bias, eps)
        - Q.layernorm(x - step * v, weight, bias, eps)
    ) / (2 * step)
    assert np.abs(analytic - numeric).max() < 1e-7


def test_the_affine_form_is_an_identity_at_x() -> None:
    """``h = J(x) x + c`` must hold exactly, not to first order."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((5, 32))
    weight = rng.uniform(0.5, 1.5, 32)
    bias = rng.standard_normal(32) * 0.1
    h, c = Q.layernorm_affine(x, weight, bias, 1e-5)
    assert np.abs(Q.layernorm_jvp(x, x, weight, bias, 1e-5) + c - h).max() < TOL


def test_the_jacobian_ignores_the_layernorm_bias() -> None:
    """The bias shifts the output, not the tangent; a test that would catch it
    being folded into the Jacobian by mistake."""
    rng = np.random.default_rng(13)
    x = rng.standard_normal(16)
    v = rng.standard_normal(16)
    weight = np.ones(16)
    a = Q.layernorm_jvp(x, v, weight, np.zeros(16), 1e-5)
    b = Q.layernorm_jvp(x, v, weight, np.full(16, 7.0), 1e-5)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# The dictionary convention
# --------------------------------------------------------------------------


def test_the_dictionary_subtracts_its_decoder_bias_before_encoding() -> None:
    """``apply_b_dec_to_input=True`` is a convention, not a detail.

    It means the encoder sees ``x - b_dec``, so an input equal to ``b_dec`` has
    zero centred residual: every unit is exactly ``relu(b_enc)``. With the
    convention reversed, that input would instead light up every unit whose
    bias is negative, and the decomposition would describe a different vector.
    """
    dictionary = Q.SparseDictionary.random(16, 32, seed=3)
    assert np.allclose(dictionary.encode(dictionary.b_dec), np.maximum(dictionary.b_enc, 0.0))
    assert np.allclose(dictionary.decode(np.zeros(32)), dictionary.b_dec)
    x = np.random.default_rng(4).standard_normal((4, 16))
    centred = np.maximum((x - dictionary.b_dec) @ dictionary.W_enc + dictionary.b_enc, 0.0)
    uncentred = np.maximum(x @ dictionary.W_enc + dictionary.b_enc, 0.0)
    # The two conventions must actually disagree on this input, or the test
    # would pass whichever one the module implemented.
    assert not np.allclose(centred, uncentred)
    assert np.allclose(dictionary.encode(x), centred)


def test_a_dictionary_with_inconsistent_shapes_is_rejected() -> None:
    with pytest.raises(ValueError, match="dictionary width"):
        Q.SparseDictionary(
            W_enc=np.zeros((16, 8)),
            b_enc=np.zeros(8),
            W_dec=np.zeros((9, 16)),
            b_dec=np.zeros(16),
        )


def test_the_safetensors_loader_records_the_tensors_it_read(tmp_path: Path) -> None:
    from safetensors.numpy import save_file

    dictionary = Q.SparseDictionary.random(8, 5, seed=2)
    path = tmp_path / "sae.safetensors"
    save_file(
        {
            "W_enc": dictionary.W_enc,
            "b_enc": dictionary.b_enc,
            "W_dec": dictionary.W_dec,
            "b_dec": dictionary.b_dec,
        },
        str(path),
    )
    loaded = Q.SparseDictionary.load_safetensors(path)
    assert loaded.n_features == 5
    assert loaded.d_model == 8
    assert np.allclose(loaded.W_dec, dictionary.W_dec)


def test_a_safetensors_file_missing_a_tensor_is_rejected(tmp_path: Path) -> None:
    from safetensors.numpy import save_file

    path = tmp_path / "broken.safetensors"
    save_file({"W_enc": np.zeros((8, 5), np.float32)}, str(path))
    with pytest.raises(ValueError, match="missing tensors"):
        Q.SparseDictionary.load_safetensors(path)


# --------------------------------------------------------------------------
# Per-head projection layout
# --------------------------------------------------------------------------


class _Fused:
    """A stand-in for a GPT-NeoX attention whose QKV rows are identifiable."""

    def __init__(self, hidden: int = 8, n_heads: int = 2) -> None:
        import torch

        self.query_key_value = torch.nn.Linear(hidden, 3 * hidden, bias=True)
        with torch.no_grad():
            # Row ``r`` has weight ``r + 1`` on *every* input channel, so the
            # row index survives the projection whatever column is read.
            for row in range(3 * hidden):
                self.query_key_value.weight[row, :] = float(row + 1)
            self.query_key_value.bias.fill_(0.0)
            for row in range(3 * hidden):
                self.query_key_value.bias[row] = float(row)


def test_the_fused_layout_is_read_per_head_not_in_thirds() -> None:
    """GPT-NeoX interleaves ``(q, k, v)`` per head.

    With ``hidden = 8`` and two heads of width 4, head 1's query rows are
    ``12..15`` and its key rows ``16..19``. A "three contiguous thirds" slice
    would give head 1 rows ``8..11`` as its query, which is head 0's value
    block, and every attention score would be computed from the wrong vectors
    while keeping the right shape.
    """
    attn = _Fused(hidden=8, n_heads=2)
    w_q, b_q, w_k, b_k = Q.head_projections(attn, hidden_size=8, n_heads=2, head_dim=4)
    assert w_q.shape == (2, 4, 8)
    assert w_k.shape == (2, 4, 8)
    # Head 0: q rows 0-3, k rows 4-7.
    assert np.allclose(w_q[0, :, 0], [1, 2, 3, 4])
    assert np.allclose(w_k[0, :, 0], [5, 6, 7, 8])
    # Head 1: q rows 12-15, k rows 16-19.
    assert np.allclose(w_q[1, :, 0], [13, 14, 15, 16])
    assert np.allclose(w_k[1, :, 0], [17, 18, 19, 20])
    assert np.allclose(b_q[1], [12, 13, 14, 15])
    assert np.allclose(b_k[1], [16, 17, 18, 19])


class _Split:
    def __init__(self, hidden: int = 8, n_heads: int = 2) -> None:
        import torch

        self.q_proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=True)
        with torch.no_grad():
            for module, offset in ((self.q_proj, 0.0), (self.k_proj, 100.0)):
                for row in range(hidden):
                    module.weight[row, :] = float(row + 1) + offset
                for row in range(hidden):
                    module.bias[row] = float(row) + offset


def test_split_projections_are_reshaped_head_major() -> None:
    attn = _Split(hidden=8, n_heads=2)
    w_q, b_q, w_k, b_k = Q.head_projections(attn, hidden_size=8, n_heads=2, head_dim=4)
    assert np.allclose(w_q[0, :, 0], [1, 2, 3, 4])
    assert np.allclose(w_q[1, :, 0], [5, 6, 7, 8])
    assert np.allclose(w_k[1, :, 0], [105, 106, 107, 108])
    assert np.allclose(b_q[1], [4, 5, 6, 7])


def test_a_fused_projection_of_the_wrong_width_is_rejected() -> None:
    attn = _Fused(hidden=8, n_heads=2)
    with pytest.raises(ValueError, match="fused QKV"):
        Q.head_projections(attn, hidden_size=9, n_heads=2, head_dim=4)


def test_an_unknown_projection_layout_is_reported_not_guessed() -> None:
    class Nothing:
        pass

    with pytest.raises(NotImplementedError, match="neither"):
        Q.head_projections(Nothing(), hidden_size=8, n_heads=2, head_dim=4)


# --------------------------------------------------------------------------
# Token buckets: the split must be exact
# --------------------------------------------------------------------------


def _buckets_and_full(seed: int = 0):
    dictionary, ln_w, ln_b, w_q, b_q, w_k, b_k, freqs = random_setup(seed)
    rng = np.random.default_rng(seed + 5)
    x = rng.standard_normal((6, dictionary.d_model))
    f = dictionary.encode(x)
    buckets = [
        Q.token_buckets(x[t], f[t], dictionary, ln_w, ln_b, 1e-5, w_q, b_q, w_k, b_k, 6)
        for t in range(x.shape[0])
    ]
    h = Q.layernorm(x, ln_w, ln_b, 1e-5)
    return buckets, h @ w_q.T + b_q, h @ w_k.T + b_k, freqs


def test_the_feature_buckets_add_up_to_the_projected_vector() -> None:
    buckets, q_full, k_full, _ = _buckets_and_full()
    for t, bucket in enumerate(buckets):
        q = bucket.base_q + bucket.feature_q.sum(axis=0)
        k = bucket.base_k + bucket.feature_k.sum(axis=0)
        assert np.abs(q - q_full[t]).max() < TOL
        assert np.abs(k - k_full[t]).max() < TOL


def test_dropping_a_feature_changes_the_reconstruction() -> None:
    """The exactness test above would also pass on a base bucket that absorbs
    everything, so pin that features carry real mass."""
    buckets, q_full, _, _ = _buckets_and_full()
    bucket = buckets[0]
    without = bucket.base_q + bucket.feature_q[1:].sum(axis=0)
    assert np.abs(without - q_full[0]).max() > 1e-8
    assert np.abs(bucket.feature_q).max() > 1e-8


# --------------------------------------------------------------------------
# Rotated dot products
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rotary_dim", [8, 4])
def test_rotated_dot_matches_rotating_the_key_explicitly(rotary_dim: int) -> None:
    rng = np.random.default_rng(21)
    a = rng.standard_normal((5, 8))
    b = rng.standard_normal((7, 8))
    freqs = R.inv_freq(rotary_dim, 10000.0)
    for delta in (0, 1, 5, 64):
        ours = Q.rotated_dot(a, b, delta, freqs, rotary_dim)
        cos, sin = R.rope_cos_sin(np.array([0, delta]), freqs)
        b_rot = R.apply_rope(b[:, :rotary_dim], cos[1:2], sin[1:2])
        expected = a[:, :rotary_dim] @ b_rot.T + a[:, rotary_dim:] @ b[:, rotary_dim:].T
        assert np.abs(ours - expected).max() < 1e-10


# --------------------------------------------------------------------------
# The attribution identity
# --------------------------------------------------------------------------


def test_the_score_decomposition_is_an_identity() -> None:
    buckets, _, _, freqs = _buckets_and_full(seed=4)
    rng = np.random.default_rng(31)
    for _ in range(50):
        m, n = rng.integers(0, len(buckets), 2)
        delta = int(m - n)
        terms = Q.attribution_terms(buckets[int(m)], buckets[int(n)], delta, freqs, 8)
        assert abs(terms.reconstructed - terms.score) <= TOL * max(abs(terms.score), 1.0)


def test_the_rotary_profile_differs_from_the_total_by_the_unrotated_dot_product() -> None:
    """``sum(total - rotary)`` must be exactly the tail channels' dot product.

    Zeroing the key's tail removes the same quantity from the base bucket, so
    the invariant is stated on the feature shares alone: their difference sums
    to ``q[rotary:] . k[rotary:]``, which no rotation touches.
    """
    dictionary, ln_w, ln_b, w_q, b_q, w_k, b_k, freqs = random_setup(seed=6, rotary_dim=4)
    rng = np.random.default_rng(61)
    x = rng.standard_normal((6, dictionary.d_model))
    f = dictionary.encode(x)
    buckets = [
        Q.token_buckets(x[t], f[t], dictionary, ln_w, ln_b, 1e-5, w_q, b_q, w_k, b_k, 6)
        for t in range(6)
    ]
    terms = Q.attribution_terms(buckets[5], buckets[2], 3, freqs, 4)
    q_features = buckets[5].feature_q.sum(axis=0)
    k = buckets[2].base_k + buckets[2].feature_k.sum(axis=0)
    # The base bucket is excluded on both sides: it carries its own tail term,
    # which the two feature sums do not contain.
    expected = float(q_features[4:] @ k[4:])
    gap = float((terms.query_features - terms.query_features_rotary).sum())
    assert abs(gap - expected) < TOL * max(abs(expected), 1.0)
    # And the gap is not trivially zero on this setup.
    assert abs(expected) > 1e-6


def test_the_unrotated_tail_is_exactly_distance_independent() -> None:
    """The property that makes the rotary-only view a different object.

    ``total - rotary`` is the contribution of the key's unrotated channels,
    which no rotation touches. It must be the same number at every distance; if
    the implementation silently rotates the tail too, this becomes a function of
    ``delta`` and the two profiles differ by more than a constant.
    """
    buckets, _, _, freqs = _buckets_and_full(seed=8)
    gaps = []
    for delta in (0, 1, 2, 7, 33):
        terms = Q.attribution_terms(buckets[5], buckets[2], delta, freqs, 8)
        gaps.append(terms.query_features - terms.query_features_rotary)
    gaps = np.stack(gaps)
    assert np.abs(gaps - gaps[0]).max() < TOL


# --------------------------------------------------------------------------
# Sampling and grids
# --------------------------------------------------------------------------


def test_the_distance_grid_drops_distances_the_passage_cannot_realize() -> None:
    grid = Q.distance_grid(10, requested=[1, 5, 10, 11, 40])
    assert grid.tolist() == [1, 5, 10]
    assert Q.distance_grid(3, requested=[9]).size == 0


def test_sampled_pairs_are_causal_and_at_exactly_the_requested_distance() -> None:
    rng = np.random.default_rng(41)
    pairs = Q.sample_pairs(20, delta=4, n_pairs=10, rng=rng)
    assert pairs.shape == (10, 2)
    assert np.all(pairs[:, 1] == pairs[:, 0] - 4)
    assert np.all(pairs >= 0)
    assert np.all(pairs[:, 0] < 20)
    # No repeats: a short passage must not report the same pair twice.
    assert len({tuple(row) for row in pairs}) == len(pairs)


def test_a_distance_longer_than_the_passage_yields_no_pairs() -> None:
    assert Q.sample_pairs(5, delta=9, n_pairs=3, rng=np.random.default_rng(0)).shape == (0, 2)


# --------------------------------------------------------------------------
# Grouping and profile metrics
# --------------------------------------------------------------------------


def test_group_mean_averages_duplicates_and_leaves_unsampled_cells_nan() -> None:
    delta_index = np.array([0, 0, 1, 1, 1])
    ids = np.array([7, 7, 7, 7, 9])
    values = np.array([1.0, 3.0, 5.0, 7.0, 100.0])
    unique, profiles = Q._group_mean(delta_index, ids, values, n_deltas=3)
    assert unique.tolist() == [7, 9]
    assert profiles[0, 0] == pytest.approx(2.0)
    assert profiles[0, 1] == pytest.approx(6.0)
    assert np.isnan(profiles[0, 2])
    assert profiles[1, 1] == pytest.approx(100.0)
    assert np.isnan(profiles[1, 0])


def test_a_constant_profile_is_reproduced_by_its_own_mean() -> None:
    profiles = np.tile(np.array([2.0, 2.0, 2.0, 2.0]), (4, 1))
    metrics = Q.profile_metrics(profiles)
    assert metrics["sign_flip_fraction"] == 0.0
    assert metrics["rel_err_median"] == pytest.approx(0.0, abs=1e-15)


def test_a_sign_change_is_counted_as_a_flip() -> None:
    profiles = np.array([[1.0, 1.0, -1.0, -1.0]])
    metrics = Q.profile_metrics(profiles)
    # The mean is zero, and a zero surrogate is explicitly not judged.
    assert metrics["n_material_cells"] == 0
    assert metrics["sign_flip_fraction"] is None


def test_a_small_oscillation_around_a_large_offset_does_not_vote() -> None:
    """The materiality convention, stated as a test rather than a comment.

    Cells below 5% of the unit's own peak are resolved by rounding, so counting
    their sign would report numerical noise as a failure of the surrogate.
    """
    profiles = np.array([[100.0, 100.0, 1.0, -1.0]])
    metrics = Q.profile_metrics(profiles)
    assert metrics["n_material_cells"] == 2
    assert metrics["sign_flip_fraction"] == 0.0


def test_a_real_oscillation_above_the_threshold_does_flip() -> None:
    """A profile with a non-zero mean and a sign change on material cells.

    ``[10, -10, 10, 3]`` has mean 3.25, so the surrogate predicts "positive"
    everywhere and is wrong at the single negative cell: one flip in four.
    """
    profiles = np.array([[10.0, -10.0, 10.0, 3.0]])
    metrics = Q.profile_metrics(profiles)
    assert metrics["n_material_cells"] == 4
    assert metrics["sign_flip_fraction"] == pytest.approx(0.25)


def test_missing_cells_do_not_count_towards_the_denominator() -> None:
    profiles = np.array([[2.0, -1.0, np.nan, np.nan]])
    metrics = Q.profile_metrics(profiles)
    assert metrics["n_material_cells"] == 2
    assert metrics["sign_flip_fraction"] == pytest.approx(0.5)


def test_profile_metrics_on_an_empty_table_reports_nothing() -> None:
    metrics = Q.profile_metrics(np.zeros((0, 5)))
    assert metrics["n_units"] == 0
    assert metrics["sign_flip_fraction"] is None


def test_top_profiles_orders_by_magnitude_and_decodes_pair_ids() -> None:
    ids = np.array([3, 300, 12])
    profiles = np.array([[0.1, 0.1], [5.0, 5.0], [1.0, 1.0]])
    rows = Q.top_profiles(ids, profiles, n=2, d_sae=100)
    assert rows[0]["feature_i"] == 3 and rows[0]["feature_j"] == 0
    assert rows[0]["mean_abs"] == pytest.approx(5.0)
    assert rows[1]["feature_i"] == 0 and rows[1]["feature_j"] == 12


# --------------------------------------------------------------------------
# The upper end of the sweep
# --------------------------------------------------------------------------


def test_the_sweep_reaches_the_distance_range_the_paper_reports() -> None:
    """The synthetic measurement goes to 8192; the sweep must answer the same
    question, not a shorter one that happens to be easier."""
    assert max(Q.SWEEP_DELTAS) == 8192
    assert min(Q.SWEEP_DELTAS) == 1
    assert len(Q.SWEEP_DELTAS) >= 40


# --------------------------------------------------------------------------
# The committed artifact
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.fail(
            "results/qk_attribution.json is absent; it is committed evidence, "
            "not an optional input"
        )
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_artifact_is_an_exact_reconstruction_of_a_real_forward_pass(
    artifact: dict,
) -> None:
    """Two residuals, and neither may be confused with the effect being measured.

    ``ln_residual`` says the residual vector the dictionary encodes is the one
    the attention layer actually consumed. ``max_relative_exactness_error`` says
    the bucket split re-sums to the score. If either is not at float scale, the
    attribution numbers describe some other computation.
    """
    assert artifact["schema"] == "qk-attribution/v1"
    assert artifact["ln_residual"] < 1e-4
    assert artifact["aggregate"]["max_relative_exactness_error"] < 1e-9
    for head in artifact["heads"]:
        assert head["max_relative_exactness_error"] < 1e-9


def test_the_artifact_runs_mid_network_with_partial_rope(artifact: dict) -> None:
    """Layer 3 of 6, and a rotation width below the head width.

    Both are choices made before the measurement, and both matter: a full-width
    rotation would make the rotary and total views identical, and the paper's
    structural claim is about the partial case.
    """
    assert artifact["sae_layer"] == 3
    assert artifact["attention_layer"] == 4
    assert artifact["rotary_dim"] < artifact["head_dim"]
    assert artifact["n_tokens"] > 100


def test_the_aggregate_is_re_derived_from_the_per_head_rows(artifact: dict) -> None:
    heads = artifact["heads"]
    assert artifact["aggregate"]["n_heads"] == len(heads)
    for family in (
        "feature",
        "feature_rotary",
        "sweep_feature",
        "sweep_rotary",
        "sweep_pair",
    ):
        cells = sum(head[f"{family}_metrics"]["n_material_cells"] for head in heads)
        assert artifact["aggregate"][family]["n_material_cells"] == cells


def test_the_sweep_covers_the_whole_grid_for_every_head(artifact: dict) -> None:
    assert len(artifact["sweep_deltas"]) == len(Q.SWEEP_DELTAS)
    assert artifact["sweep_deltas"][-1] == 8192
    for head in artifact["heads"]:
        for kind in ("sweep_feature", "sweep_rotary", "sweep_pair"):
            metrics = head[f"{kind}_metrics"]
            assert metrics["n_units"] > 0
            assert metrics["n_material_cells"] > 0


def test_the_rotary_view_changes_the_answer(artifact: dict) -> None:
    """If zeroing the key's tail did nothing, the "rotary" numbers would be a
    restatement of the total and the paper's distinction would be empty."""
    for head in artifact["heads"]:
        total = head["feature_metrics"]["rel_err_median"]
        rotary = head["feature_rotary_metrics"]["rel_err_median"]
        assert abs(total - rotary) > 1e-6, head["head"]


def test_every_head_carries_a_nonzero_share_in_the_base_bucket(artifact: dict) -> None:
    """The base bucket is the honest part of the accounting: the LayerNorm
    constant, the dictionary bias and the reconstruction residual are not
    attributed to any feature, and pretending otherwise would inflate the
    concentration numbers."""
    for head in artifact["heads"]:
        assert 0.0 < head["base_share_mean"] < 20.0
        assert 0.0 < head["top_k_share_mean"] < 50.0
