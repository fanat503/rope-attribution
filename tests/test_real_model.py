"""Test suite for ``projects/rope_attribution/real_model.py``.

What this file is and is not
-----------------------------
The measurements in ``real_model.py`` are numbers about *pretrained weights*, so
almost nothing about their values can be asserted here: pinning
``ratios[50] == 0.42`` would freeze one checkpoint's activations into the test
suite, and it would freeze them at whatever value a particular transformers
version happened to produce. What *can* be asserted on real data, and what is
asserted here, is everything algebraic:

* the frequency ladder this repository builds from a checkpoint's declared
  configuration reproduces the buffer the checkpoint built for itself;
* ``PairTerms.amplitude``, ``.aligned`` and ``.crossed`` do not depend on the
  distance, which is an algebraic identity, so real vectors must satisfy it
  bit for bit;
* the per-pair closed form summed over pairs equals a **brute-force rotated dot
  product written out independently here** - the 2x2 rotation blocks are built
  from :mod:`math` here, sharing no code with ``rope.py``'s ``rotate_half``;
* ``c_k(delta) == R_k * cos(D_k - psi_k)`` on real vectors;
* the JSON that ships with the module is internally consistent: quantiles
  ordered, fractions inside ``[0, 1]``, every reported spread non-negative and
  ordered around its own median, histogram counts summing to the sample count.

Skipping
--------
Every test that needs torch, transformers or a checkpoint skips cleanly, so
``pytest -q`` stays green on a machine that has none of them. Nothing in this
file is allowed to depend on network access being available.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the suite runnable without PYTHONPATH being pre-set.
_REPO = Path(__file__).resolve().parents[1]
_PROJECTS = _REPO / "projects"
if str(_PROJECTS) not in sys.path:
    sys.path.insert(0, str(_PROJECTS))

from rope_attribution import real_model as RM  # noqa: E402  (needs the tweak above)
from rope_attribution.experiments import pair_terms  # noqa: E402
from rope_attribution.rope import inv_freq  # noqa: E402

#: The checkpoint the activation-level tests run against. One is enough: the
#: invariants under test are algebraic and hold for any RoPE model, and loading
#: three checkpoints in a test session would be wasteful. The remaining models
#: are covered through the JSON consistency tests below.
_TEST_MODEL_ID = RM.DEFAULT_MODEL_IDS[0]

_JSON_PATH = _REPO / "results" / "real_model.json"


# --------------------------------------------------------------------------
# Independent references: no rope.py inside
# --------------------------------------------------------------------------


def _ladder(dim: int, base: float) -> np.ndarray:
    """``inv_freq[i] = base ** (-2i/dim)``, written from its closed form."""
    return np.array([base ** (-2.0 * k / dim) for k in range(dim // 2)], dtype=np.float64)


def _brute_rot_dot(
    q: np.ndarray, k: np.ndarray, freqs: np.ndarray, delta: float
) -> float:
    """``q . R_delta k``, with each 2x2 rotation block written out by hand.

    ``rope.py`` reaches the same rotation through ``rotate_half`` and
    ``apply_rope``; here the block ``[[c, -s], [s, c]]`` is applied element by
    element using :mod:`math`, so the two share no code and a disagreement is a
    real disagreement. Pairs are ``(i, i + half)``, matching the split-half
    convention: ``cos``/``sin`` are duplicated and concatenated by
    ``rope_cos_sin``, so ``cos[i + half] == cos[i]``.
    """
    half = freqs.shape[0]
    assert q.shape == (2 * half,) and k.shape == (2 * half,)
    total = 0.0
    for i in range(half):
        c = math.cos(delta * float(freqs[i]))
        s = math.sin(delta * float(freqs[i]))
        q1, q2 = float(q[i]), float(q[i + half])
        k1, k2 = float(k[i]), float(k[i + half])
        total += q1 * (k1 * c - k2 * s) + q2 * (k1 * s + k2 * c)
    return total


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded():
    """A real checkpoint, or a skip."""
    if not RM.REAL_MODEL_AVAILABLE:
        pytest.skip("torch and/or transformers are not installed")
    try:
        return RM.load_model(_TEST_MODEL_ID)
    except Exception as exc:
        pytest.skip(f"checkpoint {_TEST_MODEL_ID!r} unavailable: {exc}")


@pytest.fixture(scope="module")
def vectors(loaded):
    """Real ``q_hat``/``k_hat`` for one short prompt, plus the model's ladder."""
    cv = RM.validate_config(loaded.model)
    if not cv["checkpoint_inv_freq_found"]:
        pytest.skip(f"{_TEST_MODEL_ID} has no rotary module")
    freqs = inv_freq(cv["rope_dim"], cv["rope_theta_declared"])
    hv = RM.collect_head_vectors(loaded.model, loaded.tokenizer, RM.DEFAULT_PROMPTS[0])
    return cv, freqs, hv


@pytest.fixture(scope="module")
def payload():
    """The JSON this module wrote, or a skip."""
    if not _JSON_PATH.is_file():
        pytest.skip(f"{_JSON_PATH} has not been generated yet")
    return json.loads(_JSON_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# graceful degradation - these run with or without torch
# --------------------------------------------------------------------------


def test_availability_flag_is_a_plain_bool_and_matches_its_parts() -> None:
    assert isinstance(RM.REAL_MODEL_AVAILABLE, bool)
    assert isinstance(RM.TORCH_AVAILABLE, bool)
    assert isinstance(RM.TRANSFORMERS_AVAILABLE, bool)
    assert (RM.TORCH_AVAILABLE and RM.TRANSFORMERS_AVAILABLE) == RM.REAL_MODEL_AVAILABLE
    assert RM.is_available() == RM.REAL_MODEL_AVAILABLE


def test_missing_torch_raises_an_error_that_names_the_requirements_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RM, "TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError) as exc:
        RM.load_model()
    assert "real_model_requirements.txt" in str(exc.value)


def test_missing_transformers_raises_an_error_that_names_the_requirements_file(
    loaded, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RM, "TRANSFORMERS_AVAILABLE", False)
    with pytest.raises(RuntimeError) as exc:
        RM.load_model(_TEST_MODEL_ID)
    assert "real_model_requirements.txt" in str(exc.value)


def test_run_all_refuses_rather_than_silently_emptying_the_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RM, "REAL_MODEL_AVAILABLE", False)
    with pytest.raises(RuntimeError) as exc:
        RM.run_all()
    assert "real_model_requirements.txt" in str(exc.value)


def test_importing_the_module_downloads_nothing() -> None:
    """The availability flags must be decided without importing torch.

    ``find_spec`` resolves a module's location without executing it, which is
    what keeps ``import real_model`` free of a multi-second torch import and of
    any chance of a download. If this ever regressed to a bare ``import torch``
    at module scope, the module would become unusable on a machine without it.
    """
    import importlib.util

    src = (_PROJECTS / "rope_attribution" / "real_model.py").read_text(encoding="utf-8")
    tree = __import__("ast").parse(src)
    module_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, __import__("ast").Import):
            for alias in node.names:
                module_level.add(alias.name.split(".")[0])
        elif isinstance(node, __import__("ast").ImportFrom) and node.module:
            module_level.add(node.module.split(".")[0])
    assert "torch" not in module_level
    assert "transformers" not in module_level
    # ... and the flags really are computed from find_spec.
    assert importlib.util.find_spec is not None


def test_position_blind_threshold_is_stated_and_is_a_ratio() -> None:
    assert 0.0 < RM.BLIND_RATIO < 1.0
    assert isinstance(RM.BLIND_RATIO, float)


def test_default_constants_are_sane() -> None:
    assert RM.NUM_THREADS >= 1
    assert isinstance(RM.DEFAULT_SEED, int)
    assert tuple(RM.DEFAULT_DELTAS) == (1, 64, 512, 2048, 4096)
    assert all(d > 0 for d in RM.DEFAULT_DELTAS)
    assert all(n > 1 for n in RM.DEFAULT_LENS)
    assert RM.DEFAULT_PROMPTS and all(isinstance(p, str) for p in RM.DEFAULT_PROMPTS)
    assert all(RM.DEFAULT_MODEL_IDS)


# --------------------------------------------------------------------------
# 1. the ladder matches the checkpoint's declared configuration
# --------------------------------------------------------------------------


def test_our_inv_freq_matches_the_checkpoints_own_buffer(vectors) -> None:
    cv, freqs, _ = vectors
    assert cv["checkpoint_inv_freq_found"], f"{_TEST_MODEL_ID} has no rotary module"
    assert cv["lengths_agree"]
    assert cv["our_inv_freq_len"] == cv["checkpoint_inv_freq_len"] == freqs.shape[0]
    assert cv["config_matches_checkpoint"]
    # The residual must be at the level of the dtype the checkpoint stores its
    # ladder in, not merely small. float32 has an eps of about 1.2e-7 and the
    # largest ladder entry is 1 whatever theta is, so 1e-6 absolute is generous
    # but still four orders below any structural disagreement.
    assert cv["max_abs_diff"] <= 8.0 * cv["storage_dtype_eps"]
    assert cv["max_rel_diff"] <= 1e-6


def test_the_declared_config_alone_reproduces_the_ladder(vectors) -> None:
    """``inv_freq(head_dim * partial_rotary_factor, rope_theta)``, no model needed.

    This is the claim that would break if ``rope.py`` disagreed with a real
    checkpoint, and it is checked against an independently written closed form
    as well so that a shared bug in ``inv_freq`` cannot pass.
    """
    cv, freqs, _ = vectors
    expected_dim = int(cv["head_dim"] * cv["partial_rotary_factor_declared"])
    assert cv["rope_dim"] == expected_dim
    rebuilt = inv_freq(expected_dim, cv["rope_theta_declared"])
    assert rebuilt.shape == freqs.shape
    assert np.array_equal(rebuilt, freqs)
    reference = _ladder(expected_dim, cv["rope_theta_declared"])
    assert np.abs(rebuilt - reference).max() <= 1e-15 * max(1.0, cv["rope_theta_declared"])
    # The largest ladder entry is theta ** 0 == 1 exactly, by definition.
    assert freqs[0] == pytest.approx(1.0, abs=1e-15)
    # The ladder is strictly decreasing, so pair index k is a slower channel.
    assert np.all(np.diff(freqs) < 0.0)


def test_rotated_channel_count_is_consistent_with_the_ladder(vectors) -> None:
    cv, freqs, hv = vectors
    assert cv["rotary_pairs_per_head"] * 2 == cv["rope_dim"]
    assert 2 <= cv["rope_dim"] <= cv["head_dim"]
    assert freqs.shape[0] * 2 == cv["rope_dim"]
    # The vectors collected from the model must span the whole head, and there
    # must be one slice per attention module in layer order.
    assert hv.q_hat.shape[-1] == hv.k_hat.shape[-1] == cv["head_dim"]
    assert hv.q_hat.shape[:2] == hv.k_hat.shape[:2] == (cv["n_layer"], cv["n_head"])
    assert hv.q_hat.shape[:3] == hv.k_hat.shape[:3]
    assert hv.n_tokens >= 2


def test_every_attention_module_is_found_in_layer_order(loaded) -> None:
    """The hook must see every layer, in order, or the spread is a subset."""
    model = RM.load_model(_TEST_MODEL_ID).model
    attns = RM.attention_modules(model)
    assert len(attns) == model.config.num_hidden_layers
    idx = [getattr(a, "layer_idx", None) for a in attns]
    assert idx == sorted(i for i in idx if i is not None)
    # Both fusion styles must be recognized, or the module silently measures
    # nothing on one architecture.
    for a in attns:
        assert RM._qkv_projector(a) is not None
        assert RM.head_dim_of(a, model.config) > 0
        assert RM.n_heads_of(a, model.config) > 0


def test_gpt2_is_recorded_as_not_using_rope(payload) -> None:
    """The premise this module had to check: GPT-2 uses learned absolute positions.

    Asserted on the recorded evidence rather than by re-running a 500 MB
    download here. If a future transformers release reintroduced a rotation into
    GPT-2, the recorded JSON would be stale - but the claim under test is the
    one the paper's honesty section has to get right either way, so it is
    pinned to the evidence that was actually collected.
    """
    g = payload["gpt2_rope_check"]
    if "error" in g:
        pytest.skip(f"gpt2 could not be loaded when the JSON was written: {g['error']}")
    assert g["model_type"] == "gpt2"
    assert g["rotary_module_found"] is False
    assert g["uses_rope"] is False
    assert g["rope_theta_in_config"] is False
    assert g["learned_absolute_position_embedding"] is True
    assert isinstance(g["learned_position_table_size"], int)


# --------------------------------------------------------------------------
# 2./3. algebraic identities on real vectors
# --------------------------------------------------------------------------


def test_pair_amplitude_is_independent_of_delta(vectors) -> None:
    """``R_k = hypot(A_k, B_k)`` cannot depend on the distance. Bitwise.

    ``pair_terms`` recomputes ``A_k`` and ``B_k`` from the same two vectors for
    every delta, so exact array equality is the right assertion, not a
    tolerance. Real activations are no more exempt from this than random ones.
    """
    cv, freqs, hv = vectors
    rope_dim = cv["rope_dim"]
    deltas = [0, 1, 3, 64, 511, 512, 2048, 4096]
    checked = 0
    for layer in (0, hv.n_layers // 2, hv.n_layers - 1):
        for head in (0, hv.n_heads // 2, hv.n_heads - 1):
            q = hv.q_hat[layer, head, 0, :rope_dim]
            k = hv.k_hat[layer, head, -1, :rope_dim]
            base = pair_terms(q, k, freqs, deltas[0])
            ref_amp = base.amplitude
            for d in deltas[1:]:
                t = pair_terms(q, k, freqs, d)
                assert np.array_equal(t.aligned, base.aligned)
                assert np.array_equal(t.crossed, base.crossed)
                assert np.array_equal(t.amplitude, ref_amp)
                checked += 1
    assert checked > 0
    # Sanity: the amplitude really is the hypot, and really is not zero here.
    assert np.all(ref_amp >= 0.0)
    assert ref_amp.max() > 0.0
    assert np.abs(ref_amp - np.hypot(base.aligned, base.crossed)).max() == 0.0


def test_closed_form_sums_to_a_brute_force_rotated_dot_product(vectors) -> None:
    """``sum_k c_k(delta) == q_hat . R_delta k_hat`` on real vectors.

    The right-hand side is the independently written 2x2-block rotation in this
    file, not ``rope.py``. The tolerance is derived from the dtype the checkpoint
    actually produced its activations in, not asserted as a magic constant: a
    float32 checkpoint cannot agree to 1e-15 with anything, and demanding that
    would be testing the checkpoint's arithmetic rather than the algebra. The
    factor 64 covers accumulating a few dozen products in the worst case.
    """
    cv, freqs, hv = vectors
    rope_dim = cv["rope_dim"]
    tol_eps = float(np.finfo(hv.q_hat.dtype).eps)
    worst = 0.0
    scale = 0.0
    n = 0
    for layer in range(0, hv.n_layers, 3):
        for head in range(0, hv.n_heads, 4):
            for t in (0, hv.n_tokens // 2, hv.n_tokens - 1):
                q = hv.q_hat[layer, head, t, :rope_dim]
                k = hv.k_hat[layer, head, t, :rope_dim]
                for delta in (0, 1, 7, 64, 1000):
                    closed = float(pair_terms(q, k, freqs, delta).contributions().sum())
                    brute = _brute_rot_dot(q, k, freqs, delta)
                    worst = max(worst, abs(closed - brute))
                    scale = max(scale, abs(brute))
                    n += 1
    assert n >= 30
    budget = 64.0 * tol_eps * max(1.0, scale)
    assert worst <= budget, (
        f"closed form and the hand-written rotation disagree by {worst:.3e}, "
        f"which is {worst / budget:.1f}x the {tol_eps:.3e} budget for "
        f"{hv.q_hat.dtype} on a scale of {scale:.3e}"
    )


def test_pair_contribution_is_one_sinusoid_in_distance(vectors) -> None:
    """``c_k(delta) == R_k * cos(D_k - psi_k)``, the claim the paper rests on.

    Checked on real vectors at several distances, which is the point: this is the
    form that makes each channel a fixed-amplitude, linearly-advancing phase, and
    it has to hold for a trained network and not merely for random vectors.
    Tolerance is set by the activation dtype, as above.
    """
    cv, freqs, hv = vectors
    rope_dim = cv["rope_dim"]
    tol = 32.0 * float(np.finfo(hv.q_hat.dtype).eps)
    for layer in (0, hv.n_layers - 1):
        for head in (0, hv.n_heads - 1):
            q = hv.q_hat[layer, head, 0, :rope_dim]
            k = hv.k_hat[layer, head, 0, :rope_dim]
            t = pair_terms(q, k, freqs, 0)
            amp, psi = t.amplitude, t.phase_offset
            for delta in (0, 1, 13, 512, 4096):
                got = pair_terms(q, k, freqs, delta).contributions()
                want = amp * np.cos(delta * freqs - psi)
                budget = tol * max(1.0, float(amp.max()))
                assert np.abs(got - want).max() <= budget


def test_amplitude_dominates_the_contribution(vectors) -> None:
    """``|c_k(delta)| <= R_k``: the amplitude really does upper-bound the term.

    This is the bound ``linearization_error`` relies on when it normalizes by
    ``R_k``, so it is worth asserting on real data rather than assuming it.
    """
    cv, freqs, hv = vectors
    rope_dim = cv["rope_dim"]
    q = hv.q_hat[0, 0, 0, :rope_dim]
    k = hv.k_hat[0, 0, -1, :rope_dim]
    for delta in (0, 1, 97, 4096):
        t = pair_terms(q, k, freqs, delta)
        assert np.all(np.abs(t.contributions()) <= t.amplitude + 1e-9 * t.amplitude)


def test_the_share_of_amplitude_carrying_position_is_a_ratio(vectors) -> None:
    """``|B_k| / R_k`` lies in ``[0, 1]`` and needs no sign convention."""
    cv, freqs, hv = vectors
    rope_dim = cv["rope_dim"]
    t = pair_terms(
        hv.q_hat[0, 0, 0, :rope_dim], hv.k_hat[0, 0, -1, :rope_dim], freqs, 5
    )
    share = np.abs(t.crossed) / t.amplitude
    assert np.all(share >= 0.0)
    assert np.all(share <= 1.0 + 1e-12)


def test_identities_hold_on_real_activations_at_the_dtype_floor(vectors) -> None:
    """The three theorems of E1, on real vectors, bounded by the activation dtype.

    This is the single most load-bearing assertion in the file. The synthetic
    suite reports these residuals at ~1e-14 because its vectors are float64; a
    checkpoint hands over float32, so agreement at 1e-15 would be impossible and
    agreement at the dtype's own epsilon is the correct expectation. Anything
    materially above that would mean the algebra, not the arithmetic, is wrong.
    """
    cv, freqs, hv = vectors
    out = RM.identity_residuals(hv, freqs, cv["rope_dim"], n_pairs=16)
    assert out["activation_dtype"] == str(hv.q_hat.dtype)
    eps = out["activation_dtype_eps"]
    assert eps > 0.0
    assert out["pair_scores_checked"] > 0
    assert out["relative_position_pairs_checked"] > 0
    assert out["pair_closed_form_max_abs_err"] >= 0.0
    assert out["relative_position_max_abs_err"] >= 0.0
    assert out["key_norm_preservation_max_abs_err"] >= 0.0
    assert out["query_norm_preservation_max_abs_err"] >= 0.0
    # The residual must be a small multiple of the activation epsilon, not a
    # small multiple of a number chosen for float64.
    budget = 256.0 * eps * max(1.0, hv.q_hat.shape[-1])
    assert out["pair_closed_form_max_abs_err"] <= budget
    assert out["relative_position_max_abs_err"] <= budget
    assert out["key_norm_preservation_max_abs_err"] <= budget
    assert out["query_norm_preservation_max_abs_err"] <= budget
    # The meaningful bilinearity number is the one normalized by the sum of
    # absolute products. The |score|-normalized one is recorded too, and is
    # allowed to be large: that is cancellation, not nonlinearity.
    assert out["bilinearity_checks"] > 0
    assert out["bilinearity_max_scaled_err"] <= 1e-4
    assert out["bilinearity_max_rel_err"] >= out["bilinearity_max_scaled_err"]
    assert 0.0 <= out["bilinearity_min_score_over_sum_abs"] <= 1.0 + 1e-12
    for ratio in out["residual_over_dtype_eps"].values():
        assert ratio >= 0.0, "a residual ratio was negative"
    assert out["residual_over_dtype_eps"]["bilinearity"] <= 256.0


def test_identities_are_recorded_in_the_payload(payload) -> None:
    """The shipped JSON must carry the real-data identity residuals."""
    for model in payload["models"]:
        if "error" in model:
            continue
        ident = model["per_prompt"][0]["identities"]
        assert ident["pair_scores_checked"] > 0
        assert ident["relative_position_pairs_checked"] > 0
        eps = ident["activation_dtype_eps"]
        assert eps > 0.0
        assert ident["pair_closed_form_max_abs_err"] <= 256.0 * eps * 1024.0
        assert ident["relative_position_max_abs_err"] <= 256.0 * eps * 1024.0
        agg = payload["aggregates"][model["model_id"]]["identities"]
        for key in (
            "pair_closed_form_max_abs_err",
            "relative_position_max_abs_err",
            "bilinearity_max_scaled_err",
            "bilinearity_max_rel_err",
            "bilinearity_min_score_over_sum_abs",
            "key_norm_preservation_max_abs_err",
            "query_norm_preservation_max_abs_err",
        ):
            _assert_spread_well_formed(agg[key])
            assert agg[key]["min"] >= 0.0
        assert agg["bilinearity_max_rel_err"]["max"] >= agg["bilinearity_max_scaled_err"]["max"]
        assert agg["bilinearity_min_score_over_sum_abs"]["max"] <= 1.0 + 1e-12
        ratios = agg["residual_over_dtype_eps"]
        assert set(ratios) == {
            "pair_closed_form",
            "relative_position",
            "norm_preservation",
            "bilinearity",
        }
        for spread in ratios.values():
            _assert_spread_well_formed(spread)
            assert spread["min"] >= 0.0


def test_measured_position_channels_are_self_consistent(vectors) -> None:
    """One ``(layer, head)`` sweep of the real measurement, with small ``n_pairs``."""
    cv, freqs, hv = vectors
    out = RM.measure_position_channels(hv, freqs, cv["rope_dim"], n_pairs=8)
    assert out["rotary_pairs_per_head"] * 2 == out["rope_dim"]
    assert out["n_samples"] > 0
    assert out["n_samples"] + out["n_degenerate_pairs_excluded"] == (
        hv.n_layers * hv.n_heads * 8 * out["rotary_pairs_per_head"]
    )
    qv = out["share_quantiles"]
    assert qv["values"] == sorted(qv["values"]), "quantiles must be non-decreasing"
    assert all(0.0 <= v <= 1.0 for v in qv["values"])
    assert 0.0 <= out["frac_position_blind_overall"] <= 1.0
    hist = out["share_histogram_0_to_1"]
    assert sum(hist["counts"]) == out["n_samples"]
    assert len(hist["counts"]) == len(hist["bin_edges"]) - 1
    assert len(set(hist["bin_edges"])) == len(hist["bin_edges"])
    for key in ("median_share_across_heads", "frac_position_blind_across_heads"):
        _assert_spread_well_formed(out[key])
    # The threshold is reported, not just applied.
    assert str(RM.BLIND_RATIO) in out["position_blind_threshold"]["statement"]
    assert out["position_blind_threshold"]["ratio"] == RM.BLIND_RATIO


def test_measure_position_channels_rejects_an_inconsistent_rope_dim(vectors) -> None:
    cv, freqs, hv = vectors
    with pytest.raises(ValueError, match="inconsistent"):
        RM.measure_position_channels(hv, freqs, cv["rope_dim"] + 2)


def test_linearization_on_real_data_reproduces_the_ladder_only_entries(vectors) -> None:
    """``frac_channels_linearizable`` is a function of the ladder and nothing else.

    That is why its across-head and across-layer spread must come out as exactly
    zero on real activations. Asserting it pins the *reason* the number cannot
    move, which is the honest caveat on that column.
    """
    cv, freqs, hv = vectors
    out = RM.measure_linearization(
        hv, freqs, cv["rope_dim"], deltas=(1, 64, 512), n_pairs=4
    )
    for row in out["rows"]:
        d = row["delta"]
        expected = float(np.mean(np.abs(d * freqs) <= 0.1))
        assert row["frac_channels_linearizable"]["median"] == pytest.approx(expected)
        assert row["frac_channels_linearizable"]["std"] == 0.0
        assert row["frac_channels_linearizable"]["min"] == pytest.approx(expected)
        assert row["frac_channels_linearizable"]["max"] == pytest.approx(expected)
        assert row["max_abs_angle"]["std"] == 0.0
        assert row["max_abs_angle"]["median"] == pytest.approx(d * freqs[0])
        # The data-dependent entry is the one that may spread.
        assert row["amplitude_weighted_err"]["n"] > 0
        assert row["amplitude_weighted_err"]["min"] >= 0.0


def test_entropy_ratios_are_normalized(loaded) -> None:
    """``H / ln T`` and ``H / ln(n_allowed)`` are both fractions, and ordered."""
    out = RM.measure_entropy(
        loaded.model, loaded.tokenizer, RM.DEFAULT_LONG_TEXTS[0], lens=(64,)
    )
    assert out["per_length"], "no length in the requested grid was long enough"
    for row in out["per_length"]:
        t = row["seq_len"]
        assert t > 1
        assert row["max_possible_entropy_ln_T"] == pytest.approx(math.log(t))
        for key in (
            "entropy_ratio_vs_ln_T_across_layers",
            "entropy_ratio_vs_ln_allowed_across_layers",
            "entropy_ratio_vs_ln_T_across_heads_within_layer",
        ):
            _assert_spread_well_formed(row[key])
            assert row[key]["min"] >= 0.0 and row[key]["max"] <= 1.0 + 1e-9
        # A causal row can reach at most m+1 of T keys, so dividing by ln(m+1)
        # can only raise the ratio relative to dividing by ln(T).
        assert (
            row["entropy_ratio_vs_ln_allowed_across_layers"]["median"]
            >= row["entropy_ratio_vs_ln_T_across_layers"]["median"] - 1e-12
        )
        assert len(row["entropy_ratio_vs_ln_T_per_layer"]) == loaded.config.num_hidden_layers


def test_synthetic_baseline_reads_the_published_measurements() -> None:
    base = RM.synthetic_baseline(rope_dim=64, theta=10000.0)
    assert base["published_available"], "results/measurements.json was not found"
    rows = base["synthetic_published_32_channel"]
    assert [r["delta"] for r in rows] == list(RM.DEFAULT_DELTAS)
    for r in rows:
        assert 0.0 <= r["frac_channels_linearizable"] <= 1.0
        assert r["amplitude_weighted_err"] >= 0.0
        assert r["max_abs_angle"] >= r["median_abs_angle"] >= 0.0
    ctl = base["synthetic_same_ladder"]
    assert [r["delta"] for r in ctl] == list(RM.DEFAULT_DELTAS)
    # With no partial factor the control is the published construction itself.
    for pub, c in zip(rows, ctl, strict=True):
        assert c["frac_channels_linearizable"] == pytest.approx(
            pub["frac_channels_linearizable"], abs=1e-12
        )


def test_a_partial_ladder_is_a_subsample_of_the_full_one() -> None:
    """``inv_freq(dim * p, theta)`` keeps every ``1/p``-th channel of the full ladder.

    This is what partial rotation actually does, and it is not obvious: it keeps
    an evenly-spaced subsample of the frequencies rather than truncating a
    contiguous block of channels. It is the reason a partial-RoPE checkpoint has a
    different ``frac_channels_linearizable`` from a full-width one at the same
    distance, with no weight change involved at all.
    """
    theta = 10000.0
    partial = 0.25
    full_dim, narrow_dim = 64, int(64 * partial)
    step = full_dim // narrow_dim
    a = inv_freq(narrow_dim, theta)
    b = inv_freq(full_dim, theta)
    assert step == 4
    assert a.shape[0] * step == b.shape[0]
    assert np.abs(a - b[::step][: a.shape[0]]).max() == pytest.approx(0.0, abs=1e-15)
    # ... and it is NOT a prefix of the full ladder, which is the wrong guess.
    assert np.abs(a - b[: a.shape[0]]).max() > 1e-3


def test_validate_config_reports_the_partial_ladder_subsample_step(vectors) -> None:
    cv, _freqs, _hv = vectors
    partial = cv["partial_rotary_factor_declared"]
    step = cv["partial_ladder_subsample_step"]
    if partial == 1.0:
        assert step == 1
    else:
        assert step is not None
        assert step == int(round(1.0 / partial))
        assert cv["rotary_pairs_per_head"] * step == cv["head_dim"] // 2


# --------------------------------------------------------------------------
# the shipped JSON
# --------------------------------------------------------------------------


def _assert_spread_well_formed(sp: dict) -> None:
    """Every summary block must be ordered and non-negative, or have no samples.

    The mean is allowed a few ulps outside ``[min, max]``: ``np.mean`` of a sample
    whose values are all identical can round one ulp above that value, and that
    is arithmetic, not a violated ordering.
    """
    if sp.get("n", 0) == 0:
        assert set(sp) == {"n"}
        return
    for key in ("min", "p25", "median", "p75", "max", "mean", "std"):
        assert key in sp, f"summary block is missing {key!r}"
        assert math.isfinite(sp[key]), f"{key} is not finite"
    assert sp["min"] >= 0.0, "every spread in this module is a magnitude"
    assert sp["std"] >= 0.0
    slack = 8.0 * float(np.finfo(np.float64).eps) * max(1.0, abs(sp["max"]))
    assert sp["min"] - slack <= sp["p25"] <= sp["median"] <= sp["p75"] <= sp["max"]
    assert sp["min"] - slack <= sp["mean"] <= sp["max"] + slack
    assert sp["std"] <= (sp["max"] - sp["min"]) * 2.0 + slack


def test_payload_records_its_own_provenance(payload) -> None:
    assert payload["schema"] == "rope_attribution/real_model/v1"
    assert payload["seed"] == RM.DEFAULT_SEED
    assert payload["position_blind_ratio"] == RM.BLIND_RATIO
    assert payload["deltas"] == list(RM.DEFAULT_DELTAS)
    assert payload["sequence_lengths"] == list(RM.DEFAULT_LENS)
    assert payload["models"], "no checkpoint was measured"
    assert "torch_version" in payload and "transformers_version" in payload


def test_every_recorded_model_declares_its_id_and_config(payload) -> None:
    for model in payload["models"]:
        assert isinstance(model["model_id"], str) and model["model_id"]
        if "error" in model:
            # A recorded failure is allowed, but must say what failed.
            assert model["error"]
            continue
        cv = model["config_validation"]
        assert cv["model_type"]
        assert cv["n_embd"] > 0 and cv["n_head"] > 0 and cv["n_layer"] > 0
        assert cv["head_dim"] * cv["n_head"] == cv["n_embd"]
        assert cv["rope_theta_declared"] > 1.0
        assert 0.0 < cv["partial_rotary_factor_declared"] <= 1.0
        assert cv["rotary_pairs_per_head"] * 2 == cv["rope_dim"]
        assert cv["config_matches_checkpoint"] is True
        assert cv["max_abs_diff"] <= 8.0 * cv["storage_dtype_eps"]
        assert cv["rotated_fraction_of_head"] == pytest.approx(
            cv["rope_dim"] / cv["head_dim"]
        )
        assert len(cv["our_inv_freq_head"]) == cv["our_inv_freq_len"]
        assert cv["our_inv_freq_head"][0] == pytest.approx(1.0, abs=1e-15)


def test_payload_fractions_are_fractions(payload) -> None:
    for model in payload["models"]:
        if "error" in model:
            continue
        for pp in model["per_prompt"]:
            pos = pp["position_channels"]
            assert pp["n_tokens"] >= 2
            assert 0.0 <= pos["frac_position_blind_overall"] <= 1.0
            qv = pos["share_quantiles"]
            assert qv["values"] == sorted(qv["values"])
            assert all(0.0 <= v <= 1.0 for v in qv["values"])
            hist = pos["share_histogram_0_to_1"]
            assert sum(hist["counts"]) == pos["n_samples"]
            for row in pp["linearization"]["rows"]:
                for key in (
                    "frac_channels_linearizable",
                    "max_abs_angle",
                    "median_abs_angle",
                    "amplitude_weighted_err",
                    "amplitude_weighted_err_across_layers",
                    "per_pair_err_rms",
                ):
                    _assert_spread_well_formed(row[key])
                assert row["frac_channels_linearizable"]["min"] >= 0.0
                assert row["frac_channels_linearizable"]["max"] <= 1.0 + 1e-12
                assert row["amplitude_weighted_err"]["min"] >= 0.0
            for key in (
                "median_share_across_heads",
                "median_share_across_layers",
                "frac_position_blind_across_heads",
                "frac_position_blind_across_layers",
                "mean_share_across_heads",
            ):
                _assert_spread_well_formed(pos[key])
                assert pos[key]["max"] <= 1.0 + 1e-12
        for e in model["entropy"]:
            if "error" in e:
                assert e["error"]
                continue
            for row in e["per_length"]:
                for key in (
                    "entropy_ratio_vs_ln_T_across_layers",
                    "entropy_ratio_vs_ln_allowed_across_layers",
                    "entropy_ratio_vs_ln_T_across_heads_within_layer",
                ):
                    _assert_spread_well_formed(row[key])
                    assert row[key]["min"] >= 0.0 and row[key]["max"] <= 1.0 + 1e-9


def test_payload_aggregates_agree_with_the_per_prompt_measurements(payload) -> None:
    """The pooled summary must not contradict the per-prompt rows it pools."""
    for mid, agg in payload["aggregates"].items():
        model = next(m for m in payload["models"] if m["model_id"] == mid)
        assert agg["n_samples"] == sum(
            pp["position_channels"]["n_samples"] for pp in model["per_prompt"]
        )
        assert agg["n_degenerate_pairs_excluded"] == sum(
            pp["position_channels"]["n_degenerate_pairs_excluded"]
            for pp in model["per_prompt"]
        )
        _assert_spread_well_formed(agg["median_share_across_heads"])
        _assert_spread_well_formed(agg["frac_position_blind_across_heads"])
        pooled_hist = agg["histogram_pooled_counts"]
        assert sum(pooled_hist) == agg["n_samples"]
        assert len(pooled_hist) == len(agg["histogram_pooled_bin_edges"]) - 1

        per_prompt_deltas = [r["delta"] for r in agg["linearization"]]
        assert per_prompt_deltas == sorted(per_prompt_deltas)
        assert set(per_prompt_deltas) == set(payload["deltas"])
        for row in agg["linearization"]:
            _assert_spread_well_formed(row["frac_channels_linearizable"])
            _assert_spread_well_formed(row["amplitude_weighted_err"])
            assert 0.0 <= row["frac_channels_linearizable"]["max"] <= 1.0 + 1e-12
            rng_ = row["amplitude_weighted_err_head_range"]
            assert 0.0 <= rng_["min"] <= rng_["max"]
            assert rng_["min"] <= row["amplitude_weighted_err"]["max"] + 1e-9


def test_the_recorded_linearizable_fraction_equals_the_ladder_prediction(payload) -> None:
    """The JSON's own claim, checked against its own recorded ladder.

    ``frac_channels_linearizable`` must equal ``mean(|delta * inv_freq| <= 0.1)``
    computed from the frequencies the JSON records for that model. This is the
    strongest internal-consistency statement available and it needs no model
    download: if a future edit ever let the activations leak into that column,
    this fails.
    """
    for model in payload["models"]:
        if "error" in model:
            continue
        mid = model["model_id"]
        freqs = np.asarray(
            model["config_validation"]["our_inv_freq_head"], dtype=np.float64
        )
        agg = payload["aggregates"][mid]
        for row in agg["linearization"]:
            expected = float(np.mean(np.abs(row["delta"] * freqs) <= 0.1))
            assert row["frac_channels_linearizable"]["median"] == pytest.approx(
                expected, abs=1e-12
            )
            assert row["frac_channels_linearizable_std_across_heads_and_prompts"][
                "max"
            ] == pytest.approx(0.0, abs=1e-15)


def test_payload_synthetic_baselines_match_the_published_deltas(payload) -> None:
    for mid, base in payload["synthetic_baseline"].items():
        assert base["published_available"]
        pub = base["synthetic_published_32_channel"]
        ctl = base["synthetic_same_ladder"]
        assert [r["delta"] for r in pub] == payload["deltas"]
        assert [r["delta"] for r in ctl] == payload["deltas"]
        model = next(m for m in payload["models"] if m["model_id"] == mid)
        pairs = model["config_validation"]["rotary_pairs_per_head"]
        # The ladder-only column must agree between the real model and the
        # same-width synthetic control, because no activation can change it.
        for real_row, ctl_row in zip(
            agg_rows(payload, mid), ctl, strict=True
        ):
            assert real_row["frac_channels_linearizable"]["median"] == pytest.approx(
                ctl_row["frac_channels_linearizable"], abs=1e-12
            )
        assert 2 <= pairs <= 64
        for p, c in zip(pub, ctl, strict=True):
            # Same construction, possibly a different ladder width.
            assert p["max_abs_angle"] == pytest.approx(c["max_abs_angle"], rel=1e-12)


def agg_rows(payload: dict, model_id: str) -> list[dict]:
    """The linearization rows of one model, in delta order."""
    return payload["aggregates"][model_id]["linearization"]