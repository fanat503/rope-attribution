"""Tests for the downstream position-free surrogate."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rope_attribution import downstream as D
from rope_attribution.rope import inv_freq, rope_cos_sin

torch = pytest.importorskip("torch")


def test_the_surrogate_kernel_is_the_mean_rotation() -> None:
    freqs = inv_freq(8, 10000.0)
    deltas = np.array([1.0, 2.0, 7.0, 64.0])
    cos_mean, sin_mean = D.surrogate_kernel(deltas, freqs)
    angles = deltas[:, None] * freqs[None, :]
    np.testing.assert_allclose(cos_mean, np.cos(angles).mean(axis=0))
    np.testing.assert_allclose(sin_mean, np.sin(angles).mean(axis=0))


def test_the_circular_kernel_keeps_unit_modulus() -> None:
    freqs = inv_freq(8, 10000.0)
    cos_mean, sin_mean = D.surrogate_kernel(
        np.array([1.0, 2.0, 7.0, 64.0]), freqs, mode="circular"
    )
    np.testing.assert_allclose(cos_mean**2 + sin_mean**2, np.ones_like(freqs))


def test_the_identity_kernel_removes_the_rotation() -> None:
    freqs = inv_freq(8, 10000.0)
    cos_mean, sin_mean = D.surrogate_kernel(
        np.array([1.0, 2.0, 7.0, 64.0]), freqs, mode="identity"
    )
    np.testing.assert_array_equal(cos_mean, np.ones_like(freqs))
    np.testing.assert_array_equal(sin_mean, np.zeros_like(freqs))


def test_an_unknown_kernel_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown surrogate kernel"):
        D.surrogate_kernel(np.array([1.0]), inv_freq(8), mode="magic")


def test_the_surrogate_kernel_refuses_an_empty_distance_set() -> None:
    with pytest.raises(ValueError, match="at least one distance"):
        D.surrogate_kernel(np.array([]), inv_freq(8))


def test_observed_distances_are_the_positive_gaps() -> None:
    np.testing.assert_array_equal(D.observed_distances(5), np.array([1, 2, 3, 4]))
    with pytest.raises(ValueError, match="at least two tokens"):
        D.observed_distances(1)


def test_the_hook_transform_reconstructs_the_position_free_score() -> None:
    """``R(pos) R(-pos) = I`` and ``R(pos) (M k)`` is what the model sees.

    This is the whole mathematical claim of the replacement: the feed-forward
    transform plus the model's own RoPE must equal the surrogate score, for
    every pair of positions.
    """
    rng = np.random.default_rng(0)
    head_dim, n_tokens = 8, 6
    q = torch.tensor(rng.standard_normal((n_tokens, head_dim)))
    k = torch.tensor(rng.standard_normal((n_tokens, head_dim)))
    freqs = inv_freq(head_dim)
    cos_mean, sin_mean = D.surrogate_kernel(D.observed_distances(n_tokens), freqs)
    surrogate_k = D.apply_surrogate_torch(
        k,
        torch.tensor(cos_mean),
        torch.tensor(sin_mean),
    )
    positions = np.arange(n_tokens, dtype=np.float64)
    cos_neg, sin_neg = rope_cos_sin(-positions, freqs)
    cos_pos, sin_pos = rope_cos_sin(positions, freqs)
    q_feed = D.apply_rope_torch(q, torch.tensor(cos_neg), torch.tensor(sin_neg))
    k_feed = D.apply_rope_torch(
        surrogate_k, torch.tensor(cos_neg), torch.tensor(sin_neg)
    )
    q_model = D.apply_rope_torch(q_feed, torch.tensor(cos_pos), torch.tensor(sin_pos))
    k_model = D.apply_rope_torch(k_feed, torch.tensor(cos_pos), torch.tensor(sin_pos))
    torch.testing.assert_close(q_model, q)
    torch.testing.assert_close(k_model, surrogate_k)
    torch.testing.assert_close(q_model @ k_model.T, q @ surrogate_k.T)


def _tiny_llama():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rope_theta=10000.0,
    )
    return transformers.LlamaForCausalLM(cfg).eval(), cfg


def _tiny_gpt_neox(rotary_pct: float = 1.0):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPTNeoXConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
        rotary_pct=rotary_pct,
        rotary_emb_base=10000.0,
    )
    return transformers.GPTNeoXForCausalLM(cfg).eval(), cfg


@pytest.mark.parametrize("kernel", ["mean", "circular", "identity"])
def test_a_tiny_llama_changes_its_nll_under_the_surrogate(kernel: str) -> None:
    model, _cfg = _tiny_llama()
    ids = torch.randint(0, 64, (1, 16))
    exact = D.nll(model, ids)
    assert math.isfinite(exact)
    with D.SurrogateAttention(
        model, deltas=D.observed_distances(16), kernel=kernel
    ):
        surrogate = D.nll(model, ids)
    assert math.isfinite(surrogate)
    assert surrogate != exact
    # The hooks must be removed on exit: the exact model is back.
    assert D.nll(model, ids) == pytest.approx(exact, rel=0.0, abs=0.0)


def test_a_tiny_fused_qkv_model_uses_the_same_path() -> None:
    model, _cfg = _tiny_gpt_neox()
    ids = torch.randint(0, 64, (1, 16))
    exact = D.nll(model, ids)
    with D.SurrogateAttention(model, deltas=D.observed_distances(16)):
        surrogate = D.nll(model, ids)
    assert math.isfinite(surrogate)
    assert surrogate != exact


def test_qk_normalised_attention_is_refused() -> None:
    model, _cfg = _tiny_llama()
    attn = model.model.layers[0].self_attn
    attn.q_norm = torch.nn.Identity()
    with pytest.raises(NotImplementedError, match="QK-normal"):
        D.SurrogateAttention(model)


@pytest.mark.parametrize(
    "setter",
    [
        lambda cfg: setattr(
            cfg,
            "rope_parameters",
            {
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 0.25,
                "rope_type": "default",
            },
        ),
        lambda cfg: setattr(cfg, "rotary_pct", 0.25),
        lambda cfg: setattr(cfg, "partial_rotary_factor", 0.25),
    ],
    ids=["rope_parameters", "rotary_pct", "top_level"],
)
def test_partial_rotation_rotates_only_the_declared_prefix(setter) -> None:
    """pythia-160m keeps its 0.25 inside ``rope_parameters``.

    The surrogate must rotate the 16 of 64 channels the model actually
    rotates and pass the other 48 through untouched.  The same test pins all
    three config shapes the two families use.

    Mutation verdict: rotating the whole head fails the tail equality;
    taking the frequency ladder from ``head_dim`` instead of ``rotary_dim``
    makes ``rotary_dim`` two but builds a 32-pair kernel, so the transform
    raises a shape error.
    """
    model, cfg = _tiny_llama()
    setter(cfg)
    surrogate = D.SurrogateAttention(model)
    assert surrogate.rotary_dim == 2
    assert surrogate._freqs.shape == (1,)

    q = torch.randn(1, 5, 4, 8)
    transformed = surrogate._transform_q(q.reshape(1, 5, 32)).reshape(1, 5, 4, 8)
    # The 2 rotated channels move; the 6 unrotated channels are byte-identical.
    assert not torch.equal(transformed[..., :2], q[..., :2])
    torch.testing.assert_close(transformed[..., 2:], q[..., 2:])
    # The prefix is exactly the inverse rotation the hook promises.
    cos, sin = surrogate._cos_sin_neg(5, q.device, q.dtype)
    torch.testing.assert_close(
        transformed[..., :2], D.apply_rope_torch(q[..., :2], cos, sin)
    )


def test_a_partial_rotation_fused_model_uses_the_same_path() -> None:
    """GPT-NeoX packs q/k/v and declares the factor as ``rotary_pct``."""
    model, cfg = _tiny_gpt_neox(rotary_pct=0.25)
    ids = torch.randint(0, 64, (1, 16))
    exact = D.nll(model, ids)
    assert math.isfinite(exact)
    with D.SurrogateAttention(model, deltas=D.observed_distances(16)):
        surrogate = D.nll(model, ids)
    assert math.isfinite(surrogate)
    assert surrogate != exact
    assert D.SurrogateAttention(model).rotary_dim == 2


def test_the_rotary_base_is_read_from_the_parameter_block() -> None:
    """The same block that carries the partial factor carries the base.

    Mutation verdict: a top-level-only lookup returns the 10,000 default and
    the assertion fails.
    """

    class _Config:
        rope_parameters = {"rope_theta": 123_456.0}

    assert D._rope_theta(_Config()) == 123_456.0


def test_gemma4_is_refused_by_the_downstream_probe() -> None:
    """The surrogate builds one ladder; Gemma 4 has two layer families.

    Mutation verdict: deleting the ``unsupported_architecture`` call lets the
    stub reach ``attention_modules``, which raises ``AttributeError`` instead
    of the named refusal.
    """

    class _Stub:
        config = {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "head_dim": 256,
                "global_head_dim": 512,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1_000_000.0,
                    },
                    "sliding_attention": {"rope_theta": 10_000.0},
                },
            },
        }

    with pytest.raises(NotImplementedError, match="cannot be measured"):
        D.SurrogateAttention(_Stub())


def test_a_bad_head_width_is_refused_before_any_hook_runs() -> None:
    model, _cfg = _tiny_llama()
    ids = torch.randint(0, 64, (1, 8))
    with (
        pytest.raises(RuntimeError, match="not a multiple"),
        D.SurrogateAttention(model, head_dim=6, deltas=D.observed_distances(8)),
    ):
        model(input_ids=ids)


def test_identity_kernel_on_fused_qkv_equals_a_no_rope_model() -> None:
    """The per-head split is not cosmetic: it must reproduce the real q/k.

    With the identity kernel the hook pair cancels the model's own RoPE, so the
    surrogate NLL must equal the NLL of the same model with RoPE disabled. A
    contiguous-thirds split silently mixes heads 1..H-1 and fails this equality.
    """
    model, _cfg = _tiny_gpt_neox()
    ids = torch.randint(0, 64, (1, 16))
    import transformers.models.gpt_neox.modeling_gpt_neox as neox

    original = neox.apply_rotary_pos_emb
    neox.apply_rotary_pos_emb = lambda q, k, cos, sin, *a, **kw: (q, k)
    try:
        no_rope = D.nll(model, ids)
    finally:
        neox.apply_rotary_pos_emb = original

    with D.SurrogateAttention(
        model, deltas=D.observed_distances(16), kernel="identity"
    ):
        surrogate = D.nll(model, ids)
    assert surrogate == pytest.approx(no_rope, rel=1e-6, abs=1e-6)
