"""Faithful RoPE / YaRN / partial-RoPE reference implementation (numpy, float64).

Every formula in this file is transcribed from an original source. Nothing here is
approximated, and nothing is a hardcoded result. Sources:

RoPE
    Su, Jianlin; Lu, Yu; Pan, Shengfeng; Murtadha, Ahmed; Wen, Bo; Liu, Yunfeng.
    "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021.
    https://arxiv.org/abs/2104.09864
    Inverse frequencies and the split-half rotation convention follow
    HuggingFace `transformers`, `models/llama/modeling_llama.py`
    (`rotate_half`, `compute_default_rope_parameters`).

YaRN
    Peng, Bowen; Quesnelle, Jeffrey; Fan, Honglu; Shippole, Enrico.
    "YaRN: Efficient Context Window Extension of Large Language Models", 2023.
    https://arxiv.org/abs/2309.00071
    The ramp (`find_correction_dim`, `find_correction_range`, `linear_ramp_mask`)
    and the magnitude scaling (`get_mscale`) are transcribed verbatim from the
    authors' own implementation, `scaled_rope/LlamaYaRNScaledRotaryEmbedding.py`
    in https://github.com/jquesnelle/yarn, and agree with
    `transformers.modeling_rope_utils._compute_yarn_parameters`.

Note on a common misstatement
    YaRN does *not* change `base` from 10000 to 500000. Changing the base, or
    uniformly dividing positions by a factor, is plain Position Interpolation.
    YaRN keeps the base and instead ramps per frequency: short-wavelength
    (high-frequency) dimensions keep their original inverse frequency, while
    long-wavelength (low-frequency) dimensions are divided by `scale`, with a
    linear blend in between. It also rescales cos/sin by `mscale`.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "find_correction_dim",
    "find_correction_range",
    "get_mscale",
    "inv_freq",
    "linear_ramp_mask",
    "partial_rope_cos_sin",
    "rotate_half",
    "apply_rope",
    "apply_partial_rope",
    "rope_cos_sin",
    "rotary_angles",
    "yarn_parameters",
    "per_dim_angle",
]


# --------------------------------------------------------------------------
# Plain RoPE
# --------------------------------------------------------------------------


def inv_freq(dim: int, base: float = 10000.0) -> np.ndarray:
    """Inverse frequencies, length ``dim // 2``.

    ``inv_freq[i] = base ** (-2 * i / dim)``, matching
    ``1.0 / (base ** (arange(0, dim, 2) / dim))`` in `transformers`.
    """
    if dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {dim}")
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    return 1.0 / pos_freqs


def rotary_angles(positions: np.ndarray, inv_f: np.ndarray) -> np.ndarray:
    """Per-dimension rotation angle. Returns shape ``(len(positions), dim // 2)``."""
    return np.asarray(positions, dtype=np.float64)[:, None] * inv_f[None, :]


def rope_cos_sin(
    positions: np.ndarray, inv_f: np.ndarray, mscale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin tables of shape ``(len(positions), head_dim)``.

    Uses the split-half convention of `transformers`: the frequency vector is
    duplicated and concatenated, so dimension ``i`` is paired with ``i + dim // 2``.
    That pairing is what :func:`rotate_half` undoes.
    """
    angles = rotary_angles(positions, inv_f)
    emb = np.concatenate([angles, angles], axis=-1)
    return np.cos(emb) * mscale, np.sin(emb) * mscale


def rotate_half(x: np.ndarray) -> np.ndarray:
    """``cat((-x2, x1))`` over the last axis, exactly as `transformers` does."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply RoPE. ``x`` is ``(..., seq, head_dim)``, ``cos``/``sin`` are ``(seq, head_dim)``."""
    return x * cos + rotate_half(x) * sin


def apply_partial_rope(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray, n_rot: int
) -> np.ndarray:
    """Apply partial rotary, pairing within the rotated block.

    ``x`` is ``(..., seq, head_dim)``; ``cos``/``sin`` come from
    :func:`partial_rope_cos_sin` and are ``(seq, head_dim)`` with ``cos = 1, sin = 0``
    on the unrotated tail. Only the first ``n_rot`` channels are transformed, and they
    are paired ``i`` with ``i + n_rot // 2`` -- the GPT-NeoX layout, in which the
    rotated block is itself split in half.

    This is orthogonal for every ``n_rot``, so ``|R x| == |x|`` to machine precision
    just as for full RoPE.
    """
    half = n_rot // 2
    rotated = x[..., :n_rot]
    x1, x2 = rotated[..., :half], rotated[..., half:]
    cos_r, sin_r = cos[..., :n_rot], sin[..., :n_rot]
    c1, c2 = cos_r[..., :half], cos_r[..., half:]
    s1, s2 = sin_r[..., :half], sin_r[..., half:]
    out1 = x1 * c1 - x2 * s1
    out2 = x2 * c2 + x1 * s2
    return np.concatenate([out1, out2, x[..., n_rot:]], axis=-1)


def partial_rope_cos_sin(
    positions: np.ndarray, inv_f: np.ndarray, n_rot: int, mscale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin tables for partial rotary: only the first ``n_rot`` dimensions rotate.

    The remaining ``head_dim - n_rot`` dimensions receive ``cos = 1, sin = 0`` and are
    therefore left completely unrotated. This is the "partial rotary" arrangement used
    by GPT-NeoX and by models such as ``EleutherAI/pythia-160m`` (``rotary_pct = 0.25``):
    a fraction ``n_rot / head_dim`` of the channels carry position, the rest carry
    content only.

    **The pairing matters.** Apply these tables with :func:`apply_partial_rope`, which
    pairs channel ``i`` with channel ``i + n_rot // 2`` *within the rotated block*.
    Applying them with :func:`apply_rope` instead pairs across the whole head
    (``i`` with ``i + head_dim // 2``), which for ``n_rot < head_dim`` leaves every
    touched pair half rotated and half not. That variant is not orthogonal and is not
    what any published model does; it is a bug, not a convention.

    With the correct pairing partial rotary *is* orthogonal, exactly like full RoPE, so
    the magnitude gate survives it and ``n_rot`` channels change angle without changing
    norm.
    """
    dim = inv_f.shape[0] * 2
    if n_rot % 2 != 0:
        raise ValueError(f"n_rot must be even, got {n_rot}")
    if not 0 <= n_rot <= dim:
        raise ValueError(f"n_rot must be in [0, {dim}], got {n_rot}")
    if n_rot == 0:
        ones = np.ones((len(positions), dim), dtype=np.float64)
        return ones, np.zeros((len(positions), dim), dtype=np.float64)
    cos, sin = rope_cos_sin(positions, inv_f[: n_rot // 2], mscale)
    pad = dim - n_rot
    return (
        np.concatenate([cos, np.ones((len(positions), pad))], axis=-1),
        np.concatenate([sin, np.zeros((len(positions), pad))], axis=-1),
    )


# --------------------------------------------------------------------------
# YaRN
# --------------------------------------------------------------------------


def find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    """Inverse-dimension formula: which dim index gives ``num_rotations`` rotations.

    Verbatim from `jquesnelle/yarn` and `transformers`.
    """
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
    truncate: bool = True,
) -> tuple[float, float]:
    """Bounds of the ramp, in units of *pair index* (0 .. dim//2)."""
    low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0.0), min(high, dim - 1.0)


def linear_ramp_mask(minimum: float, maximum: float, dim: int) -> np.ndarray:
    """``clamp((arange(dim) - min) / (max - min), 0, 1)``, verbatim from upstream."""
    if minimum == maximum:
        maximum += 0.001  # Prevent singularity
    linear = (np.arange(dim, dtype=np.float64) - minimum) / (maximum - minimum)
    return np.clip(linear, 0.0, 1.0)


def get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """YaRN's magnitude (temperature) scaling: ``0.1 * log(scale) + 1``."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_parameters(
    dim: int,
    base: float,
    scale: float,
    original_max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    extrapolation_factor: float = 1.0,
    attn_factor: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Return ``(inv_freq, mscale)`` for YaRN.

    ``base`` is *not* modified. High-frequency (short-wavelength) channels keep their
    original inverse frequency; low-frequency (long-wavelength) channels are divided
    by ``scale``; the two are blended linearly in between.
    """
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scale * pos_freqs)

    low, high = find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings
    )
    ramp = linear_ramp_mask(low, high, dim // 2)
    inv_freq_mask = (1.0 - ramp) * extrapolation_factor
    freqs = inv_freq_interpolation * (1.0 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    return freqs, float(get_mscale(scale) * attn_factor)


def per_dim_angle(delta: float | np.ndarray, inv_f: np.ndarray) -> np.ndarray:
    """Rotation angle ``D_k = delta * inv_freq_k`` for every frequency channel.

    This is the quantity that decides whether ``cos``/``sin`` can be linearized.
    """
    return np.asarray(delta, dtype=np.float64)[:, None] * inv_f[None, :]
