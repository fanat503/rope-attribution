"""Exact, per-distance query-key attribution on real sparse features.

The method paper for QK attribution (Kamath et al., Transformer Circuits, 2025)
rewrites ``q . k`` as a sum over (query feature, key feature) pairs. It says
explicitly that RoPE is out of scope, because rotation makes the linear map
depend on the distance. This module does the RoPE case, and the point is that
nothing is approximated:

    score(m, n) = <q_m, R_{n-m} k_n>
                = sum_ij f_i(m) f_j(n) <a_i(m), R_delta b_j(n)> + (bias + tail)

where ``a_i(m)`` is feature ``i``'s share of the query vector and ``b_j(n)`` the
same for the key. Because the attention input is a LayerNorm of the residual
stream, the share is defined by the *exact* affine map at that token:

    h = LayerNorm(x),   h = J(x) x + c(x)   with J the Jacobian at x,

which holds identically at ``x`` for any differentiable map. Writing
``x = b_dec + sum_i f_i W_dec[i] + r`` from a sparse dictionary then distributes
``h`` over features exactly: the residual ``r``, the dictionary bias and the
LayerNorm constant form an explicit "base" bucket, and the sum of all buckets is
``h`` to machine precision. Nothing is dropped and no first-order approximation
of the network is used; the Jacobian only chooses how to split a quantity that
is already fixed.

What is measured
----------------
Two different objects, because they answer two different questions and were
previously conflated.

* **A per-distance table.** For a distance ``delta``, a feature's profile is its
  mean contribution over content pairs ``(m, n = m - delta)`` sampled from one
  real passage. This is what an attribution table indexed by distance contains,
  and it supports two claims: concentration (how much of the score the top-``k``
  features and the base bucket carry) and the cost of flattening the table to a
  single scalar per feature.
* **A content-fixed sweep.** Hold one ``(query, key)`` content pair fixed, vary
  ``delta`` over the grid the paper's synthetic measurement uses, and ask
  whether the least-squares-optimal constant has the wrong sign. Averaging over
  content first smooths the oscillation, so the table understates this cost; the
  sweep is the number comparable to the tens-of-percent synthetic rate.

Under partial RoPE the two channels classes behave differently: the unrotated
tail contributes a distance-independent term that a position-free scalar
represents perfectly, and it can dominate a feature's total. Every feature-level
number is therefore reported twice, once over the total contribution and once
over the rotary channels alone, with the rotated channels of the key zeroed
rather than the rotation dropped. The gap between the two is the paper's
structural claim, measured.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import real_model as RM
from . import rope as R

__all__ = [
    "DEFAULT_DELTAS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_SAE_LAYER",
    "DEFAULT_TOP_K",
    "MATERIALITY",
    "SparseDictionary",
    "attribution_terms",
    "distance_grid",
    "head_projections",
    "layernorm_affine",
    "layernorm_jvp",
    "measure_head",
    "measure_model",
    "profile_metrics",
    "run_all",
    "sample_pairs",
    "write_results",
]

#: The SAE hook this module uses. ``blocks.3.hook_resid_post`` is the input to
#: attention layer 4 of a 6-layer checkpoint, i.e. the middle of the network.
#: Chosen before any attribution number was computed, and not swept.
DEFAULT_SAE_LAYER = 3
DEFAULT_MODEL_ID = "EleutherAI/pythia-70m"

#: Features kept per token before the tail is folded into the base bucket. The
#: decomposition is exact for any value; this one bounds the pair matrix.
DEFAULT_TOP_K = 32

#: Content pairs sampled per distance. Each costs one ``(k, d) x (d, k)`` block,
#: so this is the runtime knob.
DEFAULT_PAIRS_PER_DELTA = 24

#: Distances reported. Log-spaced so the short range, where most attention mass
#: lives, is not crowded out by the long range.
DEFAULT_DELTAS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64, 91, 128)

#: The distance grid for the content-fixed sweep. This is the grid the paper's
#: synthetic measurement uses, so the two sign-error rates answer the same
#: question: fix one (query, key) content pair, vary the distance, and ask how
#: often the least-squares constant has the wrong sign.
SWEEP_DELTAS: tuple[int, ...] = tuple(
    int(v) for v in np.unique(np.rint(np.geomspace(1, 8192, 54)).astype(np.int64))
)

#: Content pairs used by the sweep. Each contributes one profile per feature.
DEFAULT_SWEEP_PAIRS = 24

#: A sign comparison is only meaningful where the term carries real signal. Same
#: convention as ``usefulness.measure_pair_cost``: 5% of that feature's own peak
#: across the distance grid.
MATERIALITY = 0.05

#: One original passage, written for this measurement. It is deliberately not a
#: quotation, so it can be committed with the repository.
DEFAULT_TEXT = (
    "Photosynthesis converts light energy into chemical energy inside the "
    "chloroplasts of plant cells. Chlorophyll absorbs red and blue wavelengths "
    "while reflecting green light, which is why healthy leaves appear green. "
    "The absorbed energy drives two coupled stages: the light reactions split "
    "water molecules and release oxygen, and the Calvin cycle fixes carbon "
    "dioxide into sugars. Because the two stages operate on different "
    "timescales, a plant can adjust to changing light without rebuilding its "
    "molecular machinery.\n\n"
    "The printing press spread through Europe in the second half of the "
    "fifteenth century. A single workshop could produce hundreds of identical "
    "copies of a book in the time a scribe needed for one, and the price of "
    "printed matter fell sharply. Printers clustered in cities with paper "
    "mills and reliable trade routes, and their output fed the growing market "
    "for calendars, pamphlets, and scientific tables. Historians still argue "
    "about how quickly the new medium changed reading habits, but the "
    "direction of the change is not in dispute.\n\n"
    "def merge_intervals(intervals):\n"
    "    if not intervals:\n"
    "        return []\n"
    "    intervals = sorted(intervals, key=lambda pair: pair[0])\n"
    "    merged = [list(intervals[0])]\n"
    "    for start, end in intervals[1:]:\n"
    "        if start <= merged[-1][1]:\n"
    "            merged[-1][1] = max(merged[-1][1], end)\n"
    "        else:\n"
    "            merged.append([start, end])\n"
    "    return [tuple(pair) for pair in merged]\n"
)


# --------------------------------------------------------------------------
# LayerNorm, used exactly rather than approximately
# --------------------------------------------------------------------------


def layernorm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float
) -> np.ndarray:
    """``weight * (x - mean) / sqrt(var + eps) + bias`` over the last axis."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, ddof=0, keepdims=True)
    return weight * (x - mu) / np.sqrt(var + eps) + bias


def layernorm_jvp(
    x: np.ndarray,
    v: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Jacobian-vector product ``J v`` of :func:`layernorm` at ``x``.

    Derived by hand and checked against ``torch.autograd.functional.jvp`` on a
    real checkpoint (agreement ~3e-6, the float32 source precision):

        J v = weight * [ (v - mean(v)) / sigma
                         - (x - mean(x)) * <x - mean(x), v> / (d * sigma**3) ]

    ``bias`` does not enter the Jacobian, only the constant.
    """
    d = x.shape[-1]
    mu = x.mean(axis=-1, keepdims=True)
    sigma = np.sqrt(x.var(axis=-1, ddof=0, keepdims=True) + eps)
    centred = x - mu
    scaled = (v - v.mean(axis=-1, keepdims=True)) / sigma
    along = (centred * v).sum(axis=-1, keepdims=True) / (d * sigma**3)
    return weight * (scaled - centred * along)


def layernorm_affine(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(h, c)`` with ``h = J x + c`` holding exactly at ``x``."""
    h = layernorm(x, weight, bias, eps)
    c = h - layernorm_jvp(x, x, weight, bias, eps)
    return h, c


# --------------------------------------------------------------------------
# Sparse dictionary
# --------------------------------------------------------------------------


@dataclass
class SparseDictionary:
    """A ReLU dictionary with a decoder bias, in the ``sae_lens`` convention.

    ``apply_b_dec_to_input=True`` means the bias is subtracted *before* the
    encoder, so ``x_hat = f W_dec + b_dec`` and ``f = relu((x - b_dec) W_enc +
    b_enc)``. Both conventions exist in the wild; getting this one wrong produces
    a plausible-looking decomposition of the wrong vector, so the loader records
    which one it assumed.
    """

    W_enc: np.ndarray  # (d_model, d_sae)
    b_enc: np.ndarray  # (d_sae,)
    W_dec: np.ndarray  # (d_sae, d_model)
    b_dec: np.ndarray  # (d_model,)

    def __post_init__(self) -> None:
        if self.W_enc.shape[1] != self.W_dec.shape[0]:
            raise ValueError(
                f"W_enc {self.W_enc.shape} and W_dec {self.W_dec.shape} disagree "
                "on the dictionary width"
            )
        if self.W_enc.shape[0] != self.b_dec.shape[0]:
            raise ValueError("W_enc and b_dec disagree on the model width")

    @property
    def n_features(self) -> int:
        return int(self.W_dec.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_dec.shape[1])

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.maximum((x - self.b_dec) @ self.W_enc + self.b_enc, 0.0)

    def decode(self, f: np.ndarray) -> np.ndarray:
        return f @ self.W_dec + self.b_dec

    @classmethod
    def load_safetensors(cls, path: str | Path) -> SparseDictionary:
        """Load the four tensors from a ``safetensors`` file, no unpickling."""
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as handle:
            keys = set(handle.keys())
            missing = {"W_enc", "b_enc", "W_dec", "b_dec"} - keys
            if missing:
                raise ValueError(f"{path}: missing tensors {sorted(missing)}")
            arrays = {
                name: handle.get_tensor(name).astype(np.float64)
                for name in ("W_enc", "b_enc", "W_dec", "b_dec")
            }
        return cls(**arrays)

    @classmethod
    def random(
        cls, d_model: int, n_features: int, seed: int = 0
    ) -> SparseDictionary:
        """A deterministic dictionary for tests; never used for a reported number."""
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        return cls(
            W_enc=rng.standard_normal((d_model, n_features)) * scale,
            b_enc=np.zeros(n_features),
            W_dec=rng.standard_normal((n_features, d_model)) * scale,
            b_dec=rng.standard_normal(d_model) * 0.1,
        )


# --------------------------------------------------------------------------
# Tokens as buckets of features
# --------------------------------------------------------------------------


@dataclass
class TokenBuckets:
    """One token, split into its top features plus an exact base bucket.

    ``feature_q[i]`` is feature ``i``'s share of the query vector, already
    projected through ``W_Q``; ``feature_k`` the same for the key. ``base_q`` and
    ``base_k`` hold everything else, and ``base_q + sum(feature_q) == q`` exactly.
    """

    indices: np.ndarray  # (k,) feature ids, descending activation
    activations: np.ndarray  # (k,)
    feature_q: np.ndarray  # (k, head_dim)
    feature_k: np.ndarray  # (k, head_dim)
    base_q: np.ndarray  # (head_dim,)
    base_k: np.ndarray  # (head_dim,)


def head_projections(
    attn: Any, hidden_size: int, n_heads: int, head_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-head ``(W_q, b_q, W_k, b_k)`` from a fused or split projection.

    GPT-NeoX interleaves ``(q, k, v)`` **per head**, so the flat rows of head
    ``h`` are ``h*3*head_dim .. h*3*head_dim + head_dim`` for q and the next
    ``head_dim`` for k. Slicing the flat output into three contiguous thirds
    works for head 0 and mixes heads afterwards; that bug is silent because the
    result still has the right shape.
    """
    if hasattr(attn, "query_key_value"):
        weight = attn.query_key_value.weight.detach().numpy().astype(np.float64)
        bias = attn.query_key_value.bias.detach().numpy().astype(np.float64)
        if weight.shape[0] != 3 * hidden_size:
            raise ValueError(
                f"fused QKV has {weight.shape[0]} rows, expected 3 * {hidden_size}"
            )
        rows = []
        for head in range(n_heads):
            base = head * 3 * head_dim
            rows.append(np.arange(base, base + head_dim))
            rows.append(np.arange(base + head_dim, base + 2 * head_dim))
        q_rows = np.stack(rows[0::2])
        k_rows = np.stack(rows[1::2])
        return weight[q_rows], bias[q_rows], weight[k_rows], bias[k_rows]

    if hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
        wq = attn.q_proj.weight.detach().numpy().astype(np.float64)
        wk = attn.k_proj.weight.detach().numpy().astype(np.float64)
        bq = attn.q_proj.bias.detach().numpy().astype(np.float64)
        bk = attn.k_proj.bias.detach().numpy().astype(np.float64)
        shape = (n_heads, head_dim, hidden_size)
        return wq.reshape(shape), bq.reshape(n_heads, head_dim), wk.reshape(shape), bk.reshape(
            n_heads, head_dim
        )

    raise NotImplementedError(
        f"{type(attn).__name__} has neither a fused query_key_value nor q_proj/k_proj"
    )


def token_buckets(
    x: np.ndarray,
    f: np.ndarray,
    dictionary: SparseDictionary,
    ln_weight: np.ndarray,
    ln_bias: np.ndarray,
    ln_eps: float,
    w_q: np.ndarray,
    b_q: np.ndarray,
    w_k: np.ndarray,
    b_k: np.ndarray,
    top_k: int,
) -> TokenBuckets:
    """Split one token's query and key vectors into feature shares plus a base.

    ``x`` is the residual vector, ``f`` its dictionary activations. The split is
    exact: ``base_q + sum(feature_q) == W_q LayerNorm(x) + b_q``.
    """
    k = min(top_k, f.size)
    order = np.argpartition(-f, k - 1)[:k]
    order = order[np.argsort(-f[order])]
    active = f[order]

    # h = J x + c exactly, so the features and the base below are an identity.
    _h, c = layernorm_affine(x, ln_weight, ln_bias, ln_eps)

    decoder_rows = dictionary.W_dec[order] * active[:, None]  # (k, d_model)
    feature_h = layernorm_jvp(
        x, decoder_rows, ln_weight, ln_bias, ln_eps
    )  # (k, d_model)
    # Everything the top-k does not account for, including the dictionary bias
    # and the reconstruction residual.
    tail = x - decoder_rows.sum(axis=0)
    base_h = layernorm_jvp(x, tail, ln_weight, ln_bias, ln_eps) + c

    return TokenBuckets(
        indices=order.astype(np.int64),
        activations=active,
        feature_q=feature_h @ w_q.T,
        feature_k=feature_h @ w_k.T,
        base_q=base_h @ w_q.T + b_q,
        base_k=base_h @ w_k.T + b_k,
    )


# --------------------------------------------------------------------------
# Rotated dot products
# --------------------------------------------------------------------------


def rotated_dot(
    a: np.ndarray, b: np.ndarray, delta: int, freqs: np.ndarray, rotary_dim: int
) -> np.ndarray:
    """``a @ R_delta b^T`` for stacks of vectors, shape ``(len(a), len(b))``.

    Same aligned/crossed algebra as ``attention_cost``: for rotary pair ``k``
    with channels ``(k, k + half)``,

        A_k = a1 b1 + a2 b2,   B_k = a2 b1 - a1 b2,
        contribution = A_k cos(delta th_k) + B_k sin(delta th_k),

    and the unrotated tail is an ordinary dot product.
    """
    half = rotary_dim // 2
    a1 = a[:, None, :half]
    a2 = a[:, None, half:rotary_dim]
    b1 = b[None, :, :half]
    b2 = b[None, :, half:rotary_dim]
    aligned = a1 * b1 + a2 * b2  # (na, nb, half)
    crossed = a2 * b1 - a1 * b2
    angles = delta * freqs
    out = (aligned * np.cos(angles)).sum(axis=-1) + (crossed * np.sin(angles)).sum(axis=-1)
    if rotary_dim < a.shape[1]:
        out = out + a[:, rotary_dim:] @ b[:, rotary_dim:].T
    return out


@dataclass
class AttributionTerms:
    """Every term of one ``(m, n)`` score. All shapes are per selected feature."""

    score: float
    pairs: np.ndarray  # (k_m, k_n) feature-feature
    query_features: np.ndarray  # (k_m,) feature i against the whole key
    query_features_rotary: np.ndarray  # (k_m,) the same, unrotated channels removed
    key_features: np.ndarray  # (k_n,) whole query against feature j
    base: float  # base_q against the whole key
    indices_m: np.ndarray
    indices_n: np.ndarray

    @property
    def reconstructed(self) -> float:
        """``sum(query_features) + base``: the whole score, by construction.

        The query vector is ``base_q + sum(feature_q)``, so splitting the score
        along that axis is an identity. ``key_features`` splits the *other* axis
        and is not added here; adding it would double-count the feature-feature
        block.
        """
        return float(self.query_features.sum() + self.base)


def attribution_terms(
    buckets_m: TokenBuckets,
    buckets_n: TokenBuckets,
    delta: int,
    freqs: np.ndarray,
    rotary_dim: int,
) -> AttributionTerms:
    """Exact decomposition of ``<q_m, R_delta k_n>`` for one pair of tokens.

    ``query_features_rotary`` is the contribution carried by the rotary channels
    alone: the key's unrotated tail is zeroed before the dot product, which
    removes the distance-independent part of the term without touching the
    rotation. Under partial RoPE the two differ, and the difference is the whole
    point --- the paper's structural claim is about the rotary subspace, while a
    practitioner reads off the total.
    """
    q_m = buckets_m.base_q + buckets_m.feature_q.sum(axis=0)
    k_n = buckets_n.base_k + buckets_n.feature_k.sum(axis=0)
    k_n_rotary = k_n.copy()
    k_n_rotary[rotary_dim:] = 0.0

    pairs = rotated_dot(
        buckets_m.feature_q, buckets_n.feature_k, delta, freqs, rotary_dim
    )
    query_features = rotated_dot(
        buckets_m.feature_q, k_n[None, :], delta, freqs, rotary_dim
    )[:, 0]
    query_features_rotary = rotated_dot(
        buckets_m.feature_q, k_n_rotary[None, :], delta, freqs, rotary_dim
    )[:, 0]
    key_features = rotated_dot(
        q_m[None, :], buckets_n.feature_k, delta, freqs, rotary_dim
    )[0]
    base = float(
        rotated_dot(
            buckets_m.base_q[None, :], k_n[None, :], delta, freqs, rotary_dim
        )[0, 0]
    )
    score = float(
        rotated_dot(q_m[None, :], k_n[None, :], delta, freqs, rotary_dim)[0, 0]
    )
    return AttributionTerms(
        score=score,
        pairs=pairs,
        query_features=query_features,
        query_features_rotary=query_features_rotary,
        key_features=key_features,
        base=base,
        indices_m=buckets_m.indices,
        indices_n=buckets_n.indices,
    )


# --------------------------------------------------------------------------
# Sampling and aggregation
# --------------------------------------------------------------------------


def distance_grid(max_delta: int, requested: Sequence[int] = DEFAULT_DELTAS) -> np.ndarray:
    """The requested distances that the passage can actually realize."""
    kept = [int(d) for d in requested if 1 <= int(d) <= max_delta]
    return np.unique(np.array(kept, dtype=np.int64))


def sample_pairs(
    n_tokens: int, delta: int, n_pairs: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_pairs, 2)`` of ``(query, key)`` positions at exactly ``delta``.

    Queries run over ``delta .. T-1``, so every sample is causal and every key is
    a real token of the passage. Sampling without replacement when the range is
    small enough, so a short passage does not report the same pair twice.
    """
    candidates = np.arange(delta, n_tokens, dtype=np.int64)
    if candidates.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    take = min(n_pairs, candidates.size)
    queries = rng.choice(candidates, size=take, replace=False)
    return np.stack([queries, queries - delta], axis=1)


def _sweep_profiles(
    buckets: Sequence[TokenBuckets],
    freqs: np.ndarray,
    rotary_dim: int,
    sweep_deltas: np.ndarray,
    n_content_pairs: int,
    rng: np.random.Generator,
    d_sae: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Content-fixed distance profiles, at three levels of detail.

    The realized-distance table averages over content at each distance, which
    smooths oscillation and therefore understates the cost of a position-free
    scalar. This version does what the paper's synthetic measurement does: hold
    the two tokens fixed, sweep the distance, and report one profile per unit.

    Units are ``content_index * d_sae + feature`` for the two feature views and
    ``content_index * d_sae**2 + i * d_sae + j`` for feature pairs.
    """
    n_tokens = len(buckets)
    queries = rng.integers(1, n_tokens, size=n_content_pairs)
    keys = (rng.random(n_content_pairs) * queries).astype(np.int64)

    acc = {
        "feature": ([], [], []),
        "feature_rotary": ([], [], []),
        "pair": ([], [], []),
    }

    def add(kind: str, delta_index: int, units: np.ndarray, values: np.ndarray) -> None:
        delta_list, unit_list, value_list = acc[kind]
        delta_list.append(np.full(units.size, delta_index, dtype=np.int64))
        unit_list.append(units)
        value_list.append(values)

    for content_index, (m, n) in enumerate(zip(queries, keys, strict=True)):
        if m == n:
            continue
        for delta_index, delta in enumerate(sweep_deltas):
            # ``delta`` is a distance, and the key sits *behind* the query, so
            # the angle the model rotates by is ``n - m = -delta``. Feeding
            # ``+delta`` would sweep the anti-causal rotation and mirror every
            # sine term of the profile.
            terms = attribution_terms(
                buckets[int(m)], buckets[int(n)], -int(delta), freqs, rotary_dim
            )
            feature_units = content_index * d_sae + terms.indices_m
            add("feature", delta_index, feature_units, terms.query_features)
            add("feature_rotary", delta_index, feature_units, terms.query_features_rotary)
            pair_units = (
                content_index * d_sae * d_sae
                + terms.indices_m[:, None] * d_sae
                + terms.indices_n[None, :]
            ).ravel()
            add("pair", delta_index, pair_units, terms.pairs.ravel())

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind, (delta_list, unit_list, value_list) in acc.items():
        if not delta_list:
            out[kind] = (
                np.zeros(0, np.int64),
                np.zeros((0, sweep_deltas.size), dtype=np.float64),
            )
            continue
        out[kind] = _group_mean(
            np.concatenate(delta_list),
            np.concatenate(unit_list),
            np.concatenate(value_list),
            sweep_deltas.size,
        )
    return out


def measure_head(
    buckets: Sequence[TokenBuckets],
    freqs: np.ndarray,
    rotary_dim: int,
    deltas: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
    d_sae: int,
    sweep_deltas: np.ndarray | None = None,
    n_content_pairs: int = DEFAULT_SWEEP_PAIRS,
) -> dict[str, Any]:
    """The realized-distance feature profiles, plus the content-fixed sweep.

    ``feature_profile`` averages over content at each realized distance: it is
    what a per-distance attribution *table* would contain. The sweep profiles
    hold content fixed instead, which is the only way to see a single feature's
    dependence on distance, and is the object comparable to the paper's
    synthetic sign-error rate.
    """
    n_tokens = len(buckets)
    delta_idx_all: list[np.ndarray] = []
    query_ids_all: list[np.ndarray] = []
    query_values: list[np.ndarray] = []
    query_rotary_values: list[np.ndarray] = []
    exactness = 0.0
    scale = 0.0
    base_share: list[float] = []
    top_k_share: list[float] = []

    for delta_index, delta in enumerate(deltas):
        pairs = sample_pairs(n_tokens, int(delta), n_pairs, rng)
        for m, n in pairs:
            # The pair is (query, key) with the key ``delta`` positions behind
            # it, so the model's rotation angle is ``n - m = -delta``.
            terms = attribution_terms(
                buckets[m], buckets[n], int(n - m), freqs, rotary_dim
            )
            exactness = max(
                exactness,
                abs(terms.reconstructed - terms.score) / max(abs(terms.score), 1.0),
            )
            scale = max(scale, abs(terms.score))

            delta_idx_all.append(
                np.full(terms.indices_m.size, delta_index, dtype=np.int64)
            )
            query_ids_all.append(terms.indices_m)
            query_values.append(terms.query_features)
            query_rotary_values.append(terms.query_features_rotary)

            base_share.append(abs(terms.base) / max(abs(terms.score), 1e-12))
            covered = abs(terms.query_features).sum()
            top_k_share.append(covered / max(abs(terms.score), 1e-12))

    feature_ids = np.zeros(0, np.int64)
    feature_profile = np.zeros((0, deltas.size))
    feature_rotary_profile = np.zeros((0, deltas.size))
    if delta_idx_all:
        delta_idx = np.concatenate(delta_idx_all)
        ids_flat = np.concatenate(query_ids_all)
        feature_ids, feature_profile = _group_mean(
            delta_idx, ids_flat, np.concatenate(query_values), deltas.size
        )
        _, feature_rotary_profile = _group_mean(
            delta_idx, ids_flat, np.concatenate(query_rotary_values), deltas.size
        )

    sweep: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if sweep_deltas is not None and sweep_deltas.size:
        sweep = _sweep_profiles(
            buckets, freqs, rotary_dim, sweep_deltas, n_content_pairs, rng, d_sae
        )

    return {
        "n_tokens": n_tokens,
        "n_deltas": int(deltas.size),
        "deltas": [int(d) for d in deltas],
        "max_score_magnitude": float(scale),
        "max_relative_exactness_error": float(exactness),
        "base_share_mean": float(np.mean(base_share)) if base_share else float("nan"),
        "top_k_share_mean": float(np.mean(top_k_share)) if top_k_share else float("nan"),
        "feature_ids": feature_ids,
        "feature_profile": feature_profile,
        "feature_rotary_profile": feature_rotary_profile,
        "sweep": sweep,
    }


def _group_mean(
    delta_index: np.ndarray,
    ids: np.ndarray,
    values: np.ndarray,
    n_deltas: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` per ``(id, delta)`` cell, as a compact table.

    Returns ``(unique_ids, profiles)`` with ``profiles`` of shape
    ``(len(unique_ids), n_deltas)`` and NaN where a cell was never sampled. The
    table is compact in the ids rather than dense over the dictionary, because
    only the top-``k`` features of each token can ever appear.
    """
    if ids.size == 0:
        return np.zeros(0, np.int64), np.zeros((0, n_deltas), dtype=np.float64)
    unique_ids, compact = np.unique(ids, return_inverse=True)
    key = delta_index * unique_ids.size + compact
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    sums = np.add.reduceat(sorted_values, starts)
    counts = np.diff(np.r_[starts, sorted_key.size])
    means = sums / counts

    flat_ids = sorted_key[starts]
    cell_delta = flat_ids // unique_ids.size
    cell_unit = flat_ids % unique_ids.size
    profiles = np.full((unique_ids.size, n_deltas), np.nan, dtype=np.float64)
    profiles[cell_unit, cell_delta] = means
    return unique_ids, profiles


def profile_metrics(
    profiles: np.ndarray, materiality: float = MATERIALITY
) -> dict[str, Any]:
    """How badly a position-free scalar misrepresents a set of profiles.

    ``profiles`` is ``(units, distances)`` with NaN for missing cells. The
    scalar is the mean over the distance grid --- the least-squares-optimal
    constant, i.e. the strongest possible position-free table, not a strawman.
    A sign disagreement counts only where the exact value carries at least
    ``materiality`` of that unit's own peak, and only where the surrogate is not
    itself zero: a summary that predicts "no contribution" is a different
    failure from one that predicts the opposite contribution.
    """
    if profiles.size == 0:
        return {
            "n_units": 0,
            "n_material_cells": 0,
            "sign_flip_fraction": None,
            "rel_err_median": None,
            "top1_agreement": None,
        }

    finite = np.isfinite(profiles)
    filled = np.where(finite, profiles, 0.0)
    with np.errstate(invalid="ignore"):
        surrogate = np.nanmean(profiles, axis=1)
    peak = np.abs(filled).max(axis=1)
    material = np.abs(filled) > (materiality * peak)[:, None]
    judged = material & finite & np.isfinite(surrogate)[:, None] & (surrogate != 0.0)[:, None]

    exact_sign = np.sign(filled)
    surrogate_sign = np.sign(np.where(np.isfinite(surrogate), surrogate, 0.0))
    flips = int(np.sum(judged & (exact_sign != surrogate_sign[:, None])))
    cells = int(judged.sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        error = np.abs(np.where(judged, filled - np.where(np.isfinite(surrogate), surrogate, 0.0)[:, None], np.nan))
        relative = error / (peak[:, None] + 1e-300)
    valid = np.isfinite(relative)

    top1_agree = 0
    top1_total = 0
    if np.any(finite):
        # The ranking question is about the entries a reader would actually use,
        # so compare the top-20 units by magnitude. Over the full set of units,
        # the argmax is a near-tie between cells that are numerically zero and
        # the rate measures floating-point noise rather than the surrogate.
        magnitude = np.nanmean(np.abs(profiles), axis=1)
        candidates = np.argsort(-magnitude)[:20]
        surrogate_abs = np.abs(np.where(np.isfinite(surrogate), surrogate, 0.0))
        for d in range(profiles.shape[1]):
            column = np.abs(filled[candidates, d])
            if not np.any(finite[candidates, d]):
                continue
            top1_total += 1
            if int(candidates[np.argmax(column)]) == int(np.argmax(surrogate_abs)):
                top1_agree += 1

    return {
        "n_units": int(profiles.shape[0]),
        "n_material_cells": cells,
        "sign_flip_fraction": (flips / cells) if cells else None,
        "rel_err_median": float(np.median(relative[valid])) if np.any(valid) else None,
        "top1_agreement": (top1_agree / top1_total) if top1_total else None,
    }


def top_profiles(
    ids: np.ndarray, profiles: np.ndarray, n: int = 12, d_sae: int | None = None
) -> list[dict[str, Any]]:
    """The ``n`` highest-magnitude profiles, as JSON-friendly rows."""
    if profiles.size == 0:
        return []
    magnitude = np.nanmean(np.abs(profiles), axis=1)
    order = np.argsort(-magnitude)[:n]
    rows = []
    for index in order:
        row = {"mean_abs": float(magnitude[index])}
        identifier = int(ids[index])
        if d_sae is not None:
            row["feature_i"] = identifier // d_sae
            row["feature_j"] = identifier % d_sae
        else:
            row["feature"] = identifier
        row["profile"] = [
            None if not np.isfinite(v) else float(v) for v in profiles[index]
        ]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Model-level driver
# --------------------------------------------------------------------------


def _attention_index(n_blocks: int, sae_layer: int) -> int:
    """The attention block a dictionary at ``sae_layer`` explains.

    The SAE encodes the output of block ``sae_layer``, i.e.
    ``hidden_states[sae_layer + 1]``, which is exactly the input of block
    ``sae_layer + 1``'s attention. The two indices therefore differ by one. A
    dictionary on the last block has no block after it; that is an error, not
    a reason to fall back to the default layer.
    """
    index = int(sae_layer) + 1
    if int(sae_layer) < 0 or index >= int(n_blocks):
        raise ValueError(
            f"sae_layer must be in [0, {int(n_blocks) - 2}] for a model with "
            f"{int(n_blocks)} blocks (block sae_layer + 1 is attributed), "
            f"not {sae_layer!r}"
        )
    return index


def _forward(
    model: Any,
    tokenizer: Any,
    text: str,
    attention_layer: int = DEFAULT_SAE_LAYER + 1,
) -> dict[str, Any]:
    """One forward pass, capturing the residual, the attention input and probs."""
    torch = RM._require_torch()
    if not hasattr(model, "gpt_neox"):
        raise NotImplementedError(
            f"{type(model).__name__} is not a GPT-NeoX model; the layer addressing "
            "and the fused per-head QKV layout in this module are GPT-NeoX specific"
        )
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    captured: dict[str, Any] = {}
    handles = [
        model.gpt_neox.layers[attention_layer].attention.register_forward_hook(
            lambda module, args, out: captured.__setitem__("attn_in", args[0])
        )
    ]
    try:
        with torch.no_grad():
            out = model(
                input_ids=ids,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    return {
        "ids": ids,
        "hidden_states": out.hidden_states,
        "attentions": out.attentions,
        "attn_in": captured["attn_in"],
    }


def measure_model(
    model_id: str = DEFAULT_MODEL_ID,
    sae_path: str | Path | None = None,
    text: str = DEFAULT_TEXT,
    top_k: int = DEFAULT_TOP_K,
    n_pairs: int = DEFAULT_PAIRS_PER_DELTA,
    deltas: Sequence[int] = DEFAULT_DELTAS,
    seed: int = 0,
    sae_layer: int = DEFAULT_SAE_LAYER,
) -> dict[str, Any]:
    """Full attribution sweep on one checkpoint and one passage.

    ``sae_layer`` selects which public dictionary is applied; the attention
    block attributed is the one after it (see ``_attention_index``).
    """
    if sae_path is None:
        raise ValueError(
            "measure_model needs a sparse dictionary; pass the safetensors path "
            "of the public SAE (see results/qk_attribution.json provenance)"
        )
    dictionary = SparseDictionary.load_safetensors(sae_path)
    loaded = RM.load_model(model_id, seed=seed)
    model, tokenizer = loaded.model, loaded.tokenizer
    attn_index = _attention_index(len(model.gpt_neox.layers), sae_layer)

    run = _forward(model, tokenizer, text, attn_index)
    n_tokens = int(run["ids"].shape[1])
    if n_tokens < 8:
        raise ValueError(f"the passage tokenized to {n_tokens} tokens, too few to bin")

    # The SAE hook is the output of block ``sae_layer``, i.e.
    # hidden_states[sae_layer + 1].
    x = run["hidden_states"][int(sae_layer) + 1][0].numpy().astype(np.float64)
    if x.shape[1] != dictionary.d_model:
        raise ValueError(
            f"the checkpoint has width {x.shape[1]} but the dictionary was trained "
            f"on width {dictionary.d_model}"
        )

    attn = RM.attention_modules(model)[attn_index]
    ln = model.gpt_neox.layers[attn_index].input_layernorm
    ln_weight = ln.weight.detach().numpy().astype(np.float64)
    ln_bias = ln.bias.detach().numpy().astype(np.float64)

    # The residual must be exactly what the attention layer saw, or the
    # attribution is of a different vector than the one that produced the score.
    attn_in = run["attn_in"][0].numpy().astype(np.float64)
    ln_residual = float(
        np.abs(layernorm(x, ln_weight, ln_bias, float(ln.eps)) - attn_in).max()
    )

    hidden_size = int(x.shape[1])
    n_heads = int(model.config.num_attention_heads)
    head_dim = hidden_size // n_heads
    w_q, b_q, w_k, b_k = head_projections(attn, hidden_size, n_heads, head_dim)

    cfgv = RM.validate_config(model)
    rotary_dim = int(cfgv["rope_dim"])
    freqs = R.inv_freq(rotary_dim, float(cfgv["rope_theta_declared"]))

    f = dictionary.encode(x)
    active_per_token = (f > 0).sum(axis=1)
    grid = distance_grid(n_tokens - 1, deltas)

    heads = []
    rng = np.random.default_rng(seed)
    sweep = np.array([d for d in SWEEP_DELTAS], dtype=np.int64)
    for head in range(n_heads):
        buckets = [
            token_buckets(
                x[t],
                f[t],
                dictionary,
                ln_weight,
                ln_bias,
                float(ln.eps),
                w_q[head],
                b_q[head],
                w_k[head],
                b_k[head],
                top_k,
            )
            for t in range(n_tokens)
        ]
        measured = measure_head(
            buckets,
            freqs,
            rotary_dim,
            grid,
            n_pairs,
            np.random.default_rng(rng.integers(2**63)),
            dictionary.n_features,
            sweep_deltas=sweep,
        )
        heads.append(
            {
                "head": head,
                "layer": attn_index,
                "max_score_magnitude": measured["max_score_magnitude"],
                "max_relative_exactness_error": measured["max_relative_exactness_error"],
                "base_share_mean": measured["base_share_mean"],
                "top_k_share_mean": measured["top_k_share_mean"],
                # Realized distances, averaged over content: the table a reader
                # would look up. `feature` is the total contribution, `rotary`
                # only the part carried by the rotated channels.
                "feature_metrics": profile_metrics(measured["feature_profile"]),
                "feature_rotary_metrics": profile_metrics(
                    measured["feature_rotary_profile"]
                ),
                # Content-fixed sweeps over the full grid: the object comparable
                # to the paper's synthetic sign-error rate.
                "sweep_feature_metrics": profile_metrics(measured["sweep"]["feature"][1]),
                "sweep_rotary_metrics": profile_metrics(
                    measured["sweep"]["feature_rotary"][1]
                ),
                "sweep_pair_metrics": profile_metrics(measured["sweep"]["pair"][1]),
                "top_features": top_profiles(
                    measured["feature_ids"], measured["feature_profile"]
                ),
                "top_sweep_features": top_profiles(
                    measured["sweep"]["feature"][0], measured["sweep"]["feature"][1], n=8
                ),
                "top_sweep_rotary": top_profiles(
                    measured["sweep"]["feature_rotary"][0],
                    measured["sweep"]["feature_rotary"][1],
                    n=8,
                ),
                "top_sweep_pairs": top_profiles(
                    measured["sweep"]["pair"][0],
                    measured["sweep"]["pair"][1],
                    n=8,
                    d_sae=dictionary.n_features,
                ),
            }
        )

    return {
        "model_id": model_id,
        "sae_layer": int(sae_layer),
        "attention_layer": attn_index,
        "n_tokens": n_tokens,
        "deltas": [int(d) for d in grid],
        "sweep_deltas": [int(d) for d in sweep],
        "text": text,
        "top_k": int(top_k),
        "pairs_per_delta": int(n_pairs),
        "rotary_dim": rotary_dim,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "ln_residual": ln_residual,
        "active_features_mean": float(active_per_token.mean()),
        "active_features_min": int(active_per_token.min()),
        "active_features_max": int(active_per_token.max()),
        "heads": heads,
    }


def aggregate_heads(heads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool the per-head metrics, weighting cells rather than heads."""

    def pooled(kind: str) -> dict[str, Any]:
        cells = sum(int(h[kind]["n_material_cells"]) for h in heads)
        flips = sum(
            int(round((h[kind]["sign_flip_fraction"] or 0.0) * h[kind]["n_material_cells"]))
            for h in heads
        )
        agreements = [
            float(h[kind]["top1_agreement"])
            for h in heads
            if h[kind]["top1_agreement"] is not None
        ]
        errors = [
            float(h[kind]["rel_err_median"])
            for h in heads
            if h[kind]["rel_err_median"] is not None
        ]
        return {
            "n_material_cells": cells,
            "sign_flip_fraction": (flips / cells) if cells else None,
            "top1_agreement_mean": float(np.mean(agreements)) if agreements else None,
            "rel_err_median_mean": float(np.mean(errors)) if errors else None,
        }

    return {
        "n_heads": len(heads),
        "feature": pooled("feature_metrics"),
        "feature_rotary": pooled("feature_rotary_metrics"),
        "sweep_feature": pooled("sweep_feature_metrics"),
        "sweep_rotary": pooled("sweep_rotary_metrics"),
        "sweep_pair": pooled("sweep_pair_metrics"),
        "base_share_mean": float(np.mean([h["base_share_mean"] for h in heads])),
        "top_k_share_mean": float(np.mean([h["top_k_share_mean"] for h in heads])),
        "max_relative_exactness_error": float(
            max(h["max_relative_exactness_error"] for h in heads)
        ),
        "max_score_magnitude": float(max(h["max_score_magnitude"] for h in heads)),
    }


def run_all(
    model_id: str = DEFAULT_MODEL_ID,
    sae_path: str | Path | None = None,
    text: str = DEFAULT_TEXT,
    seed: int = 0,
    sae_layer: int = DEFAULT_SAE_LAYER,
) -> dict[str, Any]:
    measured = measure_model(model_id, sae_path, text, seed=seed, sae_layer=sae_layer)
    measured["aggregate"] = aggregate_heads(measured["heads"])
    measured["schema"] = "qk-attribution/v1"
    measured["dictionary"] = (
        f"ctigges/pythia-70m-deduped__res-sm_processed, {int(sae_layer)}-res-sm "
        f"(d_sae=32768, ReLU, apply_b_dec_to_input=true)"
        if sae_path is not None
        else None
    )
    measured["estimand"] = (
        "mean contribution of a feature pair to the attention score at a fixed "
        "distance, averaged over content pairs sampled from one passage"
    )
    return measured


def write_results(data: dict[str, Any], path: Path | None = None) -> Path:
    out = path or Path(__file__).resolve().parent.parent.parent / "results" / "qk_attribution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--sae",
        type=Path,
        required=True,
        help=(
            "safetensors file with W_enc/b_enc/W_dec/b_dec. The paper used the "
            "public ctigges/pythia-70m-deduped__res-sm_processed dictionary, "
            "3-res-sm (d_sae=32768); download it from the SAE Lens release and "
            "pass the file here. It is not vendored."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--sae-layer",
        type=int,
        default=DEFAULT_SAE_LAYER,
        help=(
            "block the dictionary was trained on; attention of block "
            "sae_layer + 1 is attributed. Default 3, the layer used in the paper."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    data = run_all(args.model, args.sae, sae_layer=args.sae_layer)
    path = write_results(data, args.out)
    aggregate = data["aggregate"]
    print(
        f"{'model':<26} {'realized':>9} {'rotary':>8} {'sweep':>8} "
        f"{'sw-rot':>8} {'sw-pair':>8} {'base':>7}"
    )
    row = " ".join(
        f"{aggregate[key]['sign_flip_fraction']:>8.3f}"
        for key in (
            "feature",
            "feature_rotary",
            "sweep_feature",
            "sweep_rotary",
            "sweep_pair",
        )
    )
    print(f"{data['model_id']:<26}  {row} {aggregate['base_share_mean']:>7.3f}")
    print(
        f"tokens {data['n_tokens']}, deltas {data['deltas']}, "
        f"heads {aggregate['n_heads']}, "
        f"exactness error {aggregate['max_relative_exactness_error']:.2e}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
