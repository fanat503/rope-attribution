"""Attribution-guided KV eviction: the ranking core and its masking path.

What this module is
-------------------
The paper's attribution object is the exact, per-distance score

    A_ij(delta) = sum_k [a_k cos(theta_k delta) + b_k sin(theta_k delta)]
                  + (unrotated tail),

where ``a_k = q1 k1 + q2 k2`` and ``b_k = q2 k1 - q1 k2`` for the split-half
rotary pair ``k``, and ``delta = t - j`` is the query-to-key distance.  A
position-free summary replaces the two trigonometric weights by their means
over the observed window.  This module turns either object into a ranking of
cached keys, masks the cache to the retained set, and computes the resulting
attention.

What this module is not
-----------------------
It is **not** a 1B--8B task result.  It is the ranking and masking core the
pre-registered protocol in ``docs/DOWNSTREAM_PROTOCOL.md`` needs, tested on
synthetic tensors.  The Hugging Face cache adapter is deliberately not shipped:
evicting a real model's cache requires architecture-specific surgery on the
attention forward (Llama, Qwen, Mistral and Gemma differ in QK-normalisation,
GQA layout and cache class), and a half-working adapter would be worse than an
honest boundary.  ``--dry-run`` writes the artifact schema with no task
numbers; it never invents a measurement.

Sign convention
---------------
For a query at position ``t`` and a key at position ``j``, RoPE gives
``q_t^T R(theta (j - t)) k_j``.  With ``delta = t - j`` the angle is
``-theta delta``; feeding ``+theta delta`` mirrors every sine term and silently
describes a different model.  ``test_exact_scores_match_rope_dot`` pins this.

Mutation verdicts
-----------------
* ``test_exact_scores_match_rope_dot`` fails if the angle sign is flipped: the
  cosine term is even, so only the sine term moves, and the test compares every
  pair against ``rope.apply_rope`` rather than against a second copy of the
  same formula.
* ``test_group_scores_sum_abs_over_query_heads`` fails if grouped-query heads
  are averaged instead of summed, which would make a KV head's ranking
  independent of how many query heads share it.
* ``test_select_keep_always_keeps_prefix_and_current`` fails if the forced set
  is dropped, which would let eviction delete the first tokens or the token
  being decoded.
* ``test_masked_attention_identity_when_all_kept`` fails if the mask is applied
  to the output instead of the logits, or if the softmax normalisation changes.
* ``test_partial_rope_keeps_the_tail`` fails if the unrotated channels are
  dropped, which is the dilution-law error the protocol records.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "rope_attribution/downstream_eviction/v1"
PROTOCOL_VERSION = "1.0"
ARMS: tuple[str, ...] = (
    "exact",
    "mean",
    "identity",
    "attention",
    "recency",
    "random",
)
DEFAULT_SEED = 20261111


# ---------------------------------------------------------------------------
# Shape and parameter validation
# ---------------------------------------------------------------------------


def _as_2d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be (heads, dim) or (dim,), got {arr.shape}")
    return arr


def _as_keys(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"{name} must be (kv_heads, keys, dim), got {arr.shape}")
    return arr


def _check_rotary(head_dim: int, rotary_dim: int) -> int:
    if rotary_dim < 0 or rotary_dim > head_dim:
        raise ValueError(f"rotary_dim must be in [0, {head_dim}], got {rotary_dim}")
    if rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    return rotary_dim // 2


# ---------------------------------------------------------------------------
# Per-pair coefficients and the four score functionals
# ---------------------------------------------------------------------------


def pair_terms(
    q: np.ndarray, k: np.ndarray, rotary_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(aligned, crossed, tail)`` for ``q`` and one or more keys.

    ``q`` is ``(query_heads, dim)``; ``k`` is ``(query_heads, keys, dim)``.  The
    caller expands grouped-query keys to query heads first.  ``aligned`` and
    ``crossed`` have shape ``(query_heads, keys, rotary_dim // 2)``; ``tail`` is
    ``(query_heads, keys)`` and is zero when ``rotary_dim == dim``.
    """
    q = _as_2d(q, "q")
    k = _as_keys(k, "k")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k disagree on head_dim: {q.shape[-1]} vs {k.shape[-1]}")
    if q.shape[0] != k.shape[0]:
        raise ValueError(
            f"q and k disagree on heads after expansion: {q.shape[0]} vs {k.shape[0]}"
        )
    half = _check_rotary(q.shape[-1], rotary_dim)
    q1, q2 = q[:, :half], q[:, half:rotary_dim]
    k1, k2 = k[:, :, :half], k[:, :, half:rotary_dim]
    aligned = q1[:, None, :] * k1 + q2[:, None, :] * k2
    crossed = q2[:, None, :] * k1 - q1[:, None, :] * k2
    if rotary_dim < q.shape[-1]:
        tail = (q[:, None, rotary_dim:] * k[:, :, rotary_dim:]).sum(axis=-1)
    else:
        tail = np.zeros(q.shape[0:1] + k.shape[1:2], dtype=np.float64)
    return aligned, crossed, tail


def _expand_kv(k: np.ndarray, n_query_heads: int) -> np.ndarray:
    """Repeat each KV head for the query heads that share it."""
    k = _as_keys(k, "k")
    n_kv = k.shape[0]
    if n_query_heads % n_kv:
        raise ValueError(
            f"{n_query_heads} query heads do not divide into {n_kv} KV heads"
        )
    return np.repeat(k, n_query_heads // n_kv, axis=0)


def _angles(
    query_position: float, key_positions: np.ndarray, freqs: np.ndarray
) -> np.ndarray:
    """``-theta * (query_position - key_position)``, shape ``(keys, pairs)``."""
    delta = float(query_position) - np.asarray(key_positions, dtype=np.float64)
    return -delta[:, None] * np.asarray(freqs, dtype=np.float64)[None, :]


def exact_scores(
    q: np.ndarray,
    k: np.ndarray,
    query_position: float,
    key_positions: np.ndarray,
    freqs: np.ndarray,
    rotary_dim: int,
) -> np.ndarray:
    """Exact per-distance scores, shape ``(query_heads, keys)``.

    This is the object the paper calls ``A_ij(delta)``.  It is a closed-form
    identity, not an approximation: the unrotated tail is added separately and
    the rotary pairs use the aligned/crossed coefficients above.
    """
    k_exp = _expand_kv(k, q.shape[0] if np.asarray(q).ndim > 1 else 1)
    aligned, crossed, tail = pair_terms(q, k_exp, rotary_dim)
    angles = _angles(query_position, key_positions, freqs)
    return (
        (aligned * np.cos(angles)[None, :, :]).sum(axis=-1)
        + (crossed * np.sin(angles)[None, :, :]).sum(axis=-1)
        + tail
    )


def mean_scores(
    q: np.ndarray,
    k: np.ndarray,
    query_position: float,
    key_positions: np.ndarray,
    freqs: np.ndarray,
    rotary_dim: int,
) -> np.ndarray:
    """Best position-free constant per pair over the observed distances.

    The mean is taken over the angles the query can actually see, exactly as in
    ``qk_intervention._surrogate_vector``.  Fitting over the full sequence would
    let a short-window query be summarised by distances it cannot attend to.
    """
    k_exp = _expand_kv(k, q.shape[0] if np.asarray(q).ndim > 1 else 1)
    aligned, crossed, tail = pair_terms(q, k_exp, rotary_dim)
    angles = _angles(query_position, key_positions, freqs)
    cos_mean = np.cos(angles).mean(axis=0)
    sin_mean = np.sin(angles).mean(axis=0)
    return (
        (aligned * cos_mean[None, None, :]).sum(axis=-1)
        + (crossed * sin_mean[None, None, :]).sum(axis=-1)
        + tail
    )


def identity_scores(
    q: np.ndarray, k: np.ndarray, rotary_dim: int
) -> np.ndarray:
    """The ``delta = 0`` value: the unrotated query-key dot product."""
    k_exp = _expand_kv(k, q.shape[0] if np.asarray(q).ndim > 1 else 1)
    aligned, crossed, tail = pair_terms(q, k_exp, rotary_dim)
    del crossed
    return aligned.sum(axis=-1) + tail


def recency_scores(
    key_positions: np.ndarray, query_position: float, n_query_heads: int
) -> np.ndarray:
    """A position-only baseline: closer keys score higher."""
    delta = float(query_position) - np.asarray(key_positions, dtype=np.float64)
    return np.repeat(-delta[None, :], int(n_query_heads), axis=0)


def random_scores(
    n_query_heads: int, n_keys: int, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Uniform scores from a local generator, so the arm is reproducible."""
    rng = np.random.default_rng(int(seed))
    return rng.random((int(n_query_heads), int(n_keys)))


def group_scores(per_query: np.ndarray, n_kv_heads: int) -> np.ndarray:
    """Sum ``|score|`` over the query heads that share each KV head.

    Summing is the protocol's choice: a KV head is retained if any query head
    that reads it needs it.  Averaging would make the ranking depend on the
    group size, which is an architecture detail rather than a property of the
    attribution.
    """
    per_query = np.asarray(per_query, dtype=np.float64)
    if per_query.ndim != 2:
        raise ValueError(f"per_query must be (query_heads, keys), got {per_query.shape}")
    n_query = per_query.shape[0]
    if n_query % n_kv_heads:
        raise ValueError(f"{n_query} query heads do not divide into {n_kv_heads} KV heads")
    grouped = np.abs(per_query).reshape(n_kv_heads, n_query // n_kv_heads, -1)
    return grouped.sum(axis=1)


def arm_scores(
    arm: str,
    q: np.ndarray,
    k: np.ndarray,
    query_position: float,
    key_positions: np.ndarray,
    *,
    freqs: np.ndarray | None = None,
    rotary_dim: int | None = None,
    attention: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Per-query-head scores for one arm.

    ``exact``, ``mean`` and ``identity`` need ``freqs`` and ``rotary_dim``;
    ``attention`` needs the accumulated post-softmax weights; ``recency`` and
    ``random`` need only the geometry.  The function refuses a missing input
    rather than silently substituting another arm.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown eviction arm {arm!r}; expected one of {ARMS}")
    if arm in {"exact", "mean", "identity"}:
        if freqs is None or rotary_dim is None:
            raise ValueError(f"arm {arm!r} needs freqs and rotary_dim")
        if arm == "exact":
            return exact_scores(q, k, query_position, key_positions, freqs, rotary_dim)
        if arm == "mean":
            return mean_scores(q, k, query_position, key_positions, freqs, rotary_dim)
        return identity_scores(q, k, rotary_dim)
    if arm == "attention":
        if attention is None:
            raise ValueError("arm 'attention' needs the accumulated attention weights")
        return np.asarray(attention, dtype=np.float64)
    if arm == "recency":
        n_query = np.asarray(q).shape[0] if np.asarray(q).ndim > 1 else 1
        return recency_scores(key_positions, query_position, n_query)
    n_query = np.asarray(q).shape[0] if np.asarray(q).ndim > 1 else 1
    return random_scores(n_query, len(key_positions), seed=seed)


# ---------------------------------------------------------------------------
# Selection and masked attention
# ---------------------------------------------------------------------------


def select_keep(
    scores: np.ndarray,
    retention: float,
    *,
    prefix_tokens: int = 4,
    forced: Sequence[int] = (),
) -> np.ndarray:
    """Indices retained at ``retention`` of the cache, plus the forced set.

    The top ``ceil(retention * n)`` keys are kept by score, and the first
    ``prefix_tokens`` keys and every index in ``forced`` are kept regardless.
    The union is returned sorted, so the result is a mask rather than a
    permutation.  Ties break by index, which makes the selection deterministic.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got {scores.shape}")
    if not 0.0 < float(retention) <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    if prefix_tokens < 0:
        raise ValueError(f"prefix_tokens must be non-negative, got {prefix_tokens}")
    n = scores.shape[0]
    if n == 0:
        raise ValueError("cannot select from an empty cache")
    n_top = max(1, int(math.ceil(float(retention) * n)))
    order = np.argsort(-scores, kind="stable")
    keep = {int(i) for i in order[:n_top]}
    keep.update(range(min(int(prefix_tokens), n)))
    keep.update(int(i) for i in forced if 0 <= int(i) < n)
    return np.array(sorted(keep), dtype=np.int64)


def keep_mask(
    scores: np.ndarray,
    retention: float,
    *,
    prefix_tokens: int = 4,
    forced: Sequence[int] = (),
) -> np.ndarray:
    """Boolean mask of the retained keys, one row per KV head."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[None, :]
    if scores.ndim != 2:
        raise ValueError(f"scores must be (kv_heads, keys), got {scores.shape}")
    mask = np.zeros(scores.shape, dtype=bool)
    for head in range(scores.shape[0]):
        mask[head, select_keep(
            scores[head],
            retention,
            prefix_tokens=prefix_tokens,
            forced=forced,
        )] = True
    return mask


def masked_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    keep: np.ndarray,
    *,
    head_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax attention restricted to ``keep``, with GQA expansion.

    ``q`` is ``(query_heads, dim)``, ``k`` and ``v`` are
    ``(kv_heads, keys, dim)``, and ``keep`` is ``(keys,)`` or
    ``(kv_heads, keys)``.  With every entry of ``keep`` true this is the
    ordinary causal attention for the supplied keys; that identity is the
    mutation detector for the masking path.
    """
    q = _as_2d(q, "q")
    k = _as_keys(k, "k")
    v = _as_keys(v, "v")
    if k.shape != v.shape:
        raise ValueError(f"k and v disagree: {k.shape} vs {v.shape}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k disagree on head_dim: {q.shape[-1]} vs {k.shape[-1]}")
    k_exp = _expand_kv(k, q.shape[0])
    v_exp = _expand_kv(v, q.shape[0])
    scale = math.sqrt(float(head_dim if head_dim is not None else q.shape[-1]))
    scores = (q[:, None, :] * k_exp).sum(axis=-1) / scale

    keep = np.asarray(keep, dtype=bool)
    if keep.ndim == 1:
        keep = np.repeat(keep[None, :], k.shape[0], axis=0)
    if keep.shape != (k.shape[0], k.shape[1]):
        raise ValueError(
            f"keep must be ({k.shape[1]},) or ({k.shape[0]}, {k.shape[1]}), "
            f"got {keep.shape}"
        )
    keep_exp = np.repeat(keep, q.shape[0] // k.shape[0], axis=0)
    if not np.all(keep_exp.any(axis=1)):
        raise ValueError("every query head must retain at least one key")
    scores = np.where(keep_exp, scores, -np.inf)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights @ v_exp, weights


def evict_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    query_position: float,
    key_positions: np.ndarray,
    *,
    arm: str,
    retention: float,
    prefix_tokens: int = 4,
    freqs: np.ndarray | None = None,
    rotary_dim: int | None = None,
    attention: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Rank, mask and attend for one decode step.

    The current key is the last entry of ``k`` and is always retained; the
    first ``prefix_tokens`` keys are retained too.  The returned ``keep`` mask
    is per KV head, because grouped-query heads can retain different sets.
    """
    k_arr = _as_keys(k, "k")
    n_keys = k_arr.shape[1]
    per_query = arm_scores(
        arm,
        q,
        k_arr,
        query_position,
        key_positions,
        freqs=freqs,
        rotary_dim=rotary_dim,
        attention=attention,
        seed=seed,
    )
    scores = group_scores(per_query, k_arr.shape[0])
    keep = keep_mask(
        scores,
        retention,
        prefix_tokens=prefix_tokens,
        forced=(n_keys - 1,),
    )
    output, weights = masked_attention(q, k_arr, v, keep)
    return {
        "arm": arm,
        "retention": float(retention),
        "keep": keep,
        "scores": scores,
        "weights": weights,
        "output": output,
        "n_kept": int(keep.sum(axis=1).sum()),
    }


# ---------------------------------------------------------------------------
# Artifact schema and CLI
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return proc.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def dry_run_artifact(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """A schema-valid artifact with no model and no task number.

    The protocol's confirmatory run fills ``models``, ``cells`` and ``ceiling``.
    This function fills none of them and says so in ``note``; a reader cannot
    mistake it for a measurement.
    """
    return {
        "schema": SCHEMA,
        "status": "dry-run",
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "date": _dt.date.today().isoformat(),
        "seed": int(seed),
        "prompt_sha256": None,
        "models": [],
        "cells": [],
        "ceiling": [],
        "note": (
            "No checkpoint was run. This artifact validates the schema only; "
            "it contains no task number and is not evidence for or against the "
            "downstream hypothesis."
        ),
    }


def write_artifact(data: dict[str, Any], path: Path | None = None) -> Path:
    target = Path(path) if path is not None else Path(".tmp") / "downstream_eviction.dryrun.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="write the schema with no task numbers")
    parser.add_argument("--out", type=Path, default=None, help="artifact path for --dry-run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.dry_run:
        raise SystemExit(
            "This build ships the tested ranking core, not the 1B--8B model "
            "adapter. Use --dry-run for the artifact schema, and see "
            "docs/DOWNSTREAM_PROTOCOL.md for the deferred run."
        )
    data = dry_run_artifact(seed=args.seed)
    path = write_artifact(data, args.out)
    print(f"wrote {path} ({data['schema']}, status={data['status']})")


if __name__ == "__main__":
    main()
