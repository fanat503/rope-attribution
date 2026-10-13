"""What a position-free scalar gets wrong that a reader would act on.

Why this module exists
----------------------
The paper's headline is a *sign* error rate: the best position-free scalar has the
wrong sign on roughly a quarter of per-pair, per-distance cells. A reviewer's
first objection is that a sign error in a small term does not matter, and the
paper's answer so far is that sign is what an explanation commits to.

This module adds the answer that does not lean on that defence. What a reader
actually does with an attribution is *rank* the feature pairs and act on the top
one. So: rank the pairs by exact contribution at a given distance, rank them by
the position-free surrogate, and measure how often the two put the same pair
first. That is a downstream consequence with no downstream task - it needs no
sparse autoencoder, no benchmark, and no model beyond the activations the rest of
the repository already loads.

What is measured
----------------
``top1_agreement``
    Fraction of (query, key, distance) triples where the two rankings put the same
    pair first.
``rank_correlation``
    Spearman rho between the two orderings over all pairs, per triple.
``top1_agreement_top_decile``
    The same, restricted to the pairs the exact ranking places in its top decile
    - the pairs a reader is most likely to act on, which is where a disagreement
    is most expensive.
``top1_agreement_material``
    The same, restricted to cells where the exact term carries at least a stated
    fraction of its own peak, so the small-magnitude objection cannot be answered
    by pointing at near-zero terms.

Scope, stated once
------------------
All of this is computed on the same three trained checkpoints and the same
synthetic construction the rest of the repository uses. It is not a claim about
any downstream task's accuracy; it is a claim about how often the position-free
object answers a question about the exact one incorrectly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "TOP_DECILE",
    "average_ranks",
    "spearman",
    "top1_agreement",
    "rank_correlation",
    "agreement_profile",
]

# The decile the exact ranking must place a pair in for ``top_decile`` to count it.
# Fixed in advance rather than chosen after seeing the result.
TOP_DECILE = 0.10


def average_ranks(x: Sequence[float]) -> np.ndarray:
    """Ranks 1..n with ties sharing their average rank.

    Ties must be averaged rather than broken arbitrarily: a repeated value that
    silently became a unique rank would change the correlation in a way that
    depends on the sort order.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 2:
        raise ValueError(f"need at least 2 values, got {arr.size}")
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(arr.size, dtype=np.float64)
    sorted_arr = arr[order]
    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rho between two equal-length orderings."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.size != right.size:
        raise ValueError(f"length mismatch: {left.size} vs {right.size}")
    if left.size < 2:
        raise ValueError(f"need at least 2 values, got {left.size}")
    rx = average_ranks(left)
    ry = average_ranks(right)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom == 0.0:
        # A constant ordering carries no information about the other. Returning
        # 0 rather than NaN keeps the aggregate statistics well-defined, and the
        # tests pin the cases where the answer must be 0.
        return 0.0
    return float((rx * ry).sum() / denom)


def top1_agreement(
    exact: Sequence[Sequence[float]],
    surrogate: Sequence[Sequence[float]],
) -> float:
    """Fraction of rows whose argmax is the same index in both.

    ``exact`` and ``surrogate`` are the per-pair contributions at each distance,
    as (n_distances, n_pairs). Ties in the argmax count as agreement only if the
    same index wins in both, which is what a reader taking a top-1 would see.
    """
    e = np.asarray(exact, dtype=np.float64)
    s = np.asarray(surrogate, dtype=np.float64)
    if e.shape != s.shape:
        raise ValueError(f"shape mismatch: {e.shape} vs {s.shape}")
    if e.ndim != 2:
        raise ValueError(f"expected a 2-D (distance, pair) array, got {e.ndim}D")
    return float(np.mean(np.argmax(e, axis=1) == np.argmax(s, axis=1)))


def rank_correlation(
    exact: Sequence[Sequence[float]],
    surrogate: Sequence[Sequence[float]],
) -> dict:
    """Spearman rho between the two orderings, per row and overall."""
    e = np.asarray(exact, dtype=np.float64)
    s = np.asarray(surrogate, dtype=np.float64)
    if e.shape != s.shape:
        raise ValueError(f"shape mismatch: {e.shape} vs {s.shape}")
    per_row = [spearman(e[i], s[i]) for i in range(e.shape[0])]
    return {
        "per_row": [float(v) for v in per_row],
        "mean": float(np.mean(per_row)),
        "sd": float(np.std(per_row, ddof=1)) if len(per_row) > 1 else 0.0,
        "min": float(np.min(per_row)),
        "max": float(np.max(per_row)),
    }


def agreement_profile(
    exact: Sequence[Sequence[float]],
    surrogate: Sequence[Sequence[float]],
    materiality: float = 0.05,
) -> dict:
    """All of the agreement statistics in one call.

    ``exact`` and ``surrogate`` are (n_distances, n_pairs) arrays of the exact
    per-pair contribution magnitude and of the position-free surrogate. Rows are
    distances; pairs are ranked within a row.

    The three conditioned variants are *row subsets*, not masks. Masking a row's
    values before the argmax is a no-op for the top decile, because the row
    maximum is in the top decile by construction - the first version of this
    function did exactly that and reported three identical numbers, which is what
    gave the bug away.
    """
    e = np.asarray(exact, dtype=np.float64)
    s = np.asarray(surrogate, dtype=np.float64)
    if e.shape != s.shape:
        raise ValueError(f"shape mismatch: {e.shape} vs {s.shape}")
    if e.ndim != 2:
        raise ValueError(f"expected a 2-D (distance, pair) array, got {e.ndim}D")

    argmax_e = np.argmax(e, axis=1)
    argmax_s = np.argmax(s, axis=1)
    agree = argmax_e == argmax_s

    # Does the surrogate's chosen pair lie inside the exact top decile? A weaker
    # but meaningful question: is the surrogate at least picking from the region
    # the exact ranking counts as important?
    k = max(1, int(round(TOP_DECILE * e.shape[1])))
    threshold = np.partition(e, -k, axis=1)[:, -k][:, None]
    in_decile = e >= threshold
    surrogate_in_decile = np.array(
        [in_decile[i, argmax_s[i]] for i in range(e.shape[0])], dtype=bool
    )

    # Restrict to the rows whose own peak is at least `materiality` of the
    # largest peak across rows - that is, distances where the head says something
    # at all. Conditioning within a row is a no-op, because the argmax is the row
    # maximum by construction; the first version did that and every row qualified.
    peak = np.abs(e).max(axis=1)
    scale = float(peak.max()) if peak.size else 0.0
    top_is_material = peak >= materiality * scale

    def rate(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        return float(np.mean(agree[mask]))

    return {
        "n_distances": int(e.shape[0]),
        "n_pairs": int(e.shape[1]),
        "top1_agreement": float(np.mean(agree)),
        "top1_agreement_top_decile": float(np.mean(surrogate_in_decile)),
        "top1_agreement_material": rate(top_is_material),
        "n_material_rows": int(top_is_material.sum()),
        "rank_correlation": rank_correlation(e, s),
    }
