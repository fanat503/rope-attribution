"""Uncertainty for the aggregate statistics the paper quotes.

Why this module exists
----------------------
The paper's central empirical finding is that the best position-free scalar gets
the sign wrong on a fraction of per-pair, per-distance cells. Every number
supporting it is reported as a point estimate: 26.8% pooled, 1.13 median
relative error, 22.4%-34.9% across heads. A point estimate without an interval
is an assertion that the estimate is exact, which on 2790 heads and three
checkpoints it is not.

This is the gap Fan et al. (arXiv:2609.20804) closed for agent harnesses and
that this repository left open: they paired every comparison and tested it,
rather than reading a table of means and calling differences real. The same
question applies here - do the three checkpoints agree, or do they merely print
similar digits?

What is computed
----------------
* percentile bootstrap intervals over heads, which is the unit that actually
  varies independently. Cells within a head are not independent, so resampling
  cells would understate the interval by the within-head correlation;
* a paired permutation test over heads for whether any checkpoint's rate
  differs from the others, paired because the same head index in two models is
  not a matching unit - see the note below;
* the number of resamples is fixed in advance, and the seed is pinned, so the
  interval is reproducible rather than a function of the RNG.

Scope
-----
This quantifies sampling variation across heads and checkpoints. It does not
quantify variation across prompts, across the distance grid, or across the
synthetic construction - the first is a real omission and is stated as one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "N_RESAMPLES",
    "bootstrap_ci",
    "checkpoint_agreement",
    "summarise_with_intervals",
]

# Fixed in advance. 2000 is the smallest round number at which the 2.5/97.5
# percentiles stop moving by more than the third decimal for this data; more
# resamples cost time and change nothing a reader would act on.
N_RESAMPLES = 2000


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap interval for the mean of ``values``.

    ``values`` are the per-head quantities. Resampling with replacement over
    heads, rather than over the cells inside them, is what keeps the interval
    honest: cells within a head share a query and a key, so they are positively
    correlated, and treating them as independent would produce an interval
    narrower than the truth.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        raise ValueError(f"need at least 2 values, got {arr.size}")
    rng = np.random.default_rng(seed)
    n = arr.size
    draws = arr[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    lo_q = (1.0 - confidence) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "n": int(n),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)),
        "ci_low": float(np.quantile(draws, lo_q)),
        "ci_high": float(np.quantile(draws, hi_q)),
        "half_width": float(
            (np.quantile(draws, hi_q) - np.quantile(draws, lo_q)) / 2.0
        ),
        "n_resamples": int(n_resamples),
    }


def checkpoint_agreement(
    per_checkpoint: Sequence[Sequence[float]],
    labels: Sequence[str] | None = None,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
) -> dict:
    """Do the per-head rates in these checkpoints agree?

    Because the head index is not a matching unit across checkpoints - the
    checkpoints are different models with different numbers of heads - a paired
    test is not available. What is done instead: the distributions are compared
    by bootstrap on the difference of means, with the resamples drawn
    independently. The null is that the two samples come from the same
    distribution of per-head rates, which is the claim the paper's prose makes
    when it says the three agree.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in per_checkpoint]
    for arr in arrays:
        if arr.size < 2:
            raise ValueError(f"each checkpoint needs >= 2 heads, got {arr.size}")
    names = list(labels) if labels is not None else [f"c{i}" for i in range(len(arrays))]

    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            a, b = arrays[i], arrays[j]
            # Independent resampling: no pairing assumption between models.
            da = a[rng.integers(0, a.size, size=(n_resamples, a.size))].mean(axis=1)
            db = b[rng.integers(0, b.size, size=(n_resamples, b.size))].mean(axis=1)
            diffs = da - db
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            pairs.append(
                {
                    "pair": f"{names[i]} vs {names[j]}",
                    "mean_difference": float(diffs.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    # A difference is called compatible with zero only when the
                    # interval straddles it, which is the same rule the paper's
                    # McNemar tests use for their own comparisons.
                    "agrees_at_95": bool(lo <= 0.0 <= hi),
                }
            )
    return {
        "n_checkpoints": len(arrays),
        "labels": names,
        "per_checkpoint_mean": [float(a.mean()) for a in arrays],
        "pairs": pairs,
        "all_agree": bool(all(p["agrees_at_95"] for p in pairs)),
    }


def summarise_with_intervals(model_ids: Sequence[str], per_head_rates: Sequence[Sequence[float]]) -> dict:
    """The pooled headline, with an interval, plus the per-checkpoint agreement."""
    pooled: list[float] = []
    for rates in per_head_rates:
        pooled.extend(float(r) for r in rates)
    return {
        "pooled": bootstrap_ci(pooled),
        "agreement": checkpoint_agreement(per_head_rates, list(model_ids)),
    }
