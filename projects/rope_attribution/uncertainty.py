"""Uncertainty for the aggregate statistics the paper quotes.

Why this module exists
----------------------
The paper's central empirical finding is that the best position-free scalar gets
the sign wrong on a fraction of per-pair, per-distance cells. Every number
supporting it is reported as a point estimate: 27.0% pooled, 1.13 median
relative error, 21.3%-33.9% across heads. A point estimate without an interval
is an assertion that the estimate is exact, which on 558 heads and three
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
* for each pair of checkpoints, an unpaired bootstrap on the difference of
  means, because the head index is not a matching unit across models (they do
  not even have the same number of heads). A difference is called compatible
  with zero only when its interval straddles it - see ``checkpoint_agreement``;
* the number of resamples is fixed in advance, and the seed is pinned, so the
  interval is reproducible rather than a function of the RNG.

Replication unit
----------------
A *layer-head* is the unit. The five prompts are five observations of the same
weights, not five independent heads; the earlier artifact counted all 2790
head-prompt instances as heads, which made ``n`` five times too large and every
interval about ``sqrt(5)`` too narrow. ``head_rates_from_artifact`` reads the
v2 artifact, where each head already carries its mean over prompts, and refuses
an older schema rather than silently re-creating the error.

Scope
-----
This quantifies sampling variation across heads, conditional on the three
checkpoints. It does not quantify variation across checkpoints (three is too few
for a useful interval), across prompts, across the distance grid, or across the
synthetic construction - those are real omissions and are stated as ones.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = [
    "N_RESAMPLES",
    "bootstrap_ci",
    "checkpoint_agreement",
    "head_rates_from_artifact",
    "summarise_artifact",
    "summarise_with_intervals",
    "main",
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
                    # interval straddles it; the interval is the test, so no
                    # separate p-value is computed or reported.
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


def head_rates_from_artifact(data: dict) -> tuple[list[str], list[list[float]]]:
    """Per-checkpoint layer-head rates from a v2 usefulness artifact.

    Refuses a v1 artifact by name. In v1 ``n_heads`` counted head-prompt
    instances, so reading ``per_prompt`` here would silently reproduce the
    five-fold over-count this module exists to remove.
    """
    schema = data.get("schema")
    if schema != "rope_attribution/usefulness/v2":
        raise ValueError(
            f"need schema rope_attribution/usefulness/v2, got {schema!r}; "
            "regenerate results/usefulness.json with the current usefulness.py"
        )
    ids: list[str] = []
    rates: list[list[float]] = []
    for model in data["models"]:
        if "head_rates" not in model:
            raise ValueError(f"{model.get('model_id')}: missing head_rates")
        ids.append(str(model["model_id"]))
        rates.append(
            [
                float(h["sign_flip_fraction"])
                for h in model["head_rates"]
                if h["sign_flip_fraction"] is not None
            ]
        )
    if sum(len(r) for r in rates) != int(data["pooled"]["n_heads"]):
        raise ValueError(
            "head_rates do not sum to pooled.n_heads; the artifact is inconsistent"
        )
    return ids, rates


def summarise_artifact(data: dict) -> dict:
    """Everything the paper quotes about uncertainty, from one artifact."""
    ids, rates = head_rates_from_artifact(data)
    summary = summarise_with_intervals(ids, rates)
    equal_weight = float(np.mean([np.mean(r) for r in rates]))
    pooled_mean = float(np.mean([r for row in rates for r in row]))
    delta = equal_weight - pooled_mean
    return {
        "schema": "rope_attribution/uncertainty/v1",
        "source": "results/usefulness.json",
        "unit": "layer-head, each averaged over its prompts",
        "n_heads": sum(len(r) for r in rates),
        "n_head_prompt_instances": int(data["pooled"]["n_head_prompt_instances"]),
        "equal_weight_fraction": equal_weight,
        "head_pooled_fraction": pooled_mean,
        "equal_weight_minus_pooled": delta,
        "difference_in_pooled_half_widths": (
            delta / summary["pooled"]["half_width"]
        ),
        "pooled": summary["pooled"],
        "agreement": summary["agreement"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "results" / "usefulness.json"
    out = summarise_artifact(json.loads(src.read_text(encoding="utf-8")))
    dst = root / "results" / "uncertainty.json"
    dst.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    p = out["pooled"]
    print(
        f"{out['n_heads']} heads ({out['n_head_prompt_instances']} head-prompt "
        f"instances): mean {p['mean']:.6f}, sd {p['sd']:.6f}, "
        f"95% CI [{p['ci_low']:.6f}, {p['ci_high']:.6f}]"
    )
    for pair in out["agreement"]["pairs"]:
        print(
            f"  {pair['pair']}: {100 * pair['mean_difference']:+.2f} pp "
            f"[{100 * pair['ci_low']:+.2f}, {100 * pair['ci_high']:+.2f}] "
            f"agrees={pair['agrees_at_95']}"
        )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
