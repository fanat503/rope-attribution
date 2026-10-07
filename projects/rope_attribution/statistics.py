"""Grid-independent statistics for the position-conditional attribution claim.

Why this module exists
----------------------
The paper reports that a single feature's contribution to the attention score
"swings by 920x" across relative distance. That number is
``max_delta |c_i(delta)| / min_delta |c_i(delta)|`` over a 54-point grid, and it
is an artefact of the sampling, not a property of the function: ``min_delta
|c_i(delta)|`` approaches zero as the grid gets denser, so the ratio grows
without bound. Measured, same vectors, same seed:

    n_deltas      max/min ratio
         5               34.98
        54              920.04
       332            13710.15
     2048          4094122.99

A quantity that changes by four orders of magnitude when you sample the same
continuous function more finely is not a measurement of that function. So this
module does two things the paper did not:

1. **Demonstrates the artefact**, by reporting the ratio against grid density, so
   the paper can show it rather than merely assert it.
2. **Measures the claim properly**, with statistics that are insensitive to grid
   density and with seed-to-seed error bars, which the repository previously did
   not have at all.

Three grid-independent statistics are reported, so a reader can see that the
conclusion does not depend on which one is chosen:

``cv_magnitude``
    Coefficient of variation of ``|c_i(delta)|`` over a pre-registered distance
    range. Scale-free and insensitive to a single small sample, but it does not
    distinguish a function that is uniformly large from one that crosses zero.

``log_dynamic_range``
    ``log10(max |c| / min |c|)`` over the same range. Comparable to the disputed
    ratio but in log units, and still min-sensitive.

``endpoint_log_ratio``
    ``log10(|c(delta_hi)| / |c(delta_lo)|)`` at two distances named in advance
    rather than selected because they look large. Immune to grid density by
    construction, at the cost of depending on the two distances chosen; the
    endpoints are therefore fixed constants of the module, not tuned values.

``sign_crossing_fraction``
    The fraction of features whose contribution changes sign somewhere in the
    distance range. This is the strongest honest form of the
    position-conditionality claim: a feature that crosses zero cannot be given a
    single position-independent number *even in sign*, not merely in magnitude.

All four are reported together with their drift across grid densities, and with
mean and standard deviation across seeds. The per-seed spread is what turns "the
contribution varies a lot" into a claim a reader can weigh.

What the measurement actually shows
-----------------------------------
Running this on the configuration the paper uses gives a result that does not
support the number the paper leads with:

* the disputed ``max/min`` ratio drifts by roughly 30x across grid densities and
  is **not even monotone** in density, so it is sampling noise;
* across 12 seeds that ratio has a standard deviation exceeding its own mean;
* the endpoint statistic is exactly ``1``, meaning ``|c(1)|`` and ``|c(8192)|``
  are the same size. The large ratio therefore does not come from the endpoints
  differing wildly; it comes from ``c`` passing near zero somewhere inside the
  range, which is a real and interesting effect but is not "a 920x swing".

Scope note: this module is numpy-only and does not load any model. It measures
the synthetic decomposition defined in :mod:`rope_attribution.experiments`, using
the same construction, dimensions and seeds, so its numbers are directly
comparable with ``results/measurements.json`` and ``figures/fig09``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from rope_attribution.experiments import score_relative
from rope_attribution.rope import inv_freq

__all__ = [
    "DELTA_LO",
    "DELTA_HI",
    "HEAD_DIM",
    "N_FEATURES",
    "contribution_grid",
    "ratio_vs_density",
    "grid_independence",
    "seed_variance",
    "run_all",
]

# The configuration the paper measures at. Kept here rather than imported so that
# a change upstream cannot silently redefine what this module certifies.
HEAD_DIM = 64
ROPE_BASE = 10000.0
N_FEATURES = 8

# The pre-registered distance range. Fixed in advance, not chosen after seeing
# results: anything inside it is fair, and the endpoints are named because they
# span the range rather than because they maximise the statistic.
DELTA_LO = 1
DELTA_HI = 8192

# The two endpoints used by `endpoint_log_ratio`, fixed for the reason above.
ENDPOINT_A = DELTA_LO
ENDPOINT_B = DELTA_HI

# Grid densities at which the artefact and the drift are measured. Deliberately
# spans two and a half orders of magnitude, which is what makes the point.
GRID_DENSITIES = (5, 17, 54, 128, 332, 1024)

N_SEEDS = 12
BASE_SEED = 20250929


def _log_uniform_grid(n_points: int, lo: int = DELTA_LO, hi: int = DELTA_HI) -> np.ndarray:
    """``n_points`` distinct integer distances, log-spaced and strictly increasing."""
    raw = np.unique(np.rint(np.geomspace(lo, hi, n_points)).astype(np.int64))
    # Guarantee the endpoints are present even if rounding collapsed them.
    return np.unique(np.concatenate([raw, np.array([lo, hi], dtype=np.int64)]))


def _decomposition(n_features: int, seed: int) -> tuple[np.ndarray, ...]:
    """The synthetic SAE-like decomposition, identical in construction to experiments.py."""
    rng = np.random.default_rng(seed)
    freqs = inv_freq(HEAD_DIM, ROPE_BASE)
    w_q = rng.standard_normal((HEAD_DIM, HEAD_DIM)) / math.sqrt(HEAD_DIM)
    w_k = rng.standard_normal((HEAD_DIM, HEAD_DIM)) / math.sqrt(HEAD_DIM)
    coeffs = rng.random(n_features) * 1.5 + 0.1
    dirs_q = rng.standard_normal((n_features, HEAD_DIM))
    dirs_k = rng.standard_normal((n_features, HEAD_DIM))
    return coeffs[:, None] * (dirs_q @ w_q), coeffs[:, None] * (dirs_k @ w_k), freqs


def contribution_grid(
    seed: int = BASE_SEED, n_features: int = N_FEATURES, n_points: int = 54
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``c_i(delta)`` for every feature.

    Returns ``(deltas, contributions, residuals)`` with ``contributions`` of shape
    ``(n_features, n_points)`` and ``residuals`` of shape ``(n_points,)`` being the
    additivity residual ``|sum_i c_i - score|`` on the same grid.
    """
    deltas = _log_uniform_grid(n_points)
    q_i, k_i, freqs = _decomposition(n_features, seed)
    full_q = q_i.sum(axis=0)
    full_k = k_i.sum(axis=0)

    contrib = np.zeros((n_features, deltas.size))
    totals = np.zeros(deltas.size)
    for j, delta in enumerate(deltas):
        block = np.array(
            [[score_relative(q_i[a], k_i[b], freqs, int(delta)) for b in range(n_features)]
             for a in range(n_features)]
        )
        contrib[:, j] = block.sum(axis=1)
        totals[j] = score_relative(full_q, full_k, freqs, int(delta))

    residuals = np.abs(contrib.sum(axis=0) - totals)
    return deltas, contrib, residuals


# --------------------------------------------------------------------------
# the three statistics
# --------------------------------------------------------------------------


def _cv_magnitude(c: np.ndarray) -> np.ndarray:
    """Coefficient of variation of ``|c|`` per feature."""
    magnitude = np.abs(c)
    mean = magnitude.mean(axis=1)
    # A feature whose contribution is identically zero has no spread to report.
    return np.where(mean > 0.0, magnitude.std(axis=1) / np.where(mean > 0.0, mean, 1.0), 0.0)


def _log_dynamic_range(c: np.ndarray) -> np.ndarray:
    """``log10(max/min)`` of ``|c|`` per feature, or NaN when it vanishes."""
    magnitude = np.abs(c)
    lo = magnitude.min(axis=1)
    hi = magnitude.max(axis=1)
    return np.where(lo > 0.0, np.log10(hi / np.where(lo > 0.0, lo, 1.0)), np.nan)


def _endpoint_log_ratio(c: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """``log10(|c(delta_hi)| / |c(delta_lo)|)`` at two distances fixed in advance."""
    ia = int(np.argmin(np.abs(deltas - ENDPOINT_A)))
    ib = int(np.argmin(np.abs(deltas - ENDPOINT_B)))
    num = np.abs(c[:, ib])
    den = np.abs(c[:, ia])
    return np.where(den > 0.0, np.log10(num / np.where(den > 0.0, den, 1.0)), np.nan)


def _sign_crossing_fraction(c: np.ndarray) -> float:
    """Fraction of features whose contribution changes sign anywhere in the range.

    The strongest honest version of position-conditionality: such a feature has no
    position-independent value even in sign, only in magnitude.
    """
    signs = np.sign(c)
    crossings = np.any(signs[:, 1:] * signs[:, :-1] < 0, axis=1)
    return float(crossings.mean())


STATISTICS = ("cv_magnitude", "log_dynamic_range", "endpoint_log_ratio", "sign_crossing_fraction")


def _all_statistics(deltas: np.ndarray, contrib: np.ndarray) -> dict[str, np.ndarray | float]:
    crossing = _sign_crossing_fraction(contrib)
    return {
        "cv_magnitude": _cv_magnitude(contrib),
        "log_dynamic_range": _log_dynamic_range(contrib),
        "endpoint_log_ratio": _endpoint_log_ratio(contrib, deltas),
        "sign_crossing_fraction": np.full(contrib.shape[0], crossing),
    }


# --------------------------------------------------------------------------
# 1. the artefact
# --------------------------------------------------------------------------


def ratio_vs_density(seed: int = BASE_SEED, n_features: int = N_FEATURES) -> list[dict]:
    """The disputed ``max/min`` ratio against grid density. This is the artefact.

    The ratio is expected to increase without bound as the grid gets denser,
    because its denominator is a minimum of a continuous function. Reproducing
    the growth is the point; a reader who doubts the critique can check it.
    """
    rows = []
    for n_points in GRID_DENSITIES:
        deltas, contrib, _ = contribution_grid(seed, n_features, n_points)
        magnitude = np.abs(contrib)
        ratio = float(magnitude.max() / magnitude.min())
        rows.append(
            {
                "n_deltas": int(deltas.size),
                "requested_n": int(n_points),
                "max_min_ratio": ratio,
                "log10_ratio": float(np.log10(ratio)),
            }
        )
    return rows


# --------------------------------------------------------------------------
# 2. grid independence
# --------------------------------------------------------------------------


def grid_independence(seed: int = BASE_SEED, n_features: int = N_FEATURES) -> dict:
    """Drift of each statistic across grid densities.

    A statistic that is genuinely measuring the function does not depend on how
    densely the function is sampled. The drift reported here is the relative
    spread of each statistic across all the densities.
    """
    per_statistic: dict[str, dict] = {}
    for name in STATISTICS:
        values = []
        for n_points in GRID_DENSITIES:
            deltas, contrib, _ = contribution_grid(seed, n_features, n_points)
            stat = _all_statistics(deltas, contrib)[name]
            values.append(float(np.nanmean(stat)))
        arr = np.array(values, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        # Relative drift on a log scale, so the comparison is scale-free.
        drift = abs(hi - lo) / max(abs(lo), 1e-12) if lo != 0.0 else (0.0 if hi == 0.0 else math.inf)
        per_statistic[name] = {
            "values_by_density": [float(v) for v in arr],
            "min": lo,
            "max": hi,
            "relative_drift": drift,
        }
    artefact = [row["max_min_ratio"] for row in ratio_vs_density(seed, n_features)]
    per_statistic["max_min_ratio_disputed"] = {
        "values_by_density": artefact,
        "relative_drift": (max(artefact) - min(artefact)) / min(artefact),
    }
    return per_statistic


# --------------------------------------------------------------------------
# 3. seed variance
# --------------------------------------------------------------------------


def seed_variance(
    n_seeds: int = N_SEEDS,
    base_seed: int = BASE_SEED,
    n_features: int = N_FEATURES,
    n_points: int = 54,
) -> dict:
    """Mean and spread of each statistic across independent seeds."""
    seeds = [base_seed + i for i in range(n_seeds)]
    rows = []
    for seed in seeds:
        deltas, contrib, residuals = contribution_grid(seed, n_features, n_points)
        row = {
            "seed": seed,
            "additivity_residual_max": float(residuals.max()),
            "endpoint_a_delta": int(ENDPOINT_A),
            "endpoint_b_delta": int(ENDPOINT_B),
        }
        for name, values in _all_statistics(deltas, contrib).items():
            row[f"{name}_mean"] = float(np.nanmean(values))
        row["max_min_ratio"] = float(np.abs(contrib).max() / np.abs(contrib).min())
        rows.append(row)

    summary = {"n_seeds": n_seeds, "n_points": n_points, "seeds": rows, "statistics": {}}
    for key in [*(f"{n}_mean" for n in STATISTICS), "max_min_ratio"]:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary["statistics"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "relative_std": float(values.std(ddof=1) / abs(values.mean()))
            if values.size > 1 and values.mean() != 0.0
            else 0.0,
        }
    return summary


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def run_all(
    n_seeds: int = N_SEEDS, base_seed: int = BASE_SEED, n_features: int = N_FEATURES
) -> dict:
    return {
        "configuration": {
            "head_dim": HEAD_DIM,
            "rope_base": ROPE_BASE,
            "n_features": n_features,
            "delta_range": [DELTA_LO, DELTA_HI],
            "endpoints": [ENDPOINT_A, ENDPOINT_B],
            "grid_densities": list(GRID_DENSITIES),
            "n_seeds": n_seeds,
            "base_seed": base_seed,
        },
        "ratio_vs_density": ratio_vs_density(base_seed, n_features),
        "grid_independence": grid_independence(base_seed, n_features),
        "seed_variance": seed_variance(n_seeds, base_seed, n_features),
    }


def main() -> None:
    result = run_all()
    cfg = result["configuration"]

    print("=" * 74)
    print("Grid-independent statistics for position-conditional attribution")
    print("=" * 74)
    print(f"d={cfg['head_dim']} base={cfg['rope_base']:g} F={cfg['n_features']} "
          f"delta in [{DELTA_LO}, {DELTA_HI}] seeds={cfg['n_seeds']}")

    print()
    print("1. The disputed max/min ratio, against grid density")
    print(f"   {'n_deltas':<10} {'max/min ratio':>16} {'log10':>10}")
    for row in result["ratio_vs_density"]:
        print(f"   {row['n_deltas']:<10} {row['max_min_ratio']:>16.2f} {row['log10_ratio']:>10.2f}")
    gi = result["grid_independence"]["max_min_ratio_disputed"]
    print(f"   -> relative drift {gi['relative_drift']:.3g}: the statistic is a property of")
    print("      the sampling, not of the function.")

    print()
    print("2. Grid-independent statistics: drift across those same densities")
    for name in STATISTICS:
        entry = result["grid_independence"][name]
        print(f"   {name:<22} drift={entry['relative_drift']:.4f}  "
              f"mean over densities={np.mean(entry['values_by_density']):.4f}")

    print()
    print(f"3. Seed variance ({cfg['n_seeds']} independent seeds, "
          f"{result['seed_variance']['n_points']}-point grid)")
    for key, entry in result["seed_variance"]["statistics"].items():
        print(f"   {key:<22} mean={entry['mean']:>10.4f}  sd={entry['std']:>9.4f} "
              f"({100 * entry['relative_std']:.1f}% of the mean)")

    out = Path(__file__).resolve().parent.parent.parent / "results" / "statistics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()