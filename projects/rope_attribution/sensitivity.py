"""Is the position-free cost statistic a configuration artifact?

Why this module exists
----------------------
The paper quotes a pooled sign-error rate of the least-squares position-free
surrogate on real checkpoints. A skeptical reviewer's cheapest objection is
configurational: the rate is measured at rotary base 10000 and head dimension
64, so maybe the rate is a property of those two numbers rather than of the
decomposition. The trained checkpoints come in three architectures but all sit
at base 10000 in their rotary config, so no real model can answer the base
half of the question.

This module answers both halves synthetically, and it answers them with the
*same cell definition* the real measurement uses: ``usefulness.measure_pair_cost``
is called as-is, on synthetic tokens whose construction mirrors
``statistics._decomposition`` (a few SAE-like features with fixed directions,
per-token magnitudes, projections through 1/sqrt(d)-scaled key/query matrices).
Nothing here re-types the statistic, so a sweep that disagrees with the paper's
number is a difference in the tokens, never in the definition.

What is measured
----------------
Per (rotary base, head dimension): the pooled material-cell sign-flip fraction
of the best position-free constant, over 256 content pairs, 3 sweep seeds, on
the pre-registered distance grid the real measurement uses. Same seed gives the
same tokens across bases, so the base sweep is paired, not re-drawn.

Scope, stated once
------------------
All numbers here are properties of the synthetic construction: they say the
paper's statistic does not hinge on one configuration. They do not say what
the rate is on trained weights; those numbers live in ``usefulness.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from rope_attribution import usefulness as U
from rope_attribution.rope import inv_freq

__all__ = [
    "BASES",
    "HEAD_DIMS",
    "N_SWEEP_SEEDS",
    "SWEEP_BASE_SEED",
    "N_TOKENS",
    "N_PAIRS",
    "measure_config",
    "run_all",
    "main",
]

# Rotary bases swept, log-spaced ends of the real-world range. 10000 is the
# published configuration; 500000 is the legacy base the paper measures as a
# control (under an extension scale, which is out of scope here: this sweep
# varies the base itself, one factor at a time).
BASES: tuple[float, ...] = (1000.0, 10000.0, 100000.0, 500000.0)

# Head dimensions swept. 64 is the published configuration; 32 and 128 bracket
# it, and their rotary-pair counts (16, 64) bracket 32 pairs.
HEAD_DIMS: tuple[int, ...] = (32, 64, 128)

# Sweep seeds. Three, not one: the seed variance the sweep must not hide is
# the token draw's, and three seeds make an sd computable.
N_SWEEP_SEEDS = 3

#: Own seed space, disjoint from statistics.py's sweep (BASE_SEED 20250929) by
#: a different constant, so the two sweeps never share a draw by accident.
SWEEP_BASE_SEED = 20261010

#: Content vectors per synthetic layer -- the same order as the real
#: measurement's per-layer key pool (``real_model.HeadVectors.n_tokens``).
N_TOKENS = 256

#: Content pairs measured per (configuration, seed) -- the same order as the
#: real measurement's ``real_model.N_CONTENT_PAIRS``.
N_PAIRS = 256


def _synthetic_tokens(head_dim: int, rng: np.random.Generator) -> tuple:
    """One synthetic layer's content projections.

    Mirrors ``statistics._decomposition``: ``n_features`` fixed random feature
    directions, per-token random magnitudes in the same coefficient range, and
    independent query/key projections scaled like real weight matrices.
    Returns ``(q_hat, k_hat)``, each ``(N_TOKENS, head_dim)``.
    """
    n_features = 8
    dirs = rng.standard_normal((n_features, head_dim))
    w_q = rng.standard_normal((head_dim, head_dim)) / math.sqrt(head_dim)
    w_k = rng.standard_normal((head_dim, head_dim)) / math.sqrt(head_dim)
    acts = rng.random((N_TOKENS, n_features)) * 1.5 + 0.1
    tokens = acts @ dirs
    return tokens @ w_q, tokens @ w_k


def _draw_pairs(rng: np.random.Generator) -> list:
    """``N_PAIRS`` distinct-position content pairs ``(m, n)``, ``m != n``.

    The offset trick guarantees distinctness in closed form, so no redraw loop
    can diverge between numpy versions.
    """
    m = rng.integers(0, N_TOKENS, size=N_PAIRS)
    n = (m + 1 + rng.integers(0, N_TOKENS - 1, size=N_PAIRS)) % N_TOKENS
    return list(zip(m.tolist(), n.tolist(), strict=True))


def measure_config(head_dim: int, base: float, seed: int) -> dict:
    """The cost statistic over one synthetic configuration.

    Pools the same per-cell quantities ``usefulness._pooled`` pools -- material
    cells, sign flips, per-pair relative-error maxima -- across the content
    pairs of one synthetic layer. The pool is explicit rather than a mean of
    per-pair fractions, so a pair with no material cells cannot raise the rate.
    """
    rng = np.random.default_rng(seed)
    q_hat, k_hat = _synthetic_tokens(head_dim, rng)
    freqs = inv_freq(head_dim, base)
    deltas = np.array(U.DISTANCE_GRID, dtype=np.int64)

    n_cells = 0
    n_flips = 0
    rels = []
    pairs = _draw_pairs(rng)
    for m, n in pairs:
        cost = U.measure_pair_cost(q_hat[m], k_hat[n], freqs, deltas)
        n_cells += int(cost["n_material"].sum())
        n_flips += int(cost["sign_flip_fraction"].sum())
        live = cost["live"]
        if live.any():
            rels.append(cost["rel_err"][live])
    rel = np.concatenate(rels) if rels else np.array([])
    rate = n_flips / n_cells if n_cells else float("nan")
    return {
        "base": float(base),
        "head_dim": int(head_dim),
        "rope_pairs": int(head_dim // 2),
        "seed": int(seed),
        "n_content_pairs": len(pairs),
        "n_material_cells": n_cells,
        "n_sign_flips": n_flips,
        "sign_flip_fraction": rate,
        "rel_err_median": float(np.median(rel)) if rel.size else float("nan"),
    }


def run_all() -> dict:
    """The full sweep: every (base, head dim) configuration, 3 seeds each."""
    records = []
    for head_dim in HEAD_DIMS:
        for base in BASES:
            per_seed = [
                measure_config(head_dim, base, SWEEP_BASE_SEED + s)
                for s in range(N_SWEEP_SEEDS)
            ]
            rates = [r["sign_flip_fraction"] for r in per_seed]
            records.append(
                {
                    "base": float(base),
                    "head_dim": int(head_dim),
                    "rope_pairs": int(head_dim // 2),
                    "per_seed": per_seed,
                    "sign_flip_fraction_mean": float(np.mean(rates)),
                    "sign_flip_fraction_sd": (
                        float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
                    ),
                    "rel_err_median_mean": float(
                        np.mean([r["rel_err_median"] for r in per_seed])
                    ),
                }
            )

    def _band(rows: list) -> list:
        rates = [r["sign_flip_fraction_mean"] for r in rows]
        return [min(rates), max(rates)]

    at_64 = [r for r in records if r["head_dim"] == 64]
    at_10k = [r for r in records if r["base"] == 10000.0]
    return {
        "schema": "rope_attribution/sensitivity/v1",
        "bases": list(BASES),
        "head_dims": list(HEAD_DIMS),
        "n_sweep_seeds": N_SWEEP_SEEDS,
        "sweep_base_seed": SWEEP_BASE_SEED,
        "n_tokens": N_TOKENS,
        "n_content_pairs_per_seed": N_PAIRS,
        "distance_grid": list(U.DISTANCE_GRID),
        "records": records,
        # The two bands the paper quotes: across bases at the published head
        # dimension, and across head dimensions at the published base.
        "base_band_at_dim_64": _band(at_64),
        "dim_band_at_base_10000": _band(at_10k),
    }


def main() -> None:
    data = run_all()
    for r in data["records"]:
        print(
            f"  base {r['base']:>9.0f}  dim {r['head_dim']:>3d} "
            f"({r['rope_pairs']:>2d} pairs): rate "
            f"{100.0 * r['sign_flip_fraction_mean']:.1f}% "
            f"(sd {100.0 * r['sign_flip_fraction_sd']:.1f}pp, "
            f"rel-err median {r['rel_err_median_mean']:.2f})"
        )
    bb, db = data["base_band_at_dim_64"], data["dim_band_at_base_10000"]
    print(
        f"base band at dim 64: {100.0 * bb[0]:.1f}% to {100.0 * bb[1]:.1f}%"
    )
    print(
        f"dim band at base 1e4: {100.0 * db[0]:.1f}% to {100.0 * db[1]:.1f}%"
    )
    out = Path(__file__).resolve().parents[2] / "results" / "sensitivity.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
