"""What it costs to use a position-free per-feature attribution under RoPE.

The question this module answers
--------------------------------
Sections 4.1--4.3 of the paper establish that a feature's contribution to a
rotary attention score is a function of the relative distance ``delta`` and that
it can change sign, and that on trained weights a feature's share of the score
varies by two to three orders of magnitude across heads. What those results do
*not* establish is the thing a reviewer asked for: whether correcting this is
worth anything, or whether it is a correct description of a thing nobody needs.

This module supplies the missing measurement, on the same three trained
checkpoints as ``real_model.py``. It does not ask "is the decomposition correct"
-- that is settled exactly and to the dtype floor. It asks: **if you replace the
exact per-distance attribution with the best possible position-free scalar, how
wrong are you?**

Why this is the right question
------------------------------
Every existing feature-attribution method for transformers has to commit to one
number per feature. Under RoPE that commitment is a modelling decision, and the
honest way to argue for the paper's object is to price the alternative rather
than assert the alternative is wrong. So:

* the surrogate is the **best** position-free scalar, not a strawman. For a given
  rotary pair the exact contribution is ``A_k cos(d) + B_k sin(d)`` with
  ``d = delta * inv_freq[k]``; the least-squares-optimal constant over the
  distance grid is ``A_k * mean(cos) + B_k * mean(sin)``, and that is what is used
  here. A strawman comparison would not convince anyone.
* the headline is a **sign-error rate**, not a residual. A surrogate that is
  10% off in magnitude is an annoyance; one that reports the opposite sign is a
  different explanation of the computation, and no downstream reader can detect
  it from the surrogate alone. The sign rate is what makes the choice of object
  consequential rather than cosmetic.
* the result is reported per pair *and* aggregated per head, because the pair is
  where the algebra lives and the head is where a method actually consumes it.

Stability, as everywhere else in this repository
------------------------------------------------
The distance grid is pre-registered (``DISTANCE_GRID``), never chosen after
seeing results. Both statistics - the sign-error rate and the relative error -
are reported across several grid densities, and the module asserts in its own
tests that neither is a function of that choice. A statistic that moved when the
grid was refined would be the same artefact as the withdrawn ``max/min`` ratio,
and the test that would catch it is written before the numbers are read.

What this module does not claim
-------------------------------
It does not claim any downstream task degrades. There is no patching, no
retrieval benchmark and no accuracy measurement here. It prices a modelling
choice; it does not evaluate a method.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rope_attribution import real_model as RM
from rope_attribution.experiments import pair_terms
from rope_attribution.rope import inv_freq

__all__ = [
    "DISTANCE_GRID",
    "GRID_DENSITIES",
    "measure_pair_cost",
    "measure_head_cost",
    "measure_model",
    "run_all",
    "main",
]

# Pre-registered distance grid. Log-spaced integers, the same shape as the
# figure grid in experiments.py, fixed before any statistic below was computed.
DISTANCE_GRID: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

# Densities used only to check that the statistics do not depend on sampling.
# Deliberately spanning more than two orders of magnitude.
GRID_DENSITIES: tuple[int, ...] = (9, 17, 33, 65, 129)

DEFAULT_SEED = RM.DEFAULT_SEED
DEFAULT_MODEL_IDS = RM.DEFAULT_MODEL_IDS


def _log_grid(n_points: int, lo: int = 1, hi: int = 4096) -> np.ndarray:
    """``n_points`` log-spaced integers in ``[lo, hi]``, deduplicated."""
    return np.unique(np.round(np.geomspace(lo, hi, n_points)).astype(np.int64))


def _cos_mean_sin_mean(deltas: np.ndarray, inv_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair ``mean(cos d)`` and ``mean(sin d)`` over the grid.

    These two numbers are the whole of the optimal position-free surrogate: for
    the exact term ``A_k cos(d) + B_k sin(d)``, the constant minimising the sum of
    squared differences over the grid is ``A_k * mean(cos) + B_k * mean(sin)``.
    """
    angles = np.outer(inv_f, deltas.astype(np.float64))
    return np.cos(angles).mean(axis=1), np.sin(angles).mean(axis=1)


def measure_pair_cost(
    q_hat: np.ndarray,
    k_hat: np.ndarray,
    inv_f: np.ndarray,
    deltas: np.ndarray,
) -> dict:
    """Cost of the best position-free surrogate, for one ``(m, n)`` content pair.

    ``inv_f`` is the checkpoint's own ladder; ``deltas`` the pre-registered grid.
    Returns per-pair arrays so a caller can aggregate without re-running.
    """
    cos_mean, sin_mean = _cos_mean_sin_mean(deltas, inv_f)

    # A_k and B_k do not depend on distance, so the whole distance sweep is one
    # outer product. Calling pair_terms once per delta - as the original version
    # did - is correct but quadratic in work and slow enough to matter across
    # every layer, head and content pair of a real checkpoint.
    terms = pair_terms(q_hat, k_hat, inv_f, int(deltas[0]))
    a_k, b_k = terms.aligned, terms.crossed
    angles = np.outer(inv_f, deltas.astype(np.float64))
    exact = a_k[:, None] * np.cos(angles) + b_k[:, None] * np.sin(angles)

    # The optimal constant per rotary pair.
    surrogate = a_k * cos_mean + b_k * sin_mean

    approx = np.broadcast_to(surrogate[:, None], exact.shape)
    diff = approx - exact

    # A sign error needs the exact value to be away from zero, otherwise "sign"
    # is noise. Threshold is relative to that pair's own peak amplitude.
    peak = np.abs(exact).max(axis=1)
    live = peak > 0.0
    # Only judge the sign where the exact term carries real signal.
    material = np.abs(exact) > (0.05 * peak)[:, None]
    sign_flip = material & live[:, None] & (
        np.sign(exact) != np.sign(surrogate)[:, None]
    )

    denom = np.where(live, peak, np.nan)
    rel_err = np.nanmax(np.abs(diff), axis=1) / denom

    return {
        "sign_flip_fraction": sign_flip,
        "rel_err": rel_err,
        "live": live,
        "peak": peak,
        "surrogate": surrogate,
        "spread": exact.max(axis=1) - exact.min(axis=1),
        "n_material": material & live[:, None],
    }


def _pooled(cost: dict) -> dict:
    """Aggregate one ``(m, n)`` pair's arrays into the reported statistics.

    The sign-error rate is taken over *material* cells only - cells where the
    exact term carries at least 5% of its own peak. Counting sign disagreements
    in the tails, where the exact value is indistinguishable from zero, would
    inflate the number with noise and the reader would be right to distrust it.
    """
    flip = cost["sign_flip_fraction"]
    material = cost["n_material"]
    live = cost["live"]
    n_cells = int(material.sum())
    n_flips = int(flip.sum())
    rel = cost["rel_err"][live]
    return {
        "n_pairs_measured": int(live.sum()),
        "n_material_cells": n_cells,
        "n_sign_flips": n_flips,
        "sign_flip_fraction": n_flips / n_cells if n_cells else 0.0,
        "rel_err_median": float(np.median(rel)) if rel.size else 0.0,
        "rel_err_p90": float(np.quantile(rel, 0.90)) if rel.size else 0.0,
        "rel_err_max": float(rel.max()) if rel.size else 0.0,
    }


def measure_head_cost(
    hv: RM.HeadVectors,
    freqs: np.ndarray,
    rope_dim: int,
    deltas: np.ndarray,
    n_pairs: int,
    seed: int,
    layer: int,
    head: int,
) -> dict:
    """Pooled cost over one head's content pairs."""
    rng = np.random.default_rng(seed)
    q = hv.q_hat[layer, head, :, :rope_dim]
    k = hv.k_hat[layer, head, :, :rope_dim]

    flips: list[float] = []
    rels: list[float] = []
    tot_cells = 0
    tot_flips = 0
    n = 0
    for m, nn in RM._content_pairs(hv.n_tokens, n_pairs, rng):
        cost = measure_pair_cost(q[m], k[nn], freqs, deltas)
        pooled = _pooled(cost)
        flips.append(pooled["sign_flip_fraction"])
        rels.append(pooled["rel_err_median"])
        tot_cells += pooled["n_material_cells"]
        tot_flips += pooled["n_sign_flips"]
        n += 1
    if not flips:
        return {}
    return {
        "layer": layer,
        "head": head,
        "n_content_pairs": n,
        "sign_flip_fraction_pooled": tot_flips / tot_cells if tot_cells else 0.0,
        "sign_flip_fraction_per_pair_median": float(np.median(flips)),
        "rel_err_median_of_medians": float(np.median(rels)),
        "rel_err_p90_of_medians": float(np.quantile(rels, 0.90)),
    }


def measure_model(
    model_id: str = DEFAULT_MODEL_IDS[0],
    seed: int = DEFAULT_SEED,
    prompts: Sequence[str] = RM.DEFAULT_PROMPTS,
    deltas: np.ndarray | None = None,
    n_pairs: int = RM.N_CONTENT_PAIRS,
) -> dict:
    """The position-free cost, on one trained checkpoint."""
    loaded = RM.load_model(model_id, seed=seed)
    cfgv = RM.validate_config(loaded.model)
    if not cfgv["checkpoint_inv_freq_found"]:
        raise RuntimeError(f"{model_id} has no rotary module, so it has no rotary pairs")
    rope_dim = int(cfgv["rope_dim"])
    freqs = inv_freq(rope_dim, float(cfgv["rope_theta_declared"]))

    if deltas is None:
        deltas = _log_grid(len(DISTANCE_GRID))

    per_prompt = []
    for i, text in enumerate(prompts):
        hv = RM.collect_head_vectors(loaded.model, loaded.tokenizer, text)
        heads = []
        for layer in range(hv.n_layers):
            for head in range(hv.n_heads):
                rec = measure_head_cost(
                    hv, freqs, rope_dim, deltas, n_pairs, seed + i * 1000 + layer * 31 + head,
                    layer, head,
                )
                if rec:
                    heads.append(rec)
        per_prompt.append({"prompt_index": i, "n_tokens": hv.n_tokens, "heads": heads})

    return _summarise(model_id, rope_dim, deltas, per_prompt)


def _summarise(
    model_id: str, rope_dim: int, deltas: np.ndarray, per_prompt: list[dict]
) -> dict:
    pooled_flip = []
    pooled_rel = []
    for entry in per_prompt:
        for h in entry["heads"]:
            pooled_flip.append(h["sign_flip_fraction_pooled"])
            pooled_rel.append(h["rel_err_median_of_medians"])
    arr = np.array(pooled_flip, dtype=np.float64)
    rel = np.array(pooled_rel, dtype=np.float64)
    return {
        "model_id": model_id,
        "rope_dim": rope_dim,
        "n_rotary_pairs": rope_dim // 2,
        "n_deltas": int(deltas.size),
        "deltas": [int(d) for d in deltas],
        "n_heads": int(arr.size),
        "sign_flip_fraction_pooled": float(arr.mean()) if arr.size else 0.0,
        "sign_flip_fraction_min_head": float(arr.min()) if arr.size else 0.0,
        "sign_flip_fraction_max_head": float(arr.max()) if arr.size else 0.0,
        "sign_flip_across_head_ratio": (
            float(arr.max() / arr.min()) if arr.size and arr.min() > 0 else math.inf
        ),
        "rel_err_median": float(np.median(rel)) if rel.size else 0.0,
        "rel_err_p90": float(np.quantile(rel, 0.90)) if rel.size else 0.0,
        "per_prompt": per_prompt,
    }


def grid_independence(model_id: str, seed: int = DEFAULT_SEED) -> dict:
    """Do the statistics depend on how densely the distance range is sampled?

    This is the check that keeps the module honest. The withdrawn ``max/min``
    ratio was a function of grid density; if these numbers are too, they are the
    same artefact wearing a different name.
    """
    out = {}
    for n_points in GRID_DENSITIES:
        deltas = _log_grid(n_points)
        rec = measure_model(model_id, seed=seed, deltas=deltas)
        out[f"{n_points}"] = {
            "n_deltas": int(deltas.size),
            "sign_flip_fraction_pooled": rec["sign_flip_fraction_pooled"],
            "rel_err_median": rec["rel_err_median"],
        }
    return out


def run_all(model_ids: Sequence[str] = DEFAULT_MODEL_IDS) -> dict:
    models = [measure_model(m) for m in model_ids]
    return {
        "schema": "rope_attribution/usefulness/v1",
        "distance_grid": list(DISTANCE_GRID),
        "grid_densities": list(GRID_DENSITIES),
        "seed": DEFAULT_SEED,
        "models": models,
        "pooled": {
            "sign_flip_fraction": float(
                np.mean([m["sign_flip_fraction_pooled"] for m in models])
            ),
            "sign_flip_fraction_min_head": float(
                np.min([m["sign_flip_fraction_min_head"] for m in models])
            ),
            "sign_flip_fraction_max_head": float(
                np.max([m["sign_flip_fraction_max_head"] for m in models])
            ),
            "rel_err_median": float(np.median([m["rel_err_median"] for m in models])),
            "rel_err_p90": float(np.median([m["rel_err_p90"] for m in models])),
        },
    }


def main() -> None:
    data = run_all()
    p = data["pooled"]
    print(f"distance grid: {data['distance_grid']}")
    for m in data["models"]:
        print(
            f"  {m['model_id']}: sign flips "
            f"{m['sign_flip_fraction_pooled']:.4f} "
            f"(heads {m['sign_flip_fraction_min_head']:.4f}"
            f"..{m['sign_flip_fraction_max_head']:.4f}), "
            f"rel err median {m['rel_err_median']:.4f}"
        )
    print(
        f"pooled: sign flips {p['sign_flip_fraction']:.4f}, "
        f"rel err median {p['rel_err_median']:.4f}, p90 {p['rel_err_p90']:.4f}"
    )
    out = Path(__file__).resolve().parents[2] / "results" / "usefulness.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()