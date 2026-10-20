"""Seed-to-seed variance of the position-free cost measurement.

The paper's pooled sign-flip fraction is computed under one content-pair
seed (``usefulness.DEFAULT_SEED``). A reader is entitled to ask whether the
headline is a property of the three checkpoints or of that one draw. This
module re-runs the whole sweep under three pre-registered seeds -- the
default and its two successors, fixed here before any run, not chosen after
seeing numbers -- and writes the per-seed pooled values plus their mean and
sample standard deviation.

What varies between seeds is which content pairs are drawn (256 pairs of
token positions per head); the models are loaded in ``eval`` mode and the
measurement is otherwise deterministic. The artifact therefore reports the
sampling variance of the pooled statistic, not training or checkpoint noise
-- the docstring states the limit so the number cannot be over-read.

A full re-run costs the same as three ``usefulness`` sweeps (about ten
minutes on one CPU thread); the test suite pins the artifact's structure,
its arithmetical consistency, and its agreement with ``usefulness.json`` at
the shared seed rather than repeating the sweep.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rope_attribution import real_model as RM
from rope_attribution import usefulness as U

SCHEMA = "rope_attribution/usefulness_seeds/v1"

#: Pre-registered: the default seed and the two integers after it.
SEEDS: tuple[int, ...] = (U.DEFAULT_SEED, U.DEFAULT_SEED + 1, U.DEFAULT_SEED + 2)

#: The pooled fields across which mean and sd are reported.
POOLED_FIELDS: tuple[str, ...] = (
    "sign_flip_fraction",
    "sign_flip_fraction_min_head",
    "sign_flip_fraction_max_head",
    "rel_err_median",
    "rel_err_p90",
)

#: Per-model fields kept in the artifact (the full prompt/head records live in
#: usefulness.json; here only the aggregates the variance needs).
MODEL_FIELDS: tuple[str, ...] = (
    "sign_flip_fraction_pooled",
    "sign_flip_fraction_min_head",
    "sign_flip_fraction_max_head",
    "rel_err_median",
    "rel_err_p90",
)


def run_all(seeds: Sequence[int] = SEEDS) -> dict:
    per_seed = []
    for seed in seeds:
        data = U.run_all(seed=int(seed))
        per_seed.append(
            {
                "seed": int(seed),
                "pooled": {field: data["pooled"][field] for field in POOLED_FIELDS},
                "models": [
                    {"model_id": model["model_id"],
                     **{field: model[field] for field in MODEL_FIELDS}}
                    for model in data["models"]
                ],
            }
        )
    mean = {
        field: float(np.mean([row["pooled"][field] for row in per_seed]))
        for field in POOLED_FIELDS
    }
    sd = {
        field: float(np.std([row["pooled"][field] for row in per_seed], ddof=1))
        for field in POOLED_FIELDS
    }
    return {
        "schema": SCHEMA,
        "model_ids": list(U.DEFAULT_MODEL_IDS),
        "seeds": [int(seed) for seed in seeds],
        "n_content_pairs": int(RM.N_CONTENT_PAIRS),
        "what_varies": "content-pair draws only; checkpoints are eval-mode "
        "and the measurement is otherwise deterministic",
        "per_seed": per_seed,
        "mean": mean,
        "sd": sd,
    }


def write_results(data: dict) -> Path:
    out = Path(__file__).resolve().parents[2] / "results" / "usefulness_seeds.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return out


def main() -> None:
    data = run_all()
    for row in data["per_seed"]:
        pooled = row["pooled"]
        print(
            f"seed {row['seed']}: sign flips {pooled['sign_flip_fraction']:.4f}, "
            f"rel err median {pooled['rel_err_median']:.4f}"
        )
    for field in ("sign_flip_fraction", "rel_err_median"):
        print(
            f"across seeds: {field} = {data['mean'][field]:.4f} "
            f"+/- {data['sd'][field]:.4f} (mean +/- sd, n={len(data['seeds'])})"
        )
    write_results(data)


if __name__ == "__main__":
    main()
