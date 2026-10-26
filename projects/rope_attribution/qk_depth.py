"""Depth profile of the per-distance QK attribution on one checkpoint.

``qk_attribution`` measures one public dictionary on one block. This module
repeats that measurement once per block the checkpoint can address and records
the aggregates, so the paper's layer-3 case study can be read as a depth
profile instead of a single point.

The addressing rule is the one enforced by ``qk_attribution._attention_index``:
a dictionary trained on block ``L`` encodes ``hidden_states[L + 1]``, which is
the input to block ``L + 1``'s attention. A six-block model therefore has five
attributable dictionaries, layers 0--4; layer 5 has no attention block after it.

Every row comes from the same ``measure_model`` call as the headline artifact,
with the same passage, seed, sweep and pair count. Only ``sae_layer`` and the
dictionary file differ, so the layer-3 row is expected to equal
``results/qk_attribution.json`` exactly, and a test checks that rather than
trusting it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rope_attribution import qk_attribution as Q

__all__ = [
    "DEFAULT_LAYERS",
    "SCHEMA",
    "depth_profile",
    "main",
    "write_results",
]

SCHEMA = "qk-depth/v1"

#: Every block of pythia-70m except the last, which no dictionary can explain.
DEFAULT_LAYERS: tuple[int, ...] = (0, 1, 2, 3, 4)

_FAMILY = "ctigges/pythia-70m-deduped__res-sm_processed/{layer}-res-sm"


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _row(measured: Mapping[str, Any]) -> dict[str, Any]:
    """One layer's aggregates, named positively rather than by nested keys."""
    aggregate = Q.aggregate_heads(measured["heads"])
    return {
        "sae_layer": int(measured["sae_layer"]),
        "attention_layer": int(measured["attention_layer"]),
        "ln_residual": float(measured["ln_residual"]),
        "active_features_mean": float(measured["active_features_mean"]),
        "max_relative_exactness_error": float(aggregate["max_relative_exactness_error"]),
        "sweep_rotary_sign_flip_fraction": float(
            aggregate["sweep_rotary"]["sign_flip_fraction"]
        ),
        "sweep_feature_sign_flip_fraction": float(
            aggregate["sweep_feature"]["sign_flip_fraction"]
        ),
        "sweep_pair_sign_flip_fraction": float(
            aggregate["sweep_pair"]["sign_flip_fraction"]
        ),
        "feature_rotary_sign_flip_fraction": float(
            aggregate["feature_rotary"]["sign_flip_fraction"]
        ),
        "base_share_mean": float(aggregate["base_share_mean"]),
        "top_k_share_mean": float(aggregate["top_k_share_mean"]),
        "sweep_rotary_top1_agreement_mean": _optional_float(
            aggregate["sweep_rotary"]["top1_agreement_mean"]
        ),
        "sweep_rotary_n_material_cells": int(
            aggregate["sweep_rotary"]["n_material_cells"]
        ),
    }


def depth_profile(
    model_id: str,
    sae_paths: Mapping[int, str | Path],
    text: str = Q.DEFAULT_TEXT,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure every supplied dictionary on its own block of one checkpoint.

    ``sae_paths`` maps a block index to that block's dictionary file; the
    blocks are measured in ascending order so the artifact is deterministic.
    """
    if not sae_paths:
        raise ValueError("depth_profile needs at least one dictionary")
    rows = [
        _row(
            Q.measure_model(model_id, sae_paths[layer], text, seed=seed, sae_layer=layer)
        )
        for layer in sorted(sae_paths)
    ]
    return {
        "schema": SCHEMA,
        "model_id": model_id,
        "text": text,
        "seed": int(seed),
        "deltas": [int(d) for d in Q.DEFAULT_DELTAS],
        "sweep_deltas": [int(d) for d in Q.SWEEP_DELTAS],
        "pairs_per_delta": int(Q.DEFAULT_PAIRS_PER_DELTA),
        "top_k": int(Q.DEFAULT_TOP_K),
        "dictionary_family": _FAMILY,
        "layers": rows,
    }


def write_results(data: dict[str, Any], path: Path | None = None) -> Path:
    out = (
        path
        or Path(__file__).resolve().parent.parent.parent
        / "results"
        / "qk_depth.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=Q.DEFAULT_MODEL_ID)
    parser.add_argument(
        "--sae-dir",
        type=Path,
        required=True,
        help=(
            "directory laid out like the Hugging Face repository, i.e. "
            "{layer}-res-sm/sae_weights.safetensors for every requested layer. "
            "The dictionaries are not vendored."
        ),
    )
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for layer in DEFAULT_LAYERS),
        help="comma-separated SAE layers to measure (default 0,1,2,3,4)",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    layers = [int(part) for part in str(args.layers).split(",") if part.strip()]
    paths = {
        layer: args.sae_dir / f"{layer}-res-sm" / "sae_weights.safetensors"
        for layer in layers
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing dictionary files: " + ", ".join(missing))
    data = depth_profile(args.model, paths)
    path = write_results(data, args.out)
    print(f"{'sae':>3} {'attn':>4} {'sw-rot':>7} {'sw-feat':>7} {'sw-pair':>7} {'base':>6}")
    for row in data["layers"]:
        print(
            f"{row['sae_layer']:>3} {row['attention_layer']:>4} "
            f"{row['sweep_rotary_sign_flip_fraction']:>7.4f} "
            f"{row['sweep_feature_sign_flip_fraction']:>7.5f} "
            f"{row['sweep_pair_sign_flip_fraction']:>7.4f} "
            f"{row['base_share_mean']:>6.3f}"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
