"""What context extension does to the *measured* cost of a position-free summary.

``extension_law`` counts the rotary channels that are past ``theta L = 2*pi``
under each extension scheme.  That count is a prediction about the geometry, not
a measurement on a checkpoint.  This module closes the loop: it holds the
checkpoint's own unrotated query/key vectors fixed and re-prices the same
content pairs under the scheme's frequency ladder.  The question is

    if this head had been trained with this geometry, how much of the
    position-free cost would remain?

The measurement is deliberately a *counterfactual geometry* probe.  The
checkpoint was trained under plain RoPE; changing the ladder does not change
the weights.  It therefore measures the part of the cost that the geometry
alone controls, and it must not be read as the accuracy of a YaRN-trained
model.  The plain-RoPE row is the control, and it must reproduce the headline
artifact exactly because it runs the same code on the same vectors.

Why this is worth a separate artifact: the threshold law says extension reduces
the number of channels that sweep a full period, but it does not say what
fraction of the *measured* sign errors disappears.  The answer is not one: the
sign-error rate is a phase-sensitive functional, not the minimax error.  The
artifact reports both, so the gap between them is visible rather than assumed.

Artifact: ``results/usefulness_extension.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import extension_law as EL
from . import real_model as RM
from . import usefulness as U

__all__ = [
    "CONTEXT_LENGTHS",
    "DEFAULT_SCHEMES",
    "SCHEMA",
    "measure_model_schemes",
    "pool_schemes",
    "run_all",
    "write_results",
    "main",
]

SCHEMA = "rope-attribution/usefulness-extension/v1"

#: ``(scheme, scale)`` rows.  Scale ``1`` is the plain-RoPE control; the two
#: families are kept separate because they act on different channels.
DEFAULT_SCHEMES: tuple[tuple[str, float], ...] = (
    ("rope", 1.0),
    ("yarn", 2.0),
    ("yarn", 4.0),
    ("yarn", 8.0),
    ("yarn", 16.0),
    ("yarn", 32.0),
    ("position_interpolation", 2.0),
    ("position_interpolation", 4.0),
    ("position_interpolation", 8.0),
    ("position_interpolation", 16.0),
    ("position_interpolation", 32.0),
)

#: Context lengths at which the channel count is reported.  The measurement
#: grid ends at 4096, so that is the primary column; the longer lengths show
#: where the same ladder would stand after a further extension.
CONTEXT_LENGTHS: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768)


def _frequencies(
    scheme: str,
    scale: float,
    *,
    head_dim: int,
    base: float,
    original_context: int,
) -> np.ndarray:
    """The scheme's ladder at the checkpoint's own rotary width and base."""
    return EL.scheme_frequencies(
        scheme,
        scale,
        head_dim=head_dim,
        base=base,
        original_context=original_context,
    )


def _channel_counts(freqs: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Past-threshold counts and the two reference errors, per context length."""
    out: dict[str, dict[str, float | int]] = {}
    for length in CONTEXT_LENGTHS:
        x = freqs * float(length)
        out[str(length)] = {
            "n_past_threshold": int(np.sum(x >= EL.THRESHOLD)),
            "fraction_past_threshold": float(np.mean(x >= EL.THRESHOLD)),
            "minimax_mean": float(np.mean(EL.minimax_error(x))),
            "worst_case_mean": float(np.mean(EL.worst_case_error(x))),
        }
    return out


def _scheme_row(
    model_id: str,
    scheme: str,
    scale: float,
    freqs: np.ndarray,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """One compact model x scheme row; the per-head rates stay auditable."""
    counts = _channel_counts(freqs)
    return {
        "model_id": model_id,
        "scheme": scheme,
        "scale": float(scale),
        "n_channels": int(freqs.size),
        "channels": counts,
        "sign_flip_fraction_pooled": float(summary["sign_flip_fraction_pooled"]),
        "sign_flip_fraction_min_head": float(summary["sign_flip_fraction_min_head"]),
        "sign_flip_fraction_max_head": float(summary["sign_flip_fraction_max_head"]),
        "sign_flip_fraction_sd_head": float(summary["sign_flip_fraction_sd_head"]),
        "rel_err_median": float(summary["rel_err_median"]),
        "rel_err_p90": float(summary["rel_err_p90"]),
        "n_heads": int(summary["n_heads"]),
        "head_rates": summary["head_rates"],
    }


def measure_model_schemes(
    model_id: str = U.DEFAULT_MODEL_IDS[0],
    *,
    seed: int = U.DEFAULT_SEED,
    prompts: Sequence[str] = RM.DEFAULT_PROMPTS,
    deltas: np.ndarray | None = None,
    n_pairs: int = RM.N_CONTENT_PAIRS,
    schemes: Sequence[tuple[str, float]] = DEFAULT_SCHEMES,
) -> dict[str, Any]:
    """Re-price one checkpoint's content pairs under every ``(scheme, scale)``.

    The checkpoint is loaded once and each prompt is forwarded once; only the
    frequency ladder changes between rows.  That is what makes the comparison
    paired rather than a comparison of independently sampled heads.
    """
    loaded = RM.load_model(model_id, seed=seed)
    cfgv = RM.validate_config(loaded.model)
    if not cfgv["checkpoint_inv_freq_found"]:
        raise RuntimeError(f"{model_id} has no rotary module, so it has no rotary pairs")
    rope_dim = int(cfgv["rope_dim"])
    base = float(cfgv["rope_theta_declared"])
    original_context = int(
        getattr(loaded.model.config, "max_position_embeddings", 2048)
    )
    if deltas is None:
        deltas = np.array(U.DISTANCE_GRID, dtype=np.int64)

    # One forward per prompt, reused by every scheme.  Keeping the whole list
    # is a few megabytes on these checkpoints and avoids re-running the model.
    vectors = [
        RM.collect_head_vectors(loaded.model, loaded.tokenizer, text) for text in prompts
    ]

    rows: list[dict[str, Any]] = []
    for scheme, scale in schemes:
        freqs = _frequencies(
            scheme,
            scale,
            head_dim=rope_dim,
            base=base,
            original_context=original_context,
        )
        per_prompt: list[dict[str, Any]] = []
        for i, hv in enumerate(vectors):
            heads = []
            for layer in range(hv.n_layers):
                for head in range(hv.n_heads):
                    rec = U.measure_head_cost(
                        hv,
                        freqs,
                        rope_dim,
                        deltas,
                        n_pairs,
                        seed + i * 1000 + layer * 31 + head,
                        layer,
                        head,
                    )
                    if rec:
                        heads.append(rec)
            per_prompt.append(
                {"prompt_index": i, "n_tokens": hv.n_tokens, "heads": heads}
            )
        # ``_summarise`` is the canonical aggregator; reusing it is what makes
        # the plain-RoPE row reproduce the headline artifact instead of merely
        # resembling it.
        summary = U._summarise(model_id, rope_dim, deltas, per_prompt)
        rows.append(_scheme_row(model_id, scheme, scale, freqs, summary))

    return {
        "model_id": model_id,
        "rope_dim": rope_dim,
        "base": base,
        "original_context": original_context,
        "n_prompts": len(prompts),
        "n_content_pairs_per_head": int(n_pairs),
        "deltas": [int(d) for d in deltas],
        "rows": rows,
    }


def pool_schemes(models: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool each ``(scheme, scale)`` across models, stating the unit."""
    if not models:
        raise ValueError("no models to pool")
    keys = [(row["scheme"], row["scale"]) for row in models[0]["rows"]]
    out: list[dict[str, Any]] = []
    for scheme, scale in keys:
        rates = []
        rels = []
        fractions = []
        for model in models:
            matches = [
                row
                for row in model["rows"]
                if row["scheme"] == scheme and row["scale"] == scale
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{model['model_id']} has {len(matches)} rows for {scheme} s={scale}"
                )
            row = matches[0]
            rates.append(row["sign_flip_fraction_pooled"])
            rels.append(row["rel_err_median"])
            fractions.append(row["channels"]["4096"]["fraction_past_threshold"])
        out.append(
            {
                "scheme": scheme,
                "scale": float(scale),
                "n_models": len(models),
                "sign_flip_fraction_mean": float(np.mean(rates)),
                "sign_flip_fraction_min": float(np.min(rates)),
                "sign_flip_fraction_max": float(np.max(rates)),
                "sign_flip_fraction_sd": float(np.std(rates, ddof=1))
                if len(rates) > 1
                else 0.0,
                "rel_err_median_mean": float(np.mean(rels)),
                "fraction_past_threshold_4096_mean": float(np.mean(fractions)),
            }
        )
    return out


def run_all(
    model_ids: Sequence[str] = U.DEFAULT_MODEL_IDS,
    *,
    seed: int = U.DEFAULT_SEED,
    schemes: Sequence[tuple[str, float]] = DEFAULT_SCHEMES,
    n_pairs: int = RM.N_CONTENT_PAIRS,
) -> dict[str, Any]:
    """Measure every checkpoint and pool the scheme rows."""
    models = [
        measure_model_schemes(
            model_id, seed=seed, schemes=schemes, n_pairs=n_pairs
        )
        for model_id in model_ids
    ]
    return {
        "schema": SCHEMA,
        "seed": int(seed),
        "distance_grid": [int(d) for d in U.DISTANCE_GRID],
        "context_lengths": list(CONTEXT_LENGTHS),
        "schemes": [{"scheme": s, "scale": float(k)} for s, k in schemes],
        "models": models,
        "pooled": pool_schemes(models),
    }


def write_results(data: dict[str, Any], path: Path | None = None) -> Path:
    out = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent.parent.parent
        / "results"
        / "usefulness_extension.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=list(U.DEFAULT_MODEL_IDS))
    parser.add_argument("--seed", type=int, default=U.DEFAULT_SEED)
    parser.add_argument("--pairs", type=int, default=RM.N_CONTENT_PAIRS)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    data = run_all(args.models, seed=args.seed, n_pairs=args.pairs)
    path = write_results(data, args.out)
    print(f"wrote {path}")
    print(
        f"{'scheme':>22} {'scale':>6} {'sign flips':>11} "
        f"{'range':>15} {'past 4096':>10}"
    )
    for row in data["pooled"]:
        print(
            f"{row['scheme']:>22} {row['scale']:>6g} "
            f"{row['sign_flip_fraction_mean']:>11.4f} "
            f"{row['sign_flip_fraction_min']:.4f}..{row['sign_flip_fraction_max']:.4f} "
            f"{row['fraction_past_threshold_4096_mean']:>10.3f}"
        )


if __name__ == "__main__":
    main()
