"""How the QK attribution matrix moves with distance.

The method paper for QK attribution (Kamath et al., Transformer Circuits, 2025)
decomposes an attention score into feature-feature terms at a *fixed* position
pair and notes that RoPE is outside its scope: rotation makes the bilinear form
depend on the distance.  ``qk_attribution`` supplies the exact RoPE
decomposition for one distance.  This module asks the next question:

    As the distance sweeps, what is the dimension of the family of matrices
    that the decomposition traces out?

For a head with ``P`` rotary frequency pairs, every feature-feature entry is

    C_ij(d) = A_ij + sum_k [ X_ij,k cos(d theta_k) + Y_ij,k sin(d theta_k) ],

so the entire matrix trajectory lies in a subspace of dimension at most
``1 + 2P`` --- independent of the dictionary width.  That is an exact bound,
not a fit.  We measure how tight it is on a trained dictionary, together with
the stable rank of each matrix, its top-k mass, and the relative Frobenius cost
of replacing the trajectory by its distance-mean (the matrix analogue of the
paper's position-free scalar).

What this licenses
------------------
* A statement about the *family* of QK matrices, not about one prompt: the
  distance dependence has at most ``1 + 2P`` degrees of freedom.
* A measurement of how much of that family a position-free matrix misses.
* No claim about attention patterns: the softmax is a second obstruction and is
  deliberately outside this module.

The passage, dictionary and head are the same objects ``qk_attribution`` uses,
so the two artifacts can be read side by side.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import qk_attribution as QK
from . import real_model as RM
from . import rope as R

__all__ = [
    "DEFAULT_DELTAS",
    "analyse_pair",
    "combine_layers",
    "flatten_relative_error",
    "measure_layers",
    "measure_model",
    "participation_ratio",
    "stable_rank",
    "top_k_mass",
    "trajectory_matrices",
    "trajectory_rank_bound",
]

SCHEMA = "qk-trajectory/v1"

#: The same log-spaced grid the attribution sweep uses, so the two artifacts
#: describe the same distances.
DEFAULT_DELTAS: tuple[int, ...] = tuple(int(d) for d in QK.SWEEP_DELTAS)


# --------------------------------------------------------------------------
# Linear-algebra summaries
# --------------------------------------------------------------------------


def _singular_values(matrix: np.ndarray) -> np.ndarray:
    values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    return values[values > 0]


def stable_rank(matrix: np.ndarray) -> float:
    """``||A||_F^2 / ||A||_2^2``: how many equal singular values fit the energy.

    Unlike the algebraic rank, this is stable under tiny numerical noise and
    answers "how many directions carry the matrix", which is the quantity a
    low-rank description needs.
    """
    s = _singular_values(matrix)
    if s.size == 0:
        return 0.0
    return float((s**2).sum() / (s[0] ** 2))


def participation_ratio(matrix: np.ndarray) -> float:
    """``(sum s^2)^2 / sum s^4``: the effective number of equal directions."""
    s = _singular_values(matrix)
    if s.size == 0:
        return 0.0
    s2 = s**2
    return float(s2.sum() ** 2 / (s2**2).sum())


def top_k_mass(matrix: np.ndarray, k: int) -> float:
    """Fraction of the absolute entry mass carried by the ``k`` largest entries."""
    flat = np.abs(np.asarray(matrix, dtype=np.float64)).ravel()
    if flat.size == 0:
        return 0.0
    total = flat.sum()
    if total == 0.0:
        return 0.0
    k = min(int(k), flat.size)
    return float(np.sort(flat)[-k:].sum() / total)


def trajectory_rank_bound(rotary_dim: int) -> int:
    """At most ``1 + 2 * (rotary_dim / 2)`` degrees of freedom in distance.

    The constant is the unrotated tail (or the zero-frequency term); each
    rotary pair contributes one cosine and one sine.  The bound is independent
    of the number of features, which is the point.
    """
    if rotary_dim < 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be a non-negative even number")
    return 1 + rotary_dim


def _energy_fractions(singular: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    """Fraction of squared-singular-value energy in the top ``k`` directions."""
    if singular.size == 0:
        return {str(int(k)): 0.0 for k in ks}
    energy = singular**2
    total = float(energy.sum())
    if total == 0.0:
        return {str(int(k)): 0.0 for k in ks}
    return {
        str(int(k)): float(energy[: min(int(k), energy.size)].sum() / total)
        for k in ks
    }


def flatten_relative_error(matrices: np.ndarray) -> float:
    """``||C(d) - mean_d C(d)||_F / ||C(d)||_F``, pooled over distances."""
    stack = np.asarray(matrices, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError("matrices must have shape (distances, rows, columns)")
    mean = stack.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(stack - mean)
    denominator = np.linalg.norm(stack)
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def trajectory_matrices(
    buckets_m: QK.TokenBuckets,
    buckets_n: QK.TokenBuckets,
    deltas: Sequence[int],
    freqs: np.ndarray,
    rotary_dim: int,
) -> np.ndarray:
    """``(D, K_m, K_n)`` feature-feature matrices, one per distance.

    ``buckets_m`` and ``buckets_n`` are fixed content; only the rotation angle
    changes.  That is the content-fixed sweep the paper uses, applied to the
    matrix rather than to one feature's scalar profile.  ``deltas`` are
    distances, and the key sits behind the query, so the angle is ``-delta``
    --- the same convention ``measure_head`` uses.  Passing ``+delta`` would
    compute the anti-causal rotation.
    """
    terms = [
        QK.attribution_terms(buckets_m, buckets_n, -int(delta), freqs, rotary_dim)
        for delta in deltas
    ]
    shapes = {term.pairs.shape for term in terms}
    if len(shapes) != 1:
        raise ValueError(f"the feature blocks changed shape across distances: {shapes}")
    return np.stack([term.pairs for term in terms])


def _summarise_matrices(
    matrices: np.ndarray, ks: Sequence[int]
) -> dict[str, Any]:
    per_delta = []
    for index, matrix in enumerate(matrices):
        per_delta.append(
            {
                "index": index,
                "stable_rank": stable_rank(matrix),
                "participation_ratio": participation_ratio(matrix),
                "top_k_mass": {str(int(k)): top_k_mass(matrix, int(k)) for k in ks},
                "negative_fraction": float((matrix < 0).mean()),
                "frobenius_norm": float(np.linalg.norm(matrix)),
            }
        )
    trajectory = matrices.reshape(matrices.shape[0], -1)
    singular = _singular_values(trajectory)
    return {
        "per_delta": per_delta,
        "flatten_relative_error": flatten_relative_error(matrices),
        "trajectory_stable_rank": stable_rank(trajectory),
        "trajectory_participation_ratio": participation_ratio(trajectory),
        "trajectory_singular_values": [float(value) for value in singular],
        "trajectory_energy": _energy_fractions(singular, (1, 2, 4, 8)),
        "trajectory_rank_numerical": int(np.linalg.matrix_rank(trajectory)),
    }


def analyse_pair(
    buckets_m: QK.TokenBuckets,
    buckets_n: QK.TokenBuckets,
    deltas: Sequence[int],
    freqs: np.ndarray,
    rotary_dim: int,
    *,
    ks: Sequence[int] = (1, 2, 4, 8, 16),
) -> dict[str, Any]:
    """The full trajectory summary for one fixed content pair."""
    matrices = trajectory_matrices(buckets_m, buckets_n, deltas, freqs, rotary_dim)
    summary = _summarise_matrices(matrices, ks)
    summary.update(
        {
            "n_features_m": int(matrices.shape[1]),
            "n_features_n": int(matrices.shape[2]),
            "rank_bound": trajectory_rank_bound(rotary_dim),
        }
    )
    return summary


# --------------------------------------------------------------------------
# Model wiring
# --------------------------------------------------------------------------


def _load_context(
    model_id: str,
    sae_path: str | Path,
    text: str,
    *,
    seed: int,
    sae_layer: int,
) -> dict[str, Any]:
    """Load the checkpoint and dictionary once, and cache the shared tensors.

    Every head needs the same residual stream, dictionary activations,
    LayerNorm parameters and projection weights.  Reloading the 134 MB
    dictionary per head is both slow and a needless memory spike, so the model
    wiring is separated from the per-head bucketing.
    """
    dictionary = QK.SparseDictionary.load_safetensors(sae_path)
    loaded = RM.load_model(model_id, seed=seed)
    model, tokenizer = loaded.model, loaded.tokenizer
    if not hasattr(model, "gpt_neox"):
        raise ValueError(
            "qk_trajectory currently wires the GPT-NeoX block stack only; "
            f"{type(model).__name__} needs its own layer accessor"
        )
    attn_index = QK._attention_index(len(model.gpt_neox.layers), sae_layer)

    run = QK._forward(model, tokenizer, text, attn_index)
    n_tokens = int(run["ids"].shape[1])
    if n_tokens < 8:
        raise ValueError(f"the passage tokenized to {n_tokens} tokens, too few to bin")

    x = run["hidden_states"][int(sae_layer) + 1][0].numpy().astype(np.float64)
    if x.shape[1] != dictionary.d_model:
        raise ValueError(
            f"the checkpoint has width {x.shape[1]} but the dictionary was trained "
            f"on width {dictionary.d_model}"
        )

    attn = RM.attention_modules(model)[attn_index]
    ln = model.gpt_neox.layers[attn_index].input_layernorm
    ln_weight = ln.weight.detach().numpy().astype(np.float64)
    ln_bias = ln.bias.detach().numpy().astype(np.float64)

    hidden_size = int(x.shape[1])
    n_heads = int(model.config.num_attention_heads)
    head_dim = hidden_size // n_heads
    w_q, b_q, w_k, b_k = QK.head_projections(attn, hidden_size, n_heads, head_dim)

    cfgv = RM.validate_config(model)
    rotary_dim = int(cfgv["rope_dim"])
    freqs = R.inv_freq(rotary_dim, float(cfgv["rope_theta_declared"]))
    return {
        "dictionary": dictionary,
        "x": x,
        "f": dictionary.encode(x),
        "ln_weight": ln_weight,
        "ln_bias": ln_bias,
        "ln_eps": float(ln.eps),
        "w_q": w_q,
        "b_q": b_q,
        "w_k": w_k,
        "b_k": b_k,
        "freqs": freqs,
        "rotary_dim": rotary_dim,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "n_tokens": n_tokens,
        "attn_index": attn_index,
    }


def _buckets_for_head(
    context: dict[str, Any], head: int, top_k: int
) -> list[QK.TokenBuckets]:
    """Split every token into one head's query and key shares."""
    n_heads = int(context["n_heads"])
    if not 0 <= int(head) < n_heads:
        raise ValueError(f"head must be in [0, {n_heads}), got {head}")
    x = context["x"]
    f = context["f"]
    return [
        QK.token_buckets(
            x[t],
            f[t],
            context["dictionary"],
            context["ln_weight"],
            context["ln_bias"],
            context["ln_eps"],
            context["w_q"][int(head)],
            context["b_q"][int(head)],
            context["w_k"][int(head)],
            context["b_k"][int(head)],
            top_k,
        )
        for t in range(int(context["n_tokens"]))
    ]


def _head_buckets(
    model_id: str,
    sae_path: str | Path,
    text: str,
    *,
    top_k: int,
    seed: int,
    sae_layer: int,
    head: int,
) -> tuple[list[QK.TokenBuckets], np.ndarray, int, int, int, int]:
    """One head's buckets plus the metadata the caller needs."""
    context = _load_context(
        model_id, sae_path, text, seed=seed, sae_layer=sae_layer
    )
    return (
        _buckets_for_head(context, head, top_k),
        context["freqs"],
        context["rotary_dim"],
        context["head_dim"],
        context["n_tokens"],
        context["attn_index"],
    )


def _sample_content_pairs(
    n_tokens: int, n_pairs: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Distinct ``(query, key)`` token pairs, content fixed per pair."""
    if n_tokens < 2:
        raise ValueError("need at least two tokens")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    # A deterministic sweep is preferable to rejection sampling: every query
    # gets one key at a spread of offsets, then the RNG breaks ties.
    for query in range(n_tokens):
        key = int(rng.integers(0, query)) if query else -1
        if key < 0:
            continue
        pair = (query, key)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
        if len(pairs) >= n_pairs:
            return pairs
    return pairs


def measure_model(
    model_id: str = QK.DEFAULT_MODEL_ID,
    sae_path: str | Path | None = None,
    text: str = QK.DEFAULT_TEXT,
    *,
    top_k: int = QK.DEFAULT_TOP_K,
    sae_layer: int = QK.DEFAULT_SAE_LAYER,
    heads: Sequence[int] | None = None,
    deltas: Sequence[int] = DEFAULT_DELTAS,
    n_content_pairs: int = 8,
    seed: int = 0,
    ks: Sequence[int] = (1, 2, 4, 8, 16),
) -> dict[str, Any]:
    """Trajectory summaries for every requested head on one passage."""
    if sae_path is None:
        raise ValueError(
            "measure_model needs a sparse dictionary; pass the safetensors path "
            "of the public SAE (see results/qk_attribution.json provenance)"
        )
    if not deltas:
        raise ValueError("at least one distance is required")
    context = _load_context(
        model_id, sae_path, text, seed=seed, sae_layer=sae_layer
    )
    requested = tuple(heads) if heads is not None else tuple(range(context["n_heads"]))
    if not requested:
        raise ValueError("at least one head is required")

    rows = []
    for head in requested:
        buckets = _buckets_for_head(context, int(head), top_k)
        rng = np.random.default_rng(seed + int(head))
        content_pairs = _sample_content_pairs(
            int(context["n_tokens"]), n_content_pairs, rng
        )
        summaries = [
            analyse_pair(
                buckets[m],
                buckets[n],
                deltas,
                context["freqs"],
                context["rotary_dim"],
                ks=ks,
            )
            for m, n in content_pairs
        ]
        rows.append(
            {
                "head": int(head),
                "layer": int(context["attn_index"]),
                "n_content_pairs": len(content_pairs),
                "pairs": summaries,
                "mean": _mean_summary(summaries, ks),
            }
        )

    return {
        "schema": SCHEMA,
        "model_id": model_id,
        "text": text,
        "seed": int(seed),
        "pairs_per_head": int(n_content_pairs),
        "sae_layer": int(sae_layer),
        "attention_layer": int(context["attn_index"]),
        "n_tokens": int(context["n_tokens"]),
        "top_k": int(top_k),
        "deltas": [int(delta) for delta in deltas],
        "rotary_dim": int(context["rotary_dim"]),
        "head_dim": int(context["head_dim"]),
        "rank_bound": trajectory_rank_bound(int(context["rotary_dim"])),
        "heads": rows,
    }


def _across(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample SD, min and max of a small head-level sample."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _summarise_head(head: dict[str, Any]) -> dict[str, Any]:
    """The compact per-head record kept in the committed artifact."""
    return {
        "head": int(head["head"]),
        "n_content_pairs": int(head["n_content_pairs"]),
        "mean": head["mean"],
    }


def _summarise_layer(dataset: dict[str, Any]) -> dict[str, Any]:
    """Pool one layer's heads, keeping the head-level spread visible."""
    heads = [_summarise_head(head) for head in dataset["heads"]]
    keys = (
        "trajectory_participation_ratio",
        "flatten_relative_error",
        "trajectory_stable_rank",
        "trajectory_rank_numerical",
        "trajectory_energy_top1",
        "trajectory_energy_top2",
        "trajectory_energy_top4",
        "trajectory_energy_top8",
    )
    layer: dict[str, Any] = {
        "sae_layer": int(dataset["sae_layer"]),
        "attention_layer": int(dataset["attention_layer"]),
        "n_tokens": int(dataset["n_tokens"]),
        "n_heads": len(heads),
        "heads": heads,
    }
    for key in keys:
        layer[key] = _across([head["mean"][key]["mean"] for head in heads])
    # Per-distance curves, averaged over heads.  The spread across heads is
    # kept for the two quantities the paper quotes.
    per_delta = []
    for index in range(len(heads[0]["mean"]["per_delta"])):
        row: dict[str, Any] = {"index": index}
        for key in ("stable_rank", "participation_ratio", "negative_fraction", "frobenius_norm"):
            row[key] = float(
                np.mean([head["mean"]["per_delta"][index][key] for head in heads])
            )
        for k in ("1", "2", "4", "8", "16"):
            name = f"top_{k}_mass"
            row[name] = float(
                np.mean([head["mean"]["per_delta"][index][name] for head in heads])
            )
        per_delta.append(row)
    layer["per_delta"] = per_delta
    return layer


def combine_layers(datasets: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-layer runs into the artifact the paper quotes.

    Every dataset must come from the same checkpoint, dictionary family,
    passage, top-k and distance grid; combining different ones would average
    incomparable matrices.  The check is explicit rather than assumed.
    """
    if not datasets:
        raise ValueError("at least one layer is required")
    first = datasets[0]
    for dataset in datasets[1:]:
        for key in (
            "model_id",
            "text",
            "seed",
            "pairs_per_head",
            "n_tokens",
            "top_k",
            "deltas",
            "rotary_dim",
            "head_dim",
            "rank_bound",
        ):
            if dataset[key] != first[key]:
                raise ValueError(
                    f"cannot combine layers with different {key}: "
                    f"{first[key]!r} vs {dataset[key]!r}"
                )
    layers = [_summarise_layer(dataset) for dataset in datasets]
    all_heads = [head for layer in layers for head in layer["heads"]]
    pooled = {}
    for key in (
        "trajectory_participation_ratio",
        "flatten_relative_error",
        "trajectory_stable_rank",
        "trajectory_rank_numerical",
        "trajectory_energy_top1",
        "trajectory_energy_top2",
        "trajectory_energy_top4",
        "trajectory_energy_top8",
    ):
        pooled[key] = _across([head["mean"][key]["mean"] for head in all_heads])
    return {
        "schema": SCHEMA,
        "model_id": first["model_id"],
        "text": first["text"],
        "seed": int(first["seed"]),
        "pairs_per_head": int(first["pairs_per_head"]),
        "n_layers": len(layers),
        "n_heads": len(all_heads),
        "n_tokens": int(first["n_tokens"]),
        "top_k": int(first["top_k"]),
        "deltas": first["deltas"],
        "rotary_dim": int(first["rotary_dim"]),
        "head_dim": int(first["head_dim"]),
        "rank_bound": int(first["rank_bound"]),
        "layers": layers,
        "pooled": pooled,
    }


def measure_layers(
    model_id: str,
    sae_template: str,
    layers: Sequence[int],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run ``measure_model`` once per layer and combine the results.

    ``sae_template`` is a path with a ``{layer}`` placeholder, e.g.
    ``.tmp/research/sae_ctigges_l{layer}.safetensors``.  Requiring the
    placeholder prevents silently measuring the same dictionary five times.
    """
    if "{layer}" not in sae_template:
        raise ValueError("sae_template must contain a {layer} placeholder")
    datasets = [
        measure_model(
            model_id,
            sae_template.format(layer=int(layer)),
            sae_layer=int(layer),
            **kwargs,
        )
        for layer in layers
    ]
    return combine_layers(datasets)


def _mean_summary(
    summaries: Sequence[dict[str, Any]], ks: Sequence[int]
) -> dict[str, Any]:
    """Mean and sample SD across content pairs, field by field."""
    if not summaries:
        return {}
    scalar_keys = (
        "flatten_relative_error",
        "trajectory_stable_rank",
        "trajectory_participation_ratio",
        "trajectory_rank_numerical",
        "rank_bound",
    )
    out: dict[str, Any] = {}
    for key in scalar_keys:
        values = np.array([float(row[key]) for row in summaries], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    for k in (1, 2, 4, 8):
        out[f"trajectory_energy_top{k}"] = _across(
            [float(row["trajectory_energy"][str(k)]) for row in summaries]
        )
    out["per_delta"] = []
    n_deltas = len(summaries[0]["per_delta"])
    for index in range(n_deltas):
        row: dict[str, Any] = {"index": index}
        for key in ("stable_rank", "participation_ratio", "negative_fraction", "frobenius_norm"):
            values = np.array(
                [float(summary["per_delta"][index][key]) for summary in summaries]
            )
            row[key] = float(values.mean())
            row[f"{key}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        for k in ks:
            values = np.array(
                [
                    float(summary["per_delta"][index]["top_k_mass"][str(int(k))])
                    for summary in summaries
                ]
            )
            row[f"top_{int(k)}_mass"] = float(values.mean())
        out["per_delta"].append(row)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def write_results(data: dict[str, Any], path: Path | None = None) -> Path:
    out = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent.parent.parent / "results" / "qk_trajectory.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=QK.DEFAULT_MODEL_ID)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sae", type=Path, help="one dictionary file")
    source.add_argument(
        "--sae-template",
        help="path with a {layer} placeholder, used with --layers",
    )
    parser.add_argument("--sae-layer", type=int, default=QK.DEFAULT_SAE_LAYER)
    parser.add_argument(
        "--layers",
        help="comma-separated SAE layers for --sae-template, e.g. 0,1,2,3,4",
    )
    parser.add_argument("--head", action="append", type=int, dest="heads")
    parser.add_argument("--top-k", type=int, default=QK.DEFAULT_TOP_K)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    common = {
        "top_k": args.top_k,
        "heads": tuple(args.heads) if args.heads else None,
        "n_content_pairs": args.pairs,
        "seed": args.seed,
    }
    if args.sae_template:
        if not args.layers:
            raise SystemExit("--sae-template needs --layers")
        layers = [int(part) for part in args.layers.split(",") if part.strip()]
        data = measure_layers(args.model, args.sae_template, layers, **common)
    else:
        if args.layers:
            raise SystemExit("--layers is only valid with --sae-template")
        data = measure_model(
            args.model,
            args.sae,
            sae_layer=args.sae_layer,
            **common,
        )
    path = write_results(data, args.out)
    print(f"wrote {path}")
    if "layers" in data:
        for layer in data["layers"]:
            print(
                f"layer {layer['sae_layer']}: "
                f"trajectory PR {layer['trajectory_participation_ratio']['mean']:.2f} "
                f"[{layer['trajectory_participation_ratio']['min']:.2f}, "
                f"{layer['trajectory_participation_ratio']['max']:.2f}], "
                f"flatten {layer['flatten_relative_error']['mean']:.4f}"
            )
    else:
        for row in data["heads"]:
            mean = row["mean"]
            print(
                f"head {row['head']}: bound {mean['rank_bound']['mean']:.0f}, "
                f"trajectory PR {mean['trajectory_participation_ratio']['mean']:.2f}, "
                f"flatten {mean['flatten_relative_error']['mean']:.4f}"
            )


if __name__ == "__main__":
    main()
