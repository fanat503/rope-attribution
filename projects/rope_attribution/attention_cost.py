"""What the best position-free score costs at the attention level.

The paper's structural result says a per-feature attribution under RoPE is a
function of distance, not a number. This module prices the consequence for the
object practitioners actually read off: the attention distribution.

For one head and one prompt, the exact score is

    s(m, n) = q_m^T R_{n-m} k_n
            = sum_{rotary pairs} A_k(m, n) cos(D_k) + B_k(m, n) sin(D_k)
              + (unrotated channels),

where ``A_k`` and ``B_k`` depend on the content at the two positions and
``D_k = (n - m) * inv_freq_k``. The *best* position-free attribution replaces
each rotary pair by the least-squares constant over the distances the query can
actually see:

    s_bar(m, n) = sum_k A_k(m, n) * mean_d cos(d * inv_freq_k)
                       + B_k(m, n) * mean_d sin(d * inv_freq_k)
                  + (unrotated channels).

This is the surrogate the paper prices at the score level (27% wrong signs).
Here it is pushed through the softmax, so the numbers are attention weights:

* ``top1_flip_rate``: how often the head attends to a different key;
* ``total_variation``: half the L1 distance between the two distributions;
* ``kl_exact_surrogate``: the information lost, in nats.

Every quantity is measured against the model's own attention probabilities,
which are recomputed by ``transformers`` with ``output_attentions=True`` and
compared with the module's own softmax. That residual is reported as
``max_attention_residual``; a nonzero value would mean the score reconstruction
is wrong, not that the surrogate is bad.

The surrogate is deliberately the *per-query* least-squares optimum, not a
global one. A strawman would inflate every number.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import real_model as RM
from .rope import inv_freq

__all__ = [
    "DEFAULT_MODEL_IDS",
    "DEFAULT_PROMPTS",
    "attention_metrics",
    "distance_matrix",
    "exact_and_surrogate_scores",
    "measure_head",
    "measure_model",
    "run_all",
    "write_results",
]

#: Long enough for the distance range to matter: the short prompts in
#: ``real_model`` top out at 22 tokens, where the surrogate has almost nothing
#: to be wrong about. These are original passages, not quoted text.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "Photosynthesis converts light energy into chemical energy inside the "
    "chloroplasts of plant cells. Chlorophyll absorbs red and blue wavelengths "
    "while reflecting green light, which is why healthy leaves appear green. "
    "The absorbed energy drives two coupled stages: the light reactions split "
    "water molecules and release oxygen, and the Calvin cycle fixes carbon "
    "dioxide into sugars. Because the two stages operate on different "
    "timescales, a plant can adjust to changing light without rebuilding its "
    "molecular machinery.",
    "The printing press spread through Europe in the second half of the "
    "fifteenth century. A single workshop could produce hundreds of identical "
    "copies of a book in the time a scribe needed for one, and the price of "
    "printed matter fell sharply. Printers clustered in cities with paper "
    "mills and reliable trade routes, and their output fed the growing market "
    "for calendars, pamphlets, and scientific tables. Historians still argue "
    "about how quickly the new medium changed reading habits, but the "
    "direction of the change is not in dispute.",
    "def merge_intervals(intervals):\n"
    "    if not intervals:\n"
    "        return []\n"
    "    intervals = sorted(intervals, key=lambda pair: pair[0])\n"
    "    merged = [list(intervals[0])]\n"
    "    for start, end in intervals[1:]:\n"
    "        if start <= merged[-1][1]:\n"
    "            merged[-1][1] = max(merged[-1][1], end)\n"
    "        else:\n"
    "            merged.append([start, end])\n"
    "    return [tuple(pair) for pair in merged]\n",
)
DEFAULT_MODEL_IDS: tuple[str, ...] = tuple(RM.DEFAULT_MODEL_IDS)


def distance_matrix(n_tokens: int) -> np.ndarray:
    """``delta[m, n] = n - m``, the argument RoPE actually rotates by."""
    positions = np.arange(n_tokens, dtype=np.float64)
    return positions[None, :] - positions[:, None]


def causal_mask(n_tokens: int) -> np.ndarray:
    """``n <= m``: a query may attend to itself and to earlier keys."""
    return np.tril(np.ones((n_tokens, n_tokens), dtype=bool))


def _pair_coefficients(
    q_hat: np.ndarray, k_hat: np.ndarray, rotary_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """``A[m, n, kk]`` and ``B[m, n, kk]`` for every query/key/rotary pair.

    Shape ``(T, T, rotary_dim // 2)``. The split-half grouping is the one in
    ``experiments.pair_terms``: channels ``0..h-1`` and ``h..2h-1`` form one
    rotary pair, matching ``rotate_half``.
    """
    if rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if q_hat.shape[-1] < rotary_dim or k_hat.shape[-1] < rotary_dim:
        raise ValueError(
            f"rotary_dim={rotary_dim} exceeds head width {q_hat.shape[-1]}"
        )
    half = rotary_dim // 2
    q1, q2 = q_hat[:, :half], q_hat[:, half:rotary_dim]
    k1, k2 = k_hat[:, :half], k_hat[:, half:rotary_dim]
    aligned = np.einsum("mh,nh->mnh", q1, k1) + np.einsum("mh,nh->mnh", q2, k2)
    crossed = np.einsum("mh,nh->mnh", q2, k1) - np.einsum("mh,nh->mnh", q1, k2)
    return aligned, crossed


def _unrotated_term(
    q_hat: np.ndarray, k_hat: np.ndarray, rotary_dim: int
) -> np.ndarray:
    if rotary_dim == q_hat.shape[-1]:
        return np.zeros((q_hat.shape[0], k_hat.shape[0]), dtype=np.float64)
    return q_hat[:, rotary_dim:] @ k_hat[:, rotary_dim:].T


def _per_query_means(
    n_tokens: int, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares ``cos``/``sin`` means over the distances query ``m`` sees.

    Row ``m`` averages over ``delta = -m, ..., 0``. Using ``-delta`` rather than
    a symmetric range matters for the sine: it is odd, so a symmetric average
    would cancel the very term the surrogate is supposed to approximate.
    """
    cos_mean = np.empty((n_tokens, freqs.size), dtype=np.float64)
    sin_mean = np.empty((n_tokens, freqs.size), dtype=np.float64)
    for m in range(n_tokens):
        if m == 0:
            cos_mean[m] = 1.0
            sin_mean[m] = 0.0
            continue
        delta = -np.arange(1, m + 1, dtype=np.float64)
        angles = np.outer(delta, freqs)
        # The query also sees itself at delta = 0: cos = 1, sin = 0.
        cos_mean[m] = (1.0 + np.cos(angles).sum(axis=0)) / (m + 1)
        sin_mean[m] = np.sin(angles).sum(axis=0) / (m + 1)
    return cos_mean, sin_mean


def exact_and_surrogate_scores(
    q_hat: np.ndarray,
    k_hat: np.ndarray,
    freqs: np.ndarray,
    rotary_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact RoPE scores and the best position-free surrogate, both ``(T, T)``."""
    aligned, crossed = _pair_coefficients(q_hat, k_hat, rotary_dim)
    delta = distance_matrix(q_hat.shape[0])
    angles = delta[:, :, None] * freqs[None, None, :]
    unrotated = _unrotated_term(q_hat, k_hat, rotary_dim)

    exact = (aligned * np.cos(angles) + crossed * np.sin(angles)).sum(axis=-1)
    exact = exact + unrotated

    cos_mean, sin_mean = _per_query_means(q_hat.shape[0], freqs)
    surrogate = np.einsum("mnk,mk->mn", aligned, cos_mean)
    surrogate = surrogate + np.einsum("mnk,mk->mn", crossed, sin_mean)
    surrogate = surrogate + unrotated
    return exact, surrogate


def _softmax_rows(scores: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    z = np.where(mask, scores * scale, -np.inf)
    z = z - z.max(axis=-1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=-1, keepdims=True)


def attention_metrics(
    exact: np.ndarray, surrogate: np.ndarray, scale: float
) -> dict[str, float]:
    """Compare the two induced attention distributions, row by row.

    Rows are queries. ``top1_flip_rate`` is the fraction of queries whose
    argmax key changes; ``total_variation`` is half the L1 distance; the KL is
    ``KL(exact || surrogate)`` in nats. Queries at position 0 have a single
    admissible key and are excluded from the rate, not counted as agreements.
    """
    n = exact.shape[0]
    mask = causal_mask(n)
    p_exact = _softmax_rows(exact, mask, scale)
    p_sur = _softmax_rows(surrogate, mask, scale)

    top_exact = p_exact.argmax(axis=-1)
    top_sur = p_sur.argmax(axis=-1)
    flip = top_exact != top_sur
    tv = 0.5 * np.abs(p_exact - p_sur).sum(axis=-1)
    kl = (p_exact * (np.log(p_exact + 1e-300) - np.log(p_sur + 1e-300))).sum(axis=-1)

    # Top-5 sets: the object a circuit analysis reads off, not just the argmax.
    k = min(5, n)
    set_exact = np.argsort(-p_exact, axis=-1)[:, :k]
    set_sur = np.argsort(-p_sur, axis=-1)[:, :k]
    overlap = [
        len(set(a.tolist()) & set(b.tolist())) / k
        for a, b in zip(set_exact, set_sur, strict=True)
    ]

    return {
        "n_queries": int(n),
        "n_queries_scored": int(max(n - 1, 0)),
        "top1_flip_rate": float(flip[1:].mean()) if n > 1 else 0.0,
        "total_variation_mean": float(tv[1:].mean()) if n > 1 else 0.0,
        "total_variation_median": float(np.median(tv[1:])) if n > 1 else 0.0,
        "kl_exact_surrogate_mean": float(kl[1:].mean()) if n > 1 else 0.0,
        "kl_exact_surrogate_median": float(np.median(kl[1:])) if n > 1 else 0.0,
        "top5_overlap_mean": float(np.mean(overlap[1:])) if n > 1 else 1.0,
    }


def measure_head(
    q_hat: np.ndarray,
    k_hat: np.ndarray,
    freqs: np.ndarray,
    rotary_dim: int,
    scale: float,
) -> dict[str, float]:
    exact, surrogate = exact_and_surrogate_scores(q_hat, k_hat, freqs, rotary_dim)
    return attention_metrics(exact, surrogate, scale)


def _model_rotary(loaded: RM.Loaded) -> tuple[int, np.ndarray]:
    cfgv = RM.validate_config(loaded.model)
    if not cfgv["checkpoint_inv_freq_found"]:
        raise RuntimeError(f"{loaded.model_id} has no rotary module")
    rope_dim = int(cfgv["rope_dim"])
    freqs = inv_freq(rope_dim, float(cfgv["rope_theta_declared"]))
    return rope_dim, freqs


def _model_attentions(loaded: RM.Loaded, text: str) -> tuple[np.ndarray, int]:
    """Recompute the model's own attention probabilities, as a check.

    Returns ``(attentions, n_tokens)`` with ``attentions`` of shape
    ``(n_layers, n_heads, T, T)``. ``output_attentions=True`` is independent of
    the hook path used for ``q_hat``/``k_hat``, which is the point.
    """
    torch = RM._require_torch()
    ids = loaded.tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = loaded.model(input_ids=ids, use_cache=False, output_attentions=True)
    attn = out.attentions
    if attn is None:
        raise RuntimeError("the model returned no attention probabilities")
    stacked = np.stack([a[0].to(torch.float32).numpy() for a in attn], axis=0)
    return stacked, int(ids.shape[1])


def measure_model(
    model_id: str = DEFAULT_MODEL_IDS[0],
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    seed: int = 0,
) -> dict[str, Any]:
    """Every head of one checkpoint, on every prompt, at the attention level."""
    loaded = RM.load_model(model_id, seed=seed)
    rope_dim, freqs = _model_rotary(loaded)
    head_dim = RM.head_dim_of(RM.attention_modules(loaded.model)[0], loaded.model.config)
    scale = 1.0 / float(np.sqrt(head_dim))

    per_prompt = []
    for index, text in enumerate(prompts):
        hv = RM.collect_head_vectors(loaded.model, loaded.tokenizer, text)
        if hv.n_tokens < 2:
            continue
        model_attn, n_model = _model_attentions(loaded, text)
        if n_model != hv.n_tokens:
            raise RuntimeError(
                f"{model_id}: attention pass saw {n_model} tokens, hook pass {hv.n_tokens}"
            )
        heads = []
        max_residual = 0.0
        model_top1_disagreement = 0.0
        for layer in range(hv.n_layers):
            for head in range(hv.n_heads):
                q = hv.q_hat[layer, head].astype(np.float64)
                k = hv.k_hat[layer, head].astype(np.float64)
                exact, surrogate = exact_and_surrogate_scores(q, k, freqs, rope_dim)
                p_exact = _softmax_rows(exact, causal_mask(hv.n_tokens), scale)
                residual = float(np.abs(p_exact - model_attn[layer, head]).max())
                max_residual = max(max_residual, residual)
                agree = float(
                    (
                        p_exact.argmax(axis=-1)
                        == model_attn[layer, head].argmax(axis=-1)
                    ).mean()
                )
                model_top1_disagreement = max(model_top1_disagreement, 1.0 - agree)
                rec = measure_head(q, k, freqs, rope_dim, scale)
                rec["layer"] = layer
                rec["head"] = head
                heads.append(rec)
        per_prompt.append(
            {
                "prompt_index": index,
                "n_tokens": hv.n_tokens,
                "heads": heads,
                "max_attention_residual": max_residual,
                "max_top1_disagreement_with_model": model_top1_disagreement,
            }
        )

    return {"model_id": model_id, "rope_dim": rope_dim, "per_prompt": per_prompt,
            "by_layer": _by_layer(per_prompt)}


def _by_layer(per_prompt: list[dict]) -> list[dict]:
    """Mean flip rate and total variation for each layer, over heads and prompts."""
    layers: dict[int, dict[str, list[float]]] = {}
    for prompt in per_prompt:
        for head in prompt["heads"]:
            rec = layers.setdefault(int(head["layer"]), {"flip": [], "tv": [], "kl": []})
            rec["flip"].append(float(head["top1_flip_rate"]))
            rec["tv"].append(float(head["total_variation_mean"]))
            rec["kl"].append(float(head["kl_exact_surrogate_mean"]))
    return [
        {
            "layer": layer,
            "top1_flip_rate": float(np.mean(rec["flip"])),
            "total_variation_mean": float(np.mean(rec["tv"])),
            "kl_exact_surrogate_mean": float(np.mean(rec["kl"])),
            "n_heads": len(rec["flip"]),
        }
        for layer, rec in sorted(layers.items())
    ]


def _aggregate(per_prompt: list[dict]) -> dict[str, float]:
    keys = (
        "top1_flip_rate",
        "total_variation_mean",
        "kl_exact_surrogate_mean",
        "top5_overlap_mean",
    )
    flat = {key: [] for key in keys}
    for prompt in per_prompt:
        for head in prompt["heads"]:
            for key in keys:
                flat[key].append(float(head[key]))
    return {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in flat.items()
    } | {"n_heads": len(flat["top1_flip_rate"])}


def run_all(
    model_ids: Sequence[str] = DEFAULT_MODEL_IDS,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    seed: int = 0,
) -> dict[str, Any]:
    models = []
    for model_id in model_ids:
        measured = measure_model(model_id, prompts, seed=seed)
        measured["aggregate"] = _aggregate(measured["per_prompt"])
        models.append(measured)
    return {
        "schema": "attention-cost/v1",
        "scale": "1/sqrt(head_dim), as the model applies it",
        "surrogate": (
            "per-query least-squares constant of each rotary pair over the "
            "causal distances that query sees"
        ),
        "prompts": list(prompts),
        "models": models,
    }


def write_results(data: dict[str, Any], path: Path | None = None) -> Path:
    out = path or Path(__file__).resolve().parent.parent.parent / "results" / "attention_cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODEL_IDS))
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    data = run_all(args.models)
    path = write_results(data, args.out)
    print(f"{'model':<28} {'flip':>7} {'TV':>7} {'KL':>7} {'top5':>7}")
    for model in data["models"]:
        agg = model["aggregate"]
        print(
            f"{model['model_id']:<28} {agg['top1_flip_rate']:>7.3f} "
            f"{agg['total_variation_mean']:>7.3f} "
            f"{agg['kl_exact_surrogate_mean']:>7.3f} "
            f"{agg['top5_overlap_mean']:>7.3f}"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
