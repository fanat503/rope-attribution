"""Does the position-dependent score matter downstream?

The question
------------
The paper's attribution object is a function of the relative distance: at a
fixed distance the score is exactly bilinear in the content vectors, but the
bilinear form changes with the distance. A reader can accept the algebra and
still ask the only question that matters: if a position-free approximation is
used instead, does the model behave differently?

This module answers that with the model's own next-token loss. It replaces the
exact score ``q_i^T R(delta) k_j`` inside the forward pass with the best
position-free bilinear form,

    q_i^T M k_j,    M = mean over the observed distances of R(delta),

and measures the change in negative log-likelihood on held-out text. ``M`` is
the least-squares optimum for a position-free form with the same block
structure as RoPE: for each frequency pair, the mean rotation is
``[[cos_mean, -sin_mean], [sin_mean, cos_mean]]``. It is a *stronger* baseline
than a per-feature scalar, because it still mixes the two channels of a pair.

How the replacement is realised
-------------------------------
The model is not re-implemented. Forward hooks on the query and key projections
feed

    q_feed = R(-pos_i) q,        k_feed = R(-pos_j) (M k)

so the model's own RoPE, applied after the hooks, reconstructs ``q`` and
``M k``. The dot product is then exactly ``q_i^T M k_j``; the softmax, causal
mask, value projection and the rest of the network are untouched. This is why
the comparison is downstream rather than another score-error statistic: the
loss sees the surrogate through the model's ordinary attention path.

Scope and honesty
-----------------
Supported: full-rotation attention with separate ``q_proj``/``k_proj`` or a
fused ``query_key_value``, and no QK-normalisation. Models with QK-norm or
partial rotation are refused rather than measured incorrectly. The live run
needs torch, transformers and the checkpoints; the unit tests use a randomly
initialised tiny model, so the mathematics is checked without a download.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np

from rope_attribution.real_model import (
    DEFAULT_LONG_TEXTS,
    DEFAULT_MODEL_IDS,
    REAL_MODEL_AVAILABLE,
    attention_modules,
    head_dim_of,
    join_fused_qkv,
    load_model,
    n_heads_of,
    split_fused_qkv,
    unsupported_architecture,
)
from rope_attribution.rope import inv_freq

SCHEMA = "rope_attribution/downstream/v1"


def _rope_parameters(cfg: Any) -> dict:
    """The checkpoint's flat rotary-parameter block, or an empty dict.

    Current ``transformers`` configs store ``rope_theta`` and
    ``partial_rotary_factor`` inside this block rather than at the top level;
    older GPT-NeoX configs use top-level ``rotary_pct`` and ``rotary_emb_base``.
    Reading only the top-level names is how a partial-rotation checkpoint slips
    through a guard meant to refuse it.
    """
    params = getattr(cfg, "rope_parameters", None)
    return dict(params) if isinstance(params, dict) else {}


def _partial_rotary_factor(cfg: Any) -> float:
    """The rotated fraction, from wherever this config family declares it."""
    params = _rope_parameters(cfg)
    for name in ("partial_rotary_factor", "rotary_pct"):
        if name in params:
            return float(params[name])
        value = getattr(cfg, name, None)
        if value is not None:
            return float(value)
    return 1.0



# --------------------------------------------------------------------------
# The position-free kernel: pure NumPy, checked without torch
# --------------------------------------------------------------------------


def surrogate_kernel(
    deltas: np.ndarray,
    freqs: np.ndarray,
    *,
    mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """A position-free rotation per frequency pair.

    ``mode="mean"`` is the unconstrained least-squares optimum:
    ``(mean cos, mean sin)``. It is the best bilinear fit to the exact scores,
    but over a wide distance range its modulus shrinks towards zero, so it also
    removes score amplitude. ``mode="circular"`` keeps the mean *angle* and
    unit modulus, which removes the distance dependence while preserving the
    norm; it is the fair test of position alone. ``mode="identity"`` removes
    the rotation entirely and is the simplest position-free baseline.

    All three return ``(cos, sin)`` of length ``len(freqs)``. Empty distances
    or frequencies are an error: a mean over nothing would silently produce a
    zero kernel.
    """
    deltas = np.asarray(deltas, dtype=np.float64).reshape(-1)
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if deltas.size == 0:
        raise ValueError("surrogate_kernel needs at least one distance")
    if freqs.size == 0:
        raise ValueError("surrogate_kernel needs at least one frequency")
    if mode == "identity":
        return np.ones_like(freqs), np.zeros_like(freqs)
    angles = deltas[:, None] * freqs[None, :]
    cos_mean = np.cos(angles).mean(axis=0)
    sin_mean = np.sin(angles).mean(axis=0)
    if mode == "mean":
        return cos_mean, sin_mean
    if mode == "circular":
        angle = np.arctan2(sin_mean, cos_mean)
        return np.cos(angle), np.sin(angle)
    raise ValueError(f"unknown surrogate kernel mode {mode!r}")


def observed_distances(n_tokens: int) -> np.ndarray:
    """All positive relative distances in a causal sequence of ``n_tokens``."""
    if n_tokens < 2:
        raise ValueError("a causal sequence needs at least two tokens")
    return np.arange(1, int(n_tokens), dtype=np.int64)


# --------------------------------------------------------------------------
# Torch implementation of the hook transform
# --------------------------------------------------------------------------


def _torch():
    from rope_attribution.real_model import _require_torch

    return _require_torch()


def _rotate_half(x: Any) -> Any:
    torch = _torch()
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_torch(x: Any, cos: Any, sin: Any) -> Any:
    """``x`` is ``(T, D)`` or ``(B, T, H, D)``; ``cos``/``sin`` are ``(T, D)``.

    The two shapes are the only ones this module uses: the unit test feeds a
    single sequence, the hooks feed the projection output reshaped with an
    explicit head axis. Keeping the broadcast explicit avoids guessing which
    axis is time for an arbitrary rank.
    """
    if cos.ndim == 2:
        if x.ndim == 4:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        elif x.ndim != 2:
            raise ValueError(f"apply_rope_torch does not handle rank {x.ndim}")
    return x * cos + _rotate_half(x) * sin


def apply_surrogate_torch(x: Any, cos_mean: Any, sin_mean: Any) -> Any:
    """Apply the position-free block ``[[c, -s], [s, c]]`` per frequency pair."""
    half = x.shape[-1] // 2
    if cos_mean.ndim == 1 and x.ndim > 1:
        shape = [1] * (x.ndim - 1) + [half]
        cos_mean = cos_mean.reshape(shape)
        sin_mean = sin_mean.reshape(shape)
    x1, x2 = x[..., :half], x[..., half:]
    return torch_cat(
        (x1 * cos_mean - x2 * sin_mean, x1 * sin_mean + x2 * cos_mean)
    )


def torch_cat(parts: tuple[Any, ...]) -> Any:
    """``torch.cat`` on the last axis, kept out of the math helpers' signatures."""
    return _torch().cat(parts, dim=-1)


class SurrogateAttention(AbstractContextManager):
    """Context manager that swaps exact RoPE scores for the position-free form.

    Use it around a forward pass, with ``torch.no_grad()`` for a measurement::

        with SurrogateAttention(model, deltas=deltas):
            loss = model(input_ids=ids, labels=ids).loss

    Every hook is removed on exit, including after an exception. The exact
    model is never mutated.
    """

    def __init__(
        self,
        model: Any,
        deltas: np.ndarray | None = None,
        *,
        theta: float | None = None,
        head_dim: int | None = None,
        kernel: str = "mean",
    ) -> None:
        if kernel not in ("mean", "circular", "identity"):
            raise ValueError(f"unknown surrogate kernel mode {kernel!r}")
        self.kernel = kernel
        self.model = model
        self.cfg = getattr(model, "config", None)
        reason = unsupported_architecture(self.cfg)
        if reason:
            raise NotImplementedError(reason)
        self.attns = attention_modules(model)
        if not self.attns:
            raise RuntimeError("no attention module with a recognised q/k projection")
        self.head_dim = int(head_dim or head_dim_of(self.attns[0], self.cfg))
        self.n_heads = int(n_heads_of(self.attns[0], self.cfg))
        if self.head_dim % 2:
            raise ValueError("RoPE needs an even head dimension")
        self.theta = float(theta if theta is not None else _rope_theta(self.cfg))
        factor = _partial_rotary_factor(self.cfg)
        if factor != 1.0:
            raise NotImplementedError(
                "partial rotation is not supported by the downstream surrogate; "
                f"config says partial_rotary_factor={factor}"
            )
        for attn in self.attns:
            if hasattr(attn, "q_norm") or hasattr(attn, "k_norm"):
                raise NotImplementedError(
                    "QK-normalised attention is not supported: a hook on the "
                    "projection would transform the pre-norm vector, which is "
                    "not the vector the model rotates"
                )
        self.deltas = (
            np.asarray(deltas, dtype=np.float64).reshape(-1)
            if deltas is not None
            else None
        )
        self._handles: list[Any] = []
        self._tables: dict[tuple[int, str, Any], tuple[Any, Any]] = {}
        self._kernel: dict[tuple[str, Any], tuple[Any, Any]] = {}
        self._freqs = np.asarray(inv_freq(self.head_dim, self.theta), dtype=np.float64)

    # -- kernels ---------------------------------------------------------

    def _distances(self, n_tokens: int) -> np.ndarray:
        if self.deltas is not None:
            return self.deltas
        return observed_distances(n_tokens)

    def _cos_sin_neg(self, n_tokens: int, device: Any, dtype: Any) -> tuple[Any, Any]:
        key = (int(n_tokens), str(device), dtype)
        if key not in self._tables:
            torch = _torch()
            from rope_attribution.rope import rope_cos_sin

            positions = np.arange(int(n_tokens), dtype=np.float64)
            cos, sin = rope_cos_sin(-positions, self._freqs)
            self._tables[key] = (
                torch.as_tensor(cos, device=device, dtype=dtype),
                torch.as_tensor(sin, device=device, dtype=dtype),
            )
        return self._tables[key]

    def _surrogate_kernel(self, n_tokens: int, device: Any, dtype: Any) -> tuple[Any, Any]:
        key = (
            -1 if self.deltas is not None else int(n_tokens),
            str(device),
            dtype,
        )
        if key not in self._kernel:
            torch = _torch()
            cos_mean, sin_mean = surrogate_kernel(
                self._distances(n_tokens), self._freqs, mode=self.kernel
            )
            self._kernel[key] = (
                torch.as_tensor(cos_mean, device=device, dtype=dtype),
                torch.as_tensor(sin_mean, device=device, dtype=dtype),
            )
        return self._kernel[key]

    # -- hooks -----------------------------------------------------------

    def _transform_q(self, x: Any) -> Any:
        batch, n_tokens, inner = x.shape
        if inner % self.head_dim:
            raise RuntimeError(f"q width {inner} is not a multiple of head_dim {self.head_dim}")
        heads = inner // self.head_dim
        cos, sin = self._cos_sin_neg(n_tokens, x.device, x.dtype)
        shaped = x.reshape(batch, n_tokens, heads, self.head_dim)
        return apply_rope_torch(shaped, cos, sin).reshape(batch, n_tokens, inner)

    def _transform_k(self, x: Any) -> Any:
        batch, n_tokens, inner = x.shape
        if inner % self.head_dim:
            raise RuntimeError(f"k width {inner} is not a multiple of head_dim {self.head_dim}")
        heads = inner // self.head_dim
        cos_mean, sin_mean = self._surrogate_kernel(n_tokens, x.device, x.dtype)
        cos, sin = self._cos_sin_neg(n_tokens, x.device, x.dtype)
        shaped = x.reshape(batch, n_tokens, heads, self.head_dim)
        surrogate = apply_surrogate_torch(shaped, cos_mean, sin_mean)
        return apply_rope_torch(surrogate, cos, sin).reshape(batch, n_tokens, inner)

    def _hook_q(self, _module: Any, _inputs: Any, output: Any) -> Any:
        return self._transform_q(output)

    def _hook_k(self, _module: Any, _inputs: Any, output: Any) -> Any:
        return self._transform_k(output)

    def _hook_fused(self, attn: Any, _module: Any, _inputs: Any, output: Any) -> Any:
        q, k, v = split_fused_qkv(attn, output)
        return join_fused_qkv(
            attn,
            self._transform_q(q),
            self._transform_k(k),
            v,
        )

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> SurrogateAttention:
        for attn in self.attns:
            fused = getattr(attn, "query_key_value", None)
            if fused is not None and hasattr(fused, "weight"):

                def fused_hook(module, inputs, output, _attn=attn):
                    return self._hook_fused(_attn, module, inputs, output)

                self._handles.append(fused.register_forward_hook(fused_hook))
            else:
                self._handles.append(attn.q_proj.register_forward_hook(self._hook_q))
                self._handles.append(attn.k_proj.register_forward_hook(self._hook_k))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _rope_theta(cfg: Any) -> float:
    """The rotary base, from the block or the legacy top-level field."""
    if cfg is None:
        return 10_000.0
    params = _rope_parameters(cfg)
    for name in ("rope_theta", "rotary_emb_base"):
        if name in params:
            return float(params[name])
        value = getattr(cfg, name, None)
        if value is not None:
            return float(value)
    return 10_000.0


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def nll(model: Any, input_ids: Any) -> float:
    """Mean next-token NLL of one sequence, with the model's own forward pass."""
    import torch

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    value = float(out.loss)
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite NLL: {value}")
    return value


def measure_text(
    loaded: Any,
    text: str,
    *,
    max_tokens: int = 512,
    deltas: np.ndarray | None = None,
    kernel: str = "mean",
) -> dict:
    """Exact and surrogate NLL for one text on one loaded checkpoint."""
    loaded.model.eval()
    ids = loaded.tokenizer(text, return_tensors="pt")["input_ids"]
    if ids.shape[1] > max_tokens:
        ids = ids[:, :max_tokens]
    if ids.shape[1] < 2:
        raise ValueError("the text tokenised to fewer than two tokens")
    n_tokens = int(ids.shape[1])
    exact = nll(loaded.model, ids)
    with SurrogateAttention(
        loaded.model,
        deltas=deltas if deltas is not None else observed_distances(n_tokens),
        kernel=kernel,
    ):
        surrogate = nll(loaded.model, ids)
    delta = surrogate - exact
    return {
        "n_tokens": n_tokens,
        "exact_nll": exact,
        "surrogate_nll": surrogate,
        "delta_nll": delta,
        "perplexity_ratio": math.exp(delta),
    }


def run_all(
    model_ids: Sequence[str] = DEFAULT_MODEL_IDS,
    texts: Sequence[str] = DEFAULT_LONG_TEXTS,
    *,
    max_tokens: int = 512,
    seed: int = 0,
    kernel: str = "mean",
) -> dict:
    """Measure every model on every text. A failure is recorded per model."""
    if not REAL_MODEL_AVAILABLE:
        raise RuntimeError(
            "torch and/or transformers are unavailable; see real_model_requirements.txt"
        )
    import torch

    torch.manual_seed(seed)
    models: list[dict] = []
    for model_id in model_ids:
        try:
            loaded = load_model(model_id, seed=seed)
            rows = [
                {
                    "text_index": index,
                    **measure_text(loaded, text, max_tokens=max_tokens, kernel=kernel),
                }
                for index, text in enumerate(texts)
            ]
            deltas = [row["delta_nll"] for row in rows]
            ratios = [row["perplexity_ratio"] for row in rows]
            models.append(
                {
                    "model_id": model_id,
                    "model_class": type(loaded.model).__name__,
                    "head_dim": int(head_dim_of(attention_modules(loaded.model)[0], loaded.config)),
                    "rope_theta": _rope_theta(loaded.config),
                    "texts": rows,
                    "mean_delta_nll": sum(deltas) / len(deltas),
                    "mean_perplexity_ratio": sum(ratios) / len(ratios),
                }
            )
        except Exception as exc:
            models.append({"model_id": model_id, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "schema": SCHEMA,
        "max_tokens": int(max_tokens),
        "seed": int(seed),
        "kernel": kernel,
        "surrogate": (
            "position-free rotation per frequency pair over the observed "
            f"distances; kernel={kernel}"
        ),
        "models": models,
    }


def write_results(data: dict, path: Path | None = None) -> Path:
    out = Path(path) if path is not None else Path(__file__).resolve().parent.parent.parent / "results" / "downstream.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", dest="models", metavar="HF_ID")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--kernel",
        choices=("mean", "circular", "identity"),
        default="mean",
        help="mean: least-squares (attenuating); circular: unit-modulus mean angle; "
        "identity: no rotation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    data = run_all(
        tuple(args.models) if args.models else DEFAULT_MODEL_IDS,
        max_tokens=args.max_tokens,
        seed=args.seed,
        kernel=args.kernel,
    )
    print(f"kernel: {args.kernel}")
    print(f"{'model':<34} {'exact NLL':>10} {'surrogate':>10} {'delta':>9} {'ppl x':>8}")
    for model in data["models"]:
        if "error" in model:
            print(f"{model['model_id']:<34} FAILED: {model['error']}")
            continue
        print(
            f"{model['model_id']:<34} {model['texts'][0]['exact_nll']:>10.4f} "
            f"{model['texts'][0]['surrogate_nll']:>10.4f} "
            f"{model['mean_delta_nll']:>9.4f} {model['mean_perplexity_ratio']:>8.4f}"
        )
    if args.no_write:
        print("(results not written: --no-write)")
    else:
        print(f"wrote {write_results(data, args.out)}")


if __name__ == "__main__":
    main()
