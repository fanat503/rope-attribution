"""Where a position-free attribution stops being wrong, and when it stops working.

The question
------------
``usefulness.py`` prices the best position-free scalar and reports one number for
three checkpoints: it gets the sign wrong on 27.0% of material cells. A number
like that invites two fair questions that it cannot answer.

1. **Why 27%?** If the rate is an accident of the grid, the finding is an
   artefact. If it is a value of a function, the finding is a law.
2. **Why is the rate at the *feature* level (0.017% of cells, in
   ``results/qk_attribution.json``) three orders of magnitude below the rate at
   the *rotary-pair* level (21.4%, same file)?** Both are measured on
   ``pythia-70m``; a reader who compares them will conclude that one of them is
   broken, or that the paper is choosing whichever number suits it.

This module answers both with one object. For a single rotary pair the exact
contribution is ``c(delta) = rho * cos(theta*delta - psi)`` with ``rho`` and
``psi`` fixed by the content. Over a window of distances of length ``L``, put
``x = theta * L`` --- the number of radians the pair sweeps. Then:

* ``x >= 2*pi``: the window contains a full period, so ``c`` attains both
  ``+rho`` and ``-rho``. The minimax constant is ``0`` and the relative
  sup-error is **exactly 1**. No position-free scalar does better than saying
  "this pair contributes nothing".
* ``x < 2*pi``: the best achievable relative sup-error is
  ``(1 - cos(x/2)) / 2``, attained by centring the window on a peak or a trough.

So the rate is not a constant of nature; it is a function of one dimensionless
group, and the threshold at ``x = 2*pi`` is where it saturates. ``27.0%`` is what
that function evaluates to at ``L = 4096`` on the pre-registered log grid.

The second question has an equally clean answer, and it is not "one of the two
numbers is wrong". A head's score is the sum of its rotary pairs **plus the
unrotated tail**. Under partial RoPE (``pythia``, ``rotary_pct = 0.25``) the tail
is three quarters of the channels and contributes a distance-independent term of
the same order as the whole rotary part. The per-pair error therefore survives in
the *pair* attribution and is diluted in the *score*. This module measures the
dilution directly, per head, as

```text
score_rel = max_delta |sum_k (c_hat_k - c_k(delta))| / max_delta |score(delta)|
```

and reports it against the same window. That is the p-dependence: at ``p = 1``
there is no tail to hide behind, and the two rates must converge.

What is closed-form and what is sampled
---------------------------------------
Everything about a *single pair* is closed-form. The extremes of ``cos`` over an
interval are at the endpoints unless the interval contains a phase of ``0`` or
``pi`` mod ``2*pi``; the mean is ``(sin(b) - sin(a)) / (b - a)``. No distance
grid is sampled, so the theorem's hypothesis and the measurement's hypothesis are
the same object --- which is what makes this a check rather than an illustration.

The *score* level is a sum of sinusoids with incommensurate frequencies, so its
extremes have no closed form. There a fixed dense sample is used, the sample size
is recorded in the artefact, and the number is labelled as sampled.

What this module does not claim
-------------------------------
It does not claim the algebra is new: the per-pair closed form is RoFormer's
Eq. 13 and is restated as standard in arXiv:2607.25507. It does not claim the
approximation theory is new: the minimax constant on an interval and the
``1 - cos`` error of a centred window are textbook (Rivlin; Powell). What it
claims is narrower and checkable: the *rate* the paper reports is a value of this
function, the threshold at ``2*pi`` is exact, and the gap between the pair-level
and score-level rates on a partial-RoPE checkpoint is dilution by the unrotated
tail, not a contradiction.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rope_attribution import real_model as RM
from rope_attribution.experiments import pair_terms
from rope_attribution.rope import inv_freq

__all__ = [
    "WINDOWS",
    "DEFAULT_SEED",
    "DEFAULT_MODEL_IDS",
    "TWO_PI",
    "phase_extremes",
    "pair_profile",
    "pair_table",
    "threshold_span",
    "score_profile",
    "measure_head",
    "measure_model",
    "run_all",
    "write_results",
    "main",
]

# Contiguous windows of distances ``1 .. 1 + span``. The theorem is stated for an
# interval of length ``span``; it is the number of token steps between the
# nearest and the farthest key the window admits.
WINDOWS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

# The window always starts here. A key sits behind its query, so distances are
# positive; ``1`` is the previous token, the smallest causal distance.
WINDOW_LO = 1

TWO_PI = 2.0 * np.pi

DEFAULT_SEED = RM.DEFAULT_SEED
DEFAULT_MODEL_IDS = RM.DEFAULT_MODEL_IDS

# Dense sample used only for the score-level supremum, which has no closed form.
SCORE_SAMPLES = 256

# Number of content pairs per head. Smaller than ``usefulness`` because every
# pair is evaluated at ten window lengths instead of one grid.
N_CONTENT_PAIRS = 64

# Log-spaced bins in ``x = theta * span`` for the collapse curve.
X_BINS: tuple[float, ...] = (
    1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1,
    1.0, 2.0, 5.0, 1e1, 1e2, 1e3, 1e4, 1e5,
)


def _contains_phase(lo: np.ndarray, hi: np.ndarray, target: float, period: float) -> np.ndarray:
    """Whether ``[lo, hi]`` contains a point congruent to ``target`` mod ``period``."""
    a = np.minimum(lo, hi)
    b = np.maximum(lo, hi)
    k = np.ceil((a - target) / period)
    return target + k * period <= b


def phase_extremes(lo: float, hi: float) -> tuple[float, float]:
    """``(min, max)`` of ``cos`` over the phase interval ``[lo, hi]``.

    The extremes of ``cos`` on an interval are attained at an endpoint unless the
    interval contains a phase that is a multiple of ``2*pi`` (a maximum, value
    ``+1``) or an odd multiple of ``pi`` (a minimum, value ``-1``). Checking for
    those two congruences is exact and costs nothing, which is why this module
    never samples a distance grid for a single pair.
    """
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    ends = (float(np.cos(a)), float(np.cos(b)))
    high = 1.0 if bool(_contains_phase(a, b, 0.0, TWO_PI)) else max(ends)
    low = -1.0 if bool(_contains_phase(a, b, float(np.pi), TWO_PI)) else min(ends)
    return low, high


def pair_profile(theta: float, psi: float, span: int) -> dict[str, float]:
    """Everything closed-form about one pair over ``delta in [1, 1 + span]``.

    ``c(delta) / rho = cos(theta * delta - psi)``; all returned errors are
    relative to the peak ``rho``, so they are independent of the content
    magnitude and comparable across heads. ``rel_minimax`` is the error of the
    best possible position-free constant; ``rel_mean`` is the error of the
    least-squares constant, which is what ``usefulness.py`` actually uses. The
    second is never smaller than the first, and the gap between them is the price
    of fitting by least squares instead of by minimax.
    """
    if theta <= 0.0:
        raise ValueError(f"theta must be positive, got {theta!r}")
    lo = theta * WINDOW_LO - psi
    hi = theta * (WINDOW_LO + span) - psi
    x = hi - lo
    low, high = phase_extremes(lo, hi)
    mean = float((np.sin(hi) - np.sin(lo)) / x) if x > 0 else float(np.cos(lo))
    rel_mean = max(abs(mean - low), abs(high - mean))
    return {
        "x": float(x),
        "mean": mean,
        "minimax": float(0.5 * (low + high)),
        "rel_minimax": float(0.5 * (high - low)),
        "rel_mean": float(rel_mean),
        "past_threshold": bool(x >= TWO_PI),
    }


def pair_table(freqs: np.ndarray, psi: np.ndarray, spans: Sequence[int]) -> dict[str, np.ndarray]:
    """``pair_profile`` for every (pair, window) at once, shape ``(h, len(spans))``.

    Same formulas as the scalar version; ``tests/test_collapse.py`` asserts the
    two agree elementwise, so the vectorisation cannot drift from the reference.
    """
    theta = np.asarray(freqs, dtype=np.float64)[:, None]
    phase = np.asarray(psi, dtype=np.float64)[:, None]
    span = np.asarray(spans, dtype=np.float64)[None, :]
    lo = theta * WINDOW_LO - phase
    hi = theta * (WINDOW_LO + span) - phase
    x = hi - lo

    ends_a = np.cos(lo)
    ends_b = np.cos(hi)
    has_peak = _contains_phase(lo, hi, 0.0, TWO_PI)
    has_trough = _contains_phase(lo, hi, float(np.pi), TWO_PI)
    high = np.where(has_peak, 1.0, np.maximum(ends_a, ends_b))
    low = np.where(has_trough, -1.0, np.minimum(ends_a, ends_b))
    mean = np.where(x > 0.0, (np.sin(hi) - np.sin(lo)) / np.where(x > 0.0, x, 1.0), ends_a)
    return {
        "x": x,
        "rel_minimax": 0.5 * (high - low),
        "rel_mean": np.maximum(np.abs(mean - low), np.abs(high - mean)),
        "past_threshold": x >= TWO_PI,
    }


def threshold_span(theta: float) -> float:
    """Smallest window length at which the pair's error reaches its ceiling.

    Equivalently the number of token steps the pair needs to sweep a full turn.
    ``inf`` for a pair that never turns, which RoPE never produces but a caller
    might.
    """
    if theta <= 0.0:
        return float("inf")
    return float(TWO_PI / theta)


def score_profile(
    q_hat: np.ndarray,
    k_hat: np.ndarray,
    a_k: np.ndarray,
    b_k: np.ndarray,
    freqs: np.ndarray,
    rotary_dim: int,
    span: int,
    n_samples: int = SCORE_SAMPLES,
) -> dict[str, float]:
    """Pair-level and score-level relative error on the same window.

    ``q_hat``/``k_hat`` are one token's full unrotated head vectors and ``a_k``,
    ``b_k`` its closed-form pair coefficients. The score is ``sum_k c_k(delta) +
    tail``, where ``tail`` is the ordinary inner product of the channels RoPE does
    not rotate. The pair-level error is the mean over pairs of the least-squares
    relative error; the score-level error is the sampled sup of the summed error
    over the sup of the score.

    The ratio between them is the dilution: ``tail`` sits in the denominator and
    nowhere in the numerator, so a head whose score is mostly unrotated channels
    looks almost position-free even though every rotary pair is badly served by
    its constant.
    """
    deltas = np.linspace(WINDOW_LO, WINDOW_LO + span, int(n_samples))
    angles = np.outer(freqs, deltas)
    exact_pairs = a_k[:, None] * np.cos(angles) + b_k[:, None] * np.sin(angles)

    cos_mean = np.cos(angles).mean(axis=1)
    sin_mean = np.sin(angles).mean(axis=1)
    surrogate_pairs = a_k * cos_mean + b_k * sin_mean

    rho = np.hypot(a_k, b_k)
    pair_den = np.maximum(rho, np.finfo(np.float64).tiny)
    pair_rel = np.abs(surrogate_pairs[:, None] - exact_pairs).max(axis=1) / pair_den

    tail = float(q_hat[rotary_dim:] @ k_hat[rotary_dim:]) if rotary_dim < q_hat.size else 0.0
    exact_score = exact_pairs.sum(axis=0) + tail
    surrogate_score = float(surrogate_pairs.sum()) + tail
    peak = float(np.abs(exact_score).max())
    denom = peak if peak > 0.0 else np.finfo(np.float64).tiny
    return {
        "pair_rel_mean": float(pair_rel.mean()),
        "pair_rel_max": float(pair_rel.max()),
        "score_rel": float(np.abs(surrogate_score - exact_score).max() / denom),
        "score_peak": peak,
        "tail_share": float(
            abs(tail) / (abs(tail) + float(rho.sum()) + np.finfo(np.float64).tiny)
        ),
    }


def measure_head(
    hv: RM.HeadVectors,
    freqs: np.ndarray,
    rope_dim: int,
    windows: Sequence[int],
    n_pairs: int,
    rng: np.random.Generator,
    layer: int,
    head: int,
) -> dict[str, np.ndarray]:
    """Every (content pair, window) row for one head, as flat arrays.

    Arrays rather than dicts: the raw table is one row per (content pair, pair,
    window), which is ~10^5 rows per head, and a dict per row costs more than the
    arithmetic. The column names are stable because the artefact and the figure
    both read them.
    """
    q = hv.q_hat[layer, head, :, :rope_dim]
    k = hv.k_hat[layer, head, :, :rope_dim]
    q_full = hv.q_hat[layer, head]
    k_full = hv.k_hat[layer, head]
    spans = np.asarray(windows, dtype=np.int64)
    n_spans = spans.size
    h_pairs = freqs.size

    cols = {name: [] for name in ("x", "rel_minimax", "rel_mean", "past", "span", "pair")}
    score_cols = {name: [] for name in ("span", "score_rel", "tail_share", "pair_rel_mean")}

    for m, n in RM._content_pairs(hv.n_tokens, n_pairs, rng):
        terms = pair_terms(q[m], k[n], freqs, WINDOW_LO)
        a_k = terms.aligned
        b_k = terms.crossed
        tab = pair_table(freqs, np.arctan2(b_k, a_k), spans)

        cols["x"].append(tab["x"].ravel())
        cols["rel_minimax"].append(tab["rel_minimax"].ravel())
        cols["rel_mean"].append(tab["rel_mean"].ravel())
        cols["past"].append(tab["past_threshold"].ravel().astype(np.int64))
        # row-major ravel of an (h, n_spans) array: pair index varies slowest.
        cols["span"].append(np.tile(spans, h_pairs))
        cols["pair"].append(np.repeat(np.arange(h_pairs), n_spans))

        for span in spans:
            prof = score_profile(q_full[m], k_full[n], a_k, b_k, freqs, rope_dim, int(span))
            score_cols["span"].append(np.array([int(span)]))
            score_cols["score_rel"].append(np.array([prof["score_rel"]]))
            score_cols["tail_share"].append(np.array([prof["tail_share"]]))
            score_cols["pair_rel_mean"].append(np.array([prof["pair_rel_mean"]]))

    out = {name: np.concatenate(value) for name, value in cols.items()}
    out["span"] = out["span"].astype(np.int64)
    out["pair"] = out["pair"].astype(np.int64)
    for name, value in score_cols.items():
        out[f"score_{name}"] = np.concatenate(value)
    return out


def _bin_table(
    x: np.ndarray, minimax: np.ndarray, mean: np.ndarray, past: np.ndarray
) -> list[dict]:
    """Median relative error per log-spaced bin of ``x``, for the collapse plot.

    Bins, not a scatter: the raw cloud has one point per (head, content pair,
    pair, window) and would be unreadable as a figure. The min and max per bin
    are kept so the band is drawn honestly rather than as a single line.
    """
    out = []
    for lo, hi in zip(X_BINS[:-1], X_BINS[1:], strict=True):
        mask = (x >= lo) & (x < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(
            {
                "x_lo": float(lo),
                "x_hi": float(hi),
                "n": n,
                "rel_minimax_median": float(np.median(minimax[mask])),
                "rel_minimax_min": float(minimax[mask].min()),
                "rel_minimax_max": float(minimax[mask].max()),
                "rel_mean_median": float(np.median(mean[mask])),
                "rel_mean_max": float(mean[mask].max()),
                "past_threshold_fraction": float(past[mask].mean()),
            }
        )
    return out


def _window_table(table: dict[str, np.ndarray], windows: Sequence[int]) -> list[dict]:
    """Per-window medians at both levels, plus the fraction past threshold."""
    out = []
    for span in windows:
        mask = table["span"] == int(span)
        if not mask.any():
            continue
        score_mask = table["score_span"] == int(span)
        out.append(
            {
                "span": int(span),
                "n_pair_rows": int(mask.sum()),
                "channel_rel_minimax_median": float(np.median(table["rel_minimax"][mask])),
                "channel_rel_mean_median": float(np.median(table["rel_mean"][mask])),
                "channel_rel_minimax_min": float(table["rel_minimax"][mask].min()),
                "past_threshold_fraction": float(table["past"][mask].mean()),
                "score_rel_median": float(np.median(table["score_score_rel"][score_mask])),
                "score_rel_max": float(table["score_score_rel"][score_mask].max()),
                "tail_share_median": float(np.median(table["score_tail_share"][score_mask])),
            }
        )
    return out


def _threshold_table(freqs: np.ndarray, windows: Sequence[int]) -> list[dict]:
    """How many of a checkpoint's rotary pairs are past the threshold.

    A property of the frequency ladder alone, so it is reported for every window
    the module evaluates and costs nothing.
    """
    spans = np.array([threshold_span(float(t)) for t in freqs], dtype=np.float64)
    return [
        {
            "span": int(span),
            "pairs_past_threshold": int(np.sum(spans <= float(span))),
            "n_pairs": int(freqs.size),
            "fraction_past_threshold": float(np.mean(spans <= float(span))),
        }
        for span in windows
    ]


def measure_model(
    model_id: str = DEFAULT_MODEL_IDS[0],
    seed: int = DEFAULT_SEED,
    prompts: Sequence[str] = RM.DEFAULT_PROMPTS,
    windows: Sequence[int] = WINDOWS,
    n_pairs: int = N_CONTENT_PAIRS,
) -> dict:
    """The collapse measurement on one trained checkpoint."""
    loaded = RM.load_model(model_id, seed=seed)
    cfgv = RM.validate_config(loaded.model)
    if not cfgv["checkpoint_inv_freq_found"]:
        raise RuntimeError(f"{model_id} has no rotary module, so it has no rotary pairs")
    rope_dim = int(cfgv["rope_dim"])
    theta = float(cfgv["rope_theta_declared"])
    freqs = inv_freq(rope_dim, theta)

    parts: list[dict[str, np.ndarray]] = []
    n_heads = 0
    for i, text in enumerate(prompts):
        hv = RM.collect_head_vectors(loaded.model, loaded.tokenizer, text)
        n_heads = max(n_heads, hv.n_heads)
        for layer in range(hv.n_layers):
            for head in range(hv.n_heads):
                parts.append(
                    measure_head(
                        hv, freqs, rope_dim, windows, n_pairs,
                        np.random.default_rng(seed + i * 1000 + layer * 31 + head),
                        layer, head,
                    )
                )

    table = {
        name: np.concatenate([p[name] for p in parts]) for name in parts[0]
    }
    if table["x"].size == 0:
        raise ValueError(f"{model_id}: no pair produced a defined profile")
    return {
        "model_id": model_id,
        "rope_dim": rope_dim,
        "rope_theta": theta,
        "n_rotary_pairs": int(freqs.size),
        "n_heads": n_heads,
        "n_windows": len(windows),
        "windows": [int(w) for w in windows],
        "window_lo": WINDOW_LO,
        "score_samples": SCORE_SAMPLES,
        "freqs": [float(t) for t in freqs],
        "threshold_span_per_pair": [threshold_span(float(t)) for t in freqs],
        "threshold": _threshold_table(freqs, windows),
        "by_window": _window_table(table, windows),
        "bins": _bin_table(
            table["x"], table["rel_minimax"], table["rel_mean"], table["past"]
        ),
        "n_pair_rows": int(table["x"].size),
        "n_score_rows": int(table["score_score_rel"].size),
    }


def run_all(
    model_ids: Sequence[str] = DEFAULT_MODEL_IDS,
    seed: int = DEFAULT_SEED,
    prompts: Sequence[str] = RM.DEFAULT_PROMPTS,
    windows: Sequence[int] = WINDOWS,
    n_pairs: int = N_CONTENT_PAIRS,
) -> dict:
    """Every checkpoint, plus the cross-model summary the paper quotes."""
    models = [
        measure_model(m, seed=seed, prompts=prompts, windows=windows, n_pairs=n_pairs)
        for m in model_ids
    ]
    return {
        "schema": "rope-collapse/v1",
        "seed": int(seed),
        "window_lo": WINDOW_LO,
        "windows": [int(w) for w in windows],
        "theorem": {
            "statistic": "minimax relative sup-error of a constant approximation "
            "of rho*cos(theta*delta - psi) over a window of length L",
            "threshold": "x = theta*L >= 2*pi  =>  error == 1 exactly",
            "below_threshold": "min over placements = (1 - cos(x/2))/2",
            "textbook": "best uniform approximation of a function by a constant "
            "(Rivlin; Powell); not a new result of this repository",
        },
        "models": models,
        "pooled": {
            "n_models": len(models),
            "past_threshold_at_4096": [
                {
                    "model_id": m["model_id"],
                    "fraction": next(
                        (
                            t["fraction_past_threshold"]
                            for t in m["threshold"]
                            if t["span"] == 4096
                        ),
                        None,
                    ),
                }
                for m in models
            ],
        },
    }


def write_results(data: dict, path: Path | None = None) -> Path:
    path = path or Path("results/collapse.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODEL_IDS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pairs", type=int, default=N_CONTENT_PAIRS)
    parser.add_argument("--windows", nargs="+", type=int, default=list(WINDOWS))
    parser.add_argument("--out", type=Path, default=Path("results/collapse.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    data = run_all(
        tuple(args.models), seed=args.seed, windows=tuple(args.windows), n_pairs=args.pairs
    )
    path = write_results(data, args.out)
    print(f"wrote {path}")
    for model in data["models"]:
        last = model["by_window"][-1]
        print(
            f"{model['model_id']}: span={last['span']} "
            f"channel rel_minimax median={last['channel_rel_minimax_median']:.4f} "
            f"rel_mean median={last['channel_rel_mean_median']:.4f} "
            f"score rel median={last['score_rel_median']:.4f} "
            f"past threshold={last['past_threshold_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
