"""The threshold law applied to context extension.

Why this module exists
----------------------
The paper's Proposition 5 prices a position-free summary of one rotary pair:
with ``x = theta * L`` radians swept over a window of ``L`` positions, the
minimax relative sup-error is exactly ``1`` for ``x >= 2*pi`` and at least
``(1 - cos(x/2)) / 2`` below it.  The main text uses that law to explain the
measured 27.0% rate.  It also answers a design question that the rate cannot:

    can a context-extension scheme make a position-free attribution valid?

A scheme is *globally attributable* at context length ``L`` if every rotary
channel has ``theta_k * L < 2*pi``.  This module evaluates that predicate for
four schemes that are routinely conflated:

* ``rope`` -- plain RoPE, ``theta_k = base ** (-2k/d)``;
* ``yarn`` -- the published YaRN ramp, with scale ``s`` and original context
  ``L0`` (Peng et al., 2023);
* ``position_interpolation`` -- every position divided by ``s``, so every
  inverse frequency is divided by ``s``;
* ``base_500k`` -- the "just raise the base" heuristic, ``base = 5e5``.

The result is a negative design statement, not a benchmark: YaRN and the
raised-base heuristic leave the fastest channel untouched, so at any
``L >= 2*pi`` a finite scale cannot make every channel sub-threshold.  YaRN's
fastest ``9`` of ``32`` channels are bit-for-bit identical to RoPE at
``s = 32``, which gives it a floor of ``9/32`` past-threshold channels at
``L = 4096`` before the ramp is even considered.  Position interpolation can
reach zero, but only above ``s = L / (2*pi)``: ``652`` at ``L = 4096``, which maps
a 4096-token context onto six positions.  The obstruction is structural.

What is measured and what is assumed
------------------------------------
Everything here is arithmetic on the published frequency ladders; no model is
loaded and no number is a measurement of a checkpoint.  The window is the full
context, which is the most favourable case for a position-free summary: a
shorter window can only reduce ``x``.  The minimax error is reported at the
*best* placement (the lower end of the attainable band), so the numbers are
lower bounds on what a practitioner's fixed placement pays.

The artifact is ``results/extension_law.json``; ``tests/test_extension_law.py``
pins the published counts and the two exact identities (the YaRN fast-channel
floor and the position-interpolation critical scale).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from rope_attribution.rope import inv_freq, yarn_parameters

__all__ = [
    "BASE",
    "BETA_FAST",
    "BETA_SLOW",
    "CONTEXT_LENGTHS",
    "HEAD_DIM",
    "ORIGINAL_CONTEXT",
    "SCALES",
    "SCHEMES",
    "THRESHOLD",
    "channel_rows",
    "critical_scale",
    "minimax_error",
    "scheme_frequencies",
    "summarize",
    "run_all",
    "write_results",
    "main",
]

#: The paper's head dimension and rotary base.
HEAD_DIM = 64
BASE = 10000.0

#: YaRN's published ramp parameters and the pre-extension context.
ORIGINAL_CONTEXT = 2048
BETA_FAST = 32.0
BETA_SLOW = 1.0

#: The law's threshold, in radians swept over the window.
THRESHOLD = 2.0 * math.pi

#: Context lengths a long-context practitioner actually cares about.  The
#: smallest is the pre-extension context; the largest is a 32k target.
CONTEXT_LENGTHS: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768)

#: Extension factors.  ``1`` is plain RoPE and is included so the figure has a
#: common origin; YaRN at ``s = 1`` is identical to RoPE by construction.
SCALES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

#: The four schemes, in the order the paper discusses them.
SCHEMES: tuple[str, ...] = (
    "rope",
    "yarn",
    "position_interpolation",
    "base_500k",
)


def scheme_frequencies(
    scheme: str,
    scale: float = 1.0,
    *,
    head_dim: int = HEAD_DIM,
    base: float = BASE,
    original_context: int = ORIGINAL_CONTEXT,
) -> np.ndarray:
    """Inverse frequencies for one scheme, in ``1 / token``.

    ``scale`` is ignored by ``rope`` and ``base_500k``: the first has no scale
    parameter and the second is a different base, not a scaling.  The caller
    should read the scheme's own definition rather than treating ``scale`` as a
    universal knob.
    """
    if scheme == "rope":
        return inv_freq(head_dim, base)
    if scheme == "yarn":
        freqs, _ = yarn_parameters(
            head_dim,
            base,
            float(scale),
            int(original_context),
            beta_fast=BETA_FAST,
            beta_slow=BETA_SLOW,
        )
        return freqs
    if scheme == "position_interpolation":
        if scale <= 0:
            raise ValueError("position interpolation needs a positive scale")
        return inv_freq(head_dim, base) / float(scale)
    if scheme == "base_500k":
        return inv_freq(head_dim, 500000.0)
    raise ValueError(f"unknown scheme {scheme!r}")


def minimax_error(x: np.ndarray | float) -> np.ndarray:
    """The best-placement minimax relative error of Proposition 5.

    ``1`` for ``x >= 2*pi``; ``(1 - cos(x/2)) / 2`` below.  This is the lower
    end of the attainable band, i.e. the error of the *best* constant, so it is
    the most favourable number a position-free summary can achieve.
    """
    x = np.asarray(x, dtype=np.float64)
    below = 0.5 * (1.0 - np.cos(0.5 * x))
    return np.where(x >= THRESHOLD, 1.0, below)


def worst_case_error(x: np.ndarray | float) -> np.ndarray:
    """The worst-placement error: the upper end of the attainable band.

    For ``x < 2*pi`` this is ``sin(min(x, pi)/2)``; at and above the threshold
    the band collapses to ``1``.
    """
    x = np.asarray(x, dtype=np.float64)
    upper = np.sin(0.5 * np.minimum(x, math.pi))
    return np.where(x >= THRESHOLD, 1.0, upper)


def critical_scale(
    scheme: str,
    context: int,
    *,
    head_dim: int = HEAD_DIM,
    base: float = BASE,
) -> float | None:
    """The infimum of the scales that make every channel sub-threshold.

    ``None`` means no finite scale can: the scheme leaves the fastest channel
    at ``theta_0 = 1``, so ``x_0 = L`` and ``L >= 2*pi`` is already past the
    threshold.  Position interpolation is the only scheme here that divides the
    fastest channel as well, and its critical scale is ``L / (2*pi)``.  The
    condition is strict -- at exactly that scale the fastest channel sits *on*
    the threshold, which the law counts as past it -- so the smallest usable
    scale is any value above the returned one.
    """
    if context <= 0:
        raise ValueError("context must be positive")
    if context < THRESHOLD:
        return 1.0
    if scheme == "position_interpolation":
        return float(context) / THRESHOLD
    if scheme in ("rope", "yarn", "base_500k"):
        # theta_0 = 1 for all three.  No finite scale changes that: YaRN's ramp
        # leaves the fastest channel untouched, and raising the base does too.
        return None
    raise ValueError(f"unknown scheme {scheme!r}")


def channel_rows(
    scheme: str,
    scale: float,
    context: int,
    *,
    head_dim: int = HEAD_DIM,
    base: float = BASE,
) -> list[dict]:
    """One row per rotary channel at a fixed ``(scheme, scale, context)``."""
    freqs = scheme_frequencies(
        scheme, scale, head_dim=head_dim, base=base
    )
    x = freqs * float(context)
    lower = minimax_error(x)
    upper = worst_case_error(x)
    return [
        {
            "channel": int(k),
            "inv_freq": float(freqs[k]),
            "phase": float(x[k]),
            "past_threshold": bool(x[k] >= THRESHOLD),
            "minimax_error": float(lower[k]),
            "worst_case_error": float(upper[k]),
        }
        for k in range(len(freqs))
    ]


def _untouched_channels(
    scheme: str,
    scale: float,
    *,
    head_dim: int = HEAD_DIM,
    base: float = BASE,
) -> int:
    """How many channels a scheme leaves bit-for-bit equal to plain RoPE.

    This is the floor behind the paper's YaRN remark.  The comparison is exact
    equality, not a tolerance: the published ramp multiplies the interpolation
    branch by zero on the fast channels, so the result is the extrapolation
    value itself.  Plain RoPE leaves all of its own channels equal to itself;
    the count is reported as ``len(freqs)`` rather than special-cased to zero,
    because the table's caption is a literal statement about equality.
    """
    rope = inv_freq(head_dim, base)
    freqs = scheme_frequencies(scheme, scale, head_dim=head_dim, base=base)
    return int(np.sum(freqs == rope))


def summarize(
    scheme: str,
    scale: float,
    context: int,
    *,
    head_dim: int = HEAD_DIM,
    base: float = BASE,
) -> dict:
    """The scheme-level summary at one ``(scheme, scale, context)``."""
    rows = channel_rows(scheme, scale, context, head_dim=head_dim, base=base)
    x = np.array([row["phase"] for row in rows], dtype=np.float64)
    lower = np.array([row["minimax_error"] for row in rows], dtype=np.float64)
    upper = np.array([row["worst_case_error"] for row in rows], dtype=np.float64)
    n_past = int(np.sum(x >= THRESHOLD))
    untouched = _untouched_channels(scheme, scale, head_dim=head_dim, base=base)
    rope = inv_freq(head_dim, base)
    freqs = scheme_frequencies(scheme, scale, head_dim=head_dim, base=base)
    untouched_past = int(
        np.sum((freqs == rope) & (rope * float(context) >= THRESHOLD))
    )
    return {
        "scheme": scheme,
        "scale": float(scale),
        "context": int(context),
        "n_channels": len(rows),
        "n_past_threshold": n_past,
        "fraction_past_threshold": n_past / len(rows),
        "mean_minimax_error": float(np.mean(lower)),
        "mean_worst_case_error": float(np.mean(upper)),
        "max_phase": float(np.max(x)),
        "min_phase": float(np.min(x)),
        "untouched_channels": untouched,
        "untouched_past_threshold": untouched_past,
        "critical_scale": critical_scale(scheme, context, head_dim=head_dim, base=base),
    }


def run_all(
    *,
    head_dim: int = HEAD_DIM,
    base: float = BASE,
    contexts: tuple[int, ...] = CONTEXT_LENGTHS,
    scales: tuple[int, ...] = SCALES,
) -> dict:
    """Every summary row, plus the canonical ``L = 4096`` channel table.

    The canonical scale is ``s = 32`` because that is the scale the paper uses
    in its running example (pre-extension context 2048, target 65536 in the
    YaRN paper; the synthetic section uses the same ramp).
    """
    rows: list[dict] = []
    for scheme in SCHEMES:
        for scale in scales:
            for context in contexts:
                rows.append(
                    summarize(
                        scheme,
                        scale,
                        context,
                        head_dim=head_dim,
                        base=base,
                    )
                )
    canonical = {
        scheme: channel_rows(scheme, 32.0, 4096, head_dim=head_dim, base=base)
        for scheme in SCHEMES
    }
    by_key = {(row["scheme"], row["scale"], row["context"]): row for row in rows}
    headline = {
        "context_4096": {
            scheme: {
                "n_past_threshold": by_key[(scheme, 32.0, 4096)]["n_past_threshold"],
                "fraction_past_threshold": by_key[(scheme, 32.0, 4096)][
                    "fraction_past_threshold"
                ],
                "mean_minimax_error": by_key[(scheme, 32.0, 4096)][
                    "mean_minimax_error"
                ],
                "untouched_channels": by_key[(scheme, 32.0, 4096)][
                    "untouched_channels"
                ],
                "untouched_past_threshold": by_key[(scheme, 32.0, 4096)][
                    "untouched_past_threshold"
                ],
                "critical_scale": by_key[(scheme, 32.0, 4096)]["critical_scale"],
            }
            for scheme in SCHEMES
        },
        "yarn_floor": {
            "scale": 32.0,
            "untouched_channels": _untouched_channels("yarn", 32.0),
            "untouched_past_threshold_4096": by_key[("yarn", 32.0, 4096)][
                "untouched_past_threshold"
            ],
            "statement": (
                "YaRN leaves the fastest channels bit-for-bit equal to RoPE, so "
                "its past-threshold count cannot fall below the number of "
                "untouched channels that are already past threshold."
            ),
        },
        "position_interpolation_critical_scale_4096": critical_scale(
            "position_interpolation", 4096
        ),
    }
    return {
        "schema": "rope-extension-law/v1",
        "configuration": {
            "head_dim": int(head_dim),
            "base": float(base),
            "context_lengths": list(contexts),
            "scales": list(scales),
            "schemes": list(SCHEMES),
            "threshold_radians": THRESHOLD,
            "original_context": ORIGINAL_CONTEXT,
            "beta_fast": BETA_FAST,
            "beta_slow": BETA_SLOW,
            "window": "full context; the most favourable window for a position-free summary",
            "estimand": (
                "best-placement minimax relative sup-error of a constant "
                "approximation to one rotary pair; lower bounds, not a fit"
            ),
        },
        "headline": headline,
        "rows": rows,
        "canonical_channels_4096": canonical,
    }


def write_results(out: Path | None = None, **kwargs: object) -> Path:
    result = run_all(**kwargs)
    path = (
        Path(out)
        if out is not None
        else Path(__file__).resolve().parent.parent.parent
        / "results"
        / "extension_law.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def main(out: Path | None = None) -> None:
    result = run_all()
    cfg = result["configuration"]
    print("=" * 78)
    print("The threshold law under context extension")
    print("=" * 78)
    print(
        f"d={cfg['head_dim']} base={cfg['base']:g} "
        f"threshold={cfg['threshold_radians']:.6f} rad"
    )
    print()
    print(f"{'scheme':<24} {'L':>7} {'past':>7} {'frac':>7} {'mean err':>9}")
    for row in result["rows"]:
        if row["scale"] != 32.0:
            continue
        print(
            f"{row['scheme']:<24} {row['context']:>7} "
            f"{row['n_past_threshold']:>7} "
            f"{row['fraction_past_threshold']:>7.3f} "
            f"{row['mean_minimax_error']:>9.4f}"
        )
    print()
    print("At L=4096, s=32:")
    for scheme, entry in result["headline"]["context_4096"].items():
        print(
            f"  {scheme:<24} past={entry['n_past_threshold']:>2}/32 "
            f"untouched={entry['untouched_channels']:>2} "
            f"untouched_past={entry['untouched_past_threshold']:>2} "
            f"critical_scale={entry['critical_scale']}"
        )
    floor = result["headline"]["yarn_floor"]
    print(
        f"  YaRN floor: {floor['untouched_channels']} untouched channels, "
        f"{floor['untouched_past_threshold_4096']} already past threshold at L=4096"
    )
    path = write_results(out)
    print()
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
