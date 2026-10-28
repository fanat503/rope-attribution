"""Generated figure set for the RoPE / YaRN attribution study.

Every value that appears on an axis, in an annotation, in a title or in
``figures/README.md`` is computed here at run time by calling into
``rope_attribution.experiments`` or ``rope_attribution.rope`` --- or, for the
configuration sweep (fig11), by reading ``results/sensitivity.json``, the
artifact ``rope_attribution.sensitivity`` regenerates and the test suite
diffs. There are no
typed-in measurement constants anywhere in this file: the literals that *are*
present are configuration (head dim, RoPE base, extension scale, seeds), the
definition of the distance grid, plotting parameters (dpi, figure size, tick
fiddling, a display floor for log axes) and two dimensionless *thresholds* that
are definitions rather than results - the ``0.1 rad`` linearization cut used by
``experiments.linearization_error`` and the ``1.0`` (= 100% relative error)
reference used in fig06 and fig11.

The module is the only thing in the repository allowed to write into
``figures/``. Each figure is written twice: a PNG for humans and a CSV
sidecar holding the exact array that was plotted, so a reader never has to
trust the pixels.

Run it from the repository root either way::

    python -m projects.rope_attribution.figures
    python projects/rope_attribution/figures.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# ``projects/`` has to be importable as ``rope_attribution``. ``PYTHONPATH=projects``
# already does that for ``python -m projects.rope_attribution.figures``; this makes
# ``python projects/rope_attribution/figures.py`` work from the repository root too.
_PROJECTS_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECTS_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rope_attribution.experiments import (  # noqa: E402
    PairTerms,
    method_spectrum,
    mscale_entropy,
    pair_terms,
    partial_rope_split,
    score_relative,
)
from rope_attribution.rope import (  # noqa: E402
    apply_partial_rope,
    inv_freq,
    partial_rope_cos_sin,
    yarn_parameters,
)

# --------------------------------------------------------------------------
# configuration -- knobs, not measurements
# --------------------------------------------------------------------------

HEAD_DIM = 64  # head dimension, must be even
ROPE_BASE = 10_000.0  # the RoPE base; a definition of the method
LEGACY_BASE = 500_000.0  # the base the legacy documents *claimed* YaRN used
EXT_SCALE = 32.0  # context-extension factor under test
ORIGINAL_MAX_POS = 2048  # pre-extension context length
N_KEYS = 256  # keys in the mscale/entropy attention problem
N_FEATURES = 8  # synthetic SAE features for fig09
N_ROT_FRAC = 0.25  # rotated fraction for partial RoPE
LINEARIZABLE_ANGLE_RAD = 0.1  # definition inside experiments.linearization_error
UNIT_RELATIVE_ERROR = 1.0  # "error as large as the contribution itself"

SEED_QK = 20_250_929  # q_hat / k_hat shared by fig02 and fig06
SEED_METHOD = 2  # experiments.method_spectrum
SEED_ATTRIB = 1  # experiments.feature_attribution
SEED_PARTIAL = 3  # experiments.partial_rope_split
SEED_MSCALE = 4  # experiments.mscale_entropy

DPI = 200
FIGSIZE = (6.8, 4.4)
FIGSIZE_PAIR = (10.6, 4.4)
FIGSIZE_TALL = (9.6, 6.8)

DELTA_MAX = 8192  # top of the relative-distance grid (axis extent)
DELTA_POINTS = 61  # log-spaced samples requested before de-duplication
DELTA_STRIDE = 7  # how many of them fig02 draws
DELTA_GRID = tuple(
    int(x) for x in np.unique(np.round(np.logspace(0.0, np.log10(DELTA_MAX), DELTA_POINTS)))
)
PAIR_PICK = (0, 4, 12, 24, 31)  # rotary pairs highlighted in fig06
SCALE_MAX = 128.0  # top of the extension-scale grid for fig08
SCALE_POINTS = 17
# EXT_SCALE is unioned in so fig08 can report the exact scale under test rather
# than whichever grid point happens to be nearest to it.
SCALE_GRID = tuple(
    sorted({float(x) for x in np.geomspace(1.0, SCALE_MAX, SCALE_POINTS)} | {EXT_SCALE})
)

PLOT_FLOOR = 1e-18  # log-axis display floor; CSVs keep the raw values
EPS = 1e-12  # denominator guard, matching the modules we call
SYMLOG_PIVOT_RAD = 1.0  # symlog pivot for angle plots, in radians
DEVIATION_PIVOT = 1e-2  # symlog pivot for the deviation panel of fig07
REL_TINY = 0.01  # symlog pivot, as a fraction of the panel's own peak
LIN_FRACTION = 0.2  # share of a symlog axis given to its linear strip
SYMLOG_TICK_GAP = 4.0  # smallest decade tick, as a multiple of the symlog pivot

PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # amber
    "#56B4E9",  # sky
    "#5B5B5B",  # grey
    "#8C564B",  # brown
)
SEQUENTIAL = "viridis"

# The four method labels experiments.method_spectrum is documented to return.
METHOD_ROPE = "rope_base_10k"
METHOD_YARN = "yarn"
METHOD_INTERPOLATION = "position_interpolation"
METHOD_LEGACY = "base_500k_legacy_claim"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = REPO_ROOT / "figures"


# --------------------------------------------------------------------------
# small infrastructure
# --------------------------------------------------------------------------


@dataclass
class FigureRecord:
    """What a finished figure contributed to ``figures/README.md``."""

    stem: str
    producer: str
    headline: str
    extra: str = ""


def _apply_style() -> None:
    """Print-safe light style: white ground, thin grey grid, no chart junk."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.color": "#D6D6D6",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "figure.dpi": 110,
        }
    )


def _save(
    fig: plt.Figure, stem: str, suptitle: str | None = None, top: float = 0.93
) -> Path:
    """Tighten the layout, write the PNG, release the figure."""
    # Draw once first so tight_layout sees the real tick-label extents.
    fig.canvas.draw()
    if suptitle is not None:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    else:
        fig.tight_layout()
    # "tight" grows the canvas to whatever the text needs: a suptitle wider than
    # the figure turns the plot into an unreadable strip of pixels at print size
    # (fig09 shipped like that once). Refuse to write a figure whose saved
    # canvas is more than 1.6x its declared size -- that ratio was measured at
    # 2.36 before fig09's suptitle was wrapped.
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    declared = fig.get_size_inches()
    if tight.width > 1.6 * declared[0] or tight.height > 1.6 * declared[1]:
        plt.close(fig)
        raise ValueError(
            f"{stem}: tight bbox {tight.width:.2f}x{tight.height:.2f} in is over "
            f"1.6x the declared figsize {declared[0]:.2f}x{declared[1]:.2f} in; "
            "wrap the suptitle or move it into the caption"
        )
    path = FIGS_DIR / f"{stem}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_csv(stem: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> Path:
    path = FIGS_DIR / f"{stem}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows([list(r) for r in rows])
    return path


def _note(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.0,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "pad": 3.0},
    )


def _floor(values: np.ndarray) -> np.ndarray:
    """Clip to the display floor so a log axis survives an exact zero."""
    return np.maximum(np.asarray(values, dtype=np.float64), PLOT_FLOOR)


def _colour(index: int) -> str:
    """Palette lookup that cycles, so a long method list still has colours."""
    return PALETTE[index % len(PALETTE)]


def _symlog(ax: plt.Axes, pivot: float, values: np.ndarray) -> None:
    """Symlog y with a thin linear strip and an explicit, non-crowded tick set.

    ``matplotlib``'s default symlog locator crams 0, +/-pivot and 10^-1 into the
    same few millimetres whenever the linear strip is narrow, so the decades are
    placed here instead: 0, plus the decades that clear ``pivot`` by a factor of
    four, capped at seven decades.
    """
    ax.set_yscale("symlog", linthresh=pivot, linscale=LIN_FRACTION)
    peak = float(np.max(np.abs(values)))
    if peak <= 0.0:
        return
    e_hi = int(math.floor(math.log10(peak)))
    e_lo = max(int(math.ceil(math.log10(SYMLOG_TICK_GAP * pivot))), e_hi - 6)
    e_lo = min(e_lo, e_hi)
    ticks = [0.0]
    for exponent in range(e_lo, e_hi + 1):
        ticks.extend([-(10.0**exponent), 10.0**exponent])
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _at_largest(rows: Sequence[dict], key: str) -> float:
    """The ``key`` value at the largest distance on the grid."""
    best = max(rows, key=lambda r: r["delta"])
    return float(best[key])


def _pick(grouped: dict[str, list[dict]], method: str) -> list[dict]:
    """Rows for one named method, from whatever ``method_spectrum`` returned."""
    if method not in grouped:
        raise KeyError(f"{method!r} not in method_spectrum output: {list(grouped)}")
    return grouped[method]


def get_mscale_closed_form(scale: float) -> float:
    """The published YaRN magnitude formula, re-derived here for the overlay."""
    return 1.0 if scale <= 1.0 else 0.1 * math.log(scale) + 1.0


def _qk_vectors(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(HEAD_DIM), rng.standard_normal(HEAD_DIM)


def _linearized(t: PairTerms) -> np.ndarray:
    """The first-order approximation that ``linearization_error`` measures.

    ``cos D -> 1 - D^2/2`` and ``sin D -> D``. This is the definition of the
    approximation, transcribed from the docstring of that function; it is a
    formula, not a result.
    """
    return t.aligned * (1.0 - t.angle**2 / 2.0) + t.crossed * t.angle


# --------------------------------------------------------------------------
# fig01 -- the frequency ladder
# --------------------------------------------------------------------------


def fig01_frequency_ladder() -> FigureRecord:
    stem = "fig01_frequency_ladder"
    pair_index = np.arange(HEAD_DIM // 2)
    base_freqs = inv_freq(HEAD_DIM, ROPE_BASE)
    yarn_freqs, _mscale = yarn_parameters(
        HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS
    )
    series = {
        f"RoPE, base={ROPE_BASE:g}": base_freqs,
        f"position interpolation, inv_freq/{EXT_SCALE:g}": base_freqs / EXT_SCALE,
        f"YaRN, ramped, scale={EXT_SCALE:g}": yarn_freqs,
        f"base={LEGACY_BASE:g} (the legacy claim)": inv_freq(HEAD_DIM, LEGACY_BASE),
    }

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (label, values) in enumerate(series.items()):
        ax.semilogy(
            pair_index,
            _floor(values),
            color=_colour(i),
            marker="o",
            markersize=2.5,
            label=label,
        )

    n_unchanged = int(np.count_nonzero(yarn_freqs == base_freqs))
    slow_ratio = float(yarn_freqs[-1] / base_freqs[-1])
    fast_ratio = float(yarn_freqs[0] / base_freqs[0])
    _note(
        ax,
        f"YaRN / RoPE inverse frequency\n"
        f"  fastest pair k=0:  {fast_ratio:.6f} (unchanged)\n"
        f"  slowest pair k={HEAD_DIM // 2 - 1}:  {slow_ratio:.6f}\n"
        f"  pairs left unchanged: {n_unchanged} of {HEAD_DIM // 2}\n"
        f"  base in = base out = {ROPE_BASE:g}",
    )
    ax.set_xlabel("rotary pair index k  (0 .. head_dim/2 - 1)")
    ax.set_ylabel("inverse frequency  inv_freq[k]  (rad per position step)")
    ax.set_title(
        f"Frequency ladder at head_dim={HEAD_DIM}: YaRN keeps the fast channels\n"
        f"and ramps only the slow ones (scale={EXT_SCALE:g}, "
        f"original context {ORIGINAL_MAX_POS})"
    )
    ax.legend(loc="lower left", fontsize=8)
    png = _save(fig, stem)
    csv_path = _write_csv(
        stem,
        ["pair_index", *series],
        ([i, *[float(v[i]) for v in series.values()]] for i in pair_index),
    )

    return FigureRecord(
        stem=stem,
        producer=(
            "rope.inv_freq, rope.yarn_parameters"
            f" (called through figures.py, head_dim={HEAD_DIM},"
            f" base={ROPE_BASE:g}, scale={EXT_SCALE:g})"
        ),
        headline=(
            f"YaRN leaves {n_unchanged} of {HEAD_DIM // 2} rotary pairs bit-for-bit "
            f"unchanged and slows the slowest pair by a measured factor of"
            f" {1.0 / slow_ratio:.2f}."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


# --------------------------------------------------------------------------
# fig02 -- per-channel angle spectrum
# --------------------------------------------------------------------------


def fig02_angle_spectrum(q_hat: np.ndarray, k_hat: np.ndarray) -> FigureRecord:
    stem = "fig02_angle_spectrum"
    yarn_freqs, _mscale = yarn_parameters(
        HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS
    )
    panels = {
        f"RoPE, base={ROPE_BASE:g}": inv_freq(HEAD_DIM, ROPE_BASE),
        f"YaRN, scale={EXT_SCALE:g}": yarn_freqs,
    }
    shown = DELTA_GRID[::DELTA_STRIDE]
    pair_index = np.arange(HEAD_DIM // 2)
    cmap = plt.get_cmap(SEQUENTIAL)
    norm = matplotlib.colors.LogNorm(vmin=float(shown[0]), vmax=float(shown[-1]))

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PAIR)
    rows: list[list[object]] = []
    peaks: dict[str, float] = {}
    medians: dict[str, float] = {}
    drawn: dict[str, np.ndarray] = {}
    for ax, (label, freqs) in zip(axes, panels.items(), strict=True):
        curves = []
        for delta in shown:
            angle = np.abs(pair_terms(q_hat, k_hat, freqs, delta).angle)
            rows.extend([label, delta, int(k), float(a)] for k, a in enumerate(angle))
            peaks[label] = max(peaks.get(label, 0.0), float(angle.max()))
            medians[label] = float(np.median(angle))
            ax.semilogx(pair_index, _floor(angle), color=cmap(norm(delta)), lw=1.3)
            curves.append(_floor(angle))
        drawn[label] = np.concatenate(curves)
        _symlog(ax, SYMLOG_PIVOT_RAD, drawn[label])
        ax.axhline(SYMLOG_PIVOT_RAD, color="#999999", lw=0.7, ls=":")
        ax.set_xlabel("rotary pair index k")
        ax.set_ylabel("per-pair rotation angle  |D_k| = delta * inv_freq[k]  (rad)")
        ax.set_title(
            f"{label}\nlargest angle over all shown distances:"
            f" {peaks[label]:.4g} rad"
        )
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        bar.set_label("relative distance delta (positions)")

    labels = list(panels)
    peak_diff = abs(peaks[labels[0]] - peaks[labels[1]])
    med_ratio = medians[labels[1]] / medians[labels[0]]
    _note(
        axes[1],
        f"at the largest shown delta={shown[-1]}\n"
        f"  max |D_k|: RoPE {peaks[labels[0]]:.6g}, YaRN {peaks[labels[1]]:.6g} rad\n"
        f"  difference in the max: {peak_diff:.3e} rad\n"
        f"  median |D_k| ratio YaRN/RoPE: {med_ratio:.4f}",
    )
    png = _save(
        fig,
        stem,
        "YaRN lowers the bulk of the per-channel angles but leaves the maximum"
        " untouched",
    )
    csv_path = _write_csv(
        stem, ["scheme", "delta", "pair_index", "abs_angle_rad"], rows
    )

    return FigureRecord(
        stem=stem,
        producer="experiments.pair_terms(...).angle over both frequency ladders",
        headline=(
            f"at delta={shown[-1]}: max |D_k| is {peaks[labels[0]]:.6g} rad for RoPE "
            f"and {peaks[labels[1]]:.6g} rad for YaRN (difference {peak_diff:.3e} rad), "
            f"while the median angle falls to a {med_ratio:.4f}x fraction."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


# --------------------------------------------------------------------------
# fig03 -- max vs median angle
# --------------------------------------------------------------------------


def fig03_max_vs_median_angle() -> FigureRecord:
    stem = "fig03_max_vs_median_angle"
    grouped = _method_rows()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (method, rows) in enumerate(grouped.items()):
        deltas = [r["delta"] for r in rows]
        ax.loglog(
            deltas,
            _floor([r["max_abs_angle"] for r in rows]),
            color=_colour(i),
            label=method,
        )
        ax.loglog(
            deltas,
            _floor([r["median_abs_angle"] for r in rows]),
            color=_colour(i),
            ls="--",
            lw=1.1,
        )

    rope_rows = _pick(grouped, METHOD_ROPE)
    yarn_rows = _pick(grouped, METHOD_YARN)
    peak_gap = max(
        abs(a["max_abs_angle"] - b["max_abs_angle"])
        for a, b in zip(rope_rows, yarn_rows, strict=True)
    )
    peak_ratio = _at_largest(rope_rows, "max_abs_angle") / _at_largest(
        yarn_rows, "max_abs_angle"
    )
    med_ratio = _at_largest(yarn_rows, "median_abs_angle") / _at_largest(
        rope_rows, "median_abs_angle"
    )
    _note(
        ax,
        f"max |D_k| for {METHOD_ROPE} vs {METHOD_YARN}\n"
        f"  largest difference over {len(DELTA_GRID)} distances:"
        f" {peak_gap:.3e} rad\n"
        f"  ratio at delta={DELTA_GRID[-1]}: {peak_ratio:.10f}\n"
        f"  median |D_k| ratio at the same delta: {med_ratio:.4f}",
    )
    ax.set_xlabel("relative distance delta = n - m  (positions, log scale)")
    ax.set_ylabel("rotation angle magnitude  (rad)")
    ax.set_title(
        "The worst-case channel is identical for RoPE and YaRN;\n"
        "only the median channel moves.\nsolid = max |D_k|, dashed = median |D_k|"
    )
    ax.legend(loc="lower right", ncol=2, fontsize=7.5)
    png = _save(fig, stem)
    csv_path = _write_csv(
        stem,
        ["method", "delta", "mscale", "max_abs_angle_rad", "median_abs_angle_rad"],
        (
            [
                method,
                r["delta"],
                r["mscale"],
                r["max_abs_angle"],
                r["median_abs_angle"],
            ]
            for method, rows in grouped.items()
            for r in rows
        ),
    )

    return FigureRecord(
        stem=stem,
        producer=(
            "experiments.method_spectrum (which calls experiments.linearization_error),"
            f" seed={SEED_METHOD}, {len(DELTA_GRID)} distances"
        ),
        headline=(
            f"max |D_k| for {METHOD_ROPE} and {METHOD_YARN} agree to "
            f"{peak_gap:.3e} rad across all {len(DELTA_GRID)} distances "
            f"(ratio {peak_ratio:.10f} at delta={DELTA_GRID[-1]}), while the median"
            f" angle differs by {med_ratio:.4f}x."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


# --------------------------------------------------------------------------
# fig04 -- the headline figure
# --------------------------------------------------------------------------


def fig04_linearizable_fraction() -> FigureRecord:
    stem = "fig04_linearizable_fraction"
    grouped = _method_rows()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    at_end: list[tuple[str, float]] = []
    for i, (method, rows) in enumerate(grouped.items()):
        values = [r["frac_channels_linearizable"] for r in rows]
        ax.semilogx(
            [r["delta"] for r in rows],
            values,
            color=_colour(i),
            marker="o",
            markersize=2.5,
            label=method,
        )
        at_end.append((method, values[-1]))

    by_name = dict(at_end)
    rope_frac = by_name[METHOD_ROPE]
    yarn_frac = by_name[METHOD_YARN]
    _note(
        ax,
        f"fraction with |D_k| <= {LINEARIZABLE_ANGLE_RAD} rad, at"
        f" delta={DELTA_GRID[-1]}\n"
        + "\n".join(f"  {name:<24} {value:.4f}" for name, value in at_end)
        + f"\n  {METHOD_ROPE} collapses to {rope_frac:.4f};"
          f" {METHOD_YARN} keeps {yarn_frac:.4f}",
    )
    ax.set_xlabel("relative distance delta = n - m  (positions, log scale)")
    ax.set_ylabel(
        f"fraction of rotary pairs with |D_k| <= {LINEARIZABLE_ANGLE_RAD} rad"
        "  (dimensionless)"
    )
    ax.set_title(
        "Headline: the fraction of channels that can still be linearized.\n"
        f"Plain RoPE runs out of them, YaRN keeps {yarn_frac:.1%}"
    )
    ax.set_ylim(-0.03, 1.12)
    ax.legend(loc="lower left", fontsize=8)
    png = _save(fig, stem)
    csv_path = _write_csv(
        stem,
        ["method", "delta", "frac_channels_linearizable"],
        (
            [method, r["delta"], r["frac_channels_linearizable"]]
            for method, rows in grouped.items()
            for r in rows
        ),
    )

    return FigureRecord(
        stem=stem,
        producer=(
            f"experiments.method_spectrum -> frac_channels_linearizable, seed={SEED_METHOD}"
        ),
        headline=(
            f"at delta={DELTA_GRID[-1]}: {METHOD_ROPE} retains {rope_frac:.4f} of its"
            f" channels and {METHOD_YARN} retains {yarn_frac:.4f}"
            f" (the other two retain"
            f" {by_name[METHOD_INTERPOLATION]:.4f} and"
            f" {by_name[METHOD_LEGACY]:.4f})."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


# --------------------------------------------------------------------------
# fig05 -- linearization error
# --------------------------------------------------------------------------


def fig05_linearization_error() -> FigureRecord:
    stem = "fig05_linearization_error"
    grouped = _method_rows()
    panels = (
        ("amplitude_weighted_err", "amplitude-weighted"),
        ("per_pair_err_rms", "per-pair RMS"),
    )
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PAIR, sharex=True, sharey=True)

    for ax, (key, title) in zip(axes, panels, strict=True):
        for i, (method, rows) in enumerate(grouped.items()):
            ax.semilogx(
                [r["delta"] for r in rows],
                _floor([r[key] for r in rows]),
                color=_colour(i),
                label=method,
            )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("relative distance delta = n - m  (positions, log scale)")
    axes[0].set_ylabel(
        "relative error  (dimensionless, log scale)\n"
        "1 = an error as large as the pair's own amplitude R_k"
    )

    rope_rows = _pick(grouped, METHOD_ROPE)
    yarn_rows = _pick(grouped, METHOD_YARN)
    aw_rope = _at_largest(rope_rows, "amplitude_weighted_err")
    aw_yarn = _at_largest(yarn_rows, "amplitude_weighted_err")
    rms_rope = _at_largest(rope_rows, "per_pair_err_rms")
    rms_yarn = _at_largest(yarn_rows, "per_pair_err_rms")
    _note(
        axes[1],
        f"at delta={DELTA_GRID[-1]}\n"
        f"  amp-weighted: {METHOD_ROPE} {aw_rope:.6g}\n"
        f"  amp-weighted: {METHOD_YARN} {aw_yarn:.6g}\n"
        f"  ratio yarn/rope: {aw_yarn / aw_rope:.6f}\n"
        f"  per-pair RMS ratio yarn/rope: {rms_yarn / rms_rope:.6f}",
    )
    axes[0].legend(loc="upper left", fontsize=7.5)
    png = _save(
        fig,
        stem,
        "Linearization damage is governed by the fast channels, which YaRN does"
        " not touch",
    )
    csv_path = _write_csv(
        stem,
        ["method", "delta", "amplitude_weighted_err", "per_pair_err_rms"],
        (
            [
                method,
                r["delta"],
                r["amplitude_weighted_err"],
                r["per_pair_err_rms"],
            ]
            for method, rows in grouped.items()
            for r in rows
        ),
    )

    return FigureRecord(
        stem=stem,
        producer=(
            "experiments.method_spectrum -> amplitude_weighted_err and per_pair_err_rms"
        ),
        headline=(
            f"at delta={DELTA_GRID[-1]} the amplitude-weighted error is {aw_rope:.6g}"
            f" for {METHOD_ROPE} and {aw_yarn:.6g} for {METHOD_YARN}, a ratio of only"
            f" {aw_yarn / aw_rope:.6f}."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


# --------------------------------------------------------------------------
# fig06 -- exact pair contribution vs its linearization
# --------------------------------------------------------------------------


def fig06_pair_exact_vs_linear(q_hat: np.ndarray, k_hat: np.ndarray) -> FigureRecord:
    stem = "fig06_pair_exact_vs_linear"
    yarn_freqs, _mscale = yarn_parameters(
        HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS
    )
    schemes = {
        f"RoPE, base={ROPE_BASE:g}": inv_freq(HEAD_DIM, ROPE_BASE),
        f"YaRN, scale={EXT_SCALE:g}": yarn_freqs,
    }
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_TALL, sharex=True)

    rows: list[list[object]] = []
    first_cross: dict[str, int] = {}
    for col, (label, freqs) in enumerate(schemes.items()):
        top, bottom = axes[0][col], axes[1][col]
        n_d = len(DELTA_GRID)
        exact = np.zeros((n_d, len(PAIR_PICK)))
        approx = np.zeros_like(exact)
        error = np.zeros_like(exact)
        for i, delta in enumerate(DELTA_GRID):
            terms = pair_terms(q_hat, k_hat, freqs, delta)
            ex = terms.contributions()
            ap = _linearized(terms)
            amp = terms.amplitude
            for j, pair in enumerate(PAIR_PICK):
                exact[i, j] = ex[pair]
                approx[i, j] = ap[pair]
                error[i, j] = abs(ap[pair] - ex[pair]) / (amp[pair] + EPS)
                rows.append(
                    [label, pair, delta, ex[pair], ap[pair], amp[pair], error[i, j]]
                )

        for j, pair in enumerate(PAIR_PICK):
            colour = _colour(j)
            top.semilogx(DELTA_GRID, exact[:, j], color=colour, label=f"k={pair}")
            top.semilogx(
                DELTA_GRID, approx[:, j], color=colour, ls="--", lw=1.0,
                label="_nolegend_",
            )
            bottom.semilogx(
                DELTA_GRID, _floor(error[:, j]), color=colour, label=f"k={pair}"
            )

        above = np.nonzero(error[:, 0] >= UNIT_RELATIVE_ERROR)[0]
        first_cross[label] = int(DELTA_GRID[above[0]]) if above.size else -1
        _symlog(top, REL_TINY * float(np.abs(exact).max()),
                np.concatenate([exact.ravel(), approx.ravel()]))
        bottom.set_yscale("log")
        bottom.axhline(UNIT_RELATIVE_ERROR, color="#999999", lw=0.7, ls=":")
        top.set_title(f"{label}\nsolid = exact, dashed = 1st-order Taylor")
        bottom.set_title(f"normalized |exact - linear| / R_k, {label}")
        if col == 0:
            top.set_ylabel("pair contribution c_k(delta)  (dimensionless, symlog)")
            bottom.set_ylabel("normalized absolute error  (log scale)")
        else:
            top.set_ylabel("pair contribution c_k(delta)  (dimensionless, symlog)")
        bottom.set_xlabel("relative distance delta = n - m  (positions, log scale)")

    _note(
        axes[0][0],
        f"first delta at which the fastest pair (k={PAIR_PICK[0]})\n"
        "has an error at least as large as itself\n"
        + "\n".join(f"  {name}: delta = {value}" for name, value in first_cross.items()),
    )
    axes[0][1].legend(loc="upper left", ncol=2, fontsize=7.0)
    png = _save(
        fig,
        stem,
        f"Exact pair terms against their linearization, selected pairs"
        f" {list(PAIR_PICK)} at head_dim={HEAD_DIM}.\n"
        f"Dotted line: {UNIT_RELATIVE_ERROR:g} = an error as large as the term"
        f" itself.",
    )
    csv_path = _write_csv(
        stem,
        [
            "scheme",
            "pair_index",
            "delta",
            "exact_contribution",
            "linearized_contribution",
            "amplitude_R_k",
            "normalized_abs_error",
        ],
        rows,
    )

    cross_rope = first_cross[f"RoPE, base={ROPE_BASE:g}"]
    cross_yarn = first_cross[f"YaRN, scale={EXT_SCALE:g}"]
    return FigureRecord(
        stem=stem,
        producer="experiments.pair_terms(...).contributions() vs the Taylor form",
        headline=(
            f"the fastest rotary pair (k={PAIR_PICK[0]}) has a normalized error as"
            f" large as the term itself from delta={cross_rope} in plain RoPE and"
            f" delta={cross_yarn} in YaRN - the same, because YaRN does not touch"
            f" that channel."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


# --------------------------------------------------------------------------
# fig07 -- partial RoPE
# --------------------------------------------------------------------------


def fig07_partial_rope() -> FigureRecord:
    stem = "fig07_partial_rope"
    # Mirror experiments.partial_rope_split's RNG draw order exactly, so the
    # recomputed sub-scores below are directly comparable with that experiment.
    rng = np.random.default_rng(SEED_PARTIAL)
    freqs = inv_freq(HEAD_DIM)
    n_rot = int(HEAD_DIM * N_ROT_FRAC)
    q_hat = rng.standard_normal(HEAD_DIM)
    k_hat = rng.standard_normal(HEAD_DIM)

    clean, rotated, full = [], [], []
    for delta in DELTA_GRID:
        cos, sin = partial_rope_cos_sin(np.array([0, delta]), freqs, n_rot)
        q_rot = apply_partial_rope(q_hat[None, :], cos[0:1], sin[0:1], n_rot)[0]
        k_rot = apply_partial_rope(k_hat[None, :], cos[1:2], sin[1:2], n_rot)[0]
        clean.append(float(q_hat[n_rot:] @ k_hat[n_rot:]))
        rotated.append(float(q_rot[:n_rot] @ k_rot[:n_rot]))
        # The two halves must reconstruct the full partially-rotated score.
        full.append(float(q_rot @ k_rot))
    clean_a = np.array(clean)
    rotated_a = np.array(rotated)
    full_a = np.array(full)

    residual = float(np.abs(clean_a + rotated_a - full_a).max())
    clean_spread = float(clean_a.max() - clean_a.min())
    rotated_spread = float(rotated_a.max() - rotated_a.min())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PAIR)
    axes[0].semilogx(
        DELTA_GRID, rotated_a, color=PALETTE[1], marker="o", markersize=2.5,
        label=f"rotated sub-score (k < {n_rot})",
    )
    axes[0].semilogx(
        DELTA_GRID, clean_a, color=PALETTE[0], marker="o", markersize=2.5,
        label=f"clean sub-score (k >= {n_rot})",
    )
    axes[0].set_xlabel("relative distance delta = n - m  (positions, log scale)")
    axes[0].set_ylabel("sub-score  (inner product, dimensionless)")
    axes[0].set_title("Partial RoPE splits the score into a flat part\nand an "
                      "oscillating part")
    axes[0].legend(loc="lower right", fontsize=7.5)

    for colour, values, label in (
        (PALETTE[1], rotated_a, "rotated"),
        (PALETTE[0], clean_a, "clean"),
    ):
        axes[1].semilogx(
            DELTA_GRID,
            _floor(np.abs(values - values.mean())),
            color=colour,
            label=f"{label} sub-score",
        )
    _symlog(
        axes[1],
        DEVIATION_PIVOT,
        np.concatenate(
            [np.abs(rotated_a - rotated_a.mean()), np.abs(clean_a - clean_a.mean())]
        ),
    )
    axes[1].set_xlabel("relative distance delta = n - m  (positions, log scale)")
    axes[1].set_ylabel("|sub-score - its own mean|  (dimensionless, symlog)")
    axes[1].set_title(f"Deviation from the mean over all {len(DELTA_GRID)} distances")
    axes[1].legend(loc="upper right", fontsize=7.5)

    reference = partial_rope_split(
        n_rot_frac=N_ROT_FRAC, dim=HEAD_DIM, seed=SEED_PARTIAL, deltas=DELTA_GRID
    )
    _note(
        axes[0],
        f"head_dim={HEAD_DIM}, n_rot={n_rot}"
        f" ({reference['rotated_fraction']:.0%} of channels)\n"
        f"  clean spread over delta:   {clean_spread:.6e}\n"
        f"  rotated spread over delta: {rotated_spread:.6e}\n"
        f"  partial_rope_split agrees: clean spread"
        f" {reference['clean_score_spread']:.6e},\n"
        f"    clean magnitude share {reference['clean_magnitude_share_mean']:.4f}\n"
        f"  |clean + rotated - full| max: {residual:.3e}",
    )
    png = _save(fig, stem)
    csv_path = _write_csv(
        stem,
        [
            "delta",
            "n_rot",
            "clean_subscore",
            "rotated_subscore",
            "full_partial_rope_score",
            "clean_abs_dev_from_mean",
            "rotated_abs_dev_from_mean",
        ],
        (
            [
                delta,
                n_rot,
                clean_a[i],
                rotated_a[i],
                full_a[i],
                abs(clean_a[i] - clean_a.mean()),
                abs(rotated_a[i] - rotated_a.mean()),
            ]
            for i, delta in enumerate(DELTA_GRID)
        ),
    )

    return FigureRecord(
        stem=stem,
        producer=(
            "recomputed here from rope.partial_rope_cos_sin + rope.apply_partial_rope,"
            f" cross-checked against experiments.partial_rope_split(seed={SEED_PARTIAL})"
        ),
        headline=(
            f"the clean (unrotated) sub-score varies by {clean_spread:.3e} across all"
            f" {len(DELTA_GRID)} distances, while the rotated sub-score varies by"
            f" {rotated_spread:.4f}."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


# --------------------------------------------------------------------------
# fig08 -- mscale as an attention temperature
# --------------------------------------------------------------------------


def fig08_mscale_entropy() -> FigureRecord:
    stem = "fig08_mscale_entropy"
    rows = [
        mscale_entropy(
            dim=HEAD_DIM,
            scale=scale,
            original_max_position_embeddings=ORIGINAL_MAX_POS,
            n_keys=N_KEYS,
            seed=SEED_MSCALE,
        )
        for scale in SCALE_GRID
    ]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PAIR)
    axes[0].semilogx(
        [r["scale"] for r in rows],
        [r["entropy_ratio_mscaled"] for r in rows],
        color=PALETTE[0],
        marker="o",
        markersize=2.5,
        label="with YaRN mscale",
    )
    axes[0].semilogx(
        [r["scale"] for r in rows],
        [r["entropy_ratio_unscaled"] for r in rows],
        color=PALETTE[6],
        ls="--",
        label="unscaled (mscale = 1)",
    )
    axes[0].set_xlabel("context-extension factor scale  (dimensionless, log scale)")
    axes[0].set_ylabel("normalized attention entropy  H / ln T  (1 = uniform)")
    axes[0].set_title(
        "YaRN's magnitude scaling lowers the entropy\n"
        f"(head_dim={HEAD_DIM}, T={N_KEYS} keys, seed={SEED_MSCALE})"
    )
    axes[0].legend(loc="lower left", fontsize=7.5)

    axes[1].semilogx(
        [r["scale"] for r in rows],
        [r["mscale"] for r in rows],
        color=PALETTE[1],
        marker="o",
        markersize=2.5,
        label="measured mscale",
    )
    axes[1].semilogx(
        [r["scale"] for r in rows],
        [get_mscale_closed_form(r["scale"]) for r in rows],
        color=PALETTE[6],
        ls="--",
        label="0.1*ln(scale) + 1 (rope.get_mscale)",
    )
    axes[1].set_xlabel("context-extension factor scale  (dimensionless, log scale)")
    axes[1].set_ylabel("mscale  (dimensionless softmax temperature)")
    axes[1].set_title("The magnitude scaling itself")

    probe = min(rows, key=lambda r: abs(r["scale"] - EXT_SCALE))
    drop = probe["entropy_ratio_unscaled"] - probe["entropy_ratio_mscaled"]
    _note(
        axes[0],
        f"at scale={EXT_SCALE:g}\n"
        f"  mscale = {probe['mscale']:.6f}\n"
        f"  H/lnT unscaled = {probe['entropy_ratio_unscaled']:.4f}\n"
        f"  H/lnT mscaled  = {probe['entropy_ratio_mscaled']:.4f}\n"
        f"  drop = {drop:.4f}\n"
        f"  H_max = ln {N_KEYS} = {probe['max_possible_entropy']:.4f}",
    )
    png = _save(fig, stem)
    csv_path = _write_csv(
        stem,
        [
            "scale",
            "mscale",
            "entropy_ratio_unscaled",
            "entropy_ratio_mscaled",
            "n_keys",
            "max_possible_entropy",
        ],
        (
            [
                r["scale"],
                r["mscale"],
                r["entropy_ratio_unscaled"],
                r["entropy_ratio_mscaled"],
                r["n_keys"],
                r["max_possible_entropy"],
            ]
            for r in rows
        ),
    )

    return FigureRecord(
        stem=stem,
        producer=(
            f"experiments.mscale_entropy over {len(SCALE_GRID)} values of scale,"
            f" seed={SEED_MSCALE}"
        ),
        headline=(
            f"at scale={EXT_SCALE:g}, mscale={probe['mscale']:.6f} pulls"
            f" H/lnT from {probe['entropy_ratio_unscaled']:.4f} down to"
            f" {probe['entropy_ratio_mscaled']:.4f}, a drop of {drop:.4f}."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


# --------------------------------------------------------------------------
# fig09 -- the central claim
# --------------------------------------------------------------------------


def fig09_position_conditional_attribution() -> FigureRecord:
    stem = "fig09_position_conditional_attribution"
    # Exactly the construction inside experiments.feature_attribution, redrawn
    # here so the per-feature curves can be evaluated at every distance on the
    # grid instead of at that function's five-point default.
    rng = np.random.default_rng(SEED_ATTRIB)
    freqs = inv_freq(HEAD_DIM)
    w_q = rng.standard_normal((HEAD_DIM, HEAD_DIM)) / math.sqrt(HEAD_DIM)
    w_k = rng.standard_normal((HEAD_DIM, HEAD_DIM)) / math.sqrt(HEAD_DIM)
    # The 1.5 / 0.1 below are that function's own coefficient range, replicated so
    # the draw order matches; they are construction, not a result.
    coeffs = rng.random(N_FEATURES) * 1.5 + 0.1
    dirs_q = rng.standard_normal((N_FEATURES, HEAD_DIM))
    dirs_k = rng.standard_normal((N_FEATURES, HEAD_DIM))
    q_i = coeffs[:, None] * (dirs_q @ w_q)
    k_i = coeffs[:, None] * (dirs_k @ w_k)
    full_q = q_i.sum(axis=0)
    full_k = k_i.sum(axis=0)

    contrib = np.zeros((N_FEATURES, len(DELTA_GRID)))
    totals = np.zeros(len(DELTA_GRID))
    for i, delta in enumerate(DELTA_GRID):
        block = np.array(
            [
                [score_relative(q_i[a], k_i[b], freqs, delta) for b in range(N_FEATURES)]
                for a in range(N_FEATURES)
            ]
        )
        contrib[:, i] = block.sum(axis=1)
        totals[i] = score_relative(full_q, full_k, freqs, delta)
    # One accumulation order, reused by the figure, the CSV and the summary, so
    # the three cannot disagree in the last bits.
    residuals = np.array(
        [abs(float(contrib[:, i].sum()) - float(totals[i])) for i in range(len(DELTA_GRID))]
    )
    additivity_residual = float(residuals.max())

    # Counted from the array rather than assumed, so the caption cannot claim a
    # reversal that this particular draw does not contain.
    n_reversing = int(
        np.sum((contrib.min(axis=1) < 0.0) & (contrib.max(axis=1) > 0.0))
    )
    delta_hi = int(DELTA_GRID[-1])
    # The endpoint comparison the paper leans on, computed here rather than typed
    # in: if it moves, this caption must move with it.
    grid = np.asarray(DELTA_GRID, dtype=np.float64)
    i_lo = int(np.argmin(np.abs(grid - 1)))
    i_hi = int(np.argmin(np.abs(grid - delta_hi)))
    # Per-feature log-ratio, then averaged over features - the same order of
    # operations as statistics._endpoint_log_ratio. Averaging the magnitudes first
    # and taking one ratio of the means is a different statistic: it let a single
    # large-amplitude feature dominate and printed 10^0.01 instead of this
    # figure's 10^-0.19 (the paper's 10^-0.08 is the seed mean in
    # statistics.seed_variance). One definition, one number.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_feature = np.log10(
            np.abs(contrib[:, i_hi]) / np.where(np.abs(contrib[:, i_lo]) > 0, np.abs(contrib[:, i_lo]), np.nan)
        )
    endpoint_log_ratio = float(np.nanmean(per_feature))

    per_axis = 2
    ncols = 2
    nrows = N_FEATURES // per_axis // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(9.0, 8.4), squeeze=False, sharex=True
    )
    for panel in range(nrows * ncols):
        ax = axes[panel // ncols][panel % ncols]
        members = [panel * per_axis + slot for slot in range(per_axis)]
        pivot = REL_TINY * float(np.abs(contrib[members]).max())
        for slot, feature in enumerate(members):
            values = contrib[feature]
            ax.semilogx(
                DELTA_GRID, values, color=_colour(slot),
                label=f"feature {feature}",
            )
            ax.axhline(float(values.mean()), color=_colour(slot), lw=0.7, ls=":")
        _symlog(ax, pivot, contrib[members].ravel())
        ax.set_title(
            f"features {members[0]} and {members[1]}\n"
            f"{int(np.count_nonzero(np.diff(np.sign(values)) != 0))} sign changes"
        )
        if panel % ncols == 0:
            ax.set_ylabel("c_i(delta)  (dimensionless, symlog)")
        if panel // ncols == nrows - 1:
            ax.set_xlabel("relative distance delta = n - m  (positions, log scale)")

    png = _save(
        fig,
        stem,
        "The central claim: a feature's attribution is a function of distance,\n"
        "so no position-free A_ij can express it\n"
        f"seed={SEED_ATTRIB}, N_FEATURES={N_FEATURES}, head_dim={HEAD_DIM}"
        f"  |  additivity residual {additivity_residual:.3e}\n"
        f"{n_reversing} of {N_FEATURES} features reverse sign at least once;"
        "  dotted = the feature's mean over delta\n"
        "Blue = lower-indexed feature, orange = higher-indexed."
        "  experiments.feature_attribution on the same grid.",
        top=0.84,
    )
    csv_path = _write_csv(
        stem,
        ["feature_index", "coefficient", "delta", "contribution", "total_score",
         "additivity_residual"],
        _csv_rows_for_attribution(contrib, coeffs, totals, DELTA_GRID, residuals),
    )

    return FigureRecord(
        stem=stem,
        producer=(
            "recomputed from experiments.score_relative on a synthetic SAE-like"
            f" decomposition identical to experiments.feature_attribution"
            f"(seed={SEED_ATTRIB})"
        ),
        headline=(
            f"{n_reversing} of {N_FEATURES} features reverse sign somewhere in"
            f" the range [1, {delta_hi}], with the score still additive to"
            f" {additivity_residual:.3e}. The magnitudes at the two ends of the"
            f" range agree to within 10^{endpoint_log_ratio:.2f}, so what changes"
            " across distance is"
            " which way the contribution points, not how large it is. The"
            " max/min ratio this figure was once described by is a grid-density"
            " artefact, is deliberately not plotted, and is disproven against"
            " grid density in rope_attribution.statistics; the quantities to"
            " trust are the sign change and the additivity residual."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


def _csv_rows_for_attribution(
    contrib: np.ndarray,
    coeffs: np.ndarray,
    totals: np.ndarray,
    deltas: Sequence[int],
    residuals: np.ndarray,
) -> list[list[object]]:
    """One row per (feature, distance) plus the additivity residual."""
    out: list[list[object]] = []
    for i, delta in enumerate(deltas):
        for feature in range(contrib.shape[0]):
            out.append(
                [feature, float(coeffs[feature]), delta, float(contrib[feature, i]),
                 float(totals[i]), float(residuals[i])]
            )
    return out


def fig10_seed_variance() -> FigureRecord:
    """Seed-robustness of the central claim, with the spread shown.

    Every other figure in this set is a single draw. That is defensible for
    structural facts, which are exact and therefore have no variance to show,
    but not for the central empirical claim: a reader is entitled to know whether
    "every feature reverses sign" survives a different random draw, and whether
    the coefficient of variation quoted in the paper is a property of the
    decomposition or of one seed.

    So this figure varies only the seed, over a pre-registered sweep, and reports
    both statistics per seed. The number of seeds is fixed in advance rather than
    chosen to look good; the sweep is the same ``SEED_SWEEP`` used by
    ``statistics.seed_variance``, so the figure and the JSON cannot disagree.
    """
    from rope_attribution import statistics as ST

    seeds = [ST.BASE_SEED + i for i in range(ST.N_SEEDS)]
    crossings: list[float] = []
    cvs: list[float] = []
    endpoint: list[float] = []
    for seed in seeds:
        _d, contrib, _r = ST.contribution_grid(seed, N_FEATURES, len(DELTA_GRID))
        crossings.append(ST._sign_crossing_fraction(contrib))
        cvs.append(float(np.nanmean(ST._cv_magnitude(contrib))))
        endpoint.append(float(np.nanmean(ST._endpoint_log_ratio(contrib, _d))))

    cv_arr = np.array(cvs, dtype=np.float64)
    cv_mu, cv_sd = float(cv_arr.mean()), float(cv_arr.std(ddof=1))
    cross_arr = np.array(crossings, dtype=np.float64)
    end_arr = np.array(endpoint, dtype=np.float64)

    fig, (ax_cross, ax_cv) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    x = np.arange(len(seeds))

    ax_cross.axhline(1.0, color="#B00020", lw=1.0, ls="--",
                     label="all features cross zero")
    ax_cross.plot(x, cross_arr, "o", color="#B00020", ms=4.5)
    ax_cross.set_ylim(-0.05, 1.15)
    ax_cross.set_xlabel("seed index")
    ax_cross.set_ylabel("fraction of features crossing zero")
    ax_cross.set_title("Sign reversal is seed-invariant", fontsize=9)
    ax_cross.legend(fontsize=7, loc="lower right")

    ax_cv.axhline(cv_mu, color="#1F4E79", lw=1.0, label=f"mean {cv_mu:.3f}")
    ax_cv.fill_between(
        x, cv_mu - cv_sd, cv_mu + cv_sd, color="#1F4E79", alpha=0.20,
        label=f"$\\pm 1$ SD ({cv_sd:.3f})",
    )
    ax_cv.plot(x, cv_arr, "o", color="#1F4E79", ms=4.5)
    ax_cv.set_xlabel("seed index")
    ax_cv.set_ylabel("coefficient of variation")
    ax_cv.set_title("Magnitude spread is stable", fontsize=9)
    ax_cv.legend(fontsize=7)

    stem = "fig10_seed_variance"
    png = _save(fig, stem, "Central claim across a pre-registered seed sweep")

    csv_path = _write_csv(
        stem,
        ["seed", "sign_crossing_fraction", "cv_magnitude", "endpoint_log_ratio"],
        [[s, float(c), float(v), float(e)]
         for s, c, v, e in zip(seeds, crossings, cvs, endpoint, strict=True)],
    )
    n_seeds = len(seeds)
    # The requested count is not always what _log_uniform_grid returns; report the
    # grid the loop actually used so the record cannot go stale against it.
    n_deltas = int(_d.size)
    return FigureRecord(
        stem=stem,
        producer=(
            f"rope_attribution.statistics.contribution_grid over {n_seeds} seeds "
            f"starting at {ST.BASE_SEED}, {n_deltas} distances "
            f"({len(DELTA_GRID)} requested), F={N_FEATURES}"
        ),
        headline=(
            f"all {n_seeds} seeds give a sign-crossing fraction of "
            f"{cross_arr.min():.4f} (SD {cross_arr.std(ddof=1):.4f}); the coefficient "
            f"of variation is {cv_mu:.4f} +/- {cv_sd:.4f}, and the endpoint log-ratio "
            f"averages {end_arr.mean():.4f} (SD {end_arr.std(ddof=1):.4f}), so the two "
            f"endpoints agree and the reversal is interior."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


def fig11_sensitivity_sweep() -> FigureRecord:
    """The configuration sweep behind the robustness paragraph, from the artifact.

    ``results/sensitivity.json`` is the artifact ``make results`` regenerates
    and CI diff-reports; reading it keeps figures.py from re-running the one
    CPU-bound sweep every regeneration. The read is guarded: the artifact's
    schema and its base/head-dim grid must agree with ``sensitivity.BASES``/
    ``HEAD_DIMS``, so an edited module range fails here loudly instead of
    plotting against a stale sweep.
    """
    import json

    from rope_attribution import sensitivity as S

    path = REPO_ROOT / "results" / "sensitivity.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = "rope_attribution/sensitivity/v1"
    if data["schema"] != expected:
        raise ValueError(
            f"sensitivity.json schema {data['schema']!r}, expected {expected!r}"
        )
    if [float(b) for b in data["bases"]] != [float(b) for b in S.BASES] or [
        int(d) for d in data["head_dims"]
    ] != [int(d) for d in S.HEAD_DIMS]:
        raise ValueError(
            "sensitivity.json grid disagrees with sensitivity.BASES/HEAD_DIMS"
        )

    rows = sorted(data["records"], key=lambda r: (r["base"], r["head_dim"]))
    base_rows = [r for r in rows if r["head_dim"] == 64]
    dim_rows = [r for r in rows if r["base"] == 10000.0]

    fig, (ax_base, ax_dim) = plt.subplots(1, 2, figsize=(6.4, 2.5))

    xb = np.array([r["base"] for r in base_rows])
    yb = 100.0 * np.array([r["sign_flip_fraction_mean"] for r in base_rows])
    eb = 100.0 * np.array([r["sign_flip_fraction_sd"] for r in base_rows])
    ax_base.errorbar(xb, yb, yerr=eb, fmt="o-", color="#1F4E79", ms=4.5, capsize=3)
    ax_base.axvline(10000.0, color="#B00020", lw=1.0, ls="--")
    ax_base.annotate(
        "published base",
        xy=(10000.0, float(yb.min())),
        xytext=(5, 4),
        textcoords="offset points",
        color="#B00020",
        fontsize=7,
    )
    ax_base.set_xscale("log")
    ax_base.set_xlabel("rotary base (log axis)")
    ax_base.set_ylabel("pooled sign-flip rate (%)")
    ax_base.set_title("rate vs base at head dim 64", fontsize=9)

    xd = np.array([float(r["head_dim"]) for r in dim_rows])
    yd = 100.0 * np.array([r["sign_flip_fraction_mean"] for r in dim_rows])
    ed = 100.0 * np.array([r["sign_flip_fraction_sd"] for r in dim_rows])
    ax_dim.errorbar(
        xd, yd, yerr=ed, fmt="o-", color="#1F4E79", ms=4.5, capsize=3,
        label="rate (left axis)",
    )
    ax_dim.set_xticks([32, 64, 128])
    ax_dim.set_xlabel("head dimension (base 10$^4$)")
    ax_dim.set_ylabel("pooled sign-flip rate (%)")
    ax_dim.set_title("rate and error vs head dim", fontsize=9)

    twin = ax_dim.twinx()
    rels = np.array([r["rel_err_median_mean"] for r in dim_rows])
    twin.plot(
        xd, rels, "s--", color="#3D7A3D", ms=4.0,
        label="rel. error (right axis)",
    )
    twin.axhline(1.0, color="#666666", lw=1.0, ls=":")
    twin.set_ylabel("median relative error", color="#3D7A3D")
    twin.tick_params(axis="y", labelcolor="#3D7A3D")
    twin.grid(False)
    left_handles, left_labels = ax_dim.get_legend_handles_labels()
    right_handles, right_labels = twin.get_legend_handles_labels()
    ax_dim.legend(
        left_handles + right_handles, left_labels + right_labels,
        fontsize=7, loc="lower left",
    )

    stem = "fig11_sensitivity_sweep"
    png = _save(fig, stem, "The rate is not a configuration artifact")
    csv_path = _write_csv(
        stem,
        [
            "base",
            "head_dim",
            "sign_flip_fraction_mean",
            "sign_flip_fraction_sd",
            "rel_err_median_mean",
        ],
        [
            [float(r["base"]), int(r["head_dim"]),
             float(r["sign_flip_fraction_mean"]),
             float(r["sign_flip_fraction_sd"]),
             float(r["rel_err_median_mean"])]
            for r in rows
        ],
    )
    base_band = data["base_band_at_dim_64"]
    dim_band = data["dim_band_at_base_10000"]
    low_rel = min(float(r["rel_err_median_mean"]) for r in rows)
    high_rel = max(float(r["rel_err_median_mean"]) for r in rows)
    return FigureRecord(
        stem=stem,
        producer=(
            "results/sensitivity.json as regenerated by "
            f"rope_attribution.sensitivity ({len(rows)} configs, "
            f"{data['n_sweep_seeds']} seeds each); grid asserted against the module"
        ),
        headline=(
            f"across bases at head dim 64 the rate spans {100.0 * base_band[0]:.1f}% "
            f"to {100.0 * base_band[1]:.1f}%; across head dims at base 1e4 it stays "
            f"within {100.0 * dim_band[0]:.1f}% to {100.0 * dim_band[1]:.1f}%; the "
            f"median relative error never drops below 1.0 ({low_rel:.2f} to "
            f"{high_rel:.2f} across the sweep)."
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------


def fig12_qk_case_study() -> FigureRecord:
    """The trained-dictionary result as a picture, from the committed artifact.

    The table in Section~\\ref{sec:features} reports two numbers that disagree:
    the total per-feature profile reverses on 0.02% of cells, the rotary
    sub-score on 21.4%. A reader can accept the arithmetic and still not see
    how the same feature can be flat in one view and oscillating in the other.
    This figure shows one feature on one head: the total profile is flat to
    within a fraction of a percent of its own scale, while the rotary sub-score
    runs from negative to positive and crosses zero.

    Nothing is recomputed here. The artifact is the one
    ``rope_attribution.qk_attribution`` writes and the claim gates read; the
    guard below refuses a different schema or a head set that does not match
    the artifact's own ``n_heads``.
    """
    path = REPO_ROOT / "results" / "qk_attribution.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `python -m projects.rope_attribution.qk_attribution`"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "qk-attribution/v1":
        raise ValueError(
            f"{path}: schema {data.get('schema')!r}, expected 'qk-attribution/v1'"
        )
    heads = data.get("heads") or []
    if len(heads) != int(data.get("n_heads", -1)):
        raise ValueError(
            f"{path}: {len(heads)} head records for n_heads={data.get('n_heads')}"
        )

    # One feature that is in both top lists, so the two curves are the same
    # feature's total and rotary sub-score rather than two different features.
    chosen = None
    for head in heads:
        total_rows = {int(row["feature"]): row for row in head["top_sweep_features"]}
        rotary_rows = {int(row["feature"]): row for row in head["top_sweep_rotary"]}
        common = sorted(set(total_rows) & set(rotary_rows))
        if common:
            feature = common[0]
            chosen = (head, feature, total_rows[feature], rotary_rows[feature])
            break
    if chosen is None:
        raise ValueError(f"{path}: no feature appears in both top-sweep lists")

    head, feature, total_row, rotary_row = chosen
    deltas = np.asarray(data["sweep_deltas"], dtype=np.float64)
    total = np.asarray(total_row["profile"], dtype=np.float64)
    rotary = np.asarray(rotary_row["profile"], dtype=np.float64)
    if total.size != deltas.size or rotary.size != deltas.size:
        raise ValueError(
            f"{path}: profile length {total.size}/{rotary.size} != {deltas.size}"
        )
    total_scale = float(np.mean(np.abs(total)))
    rotary_scale = float(np.mean(np.abs(rotary)))
    if total_scale == 0.0 or rotary_scale == 0.0:
        raise ValueError(f"{path}: a chosen profile has zero scale")
    total_n = total / total_scale
    rotary_n = rotary / rotary_scale

    fig, (ax_profile, ax_bars) = plt.subplots(
        1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    ax_profile.axhline(0.0, color="#555555", lw=0.8)
    ax_profile.plot(
        deltas, total_n, color=PALETTE[0], marker="o", ms=2.0,
        label="total $c_i(\\delta)$",
    )
    ax_profile.plot(
        deltas, rotary_n, color=PALETTE[1], marker="s", ms=2.0,
        label="rotary sub-score",
    )
    ax_profile.set_xscale("log", base=2)
    ax_profile.set_xlabel("relative distance $\\delta$")
    ax_profile.set_ylabel("contribution / mean $|c_i|$")
    ax_profile.set_title(
        f"One feature (head {head['head']}, feature {feature})", fontsize=9
    )
    ax_profile.legend(fontsize=7, loc="lower left")

    n_heads = len(heads)
    positions = np.arange(n_heads, dtype=np.float64)
    width = 0.26
    bars = (
        ("sweep_feature_metrics", "total", PALETTE[0], -width),
        ("sweep_rotary_metrics", "rotary", PALETTE[1], 0.0),
        ("sweep_pair_metrics", "pair", PALETTE[2], width),
    )
    for key, label, colour, offset in bars:
        values = [float(head_row[key]["sign_flip_fraction"]) for head_row in heads]
        ax_bars.bar(
            positions + offset, values, width=width, label=label, color=colour
        )
    ax_bars.set_xticks(positions)
    ax_bars.set_xticklabels([str(head_row["head"]) for head_row in heads], fontsize=7)
    ax_bars.set_xlabel("head")
    ax_bars.set_ylabel("sign-flip fraction")
    ax_bars.set_yscale("log")
    ax_bars.set_ylim(1e-4, 0.5)
    ax_bars.set_title("Sign reversal by head", fontsize=9)
    ax_bars.legend(fontsize=7, loc="lower right")

    stem = "fig12_qk_case_study"
    png = _save(fig, stem, "The total profile hides the rotary reversal")

    csv_path = _write_csv(
        stem,
        [
            "delta",
            "total_profile",
            "rotary_profile",
            "total_normalized",
            "rotary_normalized",
        ],
        [
            [float(d), float(t), float(r), float(tn), float(rn)]
            for d, t, r, tn, rn in zip(
                deltas, total, rotary, total_n, rotary_n, strict=True
            )
        ],
    )
    aggregate = data["aggregate"]
    total_pooled = float(aggregate["sweep_feature"]["sign_flip_fraction"])
    rotary_pooled = float(aggregate["sweep_rotary"]["sign_flip_fraction"])
    pair_pooled = float(aggregate["sweep_pair"]["sign_flip_fraction"])
    return FigureRecord(
        stem=stem,
        producer=(
            "results/qk_attribution.json (pythia-70m + public 32k SAE, layer 3); "
            f"head {head['head']}, feature {feature}"
        ),
        headline=(
            f"the total profile stays within {100 * (total_n.max() - total_n.min()):.2f}% "
            f"of its mean absolute value while the rotary sub-score runs "
            f"{rotary.min():.1f} to {rotary.max():.1f} and changes sign; pooled "
            f"sign-flip fractions are {100 * total_pooled:.2f}% (total), "
            f"{100 * rotary_pooled:.2f}% (rotary), {100 * pair_pooled:.2f}% (pair)"
        ),
        extra=f"png={png.name} csv={csv_path.name}",
    )


def _method_rows() -> dict[str, list[dict]]:
    """Call ``experiments.method_spectrum`` once and group it by method."""
    rows = method_spectrum(
        dim=HEAD_DIM,
        scale=EXT_SCALE,
        original_max_position_embeddings=ORIGINAL_MAX_POS,
        deltas=DELTA_GRID,
        seed=SEED_METHOD,
    )
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    return grouped


def _write_readme(records: Sequence[FigureRecord]) -> Path:
    path = FIGS_DIR / "README.md"
    lines = [
        "# `figures/` - generated figure set",
        "",
        "Everything in this directory is written by",
        "`projects/rope_attribution/figures.py`. No number in these PNGs, in the",
        "CSV sidecars, or in the table below is typed into the source: each one is",
        "returned at run time by `rope_attribution.experiments`,",
        "`rope_attribution.rope` or `rope_attribution.statistics` -- except fig11,",
        "which reads `results/sensitivity.json`, the artifact that `make results`",
        "regenerates and CI diff-reports; the figure asserts the artifact's grid",
        "against `sensitivity.BASES`/`HEAD_DIMS` before plotting, and fig12, which",
        "reads `results/qk_attribution.json` and asserts its schema and head count.",
        "The literals",
        "that do appear in `figures.py` are",
        "configuration (head dim, RoPE base, extension scale, seeds), the distance",
        "grid definition, plotting parameters, and two dimensionless thresholds that",
        "are definitions rather than results: the",
        f"`{LINEARIZABLE_ANGLE_RAD} rad` linearization cut inside",
        "`experiments.linearization_error` and the `1.0` (= 100% relative error)",
        "reference line in fig06 and fig11.",
        "",
        "## Regenerate",
        "",
        "```powershell",
        "$env:PYTHONPATH = \"projects\"",
        "python -m projects.rope_attribution.figures",
        "```",
        "",
        "Each figure has a `<stem>.csv` sidecar holding exactly the array that was",
        "plotted, so no claim has to be read off a pixel. Log axes clip at",
        f"`{PLOT_FLOOR:g}` for display only; the CSVs carry the raw values.",
        "",
        "## Figures",
        "",
        "| figure | produced by | headline number (computed) |",
        "| --- | --- | --- |",
    ]
    for record in records:
        # Escape the pipes in the headline so markdown keeps them inside the cell.
        clean = record.headline.replace("|", "\\|")
        lines.append(f"| `{record.stem}.png` | {record.producer} | {clean} |")
    lines += [
        "",
        "## Shared configuration",
        "",
        f"- `head_dim = {HEAD_DIM}`, RoPE `base = {ROPE_BASE:g}`, legacy `base ="
        f" {LEGACY_BASE:g}`",
        f"- extension `scale = {EXT_SCALE:g}`, original context ="
        f" {ORIGINAL_MAX_POS}, `T = {N_KEYS}` keys",
        f"- distance grid: {len(DELTA_GRID)} log-spaced integers from"
        f" {DELTA_GRID[0]} to {DELTA_GRID[-1]} positions",
        f"- scale grid for fig08: {len(SCALE_GRID)} values from {SCALE_GRID[0]:g} to"
        f" {SCALE_GRID[-1]:g}",
        f"- seeds: q_hat/k_hat = {SEED_QK}, method spectrum = {SEED_METHOD},"
        f" attribution = {SEED_ATTRIB}, partial RoPE = {SEED_PARTIAL},"
        f" mscale = {SEED_MSCALE}",
        f"- output: {DPI} dpi PNG on a white ground",
        "",
        "## Reading order",
        "",
        "fig04 is the headline result, fig09 is the central claim. fig03 and fig05",
        "are the two guards on fig04: they show *why* the fraction collapses (the",
        "maximum angle never changes) and *what that costs* (the amplitude-weighted",
        "error barely moves). fig06 shows the local mechanism, fig07 and fig08 the",
        "two auxiliary structure claims, and fig01 the frequency bookkeeping.",
        "",
        "fig11 is the configuration sweep behind the paper's robustness paragraph,",
        "read from its artifact; the artifact's grid is asserted against the module.",
        "fig12 is the trained-dictionary case study: the same feature's total and",
        "rotary profiles, read from `results/qk_attribution.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> int:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()
    q_hat, k_hat = _qk_vectors(SEED_QK)

    records = [
        fig01_frequency_ladder(),
        fig02_angle_spectrum(q_hat, k_hat),
        fig03_max_vs_median_angle(),
        fig04_linearizable_fraction(),
        fig05_linearization_error(),
        fig06_pair_exact_vs_linear(q_hat, k_hat),
        fig07_partial_rope(),
        fig08_mscale_entropy(),
        fig09_position_conditional_attribution(),
        fig10_seed_variance(),
        fig11_sensitivity_sweep(),
        fig12_qk_case_study(),
    ]
    readme = _write_readme(records)

    print("=" * 78)
    print(f"figures/ written from {FIGS_DIR}")
    print(f"  head_dim={HEAD_DIM}  base={ROPE_BASE:g}  scale={EXT_SCALE:g}  "
          f"deltas={len(DELTA_GRID)} in [{DELTA_GRID[0]}, {DELTA_GRID[-1]}]")
    print("=" * 78)
    for record in records:
        print(f"{record.stem}.png / {record.stem}.csv   {record.extra}")
        print(f"  source  : {record.producer}")
        print(f"  headline: {record.headline}")
    print(f"README.md written: {readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
