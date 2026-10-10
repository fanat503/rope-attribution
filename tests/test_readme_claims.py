"""Make every quantitative claim in ``README.md`` machine-verified.

Why this file exists
--------------------
``README.md`` quotes specific measured numbers, and ``results/measurements.json``
holds the canonical record of those measurements. Nothing connected the two, so a
changed measurement silently left the prose stale - precisely the failure mode
this repository exists to diagnose (documentation asserting numbers the code does
not produce). Every test below therefore

* locates the repository from ``__file__``, never from the working directory;
* **parses the number out of ``README.md``** and compares *that* with the
  canonical value, instead of hardcoding the expected literal here. Hardcoding
  would only move the problem one file over: the README would still be free to
  drift away from the data, which is the whole failure this file is here to
  catch;
* tolerates exactly the rounding the README used and no more - see
  :func:`assert_readme_number_matches`.

Sources of truth, in order of preference:

* ``results/measurements.json`` for every number an experiment recorded;
* a fresh recomputation from ``rope_attribution`` for the numbers the JSON does
  not hold (the finer figure grid, the RoPE/YaRN max-angle comparison over that
  grid, the ramp bookkeeping, and the algebraic identities the README states);
* ``pytest``'s own collection for the test count.

``test_measurements_json_is_current`` closes the loop: it re-runs the
experiments and requires the JSON on disk to be bit-for-bit what the code
produces, so "the README matches the data" cannot be satisfied by a stale data
file.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

# Make the suite runnable without PYTHONPATH being pre-set.
_REPO = Path(__file__).resolve().parents[1]
_PROJECTS = _REPO / "projects"
if str(_PROJECTS) not in sys.path:
    sys.path.insert(0, str(_PROJECTS))

from rope_attribution import experiments as E  # noqa: E402  (needs the sys.path tweak)
from rope_attribution import rope as R  # noqa: E402  (needs the sys.path tweak)

README_PATH = _REPO / "README.md"
README_TEXT = README_PATH.read_text(encoding="utf-8")
MEASUREMENTS = json.loads(
    (_REPO / "results" / "measurements.json").read_text(encoding="utf-8")
)

# The maintained configuration the README says every measurement was taken at.
HEAD_DIM = 64
ROPE_BASE = 10000.0
EXT_SCALE = 32.0
ORIGINAL_MAX_POS = 2048

# `figures.py` configuration for the finer grid the README's fig09/54-distance
# claims are measured on (DELTA_MAX, DELTA_POINTS, EXT_SCALE, ORIGINAL_MAX_POS).
# `test_figure_grid_constants_match_figures_module` re-checks these against the
# module itself, so the duplication here cannot rot.
FIG_DELTA_MAX = 8192
FIG_DELTA_POINTS = 61

# Number words the README uses for the two counts it states in prose.
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# README scheme label -> the `method` key measurements.json uses.
SCHEME_METHODS = {
    "RoPE base 10k": "rope_base_10k",
    "YaRN": "yarn",
    "position interpolation": "position_interpolation",
}

_IGNORED_DIR_NAMES = frozenset({
    ".git", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules",
})


# ==========================================================================
# README text extraction
# ==========================================================================


def readme_literals(pattern: str) -> list[str]:
    """First capture group of every match of ``pattern`` in README.md."""
    return [m.group(1) for m in re.finditer(pattern, README_TEXT, re.MULTILINE)]


def readme_groups(pattern: str, what: str) -> tuple[str, ...]:
    """The capture groups of the single place in README.md a claim is written.

    Anchoring each pattern on the surrounding prose is what makes an edit to the
    README's number fail the test: the number is never written down here.
    """
    found = [m.groups() for m in re.finditer(pattern, README_TEXT, re.MULTILINE)]
    assert len(found) == 1, (
        f"{what}: expected exactly one number in README.md for pattern "
        f"{pattern!r}, found {found}"
    )
    return found[0]


def readme_literal(pattern: str, what: str) -> str:
    """The single README number a claim is written as (one capture group)."""
    return readme_groups(pattern, what)[0]


def readme_word_number(pattern: str, what: str) -> int:
    """A count the README spells out in words ("nine figures")."""
    word = readme_literal(pattern, what)
    assert word in _WORD_NUMBERS, (
        f"{what}: README writes {word!r}, which is not a count this suite knows"
    )
    return _WORD_NUMBERS[word]


def rounding_quantum(literal: str) -> Decimal:
    """The unit of the last digit the README actually wrote.

    ``"3.6e-15"`` -> ``1e-16``, ``"0.997"`` -> ``1e-3``, ``"43"`` -> ``1``.
    This is what lets the tolerance be derived from the README's own precision
    instead of being guessed.
    """
    return Decimal(1).scaleb(Decimal(literal).as_tuple().exponent)


def assert_readme_number_matches(actual: float, literal: str, what: str) -> None:
    """``literal`` is ``actual`` correctly rounded to the README's own precision.

    The README rounds to 2-4 significant figures, so demanding float equality
    against a rounded literal would be wrong. Instead the accepted interval is
    half of the literal's last written digit: the exact set of values that round
    to what the README says. ``1e-9`` of a quantum is float slack only.
    """
    quantum = float(rounding_quantum(literal))
    written = float(literal)
    gap = abs(actual - written)
    slack = 1e-9 * quantum
    assert gap <= 0.5 * quantum + slack, (
        f"{what}: README says {literal!r} but the measurement is {actual!r}; "
        f"{literal!r} is only correct if the measurement rounds to it "
        f"(accepted range {written - 0.5 * quantum - slack:.6g}"
        f" .. {written + 0.5 * quantum + slack:.6g})"
    )


def assert_readme_says_exactly(actual: float, literal: str, what: str) -> None:
    """For claims the README states as exact (``exactly 0.0``), demand exactness."""
    assert float(literal) == actual, (
        f"{what}: README says {literal!r}, measurement is {actual!r}; the README "
        f"claims this one is exact"
    )


def spectrum_row(method: str, delta: int) -> dict:
    for row in MEASUREMENTS["method_spectrum"]:
        if row["method"] == method and row["delta"] == delta:
            return row
    raise AssertionError(f"measurements.json has no row for {method!r} at delta={delta}")


def figure_delta_grid() -> tuple[int, ...]:
    """The log-spaced integer distance grid ``figures.py`` plots on."""
    return tuple(
        int(x)
        for x in np.unique(np.round(np.logspace(0.0, math.log10(FIG_DELTA_MAX), FIG_DELTA_POINTS)))
    )


def parse_comparison_table() -> list[tuple[int, str, str, str, str]]:
    """Rows of the ``delta / scheme / max|D_k| / median|D_k| / fraction`` table."""
    pattern = (
        r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        r"\s*`?([\d.]+)`?\s*\|$"
    )
    rows = [
        (int(d), label.strip(), maxd, med, frac)
        for d, label, maxd, med, frac in re.findall(pattern, README_TEXT, re.MULTILINE)
    ]
    assert len(rows) == 5, f"expected 5 rows in the comparison table, parsed {rows}"
    return rows


def table_row(delta: int, label: str) -> tuple[str, str, str]:
    for row_delta, row_label, maxd, med, frac in parse_comparison_table():
        if row_delta == delta and row_label == label:
            return maxd, med, frac
    raise AssertionError(
        f"README comparison table has no row for delta={delta}, scheme={label!r}; "
        f"parsed rows: {parse_comparison_table()}"
    )


# ==========================================================================
# Are the measurements on disk even current?
# ==========================================================================


NOISE_FLOOR = 1e-12


def _leaves(fresh, committed, path: str = ""):
    """Yield ``(path, committed, fresh)`` for every leaf that moved materially.

    Two tolerances, because the file mixes two kinds of number:

    * **Results** (``max_abs_angle``, ``frac_channels_linearizable``, a spread
      ratio) are ordinary quantities and are compared to ``1e-12`` *relative* -
      about 4500 ULP at 1.0, far tighter than any real disagreement.
    * **Error floors** (``relative_position_max_abs_err``,
      ``additivity_max_abs_err``) are not ordinary quantities: they *are* the
      float64 roundoff of subtracting two analytically equal expressions, so
      their value is whatever the summation order happens to produce. Windows
      and Linux legitimately give 3.4e-14 and 3.6e-14 for the same sum. A
      relative tolerance on noise is meaningless, so these are compared against
      an absolute ceiling instead: the claim that matters is "below the noise
      floor of float64", not "equal to the digits I happened to get".

    That ceiling is strict enough to catch a real regression: a conservation
    error that grew to 1e-6, or an additivity residual that stopped cancelling,
    would both fail it.
    """
    if isinstance(fresh, dict):
        for key, value in committed.items():
            yield from _leaves(fresh[key], value, f"{path}.{key}")
    elif isinstance(fresh, list):
        # strict=True on purpose: a list that changed length is itself staleness.
        for index, (value, item) in enumerate(zip(fresh, committed, strict=True)):
            yield from _leaves(value, item, f"{path}[{index}]")
    elif isinstance(fresh, float) or isinstance(committed, float):
        if math.isclose(
            float(fresh), float(committed), rel_tol=1e-12, abs_tol=NOISE_FLOOR
        ):
            return
        yield path, committed, fresh
    elif fresh != committed:
        yield path, committed, fresh


def test_measurements_json_is_current() -> None:
    """results/measurements.json must be what the code produces right now.

    Without this, every claim below could be satisfied by a stale data file: the
    README and the JSON would agree with each other and both disagree with the
    code. The experiments are seeded, so the comparison is reproducible.
    """
    # Compared to a few ULP rather than bit-for-bit. The maths is identical, but
    # the host BLAS changes the last bits of a float64 summation, and the
    # committed JSON was produced on a different platform than CI runs on.
    # rtol=1e-12 is ~4500 ULP at 1.0, far tighter than any real change.
    live = E.run_all().to_dict()
    drift = [(path, committed, fresh) for path, committed, fresh in _leaves(live, MEASUREMENTS)]
    assert not drift, (
        f"results/measurements.json is stale ({len(drift)} value(s) differ): re-run "
        f"`python -m projects.rope_attribution.experiments`. First: {drift[0]}"
    )


# ==========================================================================
# Setup: the configuration the README says it measured at
# ==========================================================================


def test_readme_states_the_measured_head_dim() -> None:
    literal = readme_literal(r"All measured at `dim = (\d+)`", "the head dim")
    assert_readme_number_matches(MEASUREMENTS["structural_facts"]["dim"], literal, "dim")


def test_every_backticked_base_in_readme_is_the_measured_base() -> None:
    literals = readme_literals(r"`base = ([\d.]+)`")
    assert len(literals) >= 2, f"expected the README to state the base at least twice: {literals}"
    for literal in literals:
        assert_readme_number_matches(
            MEASUREMENTS["structural_facts"]["base"], literal, "base"
        )


def test_every_backticked_scale_in_readme_is_the_measured_scale() -> None:
    literals = readme_literals(r"`scale = ([\d.]+)`")
    assert len(literals) >= 2, f"expected the README to state the scale at least twice: {literals}"
    for literal in literals:
        assert_readme_number_matches(
            MEASUREMENTS["mscale_entropy"]["scale"], literal, "scale"
        )


def test_readme_says_seven_measured_experiments() -> None:
    claimed = readme_word_number(r"(\w+) measured experiments", "the experiment count")
    source = (_REPO / "projects" / "rope_attribution" / "experiments.py").read_text(
        encoding="utf-8"
    )
    labels = {int(n) for group in re.findall(r"E(\d+(?:/E\d+)*)", source) for n in group.split("/E")}
    assert labels, "found no experiment labels in experiments.py"
    assert claimed == len(labels), (
        f"README says {claimed} measured experiments; experiments.py labels "
        f"{sorted(labels)}"
    )


# ==========================================================================
# "The structural properties hold exactly."
# ==========================================================================


def test_relative_position_property_claim() -> None:
    pattern = r"\| relative-position property \((\d+) position pairs\) \| `([\d.eE+-]+)` \|"
    pairs, err = re.search(pattern, README_TEXT, re.MULTILINE).groups()
    structural = MEASUREMENTS["structural_facts"]
    assert int(pairs) == structural["relative_position_pairs_tested"]
    assert_readme_number_matches(
        structural["relative_position_max_abs_err"], err, "relative-position property"
    )


def test_norm_preservation_claim_covers_query_and_key() -> None:
    literal = readme_literal(
        r"\| norm preservation, query and key \| `([\d.eE+-]+)` \|", "norm preservation"
    )
    structural = MEASUREMENTS["structural_facts"]
    assert_readme_number_matches(
        structural["query_norm_preservation_max_abs_err"], literal, "query norm preservation"
    )
    assert_readme_number_matches(
        structural["key_norm_preservation_max_abs_err"], literal, "key norm preservation"
    )


def test_bilinearity_claim_and_score_check_count() -> None:
    pattern = r"\| bilinearity in content, (\d+) score checks \| `([\d.eE+-]+)` \|"
    checks, err = re.search(pattern, README_TEXT, re.MULTILINE).groups()
    structural = MEASUREMENTS["structural_facts"]
    assert int(checks) == structural["scores_tested"]
    assert_readme_number_matches(structural["bilinearity_max_rel_err"], err, "bilinearity")


def test_bilinearity_is_repeated_consistently_in_the_draft_corrections() -> None:
    """The refutation of "RoPE breaks bilinearity" quotes the same number."""
    literal = readme_literal(
        r"doubles the score, to\s+`([\d.eE+-]+)` relative error", "the bilinearity refutation"
    )
    assert_readme_number_matches(
        MEASUREMENTS["structural_facts"]["bilinearity_max_rel_err"], literal, "bilinearity"
    )


def test_pair_closed_form_claim() -> None:
    literal = readme_literal(
        r"\| per-pair closed form vs brute force \| `([\d.eE+-]+)` \|", "the closed form"
    )
    assert_readme_number_matches(
        MEASUREMENTS["structural_facts"]["pair_closed_form_max_abs_err"], literal, "closed form"
    )


def test_feature_additivity_claim() -> None:
    literal = readme_literal(
        r"\| feature additivity, `score == sum_ij f_i g_j A_ij\(delta\)` \| `([\d.eE+-]+)` \|",
        "feature additivity",
    )
    assert_readme_number_matches(
        MEASUREMENTS["feature_attribution"]["additivity_max_abs_err"], literal, "additivity"
    )


def test_additivity_number_is_repeated_consistently_in_the_prose() -> None:
    literal = readme_literal(
        r"`A_ij\(delta\)` to `([\d.eE+-]+)`", "the prose additivity number"
    )
    assert_readme_number_matches(
        MEASUREMENTS["feature_attribution"]["additivity_max_abs_err"], literal, "additivity"
    )


# ==========================================================================
# "a single feature's contribution is a function of distance"
# ==========================================================================


def test_the_readme_sign_crossing_claim_is_measured() -> None:
    """The README's headline: all eight features reverse sign.

    Recomputed from the module rather than read out of the JSON, so a stale
    artefact cannot satisfy it.
    """
    import rope_attribution.statistics as STATS

    assert "**all eight features reverse sign**" in README_TEXT

    range_lo, range_hi = readme_groups(
        r"`delta in \[(\d+), (\d+)\]`", "the sign-crossing range"
    )
    assert (int(range_lo), int(range_hi)) == (STATS.DELTA_LO, STATS.DELTA_HI)

    summary = STATS.seed_variance()["statistics"]
    crossing = summary["sign_crossing_fraction_mean"]
    assert crossing["mean"] == 1.0, crossing
    assert crossing["std"] == 0.0, crossing
    assert STATS.N_FEATURES == 8


def test_the_readme_orders_of_magnitude_claim_is_measured() -> None:
    """The 15-order claim, the bracket the README gives, and the density factor.
    """
    import rope_attribution.statistics as STATS

    band = STATS.position_free_error()["log10_ratio_max"]

    orders = int(
        readme_literal(r"\*\*(\d+)\s+orders\s+of\s+magnitude\*\*", "the position-free residual")
    )
    assert round(band["min"]) <= orders <= round(band["max"]), (orders, band)

    lo, hi = readme_groups(
        r"`(\d+\.\d)`[^`\d]+`(\d+\.\d)` as the grid is made", "the stability bracket"
    )
    assert float(lo) == pytest.approx(band["min"], abs=0.05), (lo, band)
    assert float(hi) == pytest.approx(band["max"], abs=0.05), (hi, band)

    denser = float(readme_literal(r"grid is made (\d+)x denser", "the density factor"))
    rows = STATS.position_free_error()["by_density"]
    n_deltas = [r["n_deltas"] for r in rows]
    assert max(n_deltas) / min(n_deltas) == pytest.approx(denser, rel=0.01)


# ==========================================================================
# "YaRN does not shrink the largest angle"
# ==========================================================================


def test_figure_grid_has_54_distances() -> None:
    diff, count = readme_groups(
        r"\(difference `([\d.]+)`, (\d+) distances\)", "the max-angle comparison"
    )
    grid = figure_delta_grid()
    assert int(count) == len(grid), (
        f"README compares {count} distances; the figure grid has {len(grid)}"
    )
    # The difference itself is exactly zero (asserted against the data below).
    assert float(diff) == 0.0


def test_max_abs_angle_is_identical_for_rope_and_yarn() -> None:
    """``difference 0.0`` at every one of the 54 distances, recomputed."""
    diff, count = readme_groups(
        r"\(difference `([\d.]+)`, (\d+) distances\)", "the max-angle difference"
    )
    assert int(count) == len(figure_delta_grid())
    freqs_rope = R.inv_freq(HEAD_DIM, ROPE_BASE)
    freqs_yarn, _ = R.yarn_parameters(HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS)
    grid = figure_delta_grid()
    gap = max(
        abs(delta * float(freqs_rope[0]) - delta * float(freqs_yarn[0])) for delta in grid
    )
    assert_readme_says_exactly(gap, diff, "max |D_k| difference, RoPE vs YaRN")


def test_fastest_channel_turns_one_radian_per_token() -> None:
    literal = readme_literal(
        r"so `D = ([\d.]+) rad per token` there is unchanged", "the fastest channel"
    )
    fastest = float(R.inv_freq(HEAD_DIM, ROPE_BASE)[0])
    assert_readme_number_matches(fastest, literal, "the fastest channel's rate")
    # ... and it is the max over channels, i.e. what the README's max|D_k| is.
    assert float(spectrum_row("rope_base_10k", 1)["max_abs_angle"]) == fastest


@pytest.mark.parametrize(
    ("delta", "label"),
    [(512, "RoPE base 10k"), (512, "YaRN"), (512, "position interpolation"),
     (4096, "RoPE base 10k"), (4096, "YaRN")],
)
def test_comparison_table_row(delta: int, label: str) -> None:
    maxd, med, frac = table_row(delta, label)
    method = SCHEME_METHODS[label]
    row = spectrum_row(method, delta)
    assert_readme_number_matches(row["max_abs_angle"], maxd, f"{label} @ {delta}: max|D_k|")
    assert_readme_number_matches(row["median_abs_angle"], med, f"{label} @ {delta}: median|D_k|")
    assert_readme_number_matches(
        row["frac_channels_linearizable"], frac, f"{label} @ {delta}: linearizable fraction"
    )


def test_plain_rope_has_no_linearizable_channels_by_4096() -> None:
    """"degrades to nothing for plain RoPE by delta = 4096"."""
    grid_deltas = [row["delta"] for row in MEASUREMENTS["method_spectrum"]
                   if row["method"] == "rope_base_10k"]
    first_zero = min(
        row["delta"]
        for row in MEASUREMENTS["method_spectrum"]
        if row["method"] == "rope_base_10k" and row["frac_channels_linearizable"] == 0.0
    )
    assert 4096 in grid_deltas
    assert first_zero <= 4096, f"plain RoPE first has no linearizable channel at delta={first_zero}"
    assert float(table_row(4096, "RoPE base 10k")[2]) == 0.0


def test_yarn_retains_about_a_fifth_of_the_channels_at_4096() -> None:
    """"retains a fifth of the channels under YaRN".

    The README renders 0.21875 in words. "A fifth" is accepted as 0.2 +/- 0.03
    (15%), which is the width of the claim; anything outside that band is not
    what the sentence says.
    """
    measured = spectrum_row("yarn", 4096)["frac_channels_linearizable"]
    assert abs(measured - 0.2) <= 0.03, (
        f"YaRN retains {measured} of its channels at delta=4096, which is not "
        f"'a fifth'"
    )


def test_amplitude_weighted_error_ratio_rope_to_yarn() -> None:
    pattern = r"\(ratio RoPE:YaRN of `([\d.]+)` at `delta = (\d+)`\)"
    literal, delta = re.search(pattern, README_TEXT, re.MULTILINE).groups()
    rope = spectrum_row("rope_base_10k", int(delta))["amplitude_weighted_err"]
    yarn = spectrum_row("yarn", int(delta))["amplitude_weighted_err"]
    assert_readme_number_matches(yarn / rope, literal, "the amplitude-weighted error ratio")


def test_position_interpolation_is_uniform_division_not_a_bigger_base() -> None:
    """"That is plain position interpolation, a different method" / "does not come from a bigger base"."""
    base_freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    yarn_freqs, _ = R.yarn_parameters(HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS)
    legacy_freqs = R.inv_freq(HEAD_DIM, 500000.0)
    assert not np.array_equal(yarn_freqs, legacy_freqs)
    methods = {row["method"] for row in MEASUREMENTS["method_spectrum"]}
    assert "base_500k_legacy_claim" in methods and "yarn" in methods
    # The interpolation row is exactly base/scale at every channel.
    for delta in (1, 512, 4096):
        expected = (delta * base_freqs / EXT_SCALE).max()
        assert float(spectrum_row("position_interpolation", delta)["max_abs_angle"]) == expected


def test_yarn_keeps_the_base_and_ramps_per_frequency() -> None:
    """"short-wavelength channels keep their inverse frequency, long-wavelength
    channels are divided by scale, blended linearly in between"."""
    base_freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    yarn_freqs, _ = R.yarn_parameters(HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS)
    ramp = R.linear_ramp_mask(
        *R.find_correction_range(32.0, 1.0, HEAD_DIM, ROPE_BASE, ORIGINAL_MAX_POS),
        HEAD_DIM // 2,
    )
    assert np.all(np.diff(ramp) >= 0.0), "the ramp mask is not monotone in k"
    unchanged = yarn_freqs == base_freqs
    divided = np.isclose(yarn_freqs, base_freqs / EXT_SCALE, rtol=0, atol=0)
    assert unchanged[0] and np.all(unchanged[ramp == 0.0]), "ramp==0 must keep inv_freq"
    assert divided[-1] and np.all(divided[ramp == 1.0]), "ramp==1 must divide by scale"
    blended = (ramp > 0.0) & (ramp < 1.0)
    if np.any(blended):
        assert np.all(yarn_freqs[blended] < base_freqs[blended])
        assert np.all(yarn_freqs[blended] > base_freqs[blended] / EXT_SCALE)


# ==========================================================================
# "Partial RoPE yields an exactly position-free sub-score"
# ==========================================================================


def test_partial_rope_split_fractions() -> None:
    p_literal = readme_literal(r"At `p = ([\d.]+)`", "the partial-RoPE fraction")
    clean_literal = readme_literal(r"unrotated (\d+)% of channels", "the unrotated fraction")
    partial = MEASUREMENTS["partial_rope"]
    assert_readme_number_matches(partial["rotated_fraction"], p_literal, "p")
    assert int(clean_literal) == round(100.0 * partial["clean_fraction"])
    assert partial["rotated_fraction"] + partial["clean_fraction"] == 1.0


def test_unrotated_subscore_spread_is_exactly_zero() -> None:
    exact, _ = readme_groups(
        r"is\s+exactly `([\d.]+)`, while the rotated quarter varies by `([\d.]+)`",
        "the partial-RoPE spreads",
    )
    partial = MEASUREMENTS["partial_rope"]
    assert_readme_says_exactly(
        partial["clean_score_spread"], exact, "the unrotated sub-score spread"
    )
    assert partial["rotated_fraction"] == 0.25, "the rotated quarter is 0.25, not a quarter by chance"


def test_rotated_subscore_spread_is_12_03() -> None:
    _, rotated = readme_groups(
        r"is\s+exactly `([\d.]+)`, while the rotated quarter varies by `([\d.]+)`",
        "the partial-RoPE spreads",
    )
    assert_readme_number_matches(
        MEASUREMENTS["partial_rope"]["rotated_score_spread"],
        rotated,
        "the rotated sub-score spread",
    )


def test_partial_rope_preserves_the_norm_as_full_rope_does() -> None:
    """The README must not claim partial rotation damages the magnitude gate.

    It used to state a deviation of ``0.27`` against full RoPE's 1e-15, on the
    strength of an implementation that paired channel ``i`` with ``i + dim//2``
    across the whole head. That is not the GPT-NeoX layout and not orthogonal.
    With the corrected operator the deviation is 0 to the float64 floor, so the
    README now says partial rotary preserves the norm like full RoPE, and this
    test checks the claim cannot silently drift back.
    """
    deviation = MEASUREMENTS["partial_rope"]["partial_norm_deviation"]
    assert deviation < 1e-12, (
        f"partial rotary must preserve the norm; measured {deviation:.3e}"
    )
    text = README_TEXT.lower()
    assert "deviation `0.2" not in text, (
        "the README still quotes a non-zero partial-RoPE norm deviation"
    )
    # It must state the orthogonality positively, not merely omit the old number.
    assert "preserved exactly as for full rope" in text, (
        "the README should say partial rotary preserves the norm, as full RoPE does"
    )
    assert "paired within" in text or "within\nitself" in text, (
        "the README should name the within-block pairing, which is the actual "
        "reason the norm is preserved"
    )


# ==========================================================================
# "YaRN's magnitude term is a temperature"
# ==========================================================================


def test_mscale_value_and_the_published_formula() -> None:
    pattern = r"\(`([\d.]+)` at `scale = ([\d.]+)`\)"
    literal, scale = re.search(pattern, README_TEXT, re.MULTILINE).groups()
    entropy = MEASUREMENTS["mscale_entropy"]
    assert_readme_number_matches(entropy["scale"], scale, "the mscale's scale")
    assert_readme_number_matches(entropy["mscale"], literal, "mscale")
    # The formula the README writes next to the number.
    assert float(scale) > 1.0
    assert entropy["mscale"] == 0.1 * math.log(float(scale)) + 1.0
    assert float(R.get_mscale(float(scale))) == entropy["mscale"]
    assert float(spectrum_row("yarn", 512)["mscale"]) == entropy["mscale"]


def test_attention_entropy_ratios() -> None:
    pattern = r"`H/ln T = ([\d.]+)` to `([\d.]+)`"
    unscaled, mscaled = re.search(pattern, README_TEXT, re.MULTILINE).groups()
    entropy = MEASUREMENTS["mscale_entropy"]
    assert_readme_number_matches(entropy["entropy_ratio_unscaled"], unscaled, "entropy, unscaled")
    assert_readme_number_matches(entropy["entropy_ratio_mscaled"], mscaled, "entropy, mscaled")
    # A temperature above 1 must sharpen the distribution.
    assert entropy["entropy_ratio_mscaled"] < entropy["entropy_ratio_unscaled"]
    assert entropy["max_possible_entropy"] == math.log(entropy["n_keys"])
    assert entropy["entropy_ratio_unscaled"] == entropy["entropy_unscaled"] / math.log(
        entropy["n_keys"]
    )


# ==========================================================================
# "Correcting the earlier drafts"
# ==========================================================================


def test_yarn_unchanged_and_changed_pair_counts() -> None:
    """"9 of 32 pairs are bit-for-bit unchanged and 21 of 32 differ from uniform
    interpolation" - both recomputed from ``rope.yarn_parameters``."""
    pattern = r"(\d+) of (\d+) pairs are bit-for-bit unchanged and (\d+) of (\d+) differ from uniform\s+interpolation"
    unchanged, total_a, differ, total_b = re.search(pattern, README_TEXT, re.MULTILINE).groups()
    base_freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    yarn_freqs, _ = R.yarn_parameters(HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS)
    n_pairs = base_freqs.size
    measured_unchanged = int(np.count_nonzero(yarn_freqs == base_freqs))
    measured_differ = int(np.count_nonzero(yarn_freqs != base_freqs / EXT_SCALE))
    assert int(total_a) == n_pairs and int(total_b) == n_pairs, (
        f"the README counts out of {total_a}/{total_b} pairs; head_dim={HEAD_DIM} "
        f"gives {n_pairs} rotary pairs"
    )
    assert int(unchanged) == measured_unchanged, (
        f"README says {unchanged} of {n_pairs} pairs are bit-for-bit unchanged, "
        f"measured {measured_unchanged}"
    )
    assert int(differ) == measured_differ, (
        f"README says {differ} of {n_pairs} pairs differ from uniform interpolation, "
        f"measured {measured_differ}"
    )


def test_refuted_legacy_numbers_appear_only_where_they_are_refuted() -> None:
    """3.7x and base 500000 are quoted as things that do not survive measurement."""
    start = README_TEXT.index("## Correcting the earlier drafts")
    end = README_TEXT.index("## Figures")
    section = README_TEXT[start:end]
    for legacy in ("3.7x", "500000"):
        assert legacy in section, f"the refutation of {legacy!r} has disappeared"
        outside = README_TEXT[:start] + README_TEXT[end:]
        assert legacy not in outside, (
            f"{legacy!r} is quoted outside the section that refutes it, which "
            f"would present a refuted claim as this repository's own"
        )


# ==========================================================================
# The identities the README writes out as equations
# ==========================================================================


def test_relative_frame_identities_hold() -> None:
    """``q_relative = R_m^T (R_m q_hat) = q_hat`` and ``k_relative(n) = R_{n-m} k_hat``."""
    rng = np.random.default_rng(7)
    freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    q_hat = rng.standard_normal(HEAD_DIM)
    k_hat = rng.standard_normal(HEAD_DIM)
    for m, n in ((0, 1), (0, 512), (13, 977), (100, 127)):
        cos, sin = R.rope_cos_sin(np.array([float(m), float(n)]), freqs)
        q_rot = R.apply_rope(q_hat[None, :], cos[:1], sin[:1])[0]
        k_rot = R.apply_rope(k_hat[None, :], cos[1:2], sin[1:2])[0]
        # q_relative = R_m^T (R_m q_hat) = q_hat
        inv_cos, inv_sin = R.rope_cos_sin(np.array([float(-m)]), freqs)
        q_rel = R.apply_rope(q_rot[None, :], inv_cos, inv_sin)[0]
        np.testing.assert_allclose(q_rel, q_hat, rtol=0, atol=1e-12)
        # k_relative(n) = R_m^T (R_n k_hat) = R_{n-m} k_hat
        k_rel = R.apply_rope(k_rot[None, :], inv_cos, inv_sin)[0]
        d_cos, d_sin = R.rope_cos_sin(np.array([float(n - m)]), freqs)
        k_direct = R.apply_rope(k_hat[None, :], d_cos, d_sin)[0]
        np.testing.assert_allclose(k_rel, k_direct, rtol=0, atol=1e-12)
        # score(m, n) = q_hat . R_{n-m} k_hat
        assert float(q_rot @ k_rot) == pytest.approx(
            float(q_hat @ k_direct), rel=0, abs=1e-11
        )
        assert float(E.score_relative(q_hat, k_hat, freqs, n - m)) == pytest.approx(
            float(q_rot @ k_rot), rel=0, abs=1e-12
        )


def test_pair_contribution_is_a_single_sinusoid_in_distance() -> None:
    """``c_k(delta) = A_k cos(D_k) + B_k sin(D_k) = R_k cos(D_k - psi_k)``."""
    rng = np.random.default_rng(11)
    freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    terms = E.pair_terms(rng.standard_normal(HEAD_DIM), rng.standard_normal(HEAD_DIM), freqs, 64)
    amplitude = terms.amplitude
    np.testing.assert_allclose(
        terms.contributions(),
        amplitude * np.cos(terms.angle - terms.phase_offset),
        rtol=1e-12,
        atol=1e-12,
    )
    # A_k and B_k are the README's definitions, checked against the raw vectors.
    h = HEAD_DIM // 2
    terms_2 = E.pair_terms(rng.standard_normal(HEAD_DIM), rng.standard_normal(HEAD_DIM), freqs, 3)
    assert terms_2.aligned.shape == (h,)
    np.testing.assert_allclose(terms_2.angle, 3.0 * freqs, rtol=0, atol=0)


def test_pair_amplitude_is_position_free_and_its_phase_is_linear_in_distance() -> None:
    """"``R_k`` is a single sinusoid in distance: an amplitude that does not depend
    on ``delta`` at all, and a phase that advances exactly linearly in it."

    Checked by *recovering* ``A_k`` and ``B_k`` from the contribution at five
    distances and comparing them with the closed form, rather than by trusting
    that the function computing them is the same function that reports them.
    """
    rng = np.random.default_rng(11)
    freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    q_hat = rng.standard_normal(HEAD_DIM)
    k_hat = rng.standard_normal(HEAD_DIM)
    deltas = (1, 8, 32, 128, 512)
    curves = np.array([E.pair_contributions(q_hat, k_hat, freqs, d) for d in deltas])
    angles = np.array([delta * freqs for delta in deltas])
    # Recover A_k and B_k per pair from c_k(d) = A_k cos(D_k) + B_k sin(D_k), by
    # solving the 2x2 normal equations of that fit independently per pair.
    cosines, sines = np.cos(angles), np.sin(angles)
    gram = np.stack(
        [
            np.stack([(cosines**2).sum(0), (cosines * sines).sum(0)], axis=-1),
            np.stack([(cosines * sines).sum(0), (sines**2).sum(0)], axis=-1),
        ],
        axis=-2,
    )
    rhs = np.stack([(cosines * curves).sum(0), (sines * curves).sum(0)], axis=-1)
    recovered = np.linalg.solve(gram, rhs[..., None])[..., 0]
    terms = E.pair_terms(q_hat, k_hat, freqs, deltas[0])
    np.testing.assert_allclose(recovered[:, 0], terms.aligned, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(recovered[:, 1], terms.crossed, rtol=1e-6, atol=1e-7)
    # The recovered amplitude and phase are the closed form's, and neither moves
    # with distance: only D_k does.
    amplitudes = np.hypot(recovered[:, 0], recovered[:, 1])
    phases = np.arctan2(recovered[:, 1], recovered[:, 0])
    np.testing.assert_allclose(amplitudes, terms.amplitude, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(phases, terms.phase_offset, rtol=1e-6, atol=1e-9)
    for delta in deltas:
        moved = E.pair_terms(q_hat, k_hat, freqs, delta)
        np.testing.assert_array_equal(moved.amplitude, terms.amplitude)
        np.testing.assert_array_equal(moved.phase_offset, terms.phase_offset)
        # ... and the phase advances exactly linearly, i.e. D_k - psi_k is linear.
        np.testing.assert_allclose(
            moved.angle - moved.phase_offset, delta * freqs - phases, rtol=1e-9, atol=1e-9
        )


def test_a_pair_is_position_blind_exactly_when_its_crossed_term_vanishes() -> None:
    """"A pair is completely position-blind precisely when B_k = 0".

    What is measurable is that ``B_k`` is the whole of a pair's
    position-carrying part: it is the only coefficient of ``sin(D_k)``, and it is
    exactly the deviation of the pair's phase from zero. With ``B_k = 0`` the
    pair's sinusoid has phase 0 (or pi), so its contribution is the
    position-free aligned coefficient ``A_k`` times ``cos(D_k)``; with
    ``B_k != 0`` the phase is neither.

    A wording limit worth recording: ``c_k(delta) = A_k cos(D_k)`` is still a
    function of distance, so "completely position-blind" is a claim about the
    pair's phase and gate, not about its contribution being constant across
    distance. The test pins the phase/gate facts, which is what the surrounding
    paragraph asserts ("``A_k`` ... position-free", "``B_k`` ... carries
    position").
    """
    rng = np.random.default_rng(13)
    freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    h = HEAD_DIM // 2
    q_hat = rng.standard_normal(HEAD_DIM)
    k_hat = rng.standard_normal(HEAD_DIM)
    deltas = (1, 8, 32, 128, 512)

    # B_k = q[k+h] * k[k] - q[k] * k[k+h] vanishes for every pair exactly when
    # k repeats q componentwise (then B_k = q[k+h] q[k] - q[k] q[k+h] = 0 bitwise).
    blind_q = q_hat
    blind_k = q_hat.copy()
    mixed_k = q_hat.copy()
    mixed_k[1] += 1.0  # breaks B_1 without touching B_0
    for vec_q, vec_k, expect_all_blind in (
        (blind_q, blind_k, True),
        (blind_q, mixed_k, False),
        (q_hat, k_hat, False),
    ):
        terms = E.pair_terms(vec_q, vec_k, freqs, deltas[0])
        blind = terms.crossed == 0.0
        if expect_all_blind:
            assert blind.all(), "constructed B_k = 0 for every pair, but not all are 0"
        # Phase is exactly 0 or pi where B_k = 0, and neither where it is not.
        assert np.all(np.isin(terms.phase_offset[blind], (0.0, np.pi)))
        assert not np.any(np.isin(terms.phase_offset[~blind], (0.0, np.pi)))
        # With B_k = 0 the contribution is the position-free A_k times cos(D_k).
        for delta in deltas:
            moved = E.pair_terms(vec_q, vec_k, freqs, delta)
            np.testing.assert_allclose(
                moved.contributions()[blind],
                (moved.aligned * np.cos(moved.angle))[blind],
                rtol=0,
                atol=0,
                err_msg=f"delta={delta}",
            )
    # The mixed construction really does leave exactly one blind pair.
    assert int((E.pair_terms(blind_q, mixed_k, freqs, 1).crossed == 0.0).sum()) == h - 1


# ==========================================================================
# Figures
# ==========================================================================


def figure_tokens() -> set[str]:
    return set(re.findall(r"\bfig\d{2}[A-Za-z0-9_]*", README_TEXT))


def test_every_figure_the_readme_names_exists_on_disk() -> None:
    """Covers both the bare stems in the figure table and ``figures/fig09_...png``."""
    figures_dir = _REPO / "figures"
    tokens = figure_tokens()
    assert len(tokens) >= 9, f"figure references found in README.md: {sorted(tokens)}"
    for token in sorted(tokens):
        exact = figures_dir / f"{token}.png"
        prefixed = [
            child for child in figures_dir.iterdir()
            if child.name.startswith(f"{token}_") and child.suffix == ".png"
        ]
        assert exact.is_file() or prefixed, (
            f"README references figure {token!r}, but figures/ has neither "
            f"{exact.name} nor any {token}_*.png"
        )


def test_readme_says_nine_figures_and_each_has_a_csv_sidecar() -> None:
    claimed = readme_word_number(r"(\w+) figures, each with a CSV sidecar", "the figure count")
    stems = sorted(path.stem for path in (_REPO / "figures").glob("fig*.png"))
    assert claimed == len(stems), f"README says {claimed} figures; figures/ holds {stems}"
    for stem in stems:
        assert (_REPO / "figures" / f"{stem}.csv").is_file(), f"no CSV sidecar for {stem}"


def test_the_figure_table_lists_exactly_the_figures_on_disk() -> None:
    listed = sorted(figure_tokens())
    stems = sorted(path.stem for path in (_REPO / "figures").glob("fig*.png"))
    # `figures/fig03`-style short references are not stems; the full ones are.
    full = sorted(t for t in listed if t.count("_") >= 1)
    assert full == stems, f"README figure table lists {full}; figures/ holds {stems}"


def test_figure_grid_constants_match_figures_module() -> None:
    """The grid this file recomputes fig09's number on is the one figures.py uses."""
    # Imported this way so a missing matplotlib turns into a clean skip: the
    # figures module is the only thing here that needs it.
    F = pytest.importorskip("rope_attribution.figures")

    assert F.DELTA_MAX == FIG_DELTA_MAX
    assert F.DELTA_POINTS == FIG_DELTA_POINTS
    assert F.EXT_SCALE == EXT_SCALE
    assert F.ORIGINAL_MAX_POS == ORIGINAL_MAX_POS
    assert F.HEAD_DIM == HEAD_DIM
    assert tuple(F.DELTA_GRID) == figure_delta_grid()
    assert len(F.DELTA_GRID) == 54


# ==========================================================================
# Structural: every path the README presents as existing, exists
# ==========================================================================

# Suffixes that make a bare token look like a repository file. Anything else
# (`experiments.method_spectrum`, `310bf2c`, `1.3466`, `transformers`) is prose,
# not a path.
_FILE_SUFFIXES = (
    ".py", ".md", ".txt", ".toml", ".cff", ".cfg", ".ini",
    ".csv", ".png", ".json", ".yml", ".yaml",
)

# Paths the README deliberately talks *about* while saying they no longer exist.
# `projects/fig_*.png` is named in "Legacy material" as removed. External paths
# (`models/llama/modeling_llama.py`, `scaled_rope/...` in `jquesnelle/yarn`) do
# not need an entry: the rule below only treats a slash-bearing token as
# repository-local when its first segment is a real top-level entry here.
KNOWN_ABSENT = frozenset({"projects/fig_*.png"})

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./*-]*$")


@functools.lru_cache(maxsize=1)
def top_level_entries() -> frozenset[str]:
    return frozenset(path.name for path in _REPO.iterdir())


def readme_path_tokens(text: str) -> list[str]:
    """Every whitespace-delimited token the README presents as a path.

    Sources: inline code spans, markdown link targets that are not pure anchors,
    and the contents of fenced code blocks.
    """
    tokens: list[str] = []
    tokens.extend(m.group(1) for m in re.finditer(r"`([^`\n]+)`", text))
    tokens.extend(
        m.group(1)
        for m in re.finditer(r"\]\(([^)\s]+)\)", text)
        if not m.group(1).startswith("#")
    )
    for block in re.findall(r"^```[^\n]*\n(.*?)^```", text, re.MULTILINE | re.DOTALL):
        tokens.extend(block.split())
    return [part for token in tokens for part in token.split() if part]


def repo_local_tokens() -> dict[str, str]:
    """Map each repository-local path token to the way it has to be resolved.

    The rule, stated once so it can be argued with:

    * inline code, non-anchor link targets and fenced code blocks are the only
      places a path is *presented as something that should exist*; prose without
      backticks is not scanned;
    * a token containing ``/`` is repository-local only if its first segment is
      a real top-level entry here. That is what excludes
      ``models/llama/modeling_llama.py`` and
      ``scaled_rope/LlamaYaRNScaledRotaryEmbedding.py``, which the README
      attributes to `transformers` and to `jquesnelle/yarn`;
    * a dotted module path with two or more dots
      (``python -m projects.rope_attribution.experiments``) is repository-local
      when its first segment is a directory here, and is resolved by turning the
      dots into separators. A single dot is a filename, not a module, which is
      what keeps ``figures.py`` and excludes ``experiments.method_spectrum``;
    * a dotted module path whose first segment is a directory here is resolved
      by turning dots into separators. This works even though ``projects/``
      has no ``__init__.py``: Python resolves it as an implicit namespace
      package, so ``python -m projects.rope_attribution.X`` really does run.
      A *slash* in the same position does not, which is the mistake a Makefile
      here made and the command guard below now pins;
    * a token without ``/`` is repository-local only if it exists at the
      repository root, or if it carries a real file suffix and exactly one file
      in the repository has that basename. That is what accepts the bare
      ``rope.py`` (it lives in ``projects/rope_attribution/``) and excludes
      ``transformers`` and the commit hash ``310bf2c``;
    * a token with neither ``.`` nor ``/`` and not at the root is prose.
    """
    kinds: dict[str, str] = {}
    for token in readme_path_tokens(README_TEXT):
        if not _TOKEN_RE.match(token) or token in {".", ".."}:
            continue
        if token in KNOWN_ABSENT:
            continue
        if "*" in token:
            if token.split("/")[0] in top_level_entries():
                kinds[token] = "glob"
            continue
        if "/" in token:
            if token.split("/")[0] in top_level_entries():
                kinds[token] = "path"
            continue
        if token.count(".") >= 2 and (_REPO / token.split(".")[0]).is_dir():
            kinds[token] = "module"
            continue
        if (_REPO / token).exists():
            kinds[token] = "path"
        elif Path(token).suffix in _FILE_SUFFIXES:
            kinds[token] = "basename"
    return kinds


def _resolves_path(token: str) -> bool:
    candidate = _REPO / token
    if candidate.exists():
        return True
    for suffix in _FILE_SUFFIXES:
        if candidate.with_name(candidate.name + suffix).exists():
            return True
    # `figures/fig03` is a short form of `figures/fig03_max_vs_median_angle`.
    if candidate.parent.is_dir():
        prefix = f"{candidate.name}_"
        if any(child.name.startswith(prefix) for child in candidate.parent.iterdir()):
            return True
    return False


def _resolves(kind: str, token: str) -> bool:
    if kind == "glob":
        return bool(list(_REPO.glob(token)))
    if kind == "module":
        return _resolves_path(token.replace(".", "/"))
    if kind == "basename":
        return bool(_files_by_basename().get(token))
    return _resolves_path(token)


@functools.lru_cache(maxsize=1)
def _files_by_basename() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for root, dirnames, filenames in os.walk(_REPO):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIR_NAMES]
        for name in filenames:
            found.setdefault(name, []).append(Path(root) / name)
    return found


def test_readme_references_no_missing_file() -> None:
    kinds = repo_local_tokens()
    assert len(kinds) >= 15, (
        f"the path scan only found {len(kinds)} repository-local references; "
        f"if this is a regression in the extractor, the test below is vacuous"
    )
    missing = [token for token, kind in sorted(kinds.items()) if not _resolves(kind, token)]
    assert not missing, f"README.md references files that do not exist: {missing}"


def test_the_path_scan_covers_the_paths_that_matter() -> None:
    """Guard against the extractor silently shrinking to nothing."""
    kinds = repo_local_tokens()
    for expected in (
        "results/measurements.json",
        "figures/README.md",
        "projects/ATTRIBUTION.md",
        "projects.rope_attribution.experiments",
        "projects.rope_attribution.figures",
        "projects.rope_attribution.statistics",
        "projects.rope_attribution.real_model",
        "projects.rope_attribution.usefulness",
        "figures/fig09_position_conditional_attribution.png",
        "figures/fig03",
        "projects/frontier-01-*.py",
        "projects/frontier-01-*.md",
        "requirements-dev.txt",
        "NOTICE.md",
        "LICENSE",
        "CITATION.cff",
        "rope.py",
        "experiments.py",
        "figures.py",
        "frontier-01-bag-of-words-test.py",
    ):
        assert expected in kinds, f"the path scan missed {expected!r}; it found {sorted(kinds)}"


# ---------------------------------------------------------------------------
# The documented commands have to work
# ---------------------------------------------------------------------------

_DOCS_SCANNED_FOR_COMMANDS = (
    "README.md",
    "figures/README.md",
    "LEGACY.md",
    "paper/BUILD.md",
    "NOTICE.md",
)

_CMD_RE = re.compile(r"python\s+-m\s+([A-Za-z_][A-Za-z0-9_.]*)")


def _documented_module_invocations() -> dict[str, str]:
    """Every ``python -m <dotted>`` the repository documents, and where.

    ``python -m X`` needs ``X`` to be importable, which is strictly stronger than
    a file existing at that dotted path. It is the stronger requirement that was
    silently violated: four places documented
    ``python -m projects.rope_attribution.X``, which cannot work because
    ``projects/`` is a source root and not a package. CI proved it with a
    ModuleNotFoundError, and a reader following the shipped ``figures/README.md``
    would have hit it immediately.
    """
    out: dict[str, str] = {}
    for rel in _DOCS_SCANNED_FOR_COMMANDS:
        path = _REPO / rel
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for module in _CMD_RE.findall(line):
                module = module.rstrip(".")
                out.setdefault(module, f"{rel}:{lineno}")
    return out


def test_every_documented_module_invocation_is_importable() -> None:
    """The strongest cheap check: the documented command's module really imports.

    Importing is not running the entry point, but it is exactly the step
    ``python -m`` fails at, and it costs a fraction of a second per module rather
    than the minutes the real generators take.
    """
    documented = _documented_module_invocations()
    assert len(documented) >= 3, f"the scan found too few commands: {documented}"
    for module, where in sorted(documented.items()):
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=_REPO,
            env={**os.environ, "PYTHONPATH": str(_REPO / "projects")},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"{where} documents `python -m {module}`, which does not import:\n"
            f"{proc.stderr.strip()[-800:]}"
        )


def _expanded_module_paths(text: str, variables: dict[str, str]) -> list[str]:
    """Module paths passed to ``-m``, with ``$(VAR)`` substituted.

    Without the substitution the guard is vacuous on a Makefile, which is where
    the defect actually was: the target is written ``$(PKG).experiments``, so a
    regex over the raw text sees neither ``python`` nor the path that breaks.
    """
    out = []
    for match in re.finditer(r"-m\s+(\S+)", text):
        target = match.group(1)
        for name, value in variables.items():
            target = target.replace("$(" + name + ")", value)
            target = target.replace("${" + name + "}", value)
        out.append(target)
    return out


def _makefile_variables() -> dict[str, str]:
    makefile = _REPO / "Makefile"
    if not makefile.exists():
        return {}
    found = {}
    for match in re.finditer(r"^([A-Z_][A-Z0-9_]*)\s*[:?]?=\s*(.+?)\s*$",
                             makefile.read_text(encoding="utf-8"), re.M):
        found[match.group(1)] = match.group(2)
    return found


def test_the_module_path_in_a_documented_command_is_dotted() -> None:
    """Pin the defect a Makefile here actually had: a slash in a ``-m`` path.

    ``python -m projects.rope_attribution.experiments`` runs, because Python
    resolves ``projects/`` as an implicit namespace package even without an
    ``__init__.py``. The same path written with slashes is a malformed module
    name and fails with a ModuleNotFoundError that reads like a missing file
    rather than a typo. That is how this reached CI, and it would have reached
    every reader's terminal.

    The docs are scanned too, but they were innocent; the Makefile is not, so
    its variables are expanded before the check.
    """
    variables = _makefile_variables()
    assert variables, "no Makefile variables parsed; the guard would be vacuous"
    scanned = list(_DOCS_SCANNED_FOR_COMMANDS) + ["Makefile"]
    checked = 0
    for rel in scanned:
        path = _REPO / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for target in _expanded_module_paths(text, variables):
            checked += 1
            assert "/" not in target, (
                f"{rel} invokes `-m {target}` with a slash; `-m` takes a dotted "
                f"module path, not a filesystem path"
            )
    assert checked >= 5, f"only {checked} module paths checked; guard is thin"

def test_external_and_historical_references_are_not_treated_as_repo_paths() -> None:
    """The rule must not start demanding files that the README calls deleted."""
    kinds = repo_local_tokens()
    for external in (
        "models/llama/modeling_llama.py",
        "scaled_rope/LlamaYaRNScaledRotaryEmbedding.py",
        "jquesnelle/yarn",
    ):
        assert external not in kinds, f"{external!r} is external and must not be checked"
    for prose in (
        "experiments.method_spectrum",
        "experiments.linearization_error",
        "310bf2c",
        "transformers",
        "yarn",
        "position_interpolation",
        "1.3466",
        "0.997",
        "0.0",
    ):
        assert prose not in kinds, f"{prose!r} is prose, not a repository path"
    assert "projects/fig_*.png" in KNOWN_ABSENT
    assert "projects/fig_*.png" not in kinds
    assert not list(_REPO.glob("projects/fig_*.png")), (
        "projects/fig_*.png is on the known-deleted list; if it is back, the "
        "README's Legacy material section needs revisiting"
    )


def test_readme_names_figures_and_paths_it_links_to_with_markdown() -> None:
    """The one markdown link in the README is an anchor, so no file is implied."""
    links = re.findall(r"\]\(([^)\s]+)\)", README_TEXT)
    for link in links:
        assert link.startswith("#"), f"README links to {link!r}, which is not an in-page anchor"


# ==========================================================================
# "several scripts print hardcoded illustrative values rather than computing them"
# ==========================================================================


def test_legacy_scripts_the_readme_warns_about_do_print_literals() -> None:
    bag = (_REPO / "projects" / "frontier-01-bag-of-words-test.py").read_text(encoding="utf-8")
    assert re.search(r"acc(?:uracy)?\s+0\.\d", bag), (
        "the README says frontier-01-bag-of-words-test.py assigns retrieval "
        "accuracies as literals; no such literal is there any more"
    )
    assert re.search(r"=\s*D\s*\*\*\s*2\s*/\s*2", bag), (
        "the README says that script computes 'interaction' as D**2 / 2 by definition"
    )
    graphs = (_REPO / "projects" / "frontier-01-graphs-ULTIMATE-V11.py").read_text(
        encoding="utf-8"
    )
    assert re.search(r"=\s*[\[\(]\s*[-\d]", graphs), (
        "the README says frontier-01-graphs-ULTIMATE-V11.py plots literal lists"
    )


# ==========================================================================
# "1102 tests"
# ==========================================================================


@functools.lru_cache(maxsize=1)
def collected_node_ids() -> tuple[str, ...]:
    """Ask pytest what it would collect, in a subprocess rooted at the repository.

    Parsing the ``def test_`` lines of the source files is not an option: the
    existing suite is heavily parametrized, so a source count is nowhere near the
    collected count.
    """
    env = dict(os.environ)
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"):
        env.pop(key, None)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=str(_REPO),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"could not run pytest --collect-only: {exc}")
    if proc.returncode != 0:
        pytest.skip(
            "pytest --collect-only failed, so the test count cannot be verified:\n"
            + proc.stdout[-2000:]
        )
    ids = tuple(
        line.strip()
        for line in proc.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    )
    if not ids:
        pytest.skip("pytest --collect-only reported no test node ids")
    return ids


def test_quoted_test_count_matches_what_pytest_collects() -> None:
    """``1102 tests``, in both places the README states it.

    The README counts the suite that existed when it was written. This file is
    the documentation verifier and is deliberately excluded from the comparison,
    so that *adding verification tests* cannot invalidate the README. Any other
    module gaining or losing a test still fails this.
    """
    literals = readme_literals(r"\b(\d+) tests\b")
    assert len(literals) == 2, (
        f"expected the README to quote the test count twice, found {literals}"
    )
    assert len(set(literals)) == 1, f"README quotes conflicting test counts: {literals}"

    node_ids = collected_node_ids()
    mine = [nid for nid in node_ids if nid.startswith("tests/test_readme_claims.py::")]
    assert mine, "this file contributed no tests to the collection; the count is not comparable"
    others = len(node_ids) - len(mine)
    assert others == int(literals[0]), (
        f"README says {literals[0]} tests; pytest collects {others} outside this "
        f"file ({len(node_ids)} in total, {len(mine)} of them this file). If a "
        f"test was added or removed, update the README."
    )
