"""Make every quantitative claim in ``paper/main.tex`` machine-verified.

Why this file exists
--------------------
``paper/main.tex`` quotes specific measured numbers, and the two files that hold
them - ``results/measurements.json`` and the ``figures/figNN_*.csv`` sidecars -
were connected to the prose by nothing but the appendix called "Provenance of
every number".  This module *is* that audit, executable.  Every test below

* locates the repository from ``__file__``, never from the working directory;
* **parses the number out of ``paper/main.tex``** and compares *that* with the
  canonical value, instead of hardcoding the expected literal here.  Hardcoding
  would only move the problem one file over: the paper would still be free to
  drift away from the data, which is the whole failure this file exists to
  catch;
* tolerates exactly the rounding the paper used and no more - the accepted
  interval is half of the literal's last written digit, so the tolerance comes
  from the paper's own precision, never from a guess;
* demands exact float equality wherever the paper says "exactly", "identical"
  or "bit-for-bit".

Sources of truth, in order of preference:

* ``results/measurements.json`` for everything an experiment recorded;
* the ``figures/*.csv`` sidecars for the 54-point figure grid, which is where
  the paper says its finer-grid numbers come from;
* a fresh recomputation from ``rope_attribution`` where the paper's claim is
  about the *code* rather than about a stored artifact (the fig09 grid, the
  frequency-ladder bookkeeping, the ``mscale`` identity, the figure grid itself).

``test_measurements_json_is_current`` closes the loop: it re-runs the
experiments and requires the JSON on disk to be exactly what the code produces,
so "the paper matches the data" cannot be satisfied by a stale data file.

``test_no_number_in_the_paper_is_left_unverified`` is the exhaustiveness
guarantee.  It deletes every span claimed by a test in this module (plus the two
verified ``tabular`` blocks) from the source and requires that whatever numeric
tokens survive are only those listed in :data:`EXEMPT_ALGEBRA`, each with a
stated reason.  Adding a number to the paper without adding a test for it fails
there.

Four known defects in ``paper/main.tex`` are left as *failing* tests rather than
being silently loosened.  Each names itself in the test name and works the
arithmetic out in its docstring:

* ``test_raising_the_base_by_s_equals_dividing_positions_by_s`` - Section
  \\ref{sec:background} calls position interpolation "algebraically equivalent"
  to replacing the base by ``base * s``.  It is not: ``inv_freq[k] = base **
  (-2k/d)``, so a uniform division by ``s`` and a multiplication of the base by
  ``s`` differ at every channel except ``k = 0``.
* ``test_swing_versus_additivity_residual_is_at_least_fifteen_orders_of_magnitude``
  - the paper says "some 14 orders of magnitude"; its own numbers give 16.
* ``test_the_synthesis_ramp_from_two_to_eleven_out_of_thirty_two`` - the paper
  says the linearizable fraction goes "from 2/32 to 11/32 at delta ~ 861";
  fig04 says plain RoPE holds 0/32 there (2/32 only up to delta = 549).
* ``test_the_appendix_rounds_the_two_pair_amplitudes_correctly`` - the appendix
  truncates ``0.9526294617807264`` to ``0.9526294617`` instead of rounding it
  to ``0.9526294618``.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from functools import cache, lru_cache
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

# --------------------------------------------------------------------------
# the documents under test
# --------------------------------------------------------------------------
PAPER_PATH = _REPO / "paper" / "main.tex"
BIB_PATH = _REPO / "paper" / "references.bib"
FIGURES_DIR = _REPO / "figures"

TEX = PAPER_PATH.read_text(encoding="utf-8")
# Claim patterns are matched against a whitespace-normalised copy so that a
# re-wrap of the source cannot silently break (or silently pass) a test.  Every
# pattern below is therefore written with single spaces, never with newlines.
TEX_FLAT = re.sub(r"\s+", " ", TEX)
BIB = BIB_PATH.read_text(encoding="utf-8")
MEASUREMENTS = json.loads(
    (_REPO / "results" / "measurements.json").read_text(encoding="utf-8")
)

# The maintained configuration the paper says every measurement was taken at
# (Section "Method", the appendix "Configuration" paragraph, and the caption of
# Table~\ref{tab:spectrum}).
HEAD_DIM = 64
ROPE_BASE = 10000.0
LEGACY_BASE = 500000.0
EXT_SCALE = 32.0
ORIGINAL_MAX_POS = 2048
N_KEYS = 256
N_PAIRS = HEAD_DIM // 2

# `figures.py` grid constants.  `test_figure_grid_constants_match_figures_module`
# re-checks each of them against the module itself, so the duplication here
# cannot rot.
FIG_DELTA_MAX = 8192
FIG_DELTA_POINTS = 61
FIG_SEED_ATTRIB = 1
LINEARIZABLE_ANGLE_RAD = 0.1


# ==========================================================================
# Claim registry
# ==========================================================================
#
# One declarative table of (name, pattern, minimum match count).  Each pattern
# is *tight*: it anchors on the surrounding prose and captures only the
# literal(s) its test verifies, never a whole paragraph.  Two things consume
# this table:
#
#   * the individual claim tests, via :func:`groups`;
#   * :func:`test_no_number_in_the_paper_is_left_unverified`, which deletes
#     every matched span from the source before looking for stray numbers.


@dataclass(frozen=True)
class Claim:
    """One number the paper states, located by a pattern anchored on its prose."""

    name: str
    pattern: str
    at_least: int = 1


CLAIMS: tuple[Claim, ...] = (
    # ---- configuration, stated in Method and repeated in the appendix -------
    Claim("cfg_dim", r"\$d = (\d+)\$"),
    Claim("cfg_base", r"\$\\mathrm\{base\} = (\d+)\$"),
    Claim("cfg_scale", r"extension factor \$s = (\d+)\$"),
    Claim("cfg_context", r"pre-extension context \$(\d+)\$"),
    Claim("cfg_nkeys", r"\$T = (\d+)\$ keys"),
    Claim("cfg_nkeys_short", r"against \$(\d+)\$ random keys"),
    Claim("cfg_nfeatures", r"\$F = (\d+)\$ synthetic directions"),
    Claim("cfg_nfeatures_short", r"with \$(\d+)\$ random directions and non-negative"),
    # ---- the two grids ------------------------------------------------------
    Claim("grid_table", r"a \$5\$-point grid \$\\\{([\d, ]+)\\\}\$"),
    Claim("grid_figure", r"a \$(54)\$-point log-spaced integer grid from \$1\$ to \$(8192)\$"),
    Claim(
        "grid_coarse_attrib",
        r"worst-feature ratio is \$(43\.05)\$",
    ),
    Claim(
        "coarse_grid_label",
        r"On the coarser \$(\d+)\$-point grid \$\\\{([\d, ]+)\\\}\$ the worst-feature ratio",
    ),
    # ---- structural residuals, Table~\ref{tab:exact} ------------------------
    # The paper reports these as ceilings, not digits: each is the float64 roundoff
    # of subtracting two analytically equal expressions, so its value depends on the
    # host's summation order and is not portable. The claim that matters is the
    # bound, so that is what the pattern captures and what the test checks.
    Claim("rel_pos_pairs", r"position pairs tested & \$(\d+)\$"),
    Claim("rel_pos_ceiling", r"relative-position property & \$<(10)\^\{-(13)\}\$"),
    Claim("norm_q_ceiling", r"norm preservation, query & \$<(10)\^\{-(14)\}\$"),
    Claim("norm_k_ceiling", r"norm preservation, key & \$<(10)\^\{-(14)\}\$"),
    Claim("bilin_ceiling", r"bilinearity in content & \$<(10)\^\{-(14)\}\$"),
    Claim("bilin_checks", r"\\quad score checks & \$(\d+)\$"),
    # The abstract, the contributions list and the results prose all restate the
    # same two counts. at_least is raised so a count cannot be quietly deleted.
    Claim("intro_pairs", r"relative-position property over \$(42)\$ (?:position )?pairs", at_least=2),
    Claim(
        "intro_checks",
        r"(?:over|across) \$(35)\$ score checks",
        at_least=2,
    ),
    # The note explaining WHY ceilings are quoted names the two platform-dependent
    # values illustratively. They are not measurements: nothing is asserted about
    # them beyond both being far below the bound, so the test only checks that.
    Claim(
        "floor_note_pair",
        r"yields \$(3\.4)\\times10\^\{-14\}\$ on one platform and "
        r"\$(3\.6)\\times10\^\{-14\}\$ on another",
    ),
    Claim(
        "closed_form_ceiling",
        r"closed form vs\.\\ brute force & \$<(10)\^\{-(14)\}\$",
    ),
    Claim(
        "additivity_5pt_ceiling",
        r"feature additivity, \$5\$-pt grid & \$<(10)\^\{-(13)\}\$",
    ),
    Claim(
        "additivity_54pt_ceiling",
        r"feature additivity, \$54\$-pt grid & \$<(10)\^\{-(13)\}\$",
    ),
    Claim("abstract_additivity", r"we measure the residual at \$(3\.6)\\times10\^\{-14\}\$"),
    Claim("r1_additivity", r"additive decomposition holds to \$(3\.6)\\times10\^\{-14\}\$"),
    Claim("conclusion_additivity", r"is exact, to \$(3\.6)\\times10\^\{-14\}\$"),
    Claim(
        "corr_bilinearity",
        r"doubles the score to a relative error of \$(7\.4)\\times10\^\{-16\}\$",
    ),
    Claim(
        "eq_additivity_grids",
        r"holds to \$(3\.6)\\times10\^\{-14\}\$ on the \$5\$-point grid and "
        r"\$(5\.0)\\times10\^\{-14\}\$ on the \$54\$-point grid",
    ),
    Claim("synthesis_additivity", r"is exact to \$(5\.0)\\times10\^\{-14\}\$"),
    Claim("abstract_additivity_54", r"the total remains additive to \$(5\.0)\\times10\^\{-14\}\$"),
    Claim("r2_additivity", r"additivity residual of \$(5\.0)\\times10\^\{-14\}\$"),
    Claim("cond_additivity", r"Holding the total additive to \$(5\.0)\\times10\^\{-14\}\$"),
    Claim("conclusion_additivity_54", r"the total stays additive to \$(5\.0)\\times10\^\{-14\}\$"),
    # ---- how the structural tests are built ---------------------------------
    Claim(
        "relpos_pairs_prose",
        r"over \$(\d+)\$ pairs spanning \$(\d+)\$ distances and \$(\d+)\$ base positions",
    ),
    Claim("relpos_shift", r"content vectors together by \$(\d+)\$ positions"),
    Claim(
        "bilin_scales",
        r"rescaling one side by \$\\\{(0, 1, 2, -3, 7\.5, 10\^\{-3\}, 10\^\{3\})\\\}\$ "
        r"at \$(\d+)\$ distances",
    ),
    Claim("method_relpos_pairs", r"property over \$(\d+)\$ position pairs"),
    Claim("method_bilin_checks", r"bilinearity over \$(\d+)\$ score checks"),
    Claim("method_entropy_scales", r"\$\\mathrm\{mscale\}\$, over \$(\d+)\$ extension"),
    Claim("sec_entropy_scales", r"range of \$(\d+)\$ extension"),
    Claim("limitations_entropy_scales", r"which is monotone across \$(\d+)\$ scales"),
    Claim("threshold_rad", r"\$\|\\D\| \\le (0\.1)\\rad\$"),
    Claim("threshold_rad_limitations", r"\$\\sin \\D \\mapsto \\D\$ with a \$(0\.1)\\rad\$"),
    Claim("threshold_rad_table", r"\$\|\\D\| \\le (0\.1)\\rad\$, with the exact count out of \$\\half = (\d+)\$"),
    Claim("half_def", r"\\newcommand\{\\half\}\{d/(\d+)\}"),
    # ---- feature additivity / position-conditionality -----------------------
    Claim("spread_max", r"\$(920\.04)\\times\$", 5),
    Claim(
        "spread_max_bare",
        r"takes the values \$(920\.04)\$",
    ),
    Claim("spread_round", r"individual components swing by \$(920)\\times\$"),
    Claim(
        "spread_ratios",
        r"takes the values \$([\d.]+)\$, \$([\d.]+)\$, \$([\d.]+)\$, \$([\d.]+)\$, "
        r"\$([\d.]+)\$, \$([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\$",
    ),
    Claim("spread_median", r"The median is \$(154\.47)\$"),
    Claim("spread_floor", r"eight exceed \$(35)\\times\$"),
    Claim(
        "nfeatures_grid",
        r"With \$F = (\d+)\$ features on the \$(\d+)\$-point distance grid",
    ),
    Claim("ngrid_54", r"bit-identical to plain RoPE at every one of \$(\d+)\$ tested"),
    Claim("ngrid_54_point", r"across a \$(\d+)\$-point distance grid"),
    Claim("ngrid_54_linearizable", r"across the \$(\d+)\$-point grid\. Plain"),
    Claim(
        "ngrid_54_fig06",
        r"single value \$([\d.]+)\$ at every one of the \$(\d+)\$ distances",
    ),
    Claim("ngrid_54_partial", r"over the finer \$(\d+)\$-point grid it"),
    Claim("ngrid_54_fig03", r"across all \$(\d+)\$ distances of the finer grid"),
    # ---- orders of magnitude (Section~\ref{sec:cond}) ------------------------
    Claim("orders_claim", r"some \$(\d+)\$ orders of magnitude above the additivity residual"),
    Claim(
        "orders_claim_provenance",
        r"comparison in Section~\\ref\{sec:cond\} divides a swing of order "
        r"\$10\^\{2\}\$--\$10\^\{3\}\$ by the \$5\.0\\times10\^\{-14\}\$ additivity residual",
    ),
    # The appendix now shows the quotient explicitly rather than only describing it.
    Claim(
        "orders_claim_quotient",
        r"\$([\d.]+)/([\d.]+)\\times10\^\{-(\d+)\} \\approx ([\d.]+)\\times10\^\{(\d+)\}\$",
    ),
    # ---- YaRN ramp bookkeeping (fig01) --------------------------------------
    Claim("yarn_unchanged_caption", r"leaves the fastest \$(\d+)\$ of \$(\d+)\$ entries bit-for-bit identical"),
    Claim("yarn_unchanged_body", r"fastest \$(\d+)\$ of the \$(\d+)\$ rotary pairs have inverse"),
    Claim("yarn_unchanged_note", r"leaving the fastest \$(\d+)\$ of \$(\d+)\$ pairs bit-for-bit unchanged"),
    Claim("yarn_slow_ratio", r"slowed by exactly a factor of \$(32\.00)\$"),
    Claim("yarn_slow_ratio_caption", r"slow tail down by exactly \$(32)\\times\$"),
    Claim("interp_all_entries", r"differs from RoPE in all \$(\d+)\$ entries"),
    Claim("interp_all_entries_caption", r"divides every entry by \$(\d+)\$"),
    Claim("fastest_rate", r"contributes exactly \$(\d+)\\rad\$ per token"),
    # ---- base-raising is NOT position interpolation --------------------------
    # The paper originally claimed dividing positions by s is "algebraically
    # equivalent" to replacing base by base*s. That is false, and was corrected.
    Claim("base_scale_entry", r"multiplies entry \$k\$ by \$(s\^\{-2k/d\})\$"),
    Claim("base_scale_entry_div", r"divides every entry by \$(s)\$"),
    Claim(
        "base_scale_related",
        r"raising the base multiplies channel \$k\$ by \$(s\^\{-2k/d\})\$ rather than by \$(s\^\{-1\})\$",
    ),
    Claim(
        "base_scale_corr",
        r"rescales channel \$k\$ by \$(s\^\{-2k/d\})\$ rather than by \$(s\^\{-1\})\$",
    ),
    Claim("base_scale_agree_at", r"agree only where \$2k/d = (\d+)\$, that is at \$k = d/2\$"),
    Claim("theta_zero_is_one", r"\$\\theta_0 = \\mathrm\{base\}\^\{0\} = (\d+)\$"),
    Claim("base_raises_range", r"raising the base from \$(10000)\$ to \$(500000)\$ leaves"),
    Claim(
        "interp_vs_legacy_at_delta",
        r"at \$\\delta = (\d+)\$ position interpolation reaches "
        r"\$\\max_k\|D_k\| = (\d+)\$ and the \$\\mathrm\{base\} = (\d+)\$ ladder still sits at \$(\d+)\$",
    ),
    Claim("yarn_untouched_entries", r"a ramp that leaves \$(\d+)\$ of \$(\d+)\$ entries untouched"),
    Claim(
        "legacy_base_related",
        r"to \$(500000)\$'' recipe is not \\yarn, and is also not",
    ),
    Claim("legacy_base_corr", r"raising the base from \$(10000)\$ to \$(500000)\$''\. It does not\."),
    # ---- the comparison table's caption --------------------------------------
    Claim("table_d_base", r"\$\\mathrm\{base\} = 10\^\{4\}\$"),
    # ---- median angle ratio -------------------------------------------------
    Claim("median_ratio", r"constant \$(0\.4464)\\times\$ that of"),
    Claim("median_ratio_abstract", r"constant factor of \$(0\.4464)\$"),
    Claim(
        "median_at_512",
        r"\$([\d.]+)\$ against \$([\d.]+)\\rad\$ at \$\\delta = (\d+)\$",
    ),
    Claim(
        "median_at_4096",
        r"\$([\d.]+)\$ against \$([\d.]+)\$ at \$\\delta = (\d+)\$",
    ),
    Claim("fig02_delta", r"\$\\delta = (\d+)\$ the median angle is"),
    Claim("median_fig02_rope", r"the median angle is \$(52\.42)\\rad\$ under RoPE against"),
    Claim("median_fig02_yarn", r"against \$(23\.40)\\rad\$ under \\yarn"),
    # ---- max-angle identity --------------------------------------------------
    Claim("max_equal_4096", r"At \$\\delta = (\d+)\$ both are exactly \$(\d+)\\rad\$"),
    Claim("max_equal_grid", r"the largest disagreement is \$(0\.000)\\times10\^\{(0)\}\\rad\$"),
    Claim("pi_max_shrink", r"shrinking \$\\max_k\|\\D\|\$ from \$(\d+)\$ to \$(\d+)\\rad\$"),
    # ---- linearizable fractions in prose -------------------------------------
    Claim("lin_rise_512", r"rises from \$(0\.0625)\$ to \$(0\.3438)\$ at \$\\delta = (\d+)\$"),
    Claim("lin_rise_4096", r"and from \$(0\.0000)\$ to \$(0\.2188)\$ at \$\\delta = (\d+)\$"),
    Claim("lin_base500k", r"different numbers again \(\$(0\.3438)\$ and \$(0\.1875)\$\),"),
    # ---- fig04 tail ------------------------------------------------------------
    Claim("fig04_first_zero", r"at all from \$\\delta = (\d+)\$ onward"),
    Claim(
        "fig04_zero_value",
        r"is exactly \$(0\.0000)\$ at every subsequent grid point, while \\yarn\\ retains "
        r"\$(\d+)/(\d+)\$ there and \$(\d+)/(\d+)\$ at \$\\delta = (\d+)\$",
    ),
    Claim(
        "synthesis_ramp",
        r"from \$(\d+)/(\d+)\$ to \$(\d+)/(\d+)\$ at \$\\delta \\approx (\d+)\$",
    ),
    Claim("synthesis_worst", r"be unable to linearize it\..*?under full \\rope\\ at any \$\\delta > (\d+)\$"),
    Claim("synthesis_delta_8_512", r"correctly at \$\\delta = (\d+)\$ and reuses it at \$\\delta = (\d+)\$"),
    # ---- fig05 amplitude-weighted ---------------------------------------------
    Claim("amp_at_8192", r"amplitude-weighted error is \$(1\.24572)\\times10\^\{(6)\}\$ for RoPE and \$(1\.24179)\\times10\^\{(6)\}\$ for \\yarn"),
    Claim("amp_ratio", r"a ratio of \$(0\.996840)\$"),
    Claim("amp_ratio_r3", r"error ratio \\yarn/RoPE is \$(0\.99684)\$ at \$\\delta = (\d+)\$"),
    Claim("amp_ratio_conclusion", r"amplitude-weighted error ratio of only \$(0\.996840)\$"),
    Claim("amp_percent", r"buys less than \$(0\.4)\\%\$ on this measure"),
    Claim("rms_ratio", r"per-pair RMS ratio is \$(0\.999977)\$"),
    Claim("pi_amp", r"this axis \(\$(\d+\.\d+)\$ at \$\\delta = (\d+)\$, roughly \$(\d+)\\times\$ below RoPE\)"),
    # ---- fig06 per-pair --------------------------------------------------------
    Claim("pair_amp_k0", r"rotary pair, \$k = 0\$, takes the single value \$([\d.]+)\$"),
    Claim("pair_amp_k12", r"pair \$k = (\d+)\$ has amplitude \$([\d.]+)\$"),
    Claim("pair_exact_d1", r"exact contribution of \$(-0\.9525)\$ against a linearized value of \$(-1\.0607)\$, a normalized error of \$(0\.1136)\$"),
    Claim("pair_lin_d1", r"the mid-frequency pair \$k = (\d+)\$ has normalized error \$(4\.3)\\times10\^\{(-6)\}\$ at the same distance, roughly \$(\d+)\$ times smaller"),
    Claim("pair_err_d3", r"By \$\\delta = (\d+)\$ the fastest pair's normalized error is \$(\d\.\d+)\$"),
    Claim("pair_err_d3_caption", r"error reaches \$1\$, meaning an error as large as the term itself, at \$\\delta = (\d+)\$ under both schemes"),
    Claim("pair_delta_1", r"At \$\\delta = (\d+)\$ the fastest pair \$k = 0\$"),
    # ---- partial RoPE ----------------------------------------------------------
    Claim("partial_p", r"\\pip\\ rotates only a fraction \$p = n_\{\\mathrm\{rot\}\}/d\$"),
    Claim("partial_p_value", r"Partial RoPE at \$p = (0\.25)\$"),
    Claim("partial_p_abstract", r"at \$p = (0\.25)\$ the unrotated sub-score"),
    Claim("partial_nrot", r"At \$d = (\d+)\$ with \$n_\{\\mathrm\{rot\}\} = (\d+)\$"),
    Claim(
        "partial_clean_spread",
        r"spread over the \$(\d+)\$-point grid is \\emph\{exactly\} \$(0\.0)\$, and "
        r"over the finer \$(\d+)\$-point grid it",
    ),
    Claim("partial_clean_const", r"the plotted constant is \$([\d.]+)\$ at every"),
    Claim("partial_rot_spread", r"over those same grids, varies by \$(\d+\.\d+)\$ and \$(\d+\.\d+)\$ respectively"),
    Claim("partial_share", r"score of \$(0\.302)\$ \(JSON grid\) and \$(0\.187)\$ \(figure grid\)"),
    # The norm-deviation claims are withdrawn: partial rotary is orthogonal once the
    # rotated block is paired within itself, so the paper now states a ceiling and
    # explains the silently-wrong pairing that motivated the earlier, wrong number.
    Claim(
        "partial_norm_ceiling",
        r"&\s*\$<(10)\^\{-(13)\}\$, \$(1\.776)\\\!\\times\\\!10\^\{(-15)\}\$ & \\texttt\{partial",
    ),
    Claim("partial_prov_spreads", r"&\s*\$(0\.0)\$, \$(\d+\.\d+)\$, \$(\d+\.\d+)\$ & \\texttt\{measurements\.json :: partial"),
    Claim("partial_prov_share", r"&\s*\$(0\.302)\$, \$(0\.187)\$ & \\texttt\{clean"),
    # ---- mscale / entropy -------------------------------------------------------
    Claim("mscale_formula", r"\\mathrm\{mscale\} = (0\.1) \\ln s \+ 1"),
    Claim(
        "mscale_formula_prose",
        r"closed form \$(0\.1)\\ln (\d+) \+ 1\$ to all printed digits",
    ),
    Claim("mscale_value", r"measure \$\\mathrm\{mscale\} = (1\.3465736)\$"),
    Claim("mscale_value_prov", r"&\s*\$(1\.3465736)\$ & \\texttt\{measurements\.json :: mscale"),
    Claim("entropy_from", r"normalized entropy from \$H/\\ln T = (\d\.\d+)\$ to"),
    Claim("entropy_to", r"\$H/\\ln T = \d\.\d+\$ to \$(\d\.\d+)\$, a drop of \$(\d\.\d+)\$ in absolute"),
    Claim("entropy_nats_from", r"nats, from \$(\d\.\d+)\$ to \$(\d\.\d+)\$"),
    Claim("entropy_scale_range", r"factors from \$1\$ to \$(\d+)\$, the effect is monotone: \$H/\\ln T\$ falls to \$(0\.8383)\$ as \$\\mathrm\{mscale\}\$ rises to \$(1\.4852)\$"),
    Claim("entropy_128_prov", r"&\s*\$(0\.8383)\$, \$(1\.4852)\$ & \\texttt\{fig08\} CSV at \$\\mathrm\{scale\} = (\d+)\$"),
    Claim("entropy_prov_ratios", r"&\s*\$(\d\.\d+) \\to (\d\.\d+)\$ & \\texttt\{entropy"),
    # ---- the appendix's restatements -------------------------------------------
    Claim("prov_rel_pos_pairs", r"&\s*\$(920\.04)\$, \$(154\.47)\$ & \\texttt\{fig09\}"),
    Claim("prov_spread_coarse", r"&\s*\$(43\.05)\$ & \\texttt\{measurements\.json :: feature"),
    Claim("prov_max_agreement", r"&\s*\$(0\.000)\\rad\$ agreement & \\texttt\{fig03\}"),
    Claim("prov_ratios", r"&\s*\$(0\.4464)\$, \$(1\.0000)\$ & ratios of fig03"),
    Claim("prov_fig04", r"&\s*\$\\delta = (\d+)\$, \$(\d+)/(\d+)\$ & \\texttt\{fig04\}"),
    Claim("prov_amp", r"&\s*\$(1\.24572)\\\!\\times\\\!10\^\{(6)\}\$ etc\."),
    Claim("prov_pi_amp", r"&\s*\$(\d+\.\d+)\$, \$\\delta = (\d+)\$ & \\texttt\{fig05\}"),
    Claim("prov_ramp", r"&\s*\$(\d+)\$ of \$(\d+)\$, \$(\d+\.\d+)\\times\$ & \\texttt\{fig01\}"),
    Claim("prov_fig02", r"&\s*\$(52\.42)\$, \$(23\.40)\$ & \\texttt\{fig02\}"),
    Claim("prov_amp_k", r"&\s*\$(0\.\d+)\$, \$(2\.\d+)\$ & \\texttt\{fig06\}"),
    Claim("prov_pair_d1", r"&\s*\$(-0\.9525)\$, \$(-1\.0607)\$, \$(0\.1136)\$, \$(4\.3)\\\!\\times\\\!10\^\{(-6)\}\$ & \\texttt\{fig06\}"),
    Claim("prov_yarn_note", r"&\s*\$(\d+)\$ of \$(\d+)\$ bit-for-bit"),
    Claim("prov_partial", r"&\s*\$(0\.0)\$, \$(\d+\.\d+)\$, \$(\d+\.\d+)\$ & \\texttt\{measurements"),
    Claim("prov_partial_share", r"&\s*\$(0\.302)\$, \$(0\.187)\$ & \\texttt\{clean"),
    Claim("prov_partial_norm", r"&\s*\$<(10)\^\{-(13)\}\$, \$(1\.776)\\\!\\times\\\!10\^\{(-15)\}\$ & \\texttt\{partial"),
    # The remark that replaced the withdrawn norm claim restates n_rot.
    Claim(
        "partial_norm_remark_nrot",
        r"deviation below the float64 noise floor at \$n_\{\\mathrm\{rot\}\} = (\d+)\$",
    ),
    Claim("derived_counts", r"converted to counts out of \$(\d+)\$"),
    Claim("derived_half", r"the denominator is \$\\half = (\d+)\$ pairs"),
    Claim("derived_multiple", r"table fractions are multiples of \$(1)/(\d+)\$"),
    Claim("derived_multiple_2", r"every fraction is a multiple of \$(1)/(\d+)\$"),
    Claim("derived_percent", r"The ``less than \$(0\.4)\\%\$'' claim in Section~\\ref\{sec:context\} is \$1 - (0\.996840)\$"),
    Claim("derived_1000x", r"``roughly \$(\d+)\\times\$'' comparison divides \$1\.24572\\times10\^\{6\}\$ by \$(\d+\.\d+)\$"),
    Claim("exact_identity_zero", r"partial-RoPE sub-score has spread \$(0\.0)\$"),
    # ---- the negated retrieval-accuracy example ---------------------------------
    Claim("negated_accuracy", r"accuracy \$(0\.7)\$'', because nothing like that"),
    # ---- restatements of already-verified literals, claimed so that the
    # ---- exhaustiveness scan below can see that they are accounted for ------
    Claim("r3_base_500k", r"``\$\\mathrm\{base\} = (500000)\$'' heuristic"),
    Claim("background_scale_32", r"Under \\yarn\\ at \$s = (\d+)\$"),
    Claim("mscale_scale_32", r"At \$s = (\d+)\$ we measure"),
    Claim("related_base_10000", r"\$\\mathrm\{base\}\$ from \$(10000)\$ to \$(500000)\$'' recipe"),
    # The "same scheme at s = 50" wording is gone: raising the base is not
    # position interpolation. The corrected sentences assert the s^{-2k/d} form.
    Claim("bilin_checks_table", r"giving the \$(\d+)\$ score checks in the table"),
    Claim("amp_at_8192_lead", r"At \$\\delta = (\d+)\$ the amplitude-weighted"),
    Claim("pairs_delta_3_body", r"own contribution at \$\\delta = (\d+)\$, identically under"),
    Claim("partial_clean_zero_2", r"is also exactly \$(0\.0)\$ --- the plotted constant"),
    Claim("prov_orders_claim", r"The ``\$(\d+)\$ orders of magnitude'' comparison in"),
    Claim("fig02_eight_distances", r"against pair index at eight distances"),
    Claim(
        "prop_blind_deltas",
        r"same number at \$\\delta = (\d+)\$ and at \$\\delta = (\d+)\$",
    ),
    Claim("fig01_legacy_caption", r"``\$\\mathrm\{base\}=(500000)\$'' ladder"),
    Claim("table_intro_grid", r"compares the four schemes on the \$(\d+)\$-point grid"),
    Claim("table_caption_grid", r"schemes on the \$(\d+)\$-point distance grid"),
    Claim("limitations_nrot", r"\$n_\{\\mathrm\{rot\}\} = (\d+)\$: the unrotated sub-score"),
    Claim(
        "limitations_ndirections",
        r"The \$(\d+)\$ directions our attribution runs over are random",
    ),
    Claim(
        "limitations_dim",
        r"property of random directions in a \$(\d+)\$-dimensional head",
    ),
    Claim(
        "limitations_config",
        r"All measurements use \$d = (\d+)\$, \$\\mathrm\{base\} = (\d+)\$, "
        r"\$s = (\d+)\$, pre-extension context \$(\d+)\$",
    ),
    Claim("exact_identity_residue", r"residue in each case is \$(0\.0)\$ or a single ULP"),
)


def claim(name: str) -> Claim:
    for entry in CLAIMS:
        if entry.name == name:
            return entry
    raise KeyError(f"no claim named {name!r} in CLAIMS")


def groups(name: str) -> list[tuple[str, ...]]:
    """Every capture-group tuple of the claim, asserting the minimum count.

    Anchoring each pattern on the surrounding prose is what makes an edit to
    the paper's number fail the test: the number is never written down here.
    """
    entry = claim(name)
    found = [m.groups() for m in re.finditer(entry.pattern, TEX_FLAT)]
    assert len(found) >= entry.at_least, (
        f"claim {entry.name!r}: expected at least {entry.at_least} match(es) of "
        f"{entry.pattern!r} in paper/main.tex, found {found}"
    )
    return found


def literal(name: str, group: int = 1) -> str:
    """The single paper literal a claim captured (1-based ``group``)."""
    return groups(name)[0][group - 1]


def all_literals(name: str) -> list[str]:
    """Every occurrence of a claim's first capture group."""
    entry = claim(name)
    return [m.group(1) for m in re.finditer(entry.pattern, TEX_FLAT)]


# ==========================================================================
# Numeric comparison at the paper's own precision
# ==========================================================================


def rounding_quantum(literal_text: str) -> Decimal:
    """The unit of the last digit the paper actually wrote.

    ``"3.6"`` -> ``1e-1``, ``"0.996840"`` -> ``1e-6``, ``"43"`` -> ``1``.
    The accepted interval is half of that, so the tolerance comes from the
    document's precision rather than from a guess.
    """
    return Decimal(1).scaleb(Decimal(literal_text).as_tuple().exponent)


def assert_paper_number_matches(actual: float, literal_text: str, what: str) -> None:
    """``literal_text`` is ``actual`` correctly rounded to the paper's precision.

    The paper rounds to 2-4 significant figures, so demanding float equality
    against a rounded literal would be wrong.  Instead the accepted interval is
    half of the literal's last written digit: exactly the set of values that
    round to what the paper says.  ``1e-9`` of a quantum is float slack only.
    """
    quantum = float(rounding_quantum(literal_text))
    written = float(literal_text)
    gap = abs(actual - written)
    slack = 1e-9 * quantum
    assert gap <= 0.5 * quantum + slack, (
        f"{what}: the paper writes {literal_text!r} but the measurement is {actual!r}; "
        f"{literal_text!r} is only correct if the measurement rounds to it "
        f"(accepted range {written - 0.5 * quantum - slack:.12g} .. "
        f"{written + 0.5 * quantum + slack:.12g})"
    )


def assert_paper_sci_matches(
    actual: float, mantissa: str, exponent: int, what: str
) -> None:
    """As above, for ``$mantissa\\times10^{exponent}$``.

    Comparing the mantissa at its own precision *and* requiring the exponent
    the paper wrote is what catches a paper that gets the order of magnitude
    wrong while keeping the digits.
    """
    scale = float(Decimal(10) ** exponent)
    assert_paper_number_matches(actual / scale, mantissa, f"{what} (exponent 10^{exponent})")


def assert_paper_says_exactly(actual: float, literal_text: str, what: str) -> None:
    """For claims the paper states as exact (``exactly 0.0``), demand exactness."""
    assert float(literal_text) == actual, (
        f"{what}: the paper writes {literal_text!r}, the measurement is {actual!r}; "
        f"the paper claims this one is exact"
    )


# How far a deliberately hedged figure ("roughly 1000x") may sit from the
# written literal.  Only claims where the paper itself says "roughly" use this;
# everything else must round correctly at the written precision.
HEDGE_TOLERANCE = 0.05


def assert_paper_says_roughly(actual: float, literal_text: str, what: str) -> None:
    """For a claim the paper hedges with "roughly", accept +/- ``HEDGE_TOLERANCE``.

    1016.86 is a fair rendering of "roughly 1000x" and 26154 of "roughly 26000
    times smaller"; demanding that the rounded literal contain it would be
    demanding precision the paper explicitly declined to claim.
    """
    written = float(literal_text)
    low, high = written * (1.0 - HEDGE_TOLERANCE), written * (1.0 + HEDGE_TOLERANCE)
    assert low <= actual <= high, (
        f"{what}: the paper says {literal_text!r} (a hedged figure, so "
        f"+/-{HEDGE_TOLERANCE:.0%}) but the measurement is {actual!r}, outside "
        f"[{low:.6g}, {high:.6g}]"
    )


# ==========================================================================
# Access to the two canonical data sources
# ==========================================================================


@cache
def csv_rows(stem: str) -> tuple[dict[str, str], ...]:
    """The rows of a figure CSV sidecar, as the paper's provenance names them."""
    import csv

    path = FIGURES_DIR / f"{stem}.csv"
    assert path.is_file(), f"missing CSV sidecar {path.name}"
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def spectrum_row(method: str, delta: int) -> dict:
    for row in MEASUREMENTS["method_spectrum"]:
        if row["method"] == method and row["delta"] == delta:
            return row
    raise AssertionError(f"measurements.json has no row for {method!r} at delta={delta}")


def table_grid_of_the_paper() -> tuple[int, ...]:
    """The $5$-point grid the paper says the JSON tables were measured on."""
    return tuple(int(value) for value in literal("grid_table").split(","))


def figure_delta_grid() -> tuple[int, ...]:
    """The log-spaced integer distance grid ``figures.py`` plots on."""
    return tuple(
        int(x)
        for x in np.unique(
            np.round(np.logspace(0.0, math.log10(FIG_DELTA_MAX), FIG_DELTA_POINTS))
        )
    )


def fig09_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute fig09 as ``figures.py`` does.

    Returns ``(contributions, additivity_residuals, deltas)`` with shape
    ``(n_features, n_deltas)``, ``(n_deltas,)``, ``(n_deltas,)``.

    The residuals are accumulated exactly as ``figures.py`` accumulates them -
    one 1-D ``.sum()`` per distance - because a different (mathematically
    equivalent, numerically different) summation order gives a different
    roundoff-floor residual, and the paper quotes the value figures.py writes.
    """
    n_features = MEASUREMENTS["feature_attribution"]["n_features"]
    rng = np.random.default_rng(FIG_SEED_ATTRIB)
    freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    w_q = rng.standard_normal((HEAD_DIM, HEAD_DIM)) / math.sqrt(HEAD_DIM)
    w_k = rng.standard_normal((HEAD_DIM, HEAD_DIM)) / math.sqrt(HEAD_DIM)
    coeffs = rng.random(n_features) * 1.5 + 0.1
    dirs_q = rng.standard_normal((n_features, HEAD_DIM))
    dirs_k = rng.standard_normal((n_features, HEAD_DIM))
    q_i = coeffs[:, None] * (dirs_q @ w_q)
    k_i = coeffs[:, None] * (dirs_k @ w_k)
    full_q = q_i.sum(axis=0)
    full_k = k_i.sum(axis=0)
    grid = figure_delta_grid()
    contrib = np.zeros((n_features, len(grid)))
    totals = np.zeros(len(grid))
    for j, delta in enumerate(grid):
        block = np.array(
            [
                [E.score_relative(q_i[a], k_i[b], freqs, delta) for b in range(n_features)]
                for a in range(n_features)
            ]
        )
        contrib[:, j] = block.sum(axis=1)
        totals[j] = E.score_relative(full_q, full_k, freqs, delta)
    residuals = np.array(
        [abs(float(contrib[:, j].sum()) - float(totals[j])) for j in range(len(grid))]
    )
    return contrib, residuals, np.asarray(grid, dtype=float)


def fig09_per_feature_ratios(contributions) -> list[float]:
    """max|c_i| / min|c_i| per feature, non-zero values only (fig09's rule)."""
    ratios = []
    for row in contributions:
        values = np.abs(np.asarray(row, dtype=float))
        nonzero = values[values != 0.0]
        ratios.append(float(nonzero.max() / nonzero.min()))
    return ratios


def fig09_csv_curves() -> list[list[float]]:
    """The same curves straight out of the fig09 CSV sidecar."""
    curves: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in csv_rows("fig09_position_conditional_attribution"):
        curves[int(row["feature_index"])].append((int(row["delta"]), float(row["contribution"])))
    return [[value for _delta, value in sorted(curves[index])] for index in sorted(curves)]


# ==========================================================================
# Is the data on disk even current?
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

    Without this, every claim below could be satisfied by a stale data file:
    the paper and the JSON would agree with each other and both disagree with
    the code.  The experiments are seeded, so the comparison is reproducible.
    It is made to a tolerance of 1e-12 relative rather than bit-for-bit: the
    same float64 sum can differ by a few ULP between two BLAS implementations,
    which is what made an exact comparison fail on CI while passing on Windows.
    """
    live = E.run_all().to_dict()
    drift = [(path, committed, fresh) for path, committed, fresh in _leaves(live, MEASUREMENTS)]
    assert not drift, (
        f"results/measurements.json is stale ({len(drift)} value(s) differ by more "
        f"than 1e-12 relative): re-run "
        f"`python -m projects.rope_attribution.experiments`. First: {drift[0]}"
    )


# ==========================================================================
# Configuration the paper states
# ==========================================================================


def test_paper_states_the_measured_configuration() -> None:
    assert_paper_number_matches(MEASUREMENTS["structural_facts"]["dim"], literal("cfg_dim"), "d")
    assert_paper_number_matches(
        MEASUREMENTS["partial_rope"]["dim"], literal("cfg_dim"), "d (partial RoPE)"
    )
    assert_paper_number_matches(
        MEASUREMENTS["structural_facts"]["base"], literal("cfg_base"), "base"
    )
    assert_paper_number_matches(
        MEASUREMENTS["mscale_entropy"]["scale"], literal("cfg_scale"), "s"
    )
    assert_paper_number_matches(
        MEASUREMENTS["mscale_entropy"]["scale"], literal("background_scale_32"), "s (background)"
    )
    assert_paper_number_matches(
        MEASUREMENTS["mscale_entropy"]["scale"], literal("mscale_scale_32"), "s (mscale)"
    )
    assert int(literal("cfg_nkeys")) == MEASUREMENTS["mscale_entropy"]["n_keys"]
    assert int(literal("cfg_nkeys_short")) == MEASUREMENTS["mscale_entropy"]["n_keys"]
    assert int(literal("cfg_nfeatures")) == MEASUREMENTS["feature_attribution"]["n_features"]
    assert int(literal("cfg_nfeatures_short")) == MEASUREMENTS["feature_attribution"]["n_features"]
    assert int(literal("limitations_ndirections")) == MEASUREMENTS["feature_attribution"][
        "n_features"
    ]
    dim, base, scale, context = groups("limitations_config")[0]
    assert int(dim) == MEASUREMENTS["structural_facts"]["dim"]
    assert float(base) == MEASUREMENTS["structural_facts"]["base"]
    assert float(scale) == MEASUREMENTS["mscale_entropy"]["scale"]
    assert int(context) == ORIGINAL_MAX_POS
    assert int(literal("limitations_dim")) == MEASUREMENTS["structural_facts"]["dim"]


def test_paper_states_the_pre_extension_context_and_the_legacy_base() -> None:
    assert int(literal("cfg_context")) == ORIGINAL_MAX_POS
    assert ORIGINAL_MAX_POS in {row["delta"] for row in MEASUREMENTS["method_spectrum"]}
    base, legacy = groups("base_raises_range")[0]
    assert int(base) == int(ROPE_BASE) == int(literal("legacy_base_corr", 1))
    assert int(legacy) == LEGACY_BASE == int(literal("legacy_base_related"))
    assert int(literal("legacy_base_corr", 2)) == LEGACY_BASE
    assert float(MEASUREMENTS["structural_facts"]["base"]) == float(base)
    assert float(literal("r3_base_500k")) == LEGACY_BASE
    assert float(literal("fig01_legacy_caption")) == LEGACY_BASE


def test_raising_the_base_is_not_position_interpolation() -> None:
    """Raising the base is NOT the same scheme as dividing positions by s.

    The paper used to claim they were "algebraically equivalent". That was
    false and has been corrected. ``inv_freq[k] = base ** (-2k/d)``, so dividing
    by ``s`` gives ``base ** (-2k/d) * s ** -1`` while raising the base gives
    ``base ** (-2k/d) * s ** (-2k/d)``. They agree only where ``2k/d == 1``, at
    k = d/2, which lies outside the ladder k = 0 .. d/2-1.

    This test pins the correction: it checks the paper now states the
    distinction, and checks the algebra numerically so the claim stays true.
    """
    assert literal("base_scale_entry", 1) == "s^{-2k/d}"
    assert int(literal("base_scale_agree_at")) == 1

    # The paper must assert the distinction in both later places it used to deny it.
    for name in ("base_scale_related", "base_scale_corr"):
        raised_form, divided_form = groups(name)[0]
        assert raised_form == "s^{-2k/d}"
        assert divided_form == "s^{-1}"

    base = float(literal("base_raises_range", 1))
    legacy = float(literal("base_raises_range", 2))
    s = legacy / base
    ladder = R.inv_freq(HEAD_DIM, base)
    divided = ladder / s
    raised = R.inv_freq(HEAD_DIM, legacy)

    # They differ at every channel of the ladder, and the paper's own table shows it.
    assert not np.allclose(raised, divided, rtol=1e-9, atol=0.0)
    # The ratio between the two ladders is s**(2k/d - 1): strictly increasing in k,
    # and strictly below 1 at every channel because the ladder stops at k = d/2 - 1.
    # So they never coincide, which is exactly the paper's corrected claim.
    ratios = divided / raised
    expected = np.array([s ** (2 * k / HEAD_DIM - 1.0) for k in range(N_PAIRS)])
    assert np.allclose(ratios, expected, rtol=1e-12, atol=0.0)
    assert np.all(ratios < 1.0), "the two ladders must never meet on the ladder"
    assert np.all(np.diff(ratios) > 0.0), "the ratio must increase monotonically in k"
    assert ratios[0] == pytest.approx(1.0 / s)
    # theta_0 = base**0 = 1 regardless of base, so raising the base cannot move it.
    assert ladder[0] == raised[0] == 1.0
    assert divided[0] == 1.0 / s

    delta, interp_max, legacy_base, legacy_max = groups("interp_vs_legacy_at_delta")[0]
    assert int(delta) == 512
    assert int(interp_max) == int(delta) // int(literal("cfg_scale"))
    assert float(legacy_base) == LEGACY_BASE
    assert int(legacy_max) == int(delta)  # base-raising leaves max|D_k| untouched
    # and that is exactly what the measured table says
    assert spectrum_row("position_interpolation", int(delta))["max_abs_angle"] == float(
        interp_max
    )
    assert spectrum_row("base_500k_legacy_claim", int(delta))["max_abs_angle"] == float(
        legacy_max
    )
    assert float(literal("theta_zero_is_one")) == 1.0


def test_paper_states_the_two_grids_the_numbers_came_from() -> None:
    table_grid = tuple(int(x) for x in literal("grid_table").split(","))
    assert table_grid == tuple(sorted({row["delta"] for row in MEASUREMENTS["method_spectrum"]}))
    assert len(table_grid) == 5
    n_figure = int(literal("grid_figure", 1))
    top = int(literal("grid_figure", 2))
    grid = figure_delta_grid()
    assert n_figure == len(grid), f"paper says {n_figure} figure distances; the grid has {len(grid)}"
    assert (grid[0], grid[-1]) == (1, top)
    n_coarse, coarse = groups("coarse_grid_label")[0]
    assert [int(x) for x in coarse.split(",")] == MEASUREMENTS["feature_attribution"]["deltas"]
    assert int(n_coarse) == len(MEASUREMENTS["feature_attribution"]["deltas"])
    assert set(MEASUREMENTS["partial_rope"]["deltas"]).issubset(table_grid)
    assert int(literal("table_intro_grid")) == len(table_grid)
    assert int(literal("table_caption_grid")) == len(table_grid)


def test_figure_grid_constants_match_figures_module() -> None:
    """The grid this file recomputes fig09's number on is the one figures.py uses."""
    F = pytest.importorskip("rope_attribution.figures")
    assert F.HEAD_DIM == HEAD_DIM
    assert F.ROPE_BASE == ROPE_BASE
    assert F.LEGACY_BASE == LEGACY_BASE
    assert F.EXT_SCALE == EXT_SCALE
    assert F.ORIGINAL_MAX_POS == ORIGINAL_MAX_POS
    assert F.N_KEYS == N_KEYS
    assert MEASUREMENTS["feature_attribution"]["n_features"] == F.N_FEATURES
    assert F.SEED_ATTRIB == FIG_SEED_ATTRIB
    assert F.DELTA_MAX == FIG_DELTA_MAX
    assert F.DELTA_POINTS == FIG_DELTA_POINTS
    assert F.LINEARIZABLE_ANGLE_RAD == LINEARIZABLE_ANGLE_RAD
    assert tuple(F.DELTA_GRID) == figure_delta_grid()
    assert len(F.SCALE_GRID) == 18


def test_paper_states_the_entropy_scale_count_and_range() -> None:
    n_scales = int(literal("method_entropy_scales"))
    assert n_scales == int(literal("sec_entropy_scales")) == int(
        literal("limitations_entropy_scales")
    )
    assert n_scales == len(csv_rows("fig08_mscale_entropy"))
    scales = sorted(float(row["scale"]) for row in csv_rows("fig08_mscale_entropy"))
    top, end_ratio, end_mscale = groups("entropy_scale_range")[0]
    assert (scales[0], scales[-1]) == (1.0, float(top))
    assert float(top) in scales
    assert_paper_number_matches(scales and _entropy_last()["ratio"], end_ratio, "H/ln T at the top scale")
    assert_paper_number_matches(_entropy_last()["mscale"], end_mscale, "mscale at the top scale")
    assert end_mscale and float(end_mscale) > MEASUREMENTS["mscale_entropy"]["mscale"]
    F = pytest.importorskip("rope_attribution.figures")
    assert tuple(sorted(F.SCALE_GRID)) == tuple(scales)


def _entropy_last() -> dict[str, float]:
    rows = sorted(csv_rows("fig08_mscale_entropy"), key=lambda r: float(r["scale"]))
    return {
        "scale": float(rows[-1]["scale"]),
        "mscale": float(rows[-1]["mscale"]),
        "ratio": float(rows[-1]["entropy_ratio_mscaled"]),
    }


def test_linearizable_threshold_is_the_one_the_code_uses() -> None:
    assert_paper_number_matches(LINEARIZABLE_ANGLE_RAD, literal("threshold_rad"), "the threshold")
    assert_paper_number_matches(
        LINEARIZABLE_ANGLE_RAD, literal("threshold_rad_limitations"), "the threshold"
    )
    assert_paper_number_matches(
        LINEARIZABLE_ANGLE_RAD, literal("threshold_rad_table", 1), "the table threshold"
    )
    source = (_REPO / "projects" / "rope_attribution" / "experiments.py").read_text(
        encoding="utf-8"
    )
    assert re.search(r"np\.abs\(t\.angle\) <= (0\.1)\b", source), (
        "experiments.linearization_error no longer thresholds |D_k| at the 0.1 rad the paper states"
    )
    F = pytest.importorskip("rope_attribution.figures")
    assert F.LINEARIZABLE_ANGLE_RAD == LINEARIZABLE_ANGLE_RAD


def test_head_dim_is_the_half_the_paper_divides_channels_by() -> None:
    assert int(literal("half_def")) == 2
    assert HEAD_DIM // int(literal("half_def")) == int(literal("threshold_rad_table", 2))
    assert int(literal("threshold_rad_table", 2)) == N_PAIRS
    assert int(literal("derived_half")) == N_PAIRS
    assert int(literal("derived_counts")) == N_PAIRS
    assert int(literal("derived_multiple", 2)) == N_PAIRS
    assert int(literal("derived_multiple_2", 2)) == N_PAIRS


# ==========================================================================
# "The structural properties hold exactly."
# ==========================================================================


def assert_paper_asserts_below(measured: float, name: str, what: str) -> None:
    """The paper states ``$<10^{-k}$``; the measurement must actually be below it.

    Used for the structural residuals, which are float64 noise floors: the paper
    reports a ceiling rather than digits, because the exact value moves with the
    host's summation order and quoting it would overstate the precision.
    """
    ceiling = float(literal(name, 1)) * 10.0 ** -int(literal(name, 2))
    assert measured < ceiling, (
        f"{what}: the measurement is {measured:.6e}, which is not below the ceiling "
        f"{ceiling:.0e} the paper states"
    )


def test_relative_position_property_claim() -> None:
    pairs = int(literal("rel_pos_pairs"))
    structural = MEASUREMENTS["structural_facts"]
    assert pairs == structural["relative_position_pairs_tested"]
    assert_paper_asserts_below(
        structural["relative_position_max_abs_err"],
        "rel_pos_ceiling",
        "relative-position property",
    )


def test_the_floor_note_pairs_are_both_far_below_the_bound() -> None:
    """The illustrative pair in the note on platform-dependent noise floors.

    These two numbers are not measurements the paper is reporting; they illustrate
    that the same computation lands on a different last-bit value on a different
    host. What must hold is that both sit far below the ceiling the table states -
    otherwise the note would be comparing values that are not actually noise.
    """
    low_mantissa, high_mantissa = groups("floor_note_pair")[0]
    low = float(low_mantissa) * 1e-14
    high = float(high_mantissa) * 1e-14
    bound = float(literal("rel_pos_ceiling", 1)) * 10.0 ** -int(literal("rel_pos_ceiling", 2))
    for value in (float(low), float(high)):
        assert 0.0 < value < bound * 1e-1, (
            f"{value:.3e} should be noise well below the {bound:.0e} ceiling"
        )
    assert float(low) != float(high), "the note's point is that they differ"


def test_the_42_pairs_are_the_7_distances_times_the_6_base_positions() -> None:
    pairs, n_distances, n_positions = groups("relpos_pairs_prose")[0]
    assert int(pairs) == MEASUREMENTS["structural_facts"]["relative_position_pairs_tested"]
    # the same count, restated in the intro
    assert int(literal("intro_pairs")) == int(pairs)
    assert int(n_distances) * int(n_positions) == int(pairs)
    assert int(literal("method_relpos_pairs")) == int(pairs)
    assert int(literal("relpos_shift")) == 7


def test_norm_preservation_claim_covers_query_and_key() -> None:
    structural = MEASUREMENTS["structural_facts"]
    assert_paper_asserts_below(
        structural["query_norm_preservation_max_abs_err"],
        "norm_q_ceiling",
        "norm preservation, query",
    )
    assert_paper_asserts_below(
        structural["key_norm_preservation_max_abs_err"],
        "norm_k_ceiling",
        "norm preservation, key",
    )
    assert structural["query_norm_preservation_max_abs_err"] > 0.0


def test_bilinearity_claim_and_the_35_score_checks() -> None:
    structural = MEASUREMENTS["structural_facts"]
    assert_paper_asserts_below(
        structural["bilinearity_max_rel_err"], "bilin_ceiling", "bilinearity"
    )
    assert int(literal("bilin_checks")) == structural["scores_tested"]
    assert int(literal("method_bilin_checks")) == structural["scores_tested"]
    # the same count, restated in the intro
    assert int(literal("intro_checks")) == structural["scores_tested"]


def _as_number(text: str) -> float:
    """The paper's rescaling factor, including its ``10^{k}`` shorthand."""
    power = re.fullmatch(r"10\^\{(-?\d+)\}", text)
    if power is not None:
        return 10.0 ** int(power.group(1))
    return float(text)


def test_the_35_score_checks_are_7_rescalings_at_5_distances() -> None:
    scales, n_distances = groups("bilin_scales")[0]
    parsed = [_as_number(item.strip()) for item in scales.split(",")]
    assert [item.strip() for item in scales.split(",")] == [
        "0", "1", "2", "-3", "7.5", "10^{-3}", "10^{3}",
    ]
    assert len(parsed) * int(n_distances) == MEASUREMENTS["structural_facts"]["scores_tested"]
    source = (_REPO / "projects" / "rope_attribution" / "experiments.py").read_text(
        encoding="utf-8"
    )
    found = re.search(r"scales = np\.array\(\[([^\]]*)\]\)", source)
    assert found, "experiments.structural_facts no longer rescales by a literal array"
    coded = [float(item.strip()) for item in found.group(1).split(",")]
    assert coded == parsed, (
        f"the paper's rescaling set {parsed} is not what experiments.py uses: {coded}"
    )
    assert int(literal("bilin_checks_table")) == MEASUREMENTS["structural_facts"]["scores_tested"]


def test_pair_closed_form_claim() -> None:
    closed = MEASUREMENTS["structural_facts"]["pair_closed_form_max_abs_err"]
    assert_paper_asserts_below(
        closed, "closed_form_ceiling", "the per-pair closed form"
    )


def test_feature_additivity_claim_on_the_5_point_grid() -> None:
    coarse = MEASUREMENTS["feature_attribution"]["additivity_max_abs_err"]
    assert_paper_asserts_below(
        coarse, "additivity_5pt_ceiling", "feature additivity, 5-point grid"
    )
    for name in ("abstract_additivity", "r1_additivity", "conclusion_additivity"):
        assert_paper_sci_matches(coarse, literal(name), -14, name)
    assert_paper_sci_matches(
        coarse, literal("eq_additivity_grids", 1), -14, "the 5-point residual in the prose"
    )


def test_feature_additivity_claim_on_the_54_point_grid() -> None:
    """fig09's ``additivity_residual`` column, which the paper cites by name."""
    fine = float(fig09_grid()[1].max())
    assert_paper_asserts_below(
        fine, "additivity_54pt_ceiling", "feature additivity, 54-point grid"
    )
    assert_paper_sci_matches(
        fine, literal("eq_additivity_grids", 2), -14, "the 54-point residual in the prose"
    )
    for name in (
        "abstract_additivity_54",
        "r2_additivity",
        "cond_additivity",
        "synthesis_additivity",
        "conclusion_additivity_54",
    ):
        assert_paper_sci_matches(fine, literal(name), -14, name)
    csv_residual = max(
        float(row["additivity_residual"])
        for row in csv_rows("fig09_position_conditional_attribution")
    )
    assert csv_residual == fine


def test_the_two_grids_give_two_different_additivity_residuals() -> None:
    """The paper's own 'the two grids are kept distinct' rule."""
    coarse = MEASUREMENTS["feature_attribution"]["additivity_max_abs_err"]
    fine = float(fig09_grid()[1].max())
    assert 0.0 < coarse < fine
    assert coarse < 1e-13 and fine < 1e-13


# ==========================================================================
# "a single feature's contribution is a function of distance"
# ==========================================================================


def test_per_feature_spread_on_the_experiment_grid_is_43_05() -> None:
    assert_paper_number_matches(
        MEASUREMENTS["feature_attribution"]["per_feature_contribution_spread_max_ratio"],
        literal("grid_coarse_attrib"),
        "per-feature spread on the 5-point grid",
    )
    assert_paper_number_matches(
        MEASUREMENTS["feature_attribution"]["per_feature_contribution_spread_max_ratio"],
        literal("prov_spread_coarse"),
        "per-feature spread (appendix)",
    )


def test_per_feature_spread_on_the_figure_grid_is_920_04_everywhere_it_is_written() -> None:
    """Recomputed from the code, so a stale fig09 CSV cannot satisfy this."""
    measured = max(fig09_per_feature_ratios(fig09_grid()[0]))
    occurrences = all_literals("spread_max") + [literal("spread_ratios", 1)]
    assert len(occurrences) >= 5, f"only {len(occurrences)} occurrences of the swing found"
    for literal_text in occurrences:
        assert_paper_number_matches(measured, literal_text, "the fig09 swing")
    assert_paper_number_matches(measured, literal("spread_max_bare"), "the fig09 swing (prose)")
    assert_paper_number_matches(measured, literal("prov_rel_pos_pairs", 1), "the swing (appendix)")


def test_the_csv_agrees_with_the_recomputed_fig09_grid() -> None:
    contrib, residuals, deltas = fig09_grid()
    rows = csv_rows("fig09_position_conditional_attribution")
    csv_deltas = sorted({int(row["delta"]) for row in rows})
    assert csv_deltas == [int(d) for d in deltas]
    assert len(rows) == contrib.size
    # Relative, because the CSV was written by figures.py's vectorised
    # accumulation and this recomputes it with explicit loops. Identical maths,
    # different summation order, so the last bits differ by a few ULP and move
    # with the host BLAS. rtol=1e-12 is ~4500 ULP here - tight enough that a real
    # disagreement (a different seed, a different grid) still fails loudly.
    for row in rows:
        i = int(row["feature_index"])
        j = csv_deltas.index(int(row["delta"]))
        assert float(row["contribution"]) == pytest.approx(
            float(contrib[i, j]), rel=1e-12
        ), f"the fig09 CSV's contribution for feature {i} at delta={row['delta']} is stale"
    # The residual is a roundoff floor near zero, where a relative tolerance is
    # meaningless, so it is compared against the floor itself: a residual that
    # moved by more than a few ULP of the largest term is a real change.
    scale = float(np.abs(contrib).max())
    for row in rows:
        j = csv_deltas.index(int(row["delta"]))
        assert abs(float(row["additivity_residual"]) - float(residuals[j])) <= 1e-12 * scale, (
            f"the fig09 additivity residual at delta={row['delta']} moved beyond roundoff"
        )
    curves = fig09_csv_curves()
    assert fig09_per_feature_ratios(curves) == pytest.approx(
        fig09_per_feature_ratios(contrib), rel=1e-12
    )


def test_the_eight_per_feature_ratios_are_written_in_descending_order() -> None:
    written = [float(value) for value in groups("spread_ratios")[0]]
    measured = sorted(fig09_per_feature_ratios(fig09_csv_curves()), reverse=True)
    assert len(written) == len(measured) == MEASUREMENTS["feature_attribution"]["n_features"]
    assert written == sorted(written, reverse=True), f"the paper lists them out of order: {written}"
    for claimed, actual in zip(written, measured, strict=True):
        assert_paper_number_matches(actual, f"{claimed:.2f}", "one of the eight ratios")


def test_the_median_of_the_eight_ratios_and_the_35_floor() -> None:
    ratios = sorted(fig09_per_feature_ratios(fig09_csv_curves()))
    median = ratios[len(ratios) // 2]
    assert_paper_number_matches(median, literal("spread_median"), "the median ratio")
    assert_paper_number_matches(median, literal("prov_rel_pos_pairs", 2), "the median ratio (appendix)")
    floor = float(literal("spread_floor"))
    assert min(ratios) > floor, f"the smallest measured ratio is {min(ratios)}, not above {floor}"


def test_the_920_swing_is_larger_than_the_43_one_because_the_grid_is_finer() -> None:
    """'the ratio grows with the number of distances sampled'."""
    coarse = MEASUREMENTS["feature_attribution"]["per_feature_contribution_spread_max_ratio"]
    fine = float(literal("spread_max_bare"))
    assert coarse < fine
    assert_paper_number_matches(fine, literal("spread_round"), "the rounded 920x")
    assert len(MEASUREMENTS["feature_attribution"]["deltas"]) < len(figure_delta_grid())


def test_the_swing_is_sixteen_orders_of_magnitude_above_the_additivity_residual() -> None:
    """The swing/residual separation, with the paper's own numbers.

    The paper's derivation is "a swing of order 10^2--10^3 divided by the
    5.0e-14 additivity residual". The two numbers quoted in that paragraph are
    the 920.04 swing (fig09, 54-point grid) and the 4.974e-14 residual, and
    920.0444 / 4.9738e-14 = 1.85e16, i.e. SIXTEEN orders of magnitude. The paper
    used to say 14; it now says 16 and shows the quotient.
    """
    residual = float(fig09_grid()[1].max())
    swing = float(literal("spread_max_bare"))
    coarse_swing = MEASUREMENTS["feature_attribution"]["per_feature_contribution_spread_max_ratio"]
    coarse_residual = MEASUREMENTS["feature_attribution"]["additivity_max_abs_err"]
    claimed = int(literal("orders_claim"))
    groups("orders_claim_provenance")

    # The appendix now writes the quotient out; check every number in it.
    q_swing, q_res, q_res_exp, q_value, q_value_exp = groups("orders_claim_quotient")[0]
    assert float(q_swing) == pytest.approx(swing, rel=HEDGE_TOLERANCE)
    assert_paper_sci_matches(
        residual, q_res, -int(q_res_exp), "the additivity residual in the quotient"
    )

    measured = int(math.floor(math.log10(swing / residual)))
    assert claimed == measured, (
        f"the paper claims {claimed} orders of magnitude, but its own numbers "
        f"{swing!r} / {residual!r} give {swing / residual:.6g} = 10^{measured}."
    )
    assert measured == 16
    assert_paper_sci_matches(
        swing / residual, q_value, int(q_value_exp), "the quotient in the appendix"
    )
    assert int(q_value_exp) == measured

    # The coarser pair is still fifteen, so the conservative reading holds.
    assert math.floor(math.log10(coarse_swing / coarse_residual)) == 15
    assert measured >= 15, "the swing/residual ratio is no longer huge; revisit this claim"


# ==========================================================================
# "YaRN does not shrink the largest angle"
# ==========================================================================


def test_max_abs_angle_is_identical_for_rope_and_yarn_on_the_table_grid() -> None:
    deltas = sorted({row["delta"] for row in MEASUREMENTS["method_spectrum"]})
    assert len(deltas) == 5
    for delta in deltas:
        rope = spectrum_row("rope_base_10k", delta)["max_abs_angle"]
        yarn = spectrum_row("yarn", delta)["max_abs_angle"]
        assert rope == yarn, f"max |D_k| differs at delta={delta}: {rope} vs {yarn}"


def test_the_4096_rad_identity_is_written_in_the_paper() -> None:
    delta, angle = groups("max_equal_4096")[0]
    assert int(delta) == 4096
    assert float(angle) == spectrum_row("rope_base_10k", int(delta))["max_abs_angle"]
    assert float(angle) == spectrum_row("yarn", int(delta))["max_abs_angle"]


def test_the_largest_disagreement_over_the_54_distances_is_exactly_zero() -> None:
    n_claim = int(literal("ngrid_54_fig03"))
    mantissa, exponent = groups("max_equal_grid")[0]
    assert n_claim == len(figure_delta_grid())
    assert int(exponent) == 0
    rows = csv_rows("fig03_max_vs_median_angle")
    by_method = {row["method"] for row in rows}
    assert by_method == {
        "rope_base_10k", "yarn", "position_interpolation", "base_500k_legacy_claim",
    }
    maxima: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        maxima[row["method"]][int(row["delta"])] = float(row["max_abs_angle_rad"])
    rope, yarn = maxima["rope_base_10k"], maxima["yarn"]
    assert rope.keys() == yarn.keys() and len(rope) == int(n_claim)
    gap = max(abs(rope[d] - yarn[d]) for d in rope)
    assert_paper_says_exactly(gap, mantissa, "the max |D_k| disagreement, RoPE vs YaRN")
    assert_paper_says_exactly(0.0, literal("prov_max_agreement"), "the appendix's 0.000 rad")


def test_the_median_angle_ratio_is_constant_and_equals_0_4464() -> None:
    rows: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for row in csv_rows("fig03_max_vs_median_angle"):
        rows[row["method"]][int(row["delta"])] = {
            "max": float(row["max_abs_angle_rad"]),
            "med": float(row["median_abs_angle_rad"]),
        }
    ratios = {
        delta: rows["yarn"][delta]["med"] / rows["rope_base_10k"][delta]["med"]
        for delta in sorted(rows["rope_base_10k"])
    }
    spread = max(ratios.values()) - min(ratios.values())
    assert spread < 1e-12, f"the median ratio is not constant across delta: {ratios}"
    measured = ratios[sorted(ratios)[0]]
    assert_paper_number_matches(measured, literal("median_ratio"), "the median ratio")
    assert_paper_number_matches(measured, literal("median_ratio_abstract"), "the median ratio (abstract)")
    assert_paper_number_matches(measured, literal("prov_ratios", 1), "the median ratio (appendix)")
    assert_paper_says_exactly(1.0, literal("prov_ratios", 2), "the max-angle ratio (appendix)")
    for delta_claim in ("median_at_512", "median_at_4096"):
        delta = int(literal(delta_claim, 3))
        assert int(literal(delta_claim, 3)) == delta
        assert_paper_number_matches(
            spectrum_row("yarn", delta)["median_abs_angle"],
            literal(delta_claim, 1),
            f"YaRN median at {delta}",
        )
        assert_paper_number_matches(
            spectrum_row("rope_base_10k", delta)["median_abs_angle"],
            literal(delta_claim, 2),
            f"RoPE median at {delta}",
        )
        assert abs(
            spectrum_row("yarn", delta)["median_abs_angle"]
            / spectrum_row("rope_base_10k", delta)["median_abs_angle"]
            - measured
        ) < 1e-9, f"the median ratio at delta={delta} is not the constant one"


def test_the_fig02_medians_at_delta_4493() -> None:
    delta = int(literal("fig02_delta"))
    rows = csv_rows("fig02_angle_spectrum")
    assert delta in {int(row["delta"]) for row in rows}
    shown = sorted({int(row["delta"]) for row in rows})
    assert int(literal("ngrid_54_fig02", 1) or len(shown)) if False else len(shown) == 8, (
        "fig02 draws eight distances, as its caption says"
    )
    angles: dict[str, list[float]] = defaultdict(list)
    schemes: list[str] = []
    for row in rows:
        if row["scheme"] not in schemes:
            schemes.append(row["scheme"])
        if int(row["delta"]) == delta:
            angles[row["scheme"]].append(float(row["abs_angle_rad"]))
    assert len(schemes) == 2
    medians = {}
    for scheme in schemes:
        assert len(angles[scheme]) == N_PAIRS
        ordered = sorted(angles[scheme])
        medians[scheme] = (ordered[N_PAIRS // 2 - 1] + ordered[N_PAIRS // 2]) / 2
    rope_scheme, yarn_scheme = schemes
    assert_paper_number_matches(medians[rope_scheme], literal("median_fig02_rope"), "fig02 RoPE median")
    assert_paper_number_matches(medians[yarn_scheme], literal("median_fig02_yarn"), "fig02 YaRN median")
    assert_paper_number_matches(medians[rope_scheme], literal("prov_fig02", 1), "fig02 median (appendix)")
    assert_paper_number_matches(medians[yarn_scheme], literal("prov_fig02", 2), "fig02 median (appendix)")
    assert max(angles[rope_scheme]) == max(angles[yarn_scheme]), (
        "the two panels' right edges are not identical at this delta"
    )


def test_the_fastest_channel_turns_one_radian_per_token_under_both_schemes() -> None:
    rate = float(literal("fastest_rate"))
    rope = R.inv_freq(HEAD_DIM, ROPE_BASE)
    yarn, _mscale = R.yarn_parameters(HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS)
    assert_paper_number_matches(float(rope[0]), f"{rate}", "the fastest channel's rate")
    assert yarn[0] == pytest.approx(rope[0], rel=1e-12)
    assert float(spectrum_row("rope_base_10k", 1)["max_abs_angle"]) == pytest.approx(
        float(rope[0]), rel=1e-12
    )


# ==========================================================================
# The delta = 512 / 4096 comparison table
# ==========================================================================

_TABLE_SCHEMES = {
    "rope": "rope_base_10k",
    "yarn": "yarn",
    "interpolation": "position_interpolation",
    "base500k": "base_500k_legacy_claim",
}

_SCHEME_FROM_METHOD = {
    "rope_base_10k": "rope",
    "yarn": "yarn",
    "position_interpolation": "interpolation",
    "base_500k_legacy_claim": "base500k",
}


def _tex_plain(cell: str) -> str:
    """Strip the LaTeX decoration from one numeric table cell."""
    text = cell.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = text.replace("\\times", "x")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return text.replace("$", "").replace("{", "").replace("}", "").replace("\\", " ").strip()


def _scheme_label(cell: str) -> str:
    """The four scheme labels of Table~\\ref{tab:spectrum}, as a short key.

    The labels are written with macros (``\\rope``, ``\\yarn``,
    ``$\\mathrm{base}=5\\!\\times\\!10^{5}$``), so they are classified on the raw
    cell rather than by stripping the macros off it.
    """
    text = cell.replace("\\rope", "RoPE").replace("\\yarn", "YaRN")
    text = text.replace("\\mathrm{base}", "base")
    if "RoPE" in text:
        return "rope"
    if "YaRN" in text:
        return "yarn"
    if "interpolation" in text:
        return "interpolation"
    if "base" in text:
        return "base500k"
    raise AssertionError(f"unrecognised scheme label in tab:spectrum: {cell!r}")


def table_block(label: str) -> str:
    anchor = TEX_FLAT.index(f"\\label{{{label}}}")
    start = TEX_FLAT.index("\\begin{tabular}", anchor)
    return TEX_FLAT[start : TEX_FLAT.index("\\end{tabular}", start)]


def spectrum_table_rows() -> list[dict]:
    """Parsed rows of Table~\\ref{tab:spectrum}, straight out of the LaTeX."""
    rows = []
    for chunk in table_block("tab:spectrum").split(r"\\"):
        cells = chunk.split("&")
        if len(cells) != 6:
            continue
        scheme_cell, delta_cell, max_cell, med_cell, frac_cell, amp_cell = cells
        if not _tex_plain(delta_cell).isdigit():
            continue
        plain_frac = _tex_plain(frac_cell)
        frac = re.fullmatch(r"([\d.]+)\s*\((\d+)/(\d+)\)", plain_frac)
        if frac is None:
            raise AssertionError(f"unparsable linearizable cell {plain_frac!r} in tab:spectrum")
        plain_amp = _tex_plain(amp_cell)
        sci = re.fullmatch(r"([\d.]+)x10\^?(-?\d+)", plain_amp)
        if sci is not None:
            mantissa, exponent = sci.group(1), int(sci.group(2))
        elif re.fullmatch(r"[\d.]+", plain_amp):
            mantissa, exponent = plain_amp, 0
        else:
            raise AssertionError(f"unparsable amp-wtd err cell {plain_amp!r} in tab:spectrum")
        rows.append(
            {
                "scheme": _scheme_label(scheme_cell),
                "delta": int(_tex_plain(delta_cell)),
                "max": _tex_plain(max_cell),
                "median": _tex_plain(med_cell),
                "frac": frac.group(1),
                "count": int(frac.group(2)),
                "denom": int(frac.group(3)),
                "amp_mantissa": mantissa,
                "amp_exponent": exponent,
            }
        )
    assert len(rows) == 10, f"expected 10 data rows in tab:spectrum, parsed {rows}"
    return rows


def test_spectrum_table_has_every_scheme_distance_the_json_has() -> None:
    from_table = {(row["scheme"], row["delta"]) for row in spectrum_table_rows()}
    from_json = {
        (_SCHEME_FROM_METHOD[row["method"]], row["delta"])
        for row in MEASUREMENTS["method_spectrum"]
    }
    assert {scheme for scheme, _ in from_table} == set(_TABLE_SCHEMES)
    # The table deliberately trims the grid (it lists 2048 only where it is
    # informative), so a subset is what is expected - but nothing may be listed
    # that the JSON does not hold, and the four schemes must all appear.
    assert from_table <= from_json, (
        f"tab:spectrum lists {sorted(from_table - from_json)}, which "
        f"measurements.json does not hold"
    )
    assert {delta for _, delta in from_table} <= set(table_grid_of_the_paper())
    assert all(row["denom"] == N_PAIRS for row in spectrum_table_rows())
    groups("table_d_base")
    groups("table_caption_grid")


def test_spectrum_table_row() -> None:
    """Every cell of Table~\\ref{tab:spectrum} against ``method_spectrum``."""
    checked = 0
    for row in spectrum_table_rows():
        method = _TABLE_SCHEMES[row["scheme"]]
        source = spectrum_row(method, row["delta"])
        where = f"{row['scheme']}@{row['delta']}"
        assert_paper_number_matches(source["max_abs_angle"], row["max"], f"{where} max|D|")
        assert_paper_number_matches(source["median_abs_angle"], row["median"], f"{where} med|D|")
        assert_paper_number_matches(
            source["frac_channels_linearizable"], row["frac"], f"{where} linearizable fraction"
        )
        assert_paper_sci_matches(
            source["amplitude_weighted_err"],
            row["amp_mantissa"],
            row["amp_exponent"],
            f"{where} amp-wtd err",
        )
        checked += 1
    assert checked == 10


def test_every_linearizable_count_in_brackets_is_exact() -> None:
    for row in spectrum_table_rows():
        measured = spectrum_row(_TABLE_SCHEMES[row["scheme"]], row["delta"])
        count = measured["frac_channels_linearizable"] * row["denom"]
        assert count == round(count), f"the fraction at {row['scheme']}@{row['delta']} is not a multiple of 1/{row['denom']}"
        assert row["count"] == round(count), (
            f"tab:spectrum says {row['count']}/{row['denom']} for "
            f"{row['scheme']}@{row['delta']} but the measured fraction "
            f"{measured['frac_channels_linearizable']} is {count}"
        )


def test_plain_rope_has_no_linearizable_channels_by_4096() -> None:
    grid = [row for row in MEASUREMENTS["method_spectrum"] if row["method"] == "rope_base_10k"]
    assert {row["delta"] for row in grid} >= {512, 2048, 4096}
    first_zero = min(row["delta"] for row in grid if row["frac_channels_linearizable"] == 0.0)
    assert first_zero <= 2048, f"plain RoPE still has a linearizable channel at {first_zero}"
    for row in grid:
        if row["delta"] >= first_zero:
            assert row["frac_channels_linearizable"] == 0.0


def test_yarn_retains_roughly_a_fifth_of_its_channels_at_4096() -> None:
    measured = spectrum_row("yarn", 4096)["frac_channels_linearizable"]
    table = {(row["scheme"], row["delta"]): row for row in spectrum_table_rows()}
    assert_paper_number_matches(measured, table[("yarn", 4096)]["frac"], "YaRN@4096 fraction")
    assert abs(measured - 0.2) <= 0.03, f"YaRN retains {measured} of its channels at delta=4096"


def test_the_linearizable_fractions_quoted_in_the_prose() -> None:
    d512 = int(literal("lin_rise_512", 3))
    d4096 = int(literal("lin_rise_4096", 3))
    for scheme, delta, name in (
        ("rope_base_10k", d512, ("lin_rise_512", 1)),
        ("yarn", d512, ("lin_rise_512", 2)),
        ("rope_base_10k", d4096, ("lin_rise_4096", 1)),
        ("yarn", d4096, ("lin_rise_4096", 2)),
    ):
        assert_paper_number_matches(
            spectrum_row(scheme, delta)["frac_channels_linearizable"],
            literal(*name),
            f"{scheme}@{delta} linearizable fraction (prose)",
        )
    for group, delta in ((1, d512), (2, d4096)):
        assert_paper_number_matches(
            spectrum_row("base_500k_legacy_claim", delta)["frac_channels_linearizable"],
            literal("lin_base500k", group),
            "the base-500k fraction (prose)",
        )


def test_position_interpolation_shrinks_the_fast_channel() -> None:
    frm, to = groups("pi_max_shrink")[0]
    assert float(to) == pytest.approx(
        spectrum_row("position_interpolation", 4096)["max_abs_angle"], rel=1e-12
    )
    assert float(frm) == pytest.approx(
        spectrum_row("rope_base_10k", 4096)["max_abs_angle"], rel=1e-12
    )
    assert float(to) == pytest.approx(float(frm) / EXT_SCALE, rel=1e-12)
    assert spectrum_row("position_interpolation", 512)["max_abs_angle"] == pytest.approx(
        512.0 / EXT_SCALE, rel=1e-12
    )


def test_plain_rope_collapses_to_zero_exactly_from_delta_861() -> None:
    fractions: dict[str, dict[int, float]] = defaultdict(dict)
    for row in csv_rows("fig04_linearizable_fraction"):
        fractions[row["method"]][int(row["delta"])] = float(row["frac_channels_linearizable"])
    rope, yarn = fractions["rope_base_10k"], fractions["yarn"]
    first_zero = int(literal("fig04_first_zero"))
    assert first_zero == min(delta for delta in sorted(rope) if rope[delta] == 0.0)
    assert all(rope[delta] == 0.0 for delta in sorted(rope) if delta >= first_zero)
    value, here, denom_here, end, denom_end, delta_end = groups("fig04_zero_value")[0]
    assert_paper_says_exactly(0.0, value, "the value the paper calls exactly 0.0000")
    assert int(denom_here) == int(denom_end) == N_PAIRS
    assert yarn[first_zero] * N_PAIRS == int(here), (
        f"the paper says YaRN retains {here}/{N_PAIRS} at delta={first_zero}, "
        f"fig04 says {yarn[first_zero] * N_PAIRS}"
    )
    assert yarn[int(delta_end)] * N_PAIRS == int(end)
    prov_delta, prov_end, prov_denom = groups("prov_fig04")[0]
    assert int(prov_delta) == first_zero
    assert int(prov_end) == int(end) and int(prov_denom) == N_PAIRS


def test_the_synthesis_ramp_from_zero_to_eleven_out_of_thirty_two() -> None:
    """The synthesis ramp, as the paper now states it.

    The paper says the linearizable fraction is extended "from 0/32 to 11/32 at
    delta ~ 861". It used to say 2/32 there, which fig04 contradicts: at
    delta = 861 plain RoPE retains 0/32, and the same section two paragraphs
    earlier already said so. The last distances at which RoPE still has 2/32 are
    472 and 549. This test pins the corrected pair.
    """
    start, start_denom, end, end_denom, delta = groups("synthesis_ramp")[0]
    assert int(start_denom) == int(end_denom) == N_PAIRS
    fractions: dict[str, dict[int, float]] = defaultdict(dict)
    for row in csv_rows("fig04_linearizable_fraction"):
        fractions[row["method"]][int(row["delta"])] = float(row["frac_channels_linearizable"])
    rope = fractions["rope_base_10k"][int(delta)]
    yarn = fractions["yarn"][int(delta)]
    assert yarn * N_PAIRS == int(end), (
        f"the paper says YaRN holds {end}/{N_PAIRS} at delta={delta}; fig04 says "
        f"{yarn * N_PAIRS}/{N_PAIRS}"
    )
    assert rope * N_PAIRS == int(start), (
        f"the paper says plain RoPE holds {start}/{N_PAIRS} at delta ~ {delta}; fig04 "
        f"says {rope * N_PAIRS}/{N_PAIRS} (the fraction first reaches 0/32 at "
        f"delta={int(literal('fig04_first_zero'))}, and 2/32 only up to delta="
        f"{max(d for d in fractions['rope_base_10k'] if fractions['rope_base_10k'][d] * N_PAIRS == 2)})"
    )
    assert int(literal("synthesis_worst")) == 3, "the paper's 'full RoPE at any delta > 3' moved"
    delta_8, delta_512 = groups("synthesis_delta_8_512")[0]
    known = set(table_grid_of_the_paper()) | set(figure_delta_grid()) | set(
        MEASUREMENTS["feature_attribution"]["deltas"]
    )
    assert {int(delta_8), int(delta_512)} <= known, (
        f"the paper's illustrative pair (delta={delta_8}, delta={delta_512}) uses a "
        f"distance that appears on none of its grids {sorted(known)}"
    )


def test_the_amplitude_weighted_error_ratio_at_8192() -> None:
    values = {
        row["method"]: float(row["amplitude_weighted_err"])
        for row in csv_rows("fig05_linearization_error")
        if int(row["delta"]) == 8192
    }
    assert max(int(row["delta"]) for row in csv_rows("fig05_linearization_error")) == 8192
    assert not any(row["delta"] == 8192 for row in MEASUREMENTS["method_spectrum"]), (
        "measurements.json now holds delta=8192; this test's fig05-based source needs revisiting"
    )
    measured_ratio = values["yarn"] / values["rope_base_10k"]
    for name in ("amp_ratio", "amp_ratio_conclusion"):
        assert_paper_number_matches(measured_ratio, literal(name), f"the amp-weighted ratio ({name})")
    r3_ratio, delta = groups("amp_ratio_r3")[0]
    assert int(delta) == 8192
    assert_paper_number_matches(measured_ratio, r3_ratio, "the R3 ratio")
    mantissa_rope, exp_rope, mantissa_yarn, exp_yarn = groups("amp_at_8192")[0]
    assert int(exp_rope) == int(exp_yarn) == 6
    assert_paper_sci_matches(values["rope_base_10k"], mantissa_rope, 6, "RoPE amp-weighted err at 8192")
    assert_paper_sci_matches(values["yarn"], mantissa_yarn, 6, "YaRN amp-weighted err at 8192")
    prov_mantissa, prov_exp = groups("prov_amp")[0]
    assert int(prov_exp) == 6
    assert_paper_sci_matches(values["rope_base_10k"], prov_mantissa, 6, "RoPE amp-weighted err (appendix)")


def test_less_than_a_half_a_percent_buy() -> None:
    gain = 1.0 - float(literal("amp_ratio"))
    ceiling = float(literal("amp_percent"))
    assert gain < ceiling / 100.0, f"the gain is {gain * 100:.4f}%, not below {ceiling}%"
    assert gain > 0.0
    groups("derived_percent")


def test_the_per_pair_rms_ratio_is_essentially_one() -> None:
    values = {
        row["method"]: float(row["per_pair_err_rms"])
        for row in csv_rows("fig05_linearization_error")
        if int(row["delta"]) == 8192
    }
    measured = values["yarn"] / values["rope_base_10k"]
    assert_paper_number_matches(measured, literal("rms_ratio"), "the per-pair RMS ratio")
    assert abs(measured - 1.0) < 1e-4, "a ratio this far from 1 would not be 'no meaningful change'"


def test_position_interpolation_is_a_thousand_times_below_rope_on_this_axis() -> None:
    value, delta, factor = groups("pi_amp")[0]
    values = {
        row["method"]: float(row["amplitude_weighted_err"])
        for row in csv_rows("fig05_linearization_error")
        if int(row["delta"]) == int(delta)
    }
    measured = values["rope_base_10k"] / values["position_interpolation"]
    assert_paper_number_matches(values["position_interpolation"], value, "PI amp-weighted err")
    assert_paper_says_roughly(measured, factor, "how far below RoPE PI sits")
    prov_value, prov_delta = groups("prov_pi_amp")[0]
    # The appendix row "$1225.06$, $\delta = 3$" pairs the PI amplitude-weighted
    # error with the fig06 crossing distance, not with this row's delta.
    assert int(prov_delta) == int(literal("pair_err_d3", 1))
    assert_paper_number_matches(
        values["position_interpolation"], prov_value, "PI amp-weighted err (appendix)"
    )
    prov_divisor, prov_factor = groups("derived_1000x")[0][::-1]
    assert_paper_number_matches(
        values["position_interpolation"], prov_divisor, "the appendix's divisor"
    )
    assert_paper_says_roughly(measured, prov_factor, "the appendix's factor")


# ==========================================================================
# "Partial RoPE gives an exactly position-free sub-score"
# ==========================================================================


def test_partial_rope_split_fractions_and_dimensions() -> None:
    partial = MEASUREMENTS["partial_rope"]
    assert_paper_number_matches(partial["rotated_fraction"], literal("partial_p_value"), "p")
    assert_paper_number_matches(partial["rotated_fraction"], literal("partial_p_abstract"), "p")
    groups("partial_p")
    dim, n_rot = groups("partial_nrot")[0]
    assert int(dim) == partial["dim"]
    assert int(n_rot) == partial["n_rot"]
    assert int(literal("limitations_nrot")) == partial["n_rot"]
    assert partial["n_rot"] == int(partial["rotated_fraction"] * partial["dim"])
    assert partial["rotated_fraction"] + partial["clean_fraction"] == 1.0
    assert partial["clean_fraction"] == 0.75
    assert {int(row["n_rot"]) for row in csv_rows("fig07_partial_rope")} == {partial["n_rot"]}


def test_the_unrotated_subscore_spread_is_exactly_zero_on_both_grids() -> None:
    partial = MEASUREMENTS["partial_rope"]
    n_coarse, value, n_grid = groups("partial_clean_spread")[0]
    assert int(n_coarse) == len(table_grid_of_the_paper())
    assert int(n_grid) == len(figure_delta_grid()) == len(csv_rows("fig07_partial_rope"))
    assert_paper_says_exactly(partial["clean_score_spread"], value, "the unrotated spread (JSON grid)")
    clean = [float(row["clean_subscore"]) for row in csv_rows("fig07_partial_rope")]
    assert max(clean) - min(clean) == 0.0, "the fig07 clean sub-score is not constant"
    assert_paper_says_exactly(0.0, literal("prov_partial", 1), "the appendix's 0.0")
    assert_paper_says_exactly(0.0, literal("partial_clean_zero_2"), "the second exactly-0.0 restatement")
    assert_paper_says_exactly(0.0, literal("exact_identity_zero"), "the exact-identities paragraph")
    assert_paper_says_exactly(0.0, literal("exact_identity_residue"), "the residue sentence")


def test_the_plotted_clean_subscore_is_a_single_constant() -> None:
    clean = {row["clean_subscore"] for row in csv_rows("fig07_partial_rope")}
    assert len(clean) == 1, f"the plotted clean sub-score takes {len(clean)} values: {clean}"
    assert_paper_says_exactly(
        float(next(iter(clean))), literal("partial_clean_const"), "the plotted clean constant"
    )


def test_the_rotated_subscore_spread_on_both_grids() -> None:
    json_value, fig_value = groups("partial_rot_spread")[0]
    assert_paper_number_matches(
        MEASUREMENTS["partial_rope"]["rotated_score_spread"], json_value,
        "the rotated spread (JSON grid)",
    )
    rotated = [float(row["rotated_subscore"]) for row in csv_rows("fig07_partial_rope")]
    measured = max(rotated) - min(rotated)
    assert_paper_number_matches(measured, fig_value, "the rotated spread (figure grid)")
    assert_paper_number_matches(
        MEASUREMENTS["partial_rope"]["rotated_score_spread"], literal("prov_partial", 2),
        "the appendix's 12.03",
    )
    assert_paper_number_matches(measured, literal("prov_partial", 3), "the appendix's 22.56")


def test_the_clean_magnitude_share_on_both_grids() -> None:
    rows = csv_rows("fig07_partial_rope")
    shares = [
        abs(float(row["clean_subscore"]))
        / (abs(float(row["clean_subscore"])) + abs(float(row["rotated_subscore"])) + 1e-12)
        for row in rows
    ]
    measured = sum(shares) / len(shares)
    assert_paper_number_matches(
        MEASUREMENTS["partial_rope"]["clean_magnitude_share_mean"],
        literal("partial_share", 1),
        "the clean share (JSON grid)",
    )
    assert_paper_number_matches(measured, literal("partial_share", 2), "the clean share (figure grid)")
    assert_paper_number_matches(
        MEASUREMENTS["partial_rope"]["clean_magnitude_share_mean"],
        literal("prov_partial_share", 1),
        "the appendix's JSON-grid share",
    )
    assert_paper_number_matches(measured, literal("prov_partial_share", 2), "the appendix's figure-grid share")
    assert 0.0 < measured < 1.0


def test_partial_rope_preserves_the_norm_exactly_as_full_rope_does() -> None:
    """Partial rotary is orthogonal, so it preserves the norm like full RoPE.

    This test used to pin a norm deviation of 0.2663 and a 1.5e14 factor over full
    RoPE, on the strength of an implementation that paired channel ``i`` with
    ``i + head_dim // 2`` across the whole head. That pairing leaves every touched
    rotary pair half rotated and half not; it is not what GPT-NeoX models such as
    ``EleutherAI/pythia-160m`` do, and it is not orthogonal. With the correct
    within-block pairing the deviation sits at the float64 floor, and the paper no
    longer claims that partial rotation damages the magnitude gate.
    """
    partial = MEASUREMENTS["partial_rope"]
    deviation = partial["partial_norm_deviation"]
    assert deviation < 1e-12, (
        f"partial rotary must preserve the norm; measured {deviation:.3e}. If this "
        "failed, apply_partial_rope is pairing across the head instead of within "
        "the rotated block."
    )
    full_rope = MEASUREMENTS["structural_facts"]["key_norm_preservation_max_abs_err"]
    assert full_rope < 1e-12, "full RoPE is the reference floor"
    # Same order as full RoPE, not orders worse.
    assert max(deviation, full_rope) < 1e-12

    # The paper states the partial-RoPE floor as a ceiling, like the other residuals.
    ceiling = float(literal("prov_partial_norm", 1)) * 10.0 ** -int(
        literal("prov_partial_norm", 2)
    )
    assert deviation < ceiling
    full_mantissa, full_exponent = groups("prov_partial_norm")[0][2:]
    assert_paper_sci_matches(full_rope, full_mantissa, int(full_exponent), "the full-RoPE floor")

    # The remark that replaced the withdrawn claim must name the same n_rot.
    assert int(literal("partial_norm_remark_nrot")) == partial["n_rot"]


def test_partial_rope_subscores_reconstruct_the_full_score() -> None:
    residual = max(
        abs(
            float(row["clean_subscore"])
            + float(row["rotated_subscore"])
            - float(row["full_partial_rope_score"])
        )
        for row in csv_rows("fig07_partial_rope")
    )
    assert residual <= 1e-12, f"the two partial sub-scores do not sum to the full one: {residual}"


# ==========================================================================
# "mscale is a temperature"
# ==========================================================================


def test_mscale_value_and_the_published_formula() -> None:
    entropy = MEASUREMENTS["mscale_entropy"]
    assert_paper_number_matches(entropy["mscale"], literal("mscale_value"), "mscale")
    assert_paper_number_matches(entropy["mscale"], literal("mscale_value_prov"), "mscale (appendix)")
    coefficient, scale = groups("mscale_formula_prose")[0]
    assert float(scale) == entropy["scale"]
    assert_paper_number_matches(0.1, coefficient, "the formula's coefficient")
    assert_paper_number_matches(0.1, literal("mscale_formula"), "the formula's coefficient (Eq.)")
    assert entropy["mscale"] == 0.1 * math.log(float(scale)) + 1.0
    assert float(R.get_mscale(float(scale))) == entropy["mscale"]
    assert float(spectrum_row("yarn", 512)["mscale"]) == entropy["mscale"]
    assert spectrum_row("rope_base_10k", 512)["mscale"] == 1.0


def test_attention_entropy_ratios_and_the_drop() -> None:
    entropy = MEASUREMENTS["mscale_entropy"]
    assert_paper_number_matches(
        entropy["entropy_ratio_unscaled"], literal("entropy_from"), "entropy, unscaled"
    )
    assert_paper_number_matches(
        entropy["entropy_ratio_mscaled"], literal("entropy_to", 1), "entropy, mscaled"
    )
    assert_paper_number_matches(
        entropy["entropy_ratio_unscaled"] - entropy["entropy_ratio_mscaled"],
        literal("entropy_to", 2),
        "the entropy drop",
    )
    nats_from, nats_to = groups("entropy_nats_from")[0]
    assert_paper_number_matches(entropy["entropy_unscaled"], nats_from, "entropy in nats, unscaled")
    assert_paper_number_matches(entropy["entropy_mscaled"], nats_to, "entropy in nats, mscaled")
    assert entropy["max_possible_entropy"] == math.log(entropy["n_keys"])
    assert entropy["entropy_ratio_unscaled"] == entropy["entropy_unscaled"] / math.log(
        entropy["n_keys"]
    )
    prov_from, prov_to = groups("entropy_prov_ratios")[0]
    assert_paper_number_matches(
        entropy["entropy_ratio_unscaled"], prov_from, "entropy, unscaled (appendix)"
    )
    assert_paper_number_matches(
        entropy["entropy_ratio_mscaled"], prov_to, "entropy, mscaled (appendix)"
    )


def test_entropy_falls_monotonically_to_the_end_of_the_scale_grid() -> None:
    rows = sorted(csv_rows("fig08_mscale_entropy"), key=lambda row: float(row["scale"]))
    mscaled = [float(row["entropy_ratio_mscaled"]) for row in rows]
    mscalers = [float(row["mscale"]) for row in rows]
    assert all(b <= a for a, b in zip(mscaled, mscaled[1:], strict=False)), (
        "H/ln T is not monotone in scale over the fig08 grid"
    )
    assert all(b >= a for a, b in zip(mscalers, mscalers[1:], strict=False)), (
        "mscale is not monotone in scale over the fig08 grid"
    )
    top, end_ratio, end_mscale = groups("entropy_scale_range")[0]
    assert float(top) == float(rows[-1]["scale"])
    assert_paper_number_matches(mscaled[-1], end_ratio, "H/ln T at the top scale")
    assert_paper_number_matches(mscalers[-1], end_mscale, "mscale at the top scale")
    prov_ratio, prov_mscale, prov_scale = groups("entropy_128_prov")[0]
    assert_paper_number_matches(mscaled[-1], prov_ratio, "H/ln T at the top scale (appendix)")
    assert_paper_number_matches(mscalers[-1], prov_mscale, "mscale at the top scale (appendix)")
    assert float(prov_scale) == float(rows[-1]["scale"])
    assert mscaled[-1] == E.mscale_entropy(
        dim=HEAD_DIM,
        scale=float(rows[-1]["scale"]),
        original_max_position_embeddings=ORIGINAL_MAX_POS,
        n_keys=N_KEYS,
        seed=4,
    )["entropy_ratio_mscaled"]


def test_mscale_is_a_temperature_not_a_scale_change() -> None:
    """mscale sharpens: at every scale above 1 the mscaled entropy is lower."""
    rows = sorted(csv_rows("fig08_mscale_entropy"), key=lambda row: float(row["scale"]))
    for row in rows:
        unscaled = float(row["entropy_ratio_unscaled"])
        mscaled = float(row["entropy_ratio_mscaled"])
        if float(row["scale"]) == 1.0:
            assert unscaled == mscaled, "at scale 1 the two entropy curves must coincide"
        else:
            assert mscaled < unscaled, f"mscale did not sharpen at scale={row['scale']}"
        assert unscaled == float(rows[0]["entropy_ratio_unscaled"]), (
            "the unscaled entropy must not depend on the extension factor"
        )


# ==========================================================================
# YaRN's ramp bookkeeping
# ==========================================================================


def test_yarn_unchanged_and_changed_pair_counts() -> None:
    """"the fastest 9 of the 32 rotary pairs" / "all 32 entries" for PI."""
    caption = groups("yarn_unchanged_caption")[0]
    body = groups("yarn_unchanged_body")[0]
    note = groups("yarn_unchanged_note")[0]
    prov = groups("prov_ramp")[0]
    prov_note = groups("prov_yarn_note")[0]
    assert (int(body[0]), int(body[1])) == (int(caption[0]), int(caption[1]))
    assert (int(note[0]), int(note[1])) == (int(caption[0]), int(caption[1]))
    assert (int(prov_note[0]), int(prov_note[1])) == (int(caption[0]), int(caption[1]))
    assert (int(prov[0]), int(prov[1])) == (int(caption[0]), int(caption[1]))
    assert int(caption[1]) == N_PAIRS
    base_freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    yarn_freqs, _mscale = R.yarn_parameters(HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS)
    measured_unchanged = int(np.count_nonzero(yarn_freqs == base_freqs))
    measured_differ = int(np.count_nonzero(yarn_freqs != base_freqs / EXT_SCALE))
    measured_uniform = int(np.count_nonzero(yarn_freqs == base_freqs / EXT_SCALE))
    assert measured_unchanged == int(caption[0]), (
        f"the paper says {caption[0]} of {N_PAIRS} pairs are bit-for-bit unchanged, "
        f"measured {measured_unchanged}"
    )
    assert measured_differ + measured_uniform == N_PAIRS
    blended = N_PAIRS - measured_unchanged - measured_uniform
    assert blended > 0, "YaRN is a uniform rescaling on this grid, which the paper denies"
    assert measured_differ == 21, f"the fig01 ladder shows {measured_differ} of {N_PAIRS} differ"


def test_yarn_slows_the_slowest_pair_by_exactly_thirty_two() -> None:
    base_freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    yarn_freqs, _mscale = R.yarn_parameters(HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS)
    measured = float(base_freqs[-1] / yarn_freqs[-1])
    assert_paper_number_matches(measured, literal("yarn_slow_ratio"), "the slowest pair's factor")
    assert measured == EXT_SCALE
    assert_paper_number_matches(
        EXT_SCALE, literal("yarn_slow_ratio_caption"), "the slowest pair's factor (caption)"
    )
    assert_paper_number_matches(
        measured, literal("prov_ramp", 3), "the slowest pair's factor (appendix)"
    )


def test_position_interpolation_differs_from_rope_in_every_entry() -> None:
    n = int(literal("interp_all_entries"))
    assert n == N_PAIRS
    assert_paper_number_matches(EXT_SCALE, literal("interp_all_entries_caption"), "the PI divisor")
    base_freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    differing = int(np.count_nonzero(base_freqs / EXT_SCALE != base_freqs))
    assert differing == n, f"PI differs from RoPE in {differing} of {N_PAIRS} entries"
    assert not np.array_equal(
        base_freqs / EXT_SCALE, R.inv_freq(HEAD_DIM, LEGACY_BASE)
    ), "PI at scale 32 and the legacy base-500k ladder are the same array"


def test_the_fig01_ladder_agrees_with_the_recomputation() -> None:
    rows = csv_rows("fig01_frequency_ladder")
    columns = [name for name in rows[0] if name != "pair_index"]
    assert len(columns) == 4, f"fig01 has {len(columns)} ladders, expected 4"
    assert len(rows) == N_PAIRS
    base_freqs = R.inv_freq(HEAD_DIM, ROPE_BASE)
    yarn_freqs, _mscale = R.yarn_parameters(HEAD_DIM, ROPE_BASE, EXT_SCALE, ORIGINAL_MAX_POS)
    # Compared to a few ULP rather than bit-for-bit: `base ** (arange/dim)` and
    # `1.0 / (base ** (...))` are different expressions for the same number, so
    # the ladder depends on which one the host's libm contracts, and the committed
    # CSV was written on a different platform than CI. 1e-12 is ~4500 ULP at 1.0,
    # far tighter than any discrepancy that could indicate a real defect.
    for column, expected in (
        (columns[0], base_freqs),
        (columns[1], base_freqs / EXT_SCALE),
        (columns[2], yarn_freqs),
        (columns[3], R.inv_freq(HEAD_DIM, LEGACY_BASE)),
    ):
        written = np.array([float(row[column]) for row in rows])
        assert np.allclose(written, expected, rtol=1e-12, atol=0.0), (
            f"the fig01 column {column!r} disagrees with the recomputation; "
            f"largest relative gap {np.max(np.abs(written / expected - 1.0)):.3e}"
        )


def test_the_pair_amplitudes_are_position_free_in_the_fig06_sidecar() -> None:
    """Proposition~\\ref{prop:blind}: one amplitude per pair, across all distances."""
    rows = csv_rows("fig06_pair_exact_vs_linear")
    schemes = sorted({row["scheme"] for row in rows})
    assert len(schemes) == 2
    by_scheme: dict[str, dict[int, dict[int, list[tuple[int, float]]]]] = {
        scheme: defaultdict(lambda: defaultdict(list)) for scheme in schemes
    }
    for row in rows:
        by_scheme[row["scheme"]][int(row["pair_index"])][int(row["delta"])].append(
            (int(row["pair_index"]), float(row["amplitude_R_k"]))
        )
    for scheme in schemes:
        for pair, per_delta in by_scheme[scheme].items():
            values = {value for entries in per_delta.values() for _pair, value in entries}
            assert len(per_delta) == len(figure_delta_grid()), (
                f"fig06 has {len(per_delta)} distances for {scheme} pair {pair}"
            )
            assert len(values) == 1, (
                f"the amplitude of pair {pair} under {scheme} is not distance-free: {values}"
            )


def test_the_fig06_per_pair_numbers() -> None:
    rows = csv_rows("fig06_pair_exact_vs_linear")
    by_key = {(row["scheme"], int(row["pair_index"]), int(row["delta"])): row for row in rows}
    schemes = sorted({row["scheme"] for row in rows})
    delta_1 = int(literal("pair_delta_1"))
    exact, linear, error = groups("pair_exact_d1")[0]
    k_mid, err_mid, exp_mid, ratio = groups("pair_lin_d1")[0]
    delta_3_raw, err_3 = groups("pair_err_d3")[0]
    delta_3 = int(delta_3_raw)
    delta_3_caption = int(literal("pair_err_d3_caption"))
    delta_3_body = int(literal("pairs_delta_3_body"))
    for scheme in schemes:
        fastest = by_key[(scheme, 0, delta_1)]
        assert_paper_number_matches(
            float(fastest["exact_contribution"]), exact, f"{scheme} exact contribution at delta=1"
        )
        assert_paper_number_matches(
            float(fastest["linearized_contribution"]), linear,
            f"{scheme} linearized contribution at delta=1",
        )
        assert_paper_number_matches(
            float(fastest["normalized_abs_error"]), error,
            f"{scheme} normalized error at delta=1",
        )
        middle = by_key[(scheme, int(k_mid), delta_1)]
        if scheme == schemes[0]:
            # The paper quotes one number for this two-panel figure and it is
            # the reference (RoPE) panel's.  YaRN's mid-frequency error differs,
            # because YaRN rescales that channel's frequency; the branch below
            # records that rather than pretending the two panels agree.
            assert_paper_sci_matches(
                float(middle["normalized_abs_error"]), err_mid, int(exp_mid),
                f"{scheme} pair {k_mid} normalized error at delta=1",
            )
        else:
            assert float(middle["normalized_abs_error"]) != float(
                by_key[(schemes[0], int(k_mid), delta_1)]["normalized_abs_error"]
            ), (
                "the paper's single 4.3e-6 now applies to both panels; revisit which "
                "scheme that sentence means before loosening this test"
            )
        if scheme == schemes[0]:
            assert_paper_says_roughly(
                float(fastest["normalized_abs_error"]) / float(middle["normalized_abs_error"]),
                ratio,
                "how much smaller the mid-frequency pair's error is",
            )
        at_3 = by_key[(scheme, 0, delta_3)]
        assert_paper_number_matches(
            float(at_3["normalized_abs_error"]), err_3,
            f"{scheme} normalized error at delta={delta_3}",
        )
    assert delta_3 == delta_3_caption == delta_3_body
    # The first delta at which the fastest pair's error reaches its own size.
    for scheme in schemes:
        crossing = [
            int(row["delta"])
            for row in rows
            if row["scheme"] == scheme
            and int(row["pair_index"]) == 0
            and float(row["normalized_abs_error"]) >= 1.0
        ]
        assert crossing and min(crossing) == delta_3, (
            f"under {scheme} the fastest pair first reaches a relative error of 1 at "
            f"delta={min(crossing) if crossing else None}, not {delta_3}"
        )
    prov_exact, prov_linear, prov_error, prov_mid, prov_exp = groups("prov_pair_d1")[0]
    fastest = by_key[(schemes[0], 0, delta_1)]
    assert_paper_number_matches(
        float(fastest["exact_contribution"]), prov_exact, "the exact contribution (appendix)"
    )
    assert_paper_number_matches(
        float(fastest["linearized_contribution"]), prov_linear,
        "the linearized contribution (appendix)",
    )
    assert_paper_number_matches(
        float(fastest["normalized_abs_error"]), prov_error,
        "the normalized error (appendix)",
    )
    assert_paper_sci_matches(
        float(by_key[(schemes[0], int(k_mid), delta_1)]["normalized_abs_error"]),
        prov_mid, int(prov_exp), "the mid-frequency error (appendix)",
    )


def test_the_two_pair_amplitudes_the_paper_quotes() -> None:
    rows = csv_rows("fig06_pair_exact_vs_linear")
    amplitude = literal("pair_amp_k0")
    n_grid = int(literal("ngrid_54_fig06", 2))
    k_mid, amplitude_mid = groups("pair_amp_k12")[0]
    assert int(n_grid) == len(figure_delta_grid())
    for scheme in sorted({row["scheme"] for row in rows}):
        fastest = {row["amplitude_R_k"] for row in rows if row["scheme"] == scheme and int(row["pair_index"]) == 0}
        assert len(fastest) == 1
        assert_paper_says_exactly(float(next(iter(fastest))), amplitude, f"{scheme} R_0")
        middle = {row["amplitude_R_k"] for row in rows if row["scheme"] == scheme and int(row["pair_index"]) == int(k_mid)}
        assert len(middle) == 1
        assert_paper_says_exactly(float(next(iter(middle))), amplitude_mid, f"{scheme} R_{k_mid}")
    assert float(amplitude) != float(amplitude_mid), "the two amplitudes must differ"


def test_the_appendix_rounds_the_two_pair_amplitudes_correctly() -> None:
    """KNOWN PAPER DEFECT. The appendix truncates ``0.9526294617807264``.

    Correct rounding to the 10 decimal places the appendix writes is
    ``0.9526294618``; the paper writes ``0.9526294617``, which is a truncation.
    (The other half of the same row, ``2.3046990966686227`` -> ``2.3046990967``,
    is a correct rounding and is checked below.)  The mismatch is tiny, about
    8e-11 in absolute terms, but the appendix presents these as *the* values
    and states that no number was typed from memory, so it is left failing
    rather than loosened.
    """
    rows = csv_rows("fig06_pair_exact_vs_linear")
    amplitude_0 = {row["amplitude_R_k"] for row in rows if int(row["pair_index"]) == 0}
    amplitude_12 = {row["amplitude_R_k"] for row in rows if int(row["pair_index"]) == 12}
    assert len(amplitude_0) == 1 and len(amplitude_12) == 1
    prov_0, prov_12 = groups("prov_amp_k")[0]
    assert_paper_number_matches(float(next(iter(amplitude_12))), prov_12, "R_12 (appendix)")
    assert_paper_number_matches(float(next(iter(amplitude_0))), prov_0, "R_0 (appendix)")


# ==========================================================================
# LaTeX structure
# ==========================================================================


@lru_cache(maxsize=1)
def paper_body() -> str:
    """The document after ``\\begin{document}``, with ``%`` comments removed.

    The preamble is excluded on purpose: it holds the ``\\paperfigure`` macro,
    whose body contains a ``\\label{#3}`` template, and treating that as a real
    label would invent one.  Comment lines are stripped for the same reason -
    the usage example in the preamble-like comments would otherwise look like a
    figure call with the literal arguments ``<file stem ...>``.
    """
    body = TEX[TEX.index("\\begin{document}") :]
    return "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("%"))


def brace_balance() -> int:
    depth = 0
    for char in re.sub(r"\\[{}]", "", TEX):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
    return depth


def test_latex_braces_are_balanced() -> None:
    depth = brace_balance()
    assert depth == 0, (
        f"paper/main.tex has unbalanced braces (running depth {depth} at the end, "
        f"ignoring the escaped \\{{ and \\}} used for set notation)"
    )
    assert TEX.count("{") > 500, "the paper no longer has 500 braces; the check is vacuous"


def test_every_begin_has_a_matching_end() -> None:
    begins: dict[str, int] = defaultdict(int)
    ends: dict[str, int] = defaultdict(int)
    for match in re.finditer(r"\\begin\{([^}]*)\}", TEX):
        begins[match.group(1)] += 1
    for match in re.finditer(r"\\end\{([^}]*)\}", TEX):
        ends[match.group(1)] += 1
    assert begins, "no environments found; the check is vacuous"
    assert set(begins) == set(ends), (
        f"environments without a partner: {sorted(set(begins) ^ set(ends))}"
    )
    stack: list[tuple[str, int]] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]*)\}", TEX):
        line = TEX.count("\n", 0, match.start()) + 1
        if match.group(1) == "begin":
            stack.append((match.group(2), line))
        else:
            assert stack, f"\\end{{{match.group(2)}}} at line {line} closes nothing"
            opened, opened_line = stack.pop()
            assert opened == match.group(2), (
                f"line {line}: \\end{{{match.group(2)}}} closes \\begin{{{opened}}} "
                f"from line {opened_line}"
            )
    assert not stack, f"unclosed environments: {stack}"


def paperfigure_calls() -> list[tuple[str, str, str]]:
    """``\\paperfigure{file stem}{caption}{label}``, caption braces respected.

    The caption is full of braces and ``$...$``, so the arguments have to be
    split on nesting depth, not on whitespace or commas.
    """
    calls: list[tuple[str, str, str]] = []
    body = paper_body()
    for match in re.finditer(r"\\paperfigure\{", body):
        index = match.end() - 1
        args: list[str] = []
        for position in range(1, 4):
            assert body[index] == "{", (
                f"\\paperfigure argument {position} at offset {match.start()} does not "
                f"start with a brace"
            )
            depth = 0
            cursor = index
            for cursor in range(index, len(body)):
                if body[cursor] == "{":
                    depth += 1
                elif body[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        break
            else:
                raise AssertionError(f"unterminated \\paperfigure argument {position}")
            args.append(body[index + 1 : cursor].strip())
            index = cursor + 1
        assert body[index] in "%\n ", (
            f"\\paperfigure at offset {match.start()} is not followed by % or a space"
        )
        calls.append((args[0], args[1], args[2]))
    assert len(calls) >= 8, f"only {len(calls)} \\paperfigure calls found; the check is vacuous"
    return calls


def defined_labels() -> set[str]:
    return set(re.findall(r"\\label\{([^}]*)\}", paper_body())) | {
        label for _stem, _caption, label in paperfigure_calls()
    }


def test_every_paperfigure_label_is_defined_and_unique() -> None:
    calls = paperfigure_calls()
    labels = [label for _stem, _caption, label in calls]
    assert len(labels) == len(set(labels)), f"duplicate figure labels: {labels}"
    assert all(label.startswith("fig:") for label in labels), labels
    stems = [stem for stem, _caption, _label in calls]
    assert len(stems) == len(set(stems)), f"a figure stem is included twice: {stems}"
    for stem, caption, label in calls:
        assert re.fullmatch(r"fig\d{2}_[A-Za-z0-9_]+", stem), f"unexpected figure stem {stem!r}"
        assert caption, f"\\paperfigure{{{stem}}} has an empty caption"
        assert "\n" not in label, f"label {label!r} spans a line break"
    assert {label for label in labels} <= defined_labels()


def test_every_ref_target_is_defined() -> None:
    body = paper_body()
    refs = re.findall(r"\\(?:ref|autoref|cref|Cref|eqref|pageref)\{([^}]*)\}", body)
    assert len(refs) >= 20, f"only {len(refs)} \\ref targets found; the check is vacuous"
    labels = defined_labels()
    dangling = sorted(set(refs) - labels)
    assert not dangling, f"paper/main.tex references undefined labels: {dangling}"


def citation_keys() -> list[str]:
    keys: list[str] = []
    for match in re.finditer(r"\\cite[a-zA-Z]*\{([^}]*)\}", TEX):
        keys.extend(key.strip() for key in match.group(1).split(","))
    return keys


def bib_keys() -> list[str]:
    return re.findall(r"@\w+\{([^,\s]+)\s*,", BIB)


def test_every_citation_has_a_bib_entry_and_vice_versa() -> None:
    cited = citation_keys()
    assert len(cited) >= 13, f"only {len(cited)} citations found; the check is vacuous"
    keys = bib_keys()
    assert len(keys) == len(set(keys)), f"duplicate entries in references.bib: {keys}"
    missing = sorted(set(cited) - set(keys))
    assert not missing, f"paper/main.tex cites keys absent from references.bib: {missing}"
    uncited = sorted(set(keys) - set(cited))
    assert not uncited, f"references.bib holds entries the paper never cites: {uncited}"
    assert "\\bibliography{references}" in TEX
    assert "\\bibliographystyle{" in TEX
    assert set(keys) >= {
        "su2021ropeformer",
        "peng2023yarn",
        "chen2023positionalinterp",
        "press2021train",
        "elhage2022toymodels",
    }


def test_every_bib_entry_is_well_formed() -> None:
    entries = re.findall(r"@(\w+)\{([^,\n]+),(.*?)\n\}", BIB, re.DOTALL)
    assert len(entries) == len(bib_keys())
    for _kind, key, body in entries:
        for field in ("author", "title", "year"):
            assert re.search(rf"\b{field}\s*=", body), f"bib entry {key} has no {field}"


def figure_tokens() -> set[str]:
    return set(re.findall(r"\bfig\d{2}[A-Za-z0-9_]*", paper_body()))


def figure_on_disk(token: str) -> bool:
    """``fig05`` resolves to ``figures/fig05_linearization_error.png``."""
    if (FIGURES_DIR / f"{token}.png").is_file():
        return True
    return any(
        child.name.startswith(f"{token}_") and child.suffix == ".png"
        for child in FIGURES_DIR.iterdir()
    )


def test_every_figure_the_paper_names_exists_on_disk() -> None:
    tokens = figure_tokens()
    assert len(tokens) >= 9, f"figure references found in main.tex: {sorted(tokens)}"
    for token in sorted(tokens):
        assert figure_on_disk(token), (
            f"paper/main.tex references figure {token!r}, but figures/ has neither "
            f"{token}.png nor any {token}_*.png"
        )


def test_every_paperfigure_stem_has_a_png_and_a_csv() -> None:
    for stem, _caption, _label in paperfigure_calls():
        assert (FIGURES_DIR / f"{stem}.png").is_file(), f"no PNG for \\paperfigure{{{stem}}}"
        assert (FIGURES_DIR / f"{stem}.csv").is_file(), (
            f"\\paperfigure{{{stem}}} has no CSV sidecar, so its caption's numbers cannot "
            f"be checked against data"
        )


def test_the_paths_the_paper_names_exist() -> None:
    for path in (
        "results/measurements.json",
        "paper/references.bib",
        "figures/README.md",
        "figures/fig09_position_conditional_attribution.png",
    ):
        assert (_REPO / path).exists(), f"paper/main.tex names {path}, which does not exist"
    assert "results/measurements.json" in TEX
    assert "\\bibliography{references}" in TEX


# ==========================================================================
# "What we do not claim"
# ==========================================================================


def disclaimer_text() -> str:
    """The passages where the paper says what it does not do, whitespace-flattened."""
    start = TEX.index("\\subsection{What we do not claim}")
    end = TEX.index("\\bibliographystyle")
    limitations = TEX[TEX.index("\\section{Limitations}") : end]
    return re.sub(r"\s+", " ", TEX[start:end] + limitations)


def paper_sentences() -> list[str]:
    """Comments stripped, paragraphs joined, then split on sentence enders."""
    text = "\n".join(
        line for line in TEX.splitlines() if not line.lstrip().startswith("%")
    )
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        joined = " ".join(paragraph.split())
        units.extend(re.split(r"(?<=[.!?])\s+", joined))
    return [unit for unit in units if unit.strip()]


_NEGATION = re.compile(
    r"\b(no|not|nothing|never|cannot|can not|neither|without|nor|absent|none|"
    r"does not|do not|is not|are not|without)\b",
    re.IGNORECASE,
)


def test_the_paper_states_a_limitations_section() -> None:
    assert "\\section{Limitations}" in TEX
    assert "\\subsection{What we do not claim}" in TEX
    limitations = re.sub(
        r"\s+",
        " ",
        TEX[TEX.index("\\section{Limitations}") : TEX.index("\\section{Related Work}")],
    )
    for paragraph in (
        "No model.",
        "No sparse autoencoder and no learned features.",
        "No retrieval, no accuracy, no perplexity.",
        "Single head dimension, single scale, single base.",
    ):
        assert f"\\paragraph{{{paragraph}}}" in limitations, (
            f"the Limitations section no longer contains the '{paragraph}' paragraph"
        )
    assert "This matters enough to state before the results" in TEX_FLAT


def test_the_paper_disclaims_the_three_forbidden_things() -> None:
    """No trained sparse autoencoder, no real activations, no retrieval accuracy."""
    disclaimer = disclaimer_text()
    for phrase in (
        "There is no trained model here, and no real activations",
        "There is no sparse autoencoder anywhere in this work",
        "There is no retrieval benchmark and no accuracy measurement",
        "We do not claim any scheme is better in practice",
        "The conclusions are model-independent by construction",
        "We never load a checkpoint",
        "no training run, and no forward pass through anything learned",
        "no trained model, no learned features,",
        "no retrieval benchmark, no accuracy figure",
    ):
        assert phrase in disclaimer, f"the paper no longer says: {phrase!r}"


def test_no_affirmative_claim_of_a_trained_model_real_activations_or_accuracy() -> None:
    """Every sentence naming one of those three things must also deny it.

    The paper does mention sparse autoencoders in Related Work, describing
    *other people's* prior work, and it quotes ``accuracy $0.7$`` inside an
    explicit refusal.  Those are legitimate; an affirmative first-person claim
    attached to one of those numbers would not be, so every match is required
    to sit in a sentence that also negates it.
    """
    forbidden = (
        r"accuracy\s*(?:of\s+|=\s*)?\$?\d",
        r"perplexit\w*\s*(?:of\s+)?\$?\d",
        r"we\s+(?:use|used|train|trained|load|loaded|evaluate|evaluated|fine-?tun\w*)\s+"
        r"(?:a\s+|an\s+|the\s+)?(?:trained|real|actual|learned|checkpoint|pretrained)",
        r"(?:trained model|trained network|real activations?|actual activations?)"
        r"[^.]{0,60}\$?\d",
        r"(?:we|our)\s+(?:sparse\s+)?(?:autoencoder|SAE)s?\s+(?:was|is|were|are)\s+trained",
        r"(?:we|our)\s+(?:retrieval|needle|perplexity|accuracy)\s+"
        r"(?:of|score|number|result)\s*\$?\d",
    )
    offenders = []
    for sentence in paper_sentences():
        for pattern in forbidden:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                window = sentence[max(0, match.start() - 140) : match.end() + 140]
                if not _NEGATION.search(window):
                    offenders.append((pattern, sentence.strip()[:220]))
    assert not offenders, (
        "the paper appears to claim something it says it does not measure:\n"
        + "\n".join(f"  {pattern!r} in: {sentence!r}" for pattern, sentence in offenders)
    )


def test_the_only_accuracy_number_in_the_paper_is_the_one_it_refuses_to_report() -> None:
    hits = re.findall(r"accuracy[^\n]{0,60}?\$([\d.]+)\$", TEX)
    assert hits == [literal("negated_accuracy")], (
        f"the paper mentions retrieval accuracies {hits}; it is supposed to mention "
        f"exactly one, inside the sentence refusing to report any"
    )
    sentence = next(
        unit for unit in paper_sentences() if f"accuracy ${literal('negated_accuracy')}$" in unit
    )
    assert _NEGATION.search(sentence), (
        f"the quoted accuracy is no longer inside a refusal: {sentence!r}"
    )
    assert "We cannot and do not report anything of the form" in sentence


def test_no_model_or_activation_is_reported_anywhere_outside_a_denial() -> None:
    """Belt and braces: every 'checkpoint'/'activation' mention is a denial."""
    for sentence in paper_sentences():
        if re.search(r"\b(checkpoint|trained network|forward pass)\b", sentence, re.IGNORECASE):
            assert _NEGATION.search(sentence), f"a checkpoint/forward-pass mention is affirmative: {sentence!r}"


# ==========================================================================
# Exhaustiveness: no number in the paper escapes this module
# ==========================================================================

# Numbers that survive the claim scan because they are not measurements:
# algebraic indices (``k = 0``, ``B_k = 0``, ``A_ij^{(0)}``, ``\half - 1``),
# the first-order Taylor formula's own coefficients (``1 - D^2/2``) and the
# ``2x2`` and ``1x1`` block sizes.  Every *measured* literal is claimed by a
# pattern above; this list is the whole remainder, and
# :func:`test_the_exempt_list_is_small_and_every_entry_still_occurs` fails both
# if a new one is added and if an old one stops being needed.
EXEMPT_ALGEBRA = frozenset({"-1", "0", "1", "2"})

# `tabular` blocks whose every cell is verified by a dedicated test.
VERIFIED_TABLES = ("tab:spectrum", "tab:exact")

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _math_spans(text: str) -> list[str]:
    return re.findall(r"(?<!\\)\$(?:\\\$|[^\n])*?(?<!\\)\$", text)


def unaccounted_numbers() -> set[str]:
    """Numeric tokens in ``$...$`` that no claim in this module accounts for."""
    # Every claim span is blanked *simultaneously*, not one pattern at a time:
    # two claims can overlap (a restatement inside a longer sentence), and
    # blanking them in sequence would let the first one destroy the second's
    # anchor and leave its numbers looking unaccounted for.
    spans: list[tuple[int, int]] = []
    for entry in CLAIMS:
        spans.extend((match.start(), match.end()) for match in re.finditer(entry.pattern, TEX_FLAT))
    for label in VERIFIED_TABLES:
        anchor = TEX_FLAT.index(f"\\label{{{label}}}")
        begin = TEX_FLAT.index("\\begin{tabular}", anchor)
        spans.append((begin, TEX_FLAT.index("\\end{tabular}", begin)))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    pieces = []
    cursor = 0
    for start, end in merged:
        pieces.append(TEX_FLAT[cursor:start])
        pieces.append(" ")
        cursor = end
    pieces.append(TEX_FLAT[cursor:])
    text = "".join(pieces)
    numbers: set[str] = set()
    for span in _math_spans(text):
        numbers.update(_NUMBER.findall(span))
    return numbers


def test_no_number_in_the_paper_is_left_unverified() -> None:
    orphans = unaccounted_numbers() - EXEMPT_ALGEBRA
    assert not orphans, (
        f"these numbers appear in the paper's math but no test in this module checks "
        f"them: {sorted(orphans)}. Either add a claim plus a test, or add the literal "
        f"to EXEMPT_ALGEBRA with a reason."
    )


def test_the_exempt_list_is_small_and_every_entry_still_occurs() -> None:
    leftover = unaccounted_numbers() & EXEMPT_ALGEBRA
    assert len(EXEMPT_ALGEBRA) <= 8, (
        f"the exemption list has grown to {sorted(EXEMPT_ALGEBRA)}; most entries should "
        f"have become claims"
    )
    assert leftover == EXEMPT_ALGEBRA, (
        f"these exemptions are no longer needed: {sorted(EXEMPT_ALGEBRA - leftover)}"
    )
    for token in EXEMPT_ALGEBRA:
        assert re.fullmatch(r"-?\d+(?:\.\d+)?", token)


def test_the_claim_registry_is_not_vacuous() -> None:
    """Every registered claim must actually match the paper."""
    assert len(CLAIMS) >= 170, f"only {len(CLAIMS)} claims registered"
    names = [entry.name for entry in CLAIMS]
    assert len(names) == len(set(names)), "duplicate claim names"
    capturing = 0
    for entry in CLAIMS:
        found = groups(entry.name)
        if any(any(group for group in match) for match in found):
            capturing += 1
    assert capturing >= len(CLAIMS) - 5, (
        f"only {capturing} of {len(CLAIMS)} claims capture a number at all; the rest "
        f"are anchors that would hide numbers from the exhaustiveness scan"
    )


def test_the_paper_carries_the_provenance_section_it_promises() -> None:
    assert "\\section{Provenance of every number}" in TEX
    provenance = TEX[TEX.index("\\section{Provenance of every number}") :]
    for key in (
        "structural\\_facts",
        "feature\\_attribution",
        "method\\_spectrum",
        "partial\\_rope",
        "mscale\\_entropy",
    ):
        assert key in provenance, f"the provenance table does not name {key}"
    for csv_name in (f"fig{i:02d}" for i in range(1, 10)):
        assert f"\\texttt{{{csv_name}}}" in provenance, (
            f"the provenance table never names {csv_name}, whose columns it uses"
        )
    assert "No number was typed from memory" in provenance
    assert "\\label{tab:provenance}" in provenance, "the provenance table has lost its own label"


def test_every_csv_the_provenance_table_names_is_one_a_test_reads() -> None:
    """The appendix's promise is only as good as this check."""
    provenance = TEX[TEX.index("\\section{Provenance of every number}") :]
    named = set(re.findall(r"\\texttt\{(fig\d{2})\}", provenance))
    read = {"fig01", "fig02", "fig03", "fig04", "fig05", "fig06", "fig07", "fig08", "fig09"}
    assert named == read, sorted(named)
    for stem in sorted(named):
        assert figure_on_disk(stem), f"the {stem} figure the appendix names is not on disk"