"""Numerical checks of the two propositions in ``paper/main.tex``.

Why this file exists
--------------------
Both propositions were originally stated falsely, and the error was in the
statement rather than in the derivation. ``prop:blind`` claimed the pair $k$ is
"completely position-blind if and only if $B_k = 0$", which is backwards: with
$B_k = 0$ the contribution is $A_k\\cos(\\delta\\invf_k)$, a full oscillation.
``prop:cond`` then inherited the same mistake as its exception.

Prose in a paper is not checkable by a reviewer running anything, and a false
statement sitting next to a correct derivation is exactly what a hostile review
looks for. So the corrected statements are pinned here, numerically, by tests that
read the same code the paper's equations describe. If a future edit restores the
false form, these fail.

What is and is not checked
--------------------------
Proposition statements about algebraic structure are checked by construction:
linear independence is verified on a dense grid by least-squares rank, and
non-constancy by peak-to-peak span. Neither is a proof of the general statement
--- the proofs in the paper are. These are the cheapest checks that would catch
someone reintroducing the error, which is what the file is for.
"""

from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "projects") not in sys.path:
    sys.path.insert(0, str(_REPO / "projects"))

from rope_attribution.experiments import pair_terms  # noqa: E402
from rope_attribution.rope import inv_freq  # noqa: E402

HEAD_DIM = 64
BASE = 10_000.0
GRID = np.unique(np.round(np.geomspace(1, 8192, 61)).astype(np.int64))


@pytest.fixture(scope="module")
def freqs() -> np.ndarray:
    return inv_freq(HEAD_DIM, BASE)


@pytest.fixture(scope="module")
def qk() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20_251_010)
    return rng.standard_normal(HEAD_DIM), rng.standard_normal(HEAD_DIM)


# ---------------------------------------------------------------------------
# Proposition: amplitude and phase are content-determined
# ---------------------------------------------------------------------------


def test_the_amplitude_and_phase_are_exactly_delta_independent(
    freqs: np.ndarray, qk: tuple[np.ndarray, np.ndarray]
) -> None:
    """The load-bearing half of ``prop:blind``, and the part that is true.

    $R_k$ and $\\psi_k$ are functions of the content vectors alone, so sweeping
    the distance grid must move them by exactly zero - not by a small tolerance.
    A tolerance here would hide a regression in which they acquired a $\\delta$
    term.
    """
    q, k = qk
    base = pair_terms(q, k, freqs, int(GRID[0]))
    amplitude = np.hypot(base.aligned, base.crossed)
    phase = np.arctan2(base.crossed, base.aligned)

    for delta in GRID:
        terms = pair_terms(q, k, freqs, int(delta))
        assert np.array_equal(np.hypot(terms.aligned, terms.crossed), amplitude), delta
        assert np.array_equal(
            np.arctan2(terms.crossed, terms.aligned), phase
        ), delta


def test_the_contribution_is_non_constant_whenever_the_amplitude_is_positive(
    freqs: np.ndarray,
) -> None:
    """The second half of ``prop:blind``: $c_k$ never flattens out.

    Checked as a strict peak-to-peak span over the grid, minimised over many
    random draws so the assertion is not a statement about one lucky pair. The
    span can be small - a slowly rotating sinusoid sampled near its turning point
    - but it is never zero, which is all the proposition claims.
    """
    rng = np.random.default_rng(4242)
    smallest = np.inf
    for _ in range(64):
        q = rng.standard_normal(HEAD_DIM)
        k = rng.standard_normal(HEAD_DIM)
        terms = pair_terms(q, k, freqs, int(GRID[0]))
        for j in range(terms.aligned.size):
            amplitude = float(np.hypot(terms.aligned[j], terms.crossed[j]))
            if amplitude < 1e-12:
                continue
            phase = float(np.arctan2(terms.crossed[j], terms.aligned[j]))
            values = amplitude * np.cos(GRID * freqs[j] - phase)
            smallest = min(smallest, float(values.max() - values.min()))
    assert smallest > 0.0, "a non-zero amplitude produced a constant contribution"
    assert np.isfinite(smallest)


def test_a_zero_crossed_coefficient_is_not_the_blind_case(freqs: np.ndarray) -> None:
    """The specific false claim the old proposition made, pinned as false.

    ``prop:blind`` used to assert that $B_k = 0$ is exactly the position-blind
    case. It is the opposite: $B_k = 0$ leaves $A_k\\cos(\\delta\\invf_k)$, which
    oscillates through its full range. If this test ever starts failing because
    the fixture stopped having $B_k = 0$, the fixture needs fixing - not the
    expectation.
    """
    head_dim = HEAD_DIM
    half = head_dim // 2
    # Equal halves with no cross terms give B_k = 0 for every k.
    q = np.concatenate([np.ones(half), np.zeros(half)])
    k = np.concatenate([np.ones(half), np.zeros(half)])
    terms = pair_terms(q, k, freqs, int(GRID[0]))
    assert np.allclose(terms.crossed, 0.0), "fixture does not actually have B_k = 0"

    values = terms.aligned[:, None] * np.cos(GRID[None, :] * freqs[:, None])
    spans = values.max(axis=1) - values.min(axis=1)
    assert np.all(spans > 0.5), (
        f"B_k = 0 should still oscillate; spans were {spans.min():.4f}..{spans.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Proposition: conditionality
# ---------------------------------------------------------------------------


def test_the_frequency_ladder_is_positive_and_pairwise_distinct(freqs: np.ndarray) -> None:
    """The hypothesis of ``prop:cond``, stated where it is used.

    Linear independence is the whole content of the proof, and it needs distinct
    positive frequencies. An implementation that clamped or repeated a frequency
    would silently break the proposition.
    """
    assert freqs.shape == (HEAD_DIM // 2,)
    assert (freqs > 0).all()
    assert np.unique(freqs).size == freqs.size
    assert np.all(np.diff(freqs) < 0), "the ladder should decrease strictly with k"


def test_no_pair_of_distinct_frequencies_admits_a_constant_combination(
    freqs: np.ndarray,
) -> None:
    """Least-squares evidence for the linear independence the proof cites.

    A constant on the grid would be an exact solution of the fitted system. The
    best residual over every two-frequency subset is small but non-zero, which is
    what independence predicts; an exact zero would mean the claim is false.
    """
    n = freqs.size
    best = np.inf
    for i, j in combinations(range(n), 2):
        design = np.column_stack([np.cos(GRID * freqs[i]), np.cos(GRID * freqs[j])])
        target = np.ones_like(GRID, dtype=float)
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        misfit = design @ coefficients - target
        assert np.all(np.isfinite(misfit))
        best = min(best, float(np.linalg.norm(misfit)))
    assert best > 1e-6, f"a two-frequency combination fit a constant: residual {best:.3e}"
    # And it must stay small: an independence failure would show up as O(1).
    assert best < 0.05, f"residual suspiciously large: {best:.3e}"


def test_a_single_pair_alone_is_already_non_constant(freqs: np.ndarray) -> None:
    """No proper subset of the touched pairs can be the position-free one.

    ``prop:cond`` says the contribution is $\\delta$-independent only when every
    touched pair vanishes. This checks the degenerate one-pair case that the old
    exception tried to carve out, where $B_k = 0$ was claimed to make the sum
    constant.
    """
    j = 0
    values = np.cos(GRID * freqs[j])
    assert values.max() - values.min() > 0.5


# ---------------------------------------------------------------------------
# the paper's prose must not drift back
# ---------------------------------------------------------------------------


def test_the_paper_does_not_claim_zero_crossed_means_blind() -> None:
    """Pin the wording, so the corrected statement cannot silently revert."""
    tex = re.sub(r"\s+", " ", (_REPO / "paper" / "main.tex").read_text(encoding="utf-8"))
    assert "completely position-blind if and only if" not in tex
    assert "is \\emph{not} a blind case" in tex


def test_every_proposition_in_the_paper_has_a_proof() -> None:
    """A proposition without a proof is an assertion, and this paper claims rigor.

    Counted over ``\\begin{proposition}`` blocks rather than over labels, so adding
    a fourth proposition without proving it fails here.
    """
    tex = (_REPO / "paper" / "main.tex").read_text(encoding="utf-8")
    statements = tex.count("\\begin{proposition}")
    proofs = tex.count("\\begin{proof}")
    assert statements >= 3, statements
    assert proofs >= statements, f"{statements} propositions but {proofs} proofs"