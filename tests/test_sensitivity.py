"""Tests for ``projects/rope_attribution/sensitivity.py``.

The module answers the configurational objection against the paper's cost
statistic: the rate is measured at base 10000 and head dimension 64, so a
reviewer may ask whether the number belongs to the construction or to those
two constants. The sweep must therefore pin (a) the cell definition to
``usefulness.measure_pair_cost`` -- already tested there -- and (b) its own
bookkeeping here: the pool, the pairing across bases, the monotonicity that
the paper's sentence claims, and every printed number's re-derivation from
``results/sensitivity.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "projects") not in sys.path:
    sys.path.insert(0, str(_REPO / "projects"))

from rope_attribution import sensitivity as S  # noqa: E402
from rope_attribution import usefulness as U  # noqa: E402
from test_paper_claims import (  # noqa: E402
    assert_paper_number_matches,
    literal,
)

_JSON = _REPO / "results" / "sensitivity.json"


def _committed() -> dict:
    if not _JSON.exists():
        pytest.skip("results/sensitivity.json is absent; run sensitivity.py")
    return json.loads(_JSON.read_text(encoding="utf-8"))


# ==========================================================================
# The measurement itself, on one configuration, without the artifact
# ==========================================================================


def test_measure_config_is_deterministic() -> None:
    """Same seed, same tokens, same pairs, same pooled counts."""
    a = S.measure_config(64, 10000.0, S.SWEEP_BASE_SEED)
    b = S.measure_config(64, 10000.0, S.SWEEP_BASE_SEED)
    assert a == b


def test_measure_config_measures_every_requested_pair() -> None:
    """The pool is the requested pair count, not however many survived."""
    rec = S.measure_config(64, 10000.0, S.SWEEP_BASE_SEED)
    assert rec["n_content_pairs"] == S.N_PAIRS
    assert rec["n_material_cells"] > 0
    assert rec["rope_pairs"] == 32
    base = rec["n_material_cells"]
    assert rec["sign_flip_fraction"] == pytest.approx(
        rec["n_sign_flips"] / base
    )
    # The relative errors are per-pair maxima over the grid, so "median above
    # one" is what the paper's sentence claims for the whole sweep.
    assert rec["rel_err_median"] > 1.0


def test_the_pair_draw_never_samples_a_pair_with_itself() -> None:
    rng = np.random.default_rng(7)
    for m, n in S._draw_pairs(rng):
        assert m != n


def test_same_seed_gives_the_same_tokens_across_bases() -> None:
    """The base sweep is paired: only the frequencies change, not the draw."""
    rng_a = np.random.default_rng(S.SWEEP_BASE_SEED + 2)
    rng_b = np.random.default_rng(S.SWEEP_BASE_SEED + 2)
    qa, ka = S._synthetic_tokens(64, rng_a)
    qb, kb = S._synthetic_tokens(64, rng_b)
    assert np.array_equal(qa, qb)
    assert np.array_equal(ka, kb)


# ==========================================================================
# The committed artifact
# ==========================================================================


def test_the_committed_artifact_is_self_describing() -> None:
    data = _committed()
    assert data["schema"] == "rope_attribution/sensitivity/v1"
    assert data["bases"] == list(S.BASES)
    assert data["head_dims"] == list(S.HEAD_DIMS)
    assert data["n_sweep_seeds"] == S.N_SWEEP_SEEDS
    assert data["sweep_base_seed"] == S.SWEEP_BASE_SEED
    assert data["distance_grid"] == list(U.DISTANCE_GRID)
    assert len(data["records"]) == len(S.BASES) * len(S.HEAD_DIMS)


def test_every_record_is_internally_consistent() -> None:
    for rec in _committed()["records"]:
        head = rec["per_seed"]
        assert len(head) == S.N_SWEEP_SEEDS, rec
        seeds = {h["seed"] for h in head}
        assert seeds == {S.SWEEP_BASE_SEED + i for i in range(S.N_SWEEP_SEEDS)}
        for h in head:
            assert h["n_content_pairs"] == S.N_PAIRS, h
            assert h["rel_err_median"] > 1.0, h
        rates = [h["sign_flip_fraction"] for h in head]
        assert rec["sign_flip_fraction_mean"] == pytest.approx(
            float(np.mean(rates))
        ), rec
        assert rec["sign_flip_fraction_sd"] == pytest.approx(
            float(np.std(rates, ddof=1))
        ), rec
        # Three seeds on the same construction: the spread between them must
        # stay small, else the band the paper quotes is seed noise, not config.
        assert rec["sign_flip_fraction_sd"] < 0.02, rec


def test_the_bands_are_the_min_and_max_of_their_records() -> None:
    data = _committed()
    at_64 = [r for r in data["records"] if r["head_dim"] == 64]
    rates = [r["sign_flip_fraction_mean"] for r in at_64]
    assert data["base_band_at_dim_64"] == [min(rates), max(rates)]
    at_10k = [r for r in data["records"] if r["base"] == 10000.0]
    rates = [r["sign_flip_fraction_mean"] for r in at_10k]
    assert data["dim_band_at_base_10000"] == [min(rates), max(rates)]


def test_the_rate_is_monotone_in_the_base() -> None:
    """The paper's sentence says ``monotonically``: pin the direction.

    Higher base means slower rotation per grid distance, which is why the
    constant fits better and the flip rate drops. A sweep that stops being
    monotone breaks the printed sentence before a reader can.
    """
    at_64 = sorted(
        (r for r in _committed()["records"] if r["head_dim"] == 64),
        key=lambda r: r["base"],
    )
    rates = [r["sign_flip_fraction_mean"] for r in at_64]
    assert all(rates[i] > rates[i + 1] for i in range(len(rates) - 1))


def test_the_head_dimension_is_not_load_bearing() -> None:
    """The published answer to the 'you chose dim 64' objection: a tight band."""
    lo, hi = _committed()["dim_band_at_base_10000"]
    assert hi - lo < 0.02, (lo, hi)


# ==========================================================================
# The paper's paragraph, re-derived
# ==========================================================================


def test_the_base_claim_is_re_derived() -> None:
    r"""``from $32.0\%$ down to $18.3\%$`` -- order and values from the band."""
    data = _committed()
    lo, hi = data["base_band_at_dim_64"]
    assert_paper_number_matches(
        100.0 * hi, literal("sensitivity_base", 4), "the sweep's highest rate"
    )
    assert_paper_number_matches(
        100.0 * lo, literal("sensitivity_base", 5), "the sweep's lowest rate"
    )
    # The written order is "from ... down to ...": the band must actually run
    # downhill, which is the monotonicity test's whole point.
    assert hi > lo


def test_the_base_claim_names_bases_and_a_head_dimension() -> None:
    """The written config literals must be the swept ones, not near-misses."""
    data = _committed()
    assert literal("sensitivity_base", 1) == "10^{3}"
    assert literal("sensitivity_base", 2) == "5\\times10^{5}"
    assert literal("sensitivity_base", 3) == "64"
    assert data["bases"][0] == 1000.0
    assert data["bases"][-1] == 500000.0


def test_the_dim_claim_is_re_derived() -> None:
    r"""``at $25.6\%$ to $26.3\%$`` -- ascending, from the committed band."""
    data = _committed()
    lo, hi = data["dim_band_at_base_10000"]
    assert_paper_number_matches(
        100.0 * lo, literal("sensitivity_dim", 4), "the dim sweep's lowest rate"
    )
    assert_paper_number_matches(
        100.0 * hi, literal("sensitivity_dim", 5), "the dim sweep's highest rate"
    )
    assert literal("sensitivity_dim", 1) == "32"
    assert literal("sensitivity_dim", 2) == "128"
    assert literal("sensitivity_dim", 3) == "10^{4}"


def test_the_rel_err_claim_is_re_derived() -> None:
    """``above $1.0$ everywhere, from $1.03$ to $1.18$``."""
    medians = [r["rel_err_median_mean"] for r in _committed()["records"]]
    assert_paper_number_matches(
        min(medians), literal("sensitivity_relerr", 2), "the lowest rel-err median"
    )
    assert_paper_number_matches(
        max(medians), literal("sensitivity_relerr", 3), "the highest rel-err median"
    )
    # The claim of the sentence's first half: every configuration's median
    # relative error clears the signal, no matter the base or the dimension.
    assert min(medians) > float(literal("sensitivity_relerr", 1))


def test_the_synthetic_construction_is_calibrated() -> None:
    """At the published configuration the synthetic rate matches the trained one.

    ``usefulness.json``'s pooled rate comes from trained weights on three
    checkpoints; this sweep's mean at (base 10000, head dimension 64) is the
    same configuration drawn synthetically. The construction was designed to
    mirror a real layer -- it was never tuned to the trained number -- so the
    two must agree to about a percentage point. A drift here means the token
    construction or the projections changed without the sweep noticing.
    """
    trained = json.loads(
        (_REPO / "results" / "usefulness.json").read_text(encoding="utf-8")
    )
    trained_rate = 100.0 * trained["pooled"]["sign_flip_fraction"]
    row = next(
        r
        for r in _committed()["records"]
        if r["base"] == 10000.0 and r["head_dim"] == 64
    )
    assert abs(100.0 * row["sign_flip_fraction_mean"] - trained_rate) < 1.5, (
        row["sign_flip_fraction_mean"],
        trained_rate,
    )


def test_the_module_reproduces_the_committed_artifact() -> None:
    """results/sensitivity.json must be what the module produces right now.

    Every other test in this file reads the JSON, so without this one the
    artifact could silently diverge from the code the paper's numbers are
    re-derived from. The sweep is seeded: its counts are integers and are
    compared exactly, and the rates built from them are compared to 1e-12
    relative, the way test_readme_claims compares measurements.json, because
    the last bits of a float64 sum depend on the host BLAS.
    """
    committed = _committed()
    fresh = S.run_all()
    assert fresh["schema"] == committed["schema"]
    assert fresh["bases"] == committed["bases"]
    assert fresh["head_dims"] == committed["head_dims"]
    old_rows = {(r["base"], r["head_dim"]): r for r in committed["records"]}
    assert len(old_rows) == len(fresh["records"]), (
        "sensitivity.json lists a different number of configurations than the "
        "module sweeps"
    )
    for row in fresh["records"]:
        old = old_rows[(row["base"], row["head_dim"])]
        for new_seed, old_seed in zip(row["per_seed"], old["per_seed"], strict=True):
            for field in ("n_material_cells", "n_sign_flips"):
                assert new_seed[field] == old_seed[field], (
                    f"sensitivity.json is stale at base {row['base']} / dim "
                    f"{row['head_dim']}: {field} {old_seed[field]} committed vs "
                    f"{new_seed[field]} fresh"
                )
            for field in ("sign_flip_fraction", "rel_err_median"):
                assert math.isclose(
                    new_seed[field], old_seed[field], rel_tol=1e-12, abs_tol=1e-12
                ), (row["base"], row["head_dim"], field, old_seed[field], new_seed[field])
        for field in (
            "sign_flip_fraction_mean",
            "sign_flip_fraction_sd",
            "rel_err_median_mean",
        ):
            assert math.isclose(
                row[field], old[field], rel_tol=1e-12, abs_tol=1e-12
            ), (row["base"], row["head_dim"], field, old[field], row[field])
    for key in ("base_band_at_dim_64", "dim_band_at_base_10000"):
        for new_value, old_value in zip(fresh[key], committed[key], strict=True):
            assert math.isclose(
                new_value, old_value, rel_tol=1e-12, abs_tol=1e-12
            ), (key, old_value, new_value)
