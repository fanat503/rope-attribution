"""Tests for ``projects/rope_attribution/seed_variance.py``.

The artifact costs three full usefulness sweeps to regenerate (about ten
minutes on one thread), so the suite does not repeat the sweep; the
reproduce-grade check of the underlying sweep is ``test_usefulness``'s.
What is pinned here is everything a reader of the artifact could be misled
by; the paper sentence that quotes it is re-derived from the same artifact
in ``test_paper_trained_claims.test_the_seed_variance_is_as_printed``:

* the schema, the pre-registered seed list, and the shared sampling
  parameters are what the module says they are;
* the row at the default seed reproduces ``usefulness.json`` exactly, so
  the two artifacts cannot drift apart silently;
* the reported mean and sample standard deviation are the arithmetic of
  the rows the artifact ships, recomputed here rather than trusted;
* the seeds actually produced different draws: a dead seed path would
  otherwise ship three identical rows and ``sd = 0.0``, which would look
  like perfect stability;
* ``run_all`` forwards its seed to ``measure_model`` -- the one-line join
  that, if dropped, makes the previous property pass on a repeated row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_PROJECTS = _REPO / "projects"
if str(_PROJECTS) not in sys.path:
    sys.path.insert(0, str(_PROJECTS))

from rope_attribution import real_model as RM  # noqa: E402
from rope_attribution import seed_variance as S  # noqa: E402
from rope_attribution import usefulness as U  # noqa: E402

_JSON = _REPO / "results" / "usefulness_seeds.json"
_USEFULNESS = _REPO / "results" / "usefulness.json"


def _artifact() -> dict:
    return json.loads(_JSON.read_text(encoding="utf-8"))


def _usefulness() -> dict:
    return json.loads(_USEFULNESS.read_text(encoding="utf-8"))


def test_the_artifact_has_the_promised_shape() -> None:
    artifact = _artifact()
    assert artifact["schema"] == "rope_attribution/usefulness_seeds/v1"
    assert artifact["model_ids"] == list(U.DEFAULT_MODEL_IDS)
    assert artifact["n_content_pairs"] == RM.N_CONTENT_PAIRS
    assert [row["seed"] for row in artifact["per_seed"]] == list(S.SEEDS)
    assert artifact["seeds"] == list(S.SEEDS)
    for row in artifact["per_seed"]:
        assert set(row["pooled"]) == set(S.POOLED_FIELDS)
        assert [m["model_id"] for m in row["models"]] == list(U.DEFAULT_MODEL_IDS)
        for model in row["models"]:
            assert set(model) == {"model_id", *S.MODEL_FIELDS}


def test_the_shared_seed_reproduces_the_published_sweep() -> None:
    """The default-seed row and usefulness.json are the same run.

    The row stores the five fields in ``MODEL_FIELDS``/``POOLED_FIELDS``, not the
    whole model record: ``usefulness.json`` also carries ``head_rates`` and the
    per-prompt rows, which the seed artifact deliberately omits. The comparison
    is therefore over exactly the stored fields, and each one must also exist in
    the published artifact -- that is the join that could break silently.

    Mutation verdict: changing any pooled rate or any per-model field in the seed
    artifact makes one of these equality assertions fail; dropping a field from
    ``usefulness.json`` makes the membership assertion fail.
    """
    artifact = _artifact()
    usefulness = _usefulness()
    row = next(r for r in artifact["per_seed"] if r["seed"] == U.DEFAULT_SEED)
    for field in S.POOLED_FIELDS:
        assert field in usefulness["pooled"], f"usefulness.json lost {field}"
        assert row["pooled"][field] == usefulness["pooled"][field], field
    for stored, published in zip(row["models"], usefulness["models"], strict=True):
        assert stored["model_id"] == published["model_id"]
        for field in S.MODEL_FIELDS:
            assert field in published, f"usefulness.json lost {field}"
            assert stored[field] == published[field], field


def test_the_mean_and_sd_are_the_arithmetic_of_the_rows() -> None:
    artifact = _artifact()
    for field in S.POOLED_FIELDS:
        values = [row["pooled"][field] for row in artifact["per_seed"]]
        assert artifact["mean"][field] == float(np.mean(values))
        assert artifact["sd"][field] == float(np.std(values, ddof=1))


def test_the_seeds_actually_produced_different_draws() -> None:
    """A dead seed path would ship identical rows and sd = 0.0."""
    artifact = _artifact()
    rates = [row["pooled"]["sign_flip_fraction"] for row in artifact["per_seed"]]
    assert len(set(rates)) == len(rates), (
        "the three seeds gave identical pooled rates; the seed is not reaching "
        "the content-pair draw"
    )
    assert artifact["sd"]["sign_flip_fraction"] > 0.0


def test_run_all_forwards_the_seed_to_measure_model(monkeypatch) -> None:
    """``_pool`` needs the head-level records, so the stub must carry them.

    ``measure_model`` returns a full record; this stub is the minimum
    ``run_all`` actually consumes. A stub missing ``head_rates`` fails inside
    ``_pool``, which is the failure this test caught when the pooling unit
    changed from head-prompt instances to unique layer-heads.

    Mutation verdict: dropping the ``seed=seed`` forwarding makes ``seen`` hold
    the default instead of 123.
    """
    seen: list[int] = []

    def fake(model_id, seed=U.DEFAULT_SEED):
        seen.append(seed)
        return {
            "model_id": model_id,
            "sign_flip_fraction_pooled": 0.5,
            "sign_flip_fraction_min_head": 0.1,
            "sign_flip_fraction_max_head": 0.9,
            "rel_err_median": 1.0,
            "rel_err_p90": 1.5,
            "n_head_prompt_instances": 2,
            "head_rates": [
                {"sign_flip_fraction": 0.1, "rel_err_median": 1.0},
                {"sign_flip_fraction": 0.9, "rel_err_median": 1.5},
            ],
        }

    monkeypatch.setattr(U, "measure_model", fake)
    U.run_all(model_ids=["stub"], seed=123)
    assert seen == [123]
