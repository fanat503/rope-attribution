"""Tests for ``projects/rope_attribution/figures.py``.

The figure set's headline contract -- every plotted value is computed at run
time or read from a regenerable artifact -- is checked where it is observable:
the CSV sidecars are re-derived against the paper's claims in
``test_paper_claims.py``, and the README's figure table is checked in
``test_readme_claims.py``. This file covers what those cannot see: the guard
that keeps fig11 from plotting a sweep its own module no longer produces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "projects") not in sys.path:
    sys.path.insert(0, str(_REPO / "projects"))


def test_fig11_refuses_a_sweep_artifact_that_disagrees_with_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is the only thing that catches a stale sweep before plotting.

    ``fig11_sensitivity_sweep`` reads ``results/sensitivity.json`` instead of
    re-running the sweep, and it checks the artifact's schema and its
    base/head-dim grid against ``sensitivity.BASES`` / ``HEAD_DIMS``. Both
    doctored artifacts below would plot without an error if that check were
    removed, so ``pytest.raises`` is the detector: delete the guard and this
    test fails. Plotting and writing are stubbed so that a removed guard cannot
    touch the committed figures while the test is busy failing.
    """
    F = pytest.importorskip("rope_attribution.figures")
    source = _REPO / "results" / "sensitivity.json"
    if not source.exists():
        pytest.skip("results/sensitivity.json is absent; run sensitivity.py")
    good = json.loads(source.read_text(encoding="utf-8"))

    def plant(data: dict) -> None:
        target = tmp_path / "results"
        target.mkdir(exist_ok=True)
        (target / "sensitivity.json").write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(F, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(F, "_save", lambda *args, **kwargs: Path("stub.png"))
    monkeypatch.setattr(F, "_write_csv", lambda *args, **kwargs: Path("stub.csv"))

    foreign_schema = json.loads(json.dumps(good))
    foreign_schema["schema"] = "rope_attribution/sensitivity/other"
    plant(foreign_schema)
    with pytest.raises(ValueError, match="schema"):
        F.fig11_sensitivity_sweep()

    shifted_grid = json.loads(json.dumps(good))
    shifted_grid["bases"] = [1234567.0] + list(shifted_grid["bases"][1:])
    plant(shifted_grid)
    with pytest.raises(ValueError, match="disagrees"):
        F.fig11_sensitivity_sweep()
