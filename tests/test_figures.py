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


def test_fig12_reads_the_committed_qk_artifact_and_refuses_a_foreign_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The figure must plot the artifact the claim gates read, not a copy.

    ``fig12_qk_case_study`` reads ``results/qk_attribution.json`` and checks its
    schema and head count. Both doctored artifacts below would plot without an
    error if those checks were removed, so ``pytest.raises`` is the detector.
    Plotting and writing are stubbed so a removed guard cannot touch the
    committed figures while the test is busy failing.

    Mutation verdict: delete the schema check or the ``n_heads`` check and the
    corresponding ``pytest.raises`` block fails.
    """
    F = pytest.importorskip("rope_attribution.figures")
    source = _REPO / "results" / "qk_attribution.json"
    if not source.exists():
        pytest.skip("results/qk_attribution.json is absent; run qk_attribution.py")
    good = json.loads(source.read_text(encoding="utf-8"))

    def plant(data: dict) -> None:
        target = tmp_path / "results"
        target.mkdir(exist_ok=True)
        (target / "qk_attribution.json").write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(F, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(F, "_save", lambda *args, **kwargs: Path("stub.png"))
    monkeypatch.setattr(F, "_write_csv", lambda *args, **kwargs: Path("stub.csv"))

    plant(good)
    record = F.fig12_qk_case_study()
    assert record.stem == "fig12_qk_case_study"
    aggregate = good["aggregate"]
    assert f"{100 * aggregate['sweep_rotary']['sign_flip_fraction']:.2f}%" in record.headline

    foreign_schema = json.loads(json.dumps(good))
    foreign_schema["schema"] = "qk-attribution/other"
    plant(foreign_schema)
    with pytest.raises(ValueError, match="schema"):
        F.fig12_qk_case_study()

    short_heads = json.loads(json.dumps(good))
    short_heads["heads"] = short_heads["heads"][:-1]
    plant(short_heads)
    with pytest.raises(ValueError, match="head records"):
        F.fig12_qk_case_study()


def test_save_refuses_a_suptitle_that_stretches_the_canvas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fig09 shipped for a while with a 281-character suptitle line and a saved
    canvas 2.36x its declared size: the plot was 21% of the image width and one
    panel printed 0.28 in wide. ``_save`` now measures the tight bbox against
    the declared figsize and refuses anything over 1.6x (wrapping the suptitle
    brought fig09 to 0.99x). Both controls run the real save path, so deleting
    the guard makes the negative control fail by not raising.
    """
    F = pytest.importorskip("rope_attribution.figures")
    plt = pytest.importorskip("matplotlib.pyplot")
    monkeypatch.setattr(F, "FIGS_DIR", tmp_path)

    ok_fig, ok_ax = plt.subplots(figsize=(4.0, 3.0))
    ok_ax.plot([0.0, 1.0], [0.0, 1.0])
    path = F._save(ok_fig, "canvas_ok", "a short title")
    assert path == tmp_path / "canvas_ok.png"
    assert path.exists()

    tall_fig, tall_ax = plt.subplots(figsize=(4.0, 3.0))
    tall_ax.plot([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="1.6x the declared figsize"):
        F._save(tall_fig, "canvas_blown", "x" * 300)
    assert not (tmp_path / "canvas_blown.png").exists()
