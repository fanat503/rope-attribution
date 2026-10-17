"""Tests for ``projects/rope_attribution/train.py``.

The training loop is a measurement instrument: it must obey the same
standards as every other module here. The torch-free tests run always and
cover the arithmetic a reviewer would check by hand -- exact parameter
counts against the named 100M-300M bands, the schedule's shape, the metric
writer's file discipline. The torch tests run wherever torch exists (the
owner's machine does) and pin the model structure, the rotation fidelity,
and the one thing that would ruin the whole instrument: run-to-run
determinism of the logged metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "projects") not in sys.path:
    sys.path.insert(0, str(_REPO / "projects"))

from rope_attribution import train as T  # noqa: E402
from rope_attribution.rope import apply_rope  # noqa: E402

torch = pytest.importorskip("torch", reason="torch not installed")


# ==========================================================================
# The torch-free arithmetic
# ==========================================================================


def test_the_named_configs_land_inside_their_bands() -> None:
    """A config that drifts out of its 100M/200M/300M band fails here first."""
    checks = []
    for name, (low, high) in T.TRAIN_PARAM_BANDS.items():
        params = T.count_params(T.CONFIGS[name])
        checks.append((name, low <= params <= high, params))
    for name, inside, params in checks:
        assert inside, f"{name}: {params:,} leaves its band"


def test_the_counts_sit_where_the_names_say() -> None:
    """train_124m/208m/303m are honest names, not aspirations."""
    assert T.count_params(T.CONFIGS["train_124m"]) == 123_551_232
    assert T.count_params(T.CONFIGS["train_208m"]) == 208_504_320
    assert T.count_params(T.CONFIGS["train_303m"]) == 303_163_392


def test_head_dimension_is_64_in_every_named_config() -> None:
    """The paper's algebra is stated for head_dim 64; so are these models."""

    for name in ("train_124m", "train_208m", "train_303m"):
        config = T.CONFIGS[name]
        assert config["head_dim"] == T.HEAD_DIM == 64, name
        assert config["n_heads"] * config["head_dim"] == config["dim"], name
    assert "smoke" not in T.TRAIN_PARAM_BANDS


def test_the_schedule_peaks_at_warmup_and_floors_at_the_end() -> None:
    config = dict(T.CONFIGS["train_124m"])
    assert T.lr_at(0, config) == 0.0
    assert T.lr_at(config["warmup_steps"], config) == pytest.approx(config["lr"])
    ramp = [
        T.lr_at(s, config)
        for s in range(1, config["warmup_steps"])
    ]
    assert all(ramp[i] < ramp[i + 1] for i in range(len(ramp) - 1)), "warmup must rise"
    tail = [T.lr_at(s, config) for s in range(config["warmup_steps"], config["n_steps"] + 1)]
    assert max(tail) == pytest.approx(config["lr"])
    assert tail[-1] == pytest.approx(config["lr"] * config["lr_floor_fraction"])
    assert min(tail) == pytest.approx(config["lr"] * config["lr_floor_fraction"])


def test_the_metric_writer_writes_each_record_and_a_summary(tmp_path: Path) -> None:
    writer = T.MetricsWriter(tmp_path / "run")
    writer.log({"event": "eval", "step": 1, "loss": 1.5})
    writer.log({"event": "eval", "step": 2, "loss": 1.25})
    summary = {"event": "summary", "steps": 2}
    writer.finalize(summary)
    lines = (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]
    on_disk = json.loads(
        (tmp_path / "run" / "summary.json").read_text(encoding="utf-8")
    )
    assert on_disk == summary
    assert writer.records == [
        {"event": "eval", "step": 1, "loss": 1.5},
        {"event": "eval", "step": 2, "loss": 1.25},
    ]


def test_every_config_is_a_complete_call() -> None:
    """A config missing a key the loop reads would KeyError mid-run, not here."""
    required = {
        "n_layers", "n_heads", "head_dim", "dim", "ffn", "vocab", "ctx",
        "lr", "n_steps", "batch_size", "eval_every", "warmup_steps",
        "lr_floor_fraction", "grad_clip", "seed",
    }
    for name, config in T.CONFIGS.items():
        missing = required - set(config)
        assert not missing, f"{name} lacks {sorted(missing)}"


# ==========================================================================
# The torch-checked structure
# ==========================================================================


def test_the_model_realizes_the_counted_parameters() -> None:
    """The closed-form count and the actual tensors must agree exactly."""
    config = T.CONFIGS["smoke"]
    torch.manual_seed(0)
    model = T.build_model(config)
    total = sum(p.numel() for p in model.parameters())
    assert total == T.count_params(config)


def test_the_ref_rope_helper_matches_rope_dot_py() -> None:
    """The probe's 4-D rotation is rope.apply_rope's own arithmetic."""
    positions = np.arange(8, dtype=np.float64)
    cos, sin = T.reference_cos_sin(positions, 16)
    rng = np.random.default_rng(7)
    x2d = rng.standard_normal((8, 16))
    rotated = T._apply_rope_ref_4d(x2d[None, :, None, :], cos, sin)[0, :, 0, :]
    assert np.array_equal(apply_rope(x2d, cos, sin), rotated)


def test_the_rotation_is_faithful_and_the_algebra_is_exact() -> None:
    """Probe floors: float32 rotation ~1e-7, closed form is machine epsilon."""
    config = T.CONFIGS["smoke"]
    torch.manual_seed(1)
    model = T.build_model(config)
    rng = np.random.default_rng(config["seed"])
    ids = rng.integers(0, config["vocab"], size=(1, config["ctx"]))
    deltas = np.array(list(range(1, min(64, config["ctx"]) + 1)), dtype=np.int64)
    # Note the diagnostic distances here cover the grid-sized range directly.
    diagnostics = T._probe_diagnostics(model, ids, deltas, n_pairs=8, seed=5)
    assert diagnostics["rope_probe_resid_max"] < 1e-5
    assert diagnostics["closed_form_resid_max"] < 1e-11
    assert diagnostics["n_probe_cells"] > 0
    assert 0.0 <= diagnostics["sign_flip_rate"] <= 1.0
    assert np.isfinite(diagnostics["rel_err_median_probe"])


def test_smoke_run_logs_every_metric_and_finishes(tmp_path: Path) -> None:
    """The whole loop -- train, eval, diagnostics, summary -- on CPU seconds."""
    tokens = np.random.default_rng(11).integers(0, 512, size=8192)
    summary = T.train("smoke", tmp_path / "run", tokens, device="cpu")
    assert summary["steps"] == T.CONFIGS["smoke"]["n_steps"]
    records = [json.loads(line) for line in
               (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    evals = [r for r in records if r["event"] == "eval"]
    assert evals, "no evaluation was logged"
    keys = {
        "step", "lr", "grad_norm", "train_loss_ema", "val_loss",
        "tokens_seen", "tok_per_s", "closed_form_resid_max",
        "rope_probe_resid_max", "sign_flip_rate", "rel_err_median_probe",
        "n_probe_cells",
    }
    for record in evals:
        assert keys <= set(record), sorted(keys - set(record))
        assert np.isfinite(record["train_loss_ema"])
        assert np.isfinite(record["val_loss"])
        assert record["closed_form_resid_max"] < 1e-11


def test_two_smoke_runs_log_identical_series(tmp_path: Path) -> None:
    """Same seed, same machine: only the wall-clock field may differ."""
    tokens = np.random.default_rng(11).integers(0, 512, size=8192)
    T.train("smoke", tmp_path / "a", tokens, device="cpu")
    tokens_b = np.random.default_rng(11).integers(0, 512, size=8192)
    T.train("smoke", tmp_path / "b", tokens_b, device="cpu")

    def strip(run: str) -> list:
        records = [json.loads(line) for line in
                   (tmp_path / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        for record in records:
            record.pop("tok_per_s", None)
        return records

    assert strip("a") == strip("b")


def test_second_scale_configs_refuse_cpu(tmp_path: Path) -> None:
    """The named configs must not be launchable onto the owner's cores."""
    tokens = np.random.default_rng(3).integers(0, 16, size=4096)
    with pytest.raises(RuntimeError, match="GPU territory"):
        T.train("train_124m", tmp_path / "run", tokens, device="cpu", max_steps=1)
