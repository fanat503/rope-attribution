"""Tests for ``projects/rope_attribution/train.py``.

The training loop is a measurement instrument: it must obey the same
standards as every other module here. The torch-free tests run always and
cover the arithmetic a reviewer would check by hand -- exact parameter
counts against the named 100M-300M bands, the schedule's shape, the
next-token window construction, the CPU-refusal matrix, the metric
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
    """The six named configs are honest names, not aspirations."""
    assert T.count_params(T.CONFIGS["train_102m"]) == 102_312_960
    assert T.count_params(T.CONFIGS["train_124m"]) == 123_551_232
    assert T.count_params(T.CONFIGS["train_152m"]) == 151_868_928
    assert T.count_params(T.CONFIGS["train_208m"]) == 208_504_320
    assert T.count_params(T.CONFIGS["train_253m"]) == 252_823_552
    assert T.count_params(T.CONFIGS["train_303m"]) == 303_163_392


def test_head_dimension_is_64_in_every_named_config() -> None:
    """The paper's algebra is stated for head_dim 64; so are these models."""

    assert set(T.TRAIN_PARAM_BANDS) <= set(T.CONFIGS)
    for name in T.TRAIN_PARAM_BANDS:
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
    run = tmp_path / "run"
    # Files from an earlier run of the same name -- including a summary
    # left by a run that crashed before finalize -- must not survive into
    # this one: a stale line would make the records unfaithful, and a stale
    # summary would outlive the run it describes.
    run.mkdir()
    (run / "metrics.jsonl").write_text('{"event": "stale"}\n', encoding="utf-8")
    (run / "summary.json").write_text('{"event": "stale"}', encoding="utf-8")
    writer = T.MetricsWriter(run)
    assert not (run / "summary.json").exists(), "stale summary survived init"
    writer.log({"event": "eval", "step": 1, "loss": 1.5})
    writer.log({"event": "eval", "step": 2, "loss": 1.25})
    summary = {"event": "summary", "steps": 2}
    writer.finalize(summary)
    lines = (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]
    on_disk = json.loads(
        (run / "summary.json").read_text(encoding="utf-8")
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
        "lr_floor_fraction", "grad_clip", "seed", "checkpoint_every",
        "val_batches",
    }
    for name, config in T.CONFIGS.items():
        missing = required - set(config)
        assert not missing, f"{name} lacks {sorted(missing)}"


def test_the_cpu_guard_blocks_named_configs_and_honours_the_override() -> None:
    """--force-cpu must actually reach the guard, on every device string."""
    params = T.count_params(T.CONFIGS["train_124m"])
    assert params > T.CPU_PARAM_LIMIT
    assert T._cpu_refusal("train_124m", params, "cpu", force_cpu=False) is not None
    assert T._cpu_refusal("train_124m", params, "cpu", force_cpu=True) is None
    assert T._cpu_refusal("train_124m", params, "cuda", force_cpu=False) is None
    smoke = T.count_params(T.CONFIGS["smoke"])
    assert T._cpu_refusal("smoke", smoke, "cpu", force_cpu=False) is None


def test_windows_are_next_token_and_the_validation_block_is_disjoint() -> None:
    """No window may predict its own first token; no held target in train."""
    tokens = np.arange(2000, dtype=np.int64)
    train_in, train_tg, held_in, held_tg = T._split_windows(tokens, 64)
    assert train_in.shape == train_tg.shape
    assert held_in.shape == held_tg.shape
    assert train_in.size > 0 and held_in.size > 0
    assert np.array_equal(train_tg, train_in + 1), "targets must follow their inputs"
    assert np.array_equal(held_tg, held_in + 1)
    assert train_tg.ravel()[-1] < held_in.ravel()[0], (
        "the held-out block must be the tail of the stream, not of the window list"
    )


# ==========================================================================
# The torch-checked structure
# ==========================================================================


def test_the_model_realizes_the_counted_parameters() -> None:
    """The closed-form count and the actual tensors must agree exactly."""
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    import torch

    config = T.CONFIGS["smoke"]
    torch.manual_seed(0)
    model = T.build_model(config)
    total = sum(p.numel() for p in model.parameters())
    assert total == T.count_params(config)
    buffers = dict(model.named_buffers())
    assert set(buffers) == {"cos_buf", "sin_buf"}, "tables must be registered"
    assert "cos_buf" not in model.state_dict(), "tables must stay non-persistent"


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
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    import torch

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
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    tokens = np.random.default_rng(11).integers(0, 512, size=8192)
    summary = T.train("smoke", tmp_path / "run", tokens, device="cpu")
    assert summary["steps"] == T.CONFIGS["smoke"]["n_steps"]
    records = [json.loads(line) for line in
               (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["event"] == "run_start"
    assert records[0]["dtype"] == "fp32", "the run must record the arithmetic it used"
    evals = [r for r in records if r["event"] == "eval"]
    assert evals, "no evaluation was logged"
    keys = {
        "step", "lr", "grad_norm", "train_loss", "train_loss_ema", "val_loss",
        "val_perplexity", "tokens_seen", "tok_per_s", "closed_form_resid_max",
        "rope_probe_resid_max", "sign_flip_rate", "rel_err_median_probe",
        "n_probe_cells", "val_batches", "val_targets", "probe_dtype",
    }
    for record in evals:
        assert keys <= set(record), sorted(keys - set(record))
        assert record["probe_dtype"] == "fp32", (
            "the eval record must name the arithmetic the probe ran in"
        )
        assert record["val_batches"] >= 1
        assert record["val_targets"] > 0
        assert np.isfinite(record["train_loss"])
        assert np.isfinite(record["train_loss_ema"])
        assert np.isfinite(record["val_loss"])
        assert np.isfinite(record["val_perplexity"])
        assert record["closed_form_resid_max"] < 1e-11
        assert record["lr"] == pytest.approx(
            T.lr_at(record["step"], T.CONFIGS["smoke"])
        ), "the logged lr must be the lr the step actually ran at"
    assert summary["final_val_loss"] == pytest.approx(evals[-1]["val_loss"])
    assert summary["best_val_loss"] <= summary["final_val_loss"]
    assert summary["best_step"] in {record["step"] for record in evals}


def test_the_run_writes_a_loadable_checkpoint(tmp_path: Path) -> None:
    """A run without weights is not reusable, and a silently half-written
    checkpoint is worse than none. The final file must load, name its config,
    and contain CPU tensors with the exact parameter count reported."""
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    import torch

    tokens = np.random.default_rng(17).integers(0, 512, size=4096)
    summary = T.train("smoke", tmp_path / "run", tokens, device="cpu")
    path = tmp_path / "run" / "checkpoint.pt"
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()
    assert summary["checkpoint"] == "checkpoint.pt"

    payload = torch.load(path, map_location="cpu")
    assert payload["format"] == "rope_attribution/train-checkpoint/v1"
    assert payload["config"] == T.CONFIGS["smoke"]
    assert payload["step"] == T.CONFIGS["smoke"]["n_steps"]
    assert payload["n_params"] == T.count_params(T.CONFIGS["smoke"])
    assert payload["model"]["embed.weight"].device.type == "cpu"
    assert payload["model"]["embed.weight"].shape == (
        T.CONFIGS["smoke"]["vocab"],
        T.CONFIGS["smoke"]["dim"],
    )

    T.train(
        "smoke", tmp_path / "no_checkpoint", tokens, device="cpu",
        save_checkpoint=False,
    )
    assert not (tmp_path / "no_checkpoint" / "checkpoint.pt").exists()


def test_token_accounting_is_global_across_data_parallel_ranks(tmp_path: Path) -> None:
    """Eight chips consume eight batches per step, and the log must say so.

    Mutation verdict: reverting ``tokens_seen += ids.numel() * world_size``
    to the per-rank ``ids.numel()`` makes the expected count eight times
    larger than the logged one.
    """
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    config = dict(T.CONFIGS["smoke"])
    config.update({"n_steps": 1, "eval_every": 1})
    tokens = np.random.default_rng(13).integers(0, 512, size=4096)
    summary = T.train(
        "smoke", tmp_path / "run", tokens, device="cpu",
        max_steps=1, override={"eval_every": 1}, world_size=8,
    )
    expected = config["batch_size"] * config["ctx"] * 8
    assert summary["tokens_seen"] == expected
    assert summary["world_size"] == 8

    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    start = records[0]
    evaluation = next(record for record in records if record["event"] == "eval")
    assert start["world_size"] == 8
    assert start["token_accounting"].startswith("global")
    assert evaluation["tokens_seen"] == expected


def test_a_fresh_run_removes_a_stale_checkpoint(tmp_path: Path) -> None:
    """A checkpoint from an earlier run of the same name is not this run's.

    Mutation verdict: deleting the ``checkpoint*.pt`` cleanup leaves both
    stale files in place, and with ``save_checkpoint=False`` they would be
    the only weights in the directory.
    """
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoint.pt").write_bytes(b"stale")
    (run / "checkpoint_resume.pt").write_bytes(b"stale")
    tokens = np.random.default_rng(15).integers(0, 512, size=4096)
    T.train(
        "smoke", run, tokens, device="cpu", max_steps=1, save_checkpoint=False,
    )
    assert not (run / "checkpoint.pt").exists(), "stale weights survived"
    assert not (run / "checkpoint_resume.pt").exists(), "stale resume state survived"


def test_resume_restores_the_optimizer_and_the_data_order(tmp_path: Path) -> None:
    """A resumed run must be the same run, not a lookalike.

    The test trains straight to eight steps, then trains to four and resumes
    to eight. The step-8 evaluation must match the straight run field for
    field. AdamW's moments and the numpy generator are the two things that
    would silently differ if the checkpoint held weights only.

    Mutation verdict: dropping ``opt.load_state_dict`` or the
    ``rng.bit_generator.state`` restore changes the step-8 losses; dropping
    both changes them further. The exact comparison is the detector.
    """
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    import torch

    tokens = np.random.default_rng(19).integers(0, 512, size=8192)
    straight = T.train(
        "smoke", tmp_path / "straight", tokens, device="cpu", max_steps=8
    )
    assert straight["steps"] == 8
    T.train("smoke", tmp_path / "part", tokens, device="cpu", max_steps=4)
    resumed = T.train(
        "smoke", tmp_path / "part", tokens, device="cpu", max_steps=8, resume=True
    )
    assert resumed["resumed_from_step"] == 4
    assert resumed["steps"] == 8

    def eval_at(run: str, step: int) -> dict:
        records = [
            json.loads(line)
            for line in (tmp_path / run / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        matches = [
            record
            for record in records
            if record["event"] == "eval" and record["step"] == step
        ]
        assert len(matches) == 1, (run, step, [r["step"] for r in records])
        return matches[0]

    reference = eval_at("straight", 8)
    actual = eval_at("part", 8)
    for key in (
        "lr",
        "grad_norm",
        "train_loss",
        "train_loss_ema",
        "val_loss",
        "tokens_seen",
        "closed_form_resid_max",
        "rope_probe_resid_max",
        "sign_flip_rate",
        "rel_err_median_probe",
        "n_probe_cells",
    ):
        assert actual[key] == reference[key], (key, actual[key], reference[key])

    payload = torch.load(
        tmp_path / "part" / "checkpoint_resume.pt", map_location="cpu"
    )
    assert payload["format"] == "rope_attribution/train-state/v1"
    assert payload["step"] == 8
    assert payload["optimizer"]["state"], "the resume file carries no optimizer state"
    assert "numpy_rng" in payload and "torch_rng" in payload


def test_validation_averages_several_batches(tmp_path: Path) -> None:
    """A single validation batch is noise, and the average must be real.

    The expected value is recomputed here from the saved checkpoint over the
    same three batches. Mutation verdict: using ``held_in[:batch_size]``
    alone both drops the ``val_batches`` field and changes ``val_loss`` away
    from the target-weighted mean checked below.
    """
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    import torch
    import torch.nn.functional as F

    tokens = np.random.default_rng(21).integers(0, 512, size=8192)
    summary = T.train(
        "smoke", tmp_path / "run", tokens, device="cpu", max_steps=1,
        override={"eval_every": 1, "val_batches": 3},
    )
    record = next(
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if json.loads(line)["event"] == "eval"
    )
    assert record["val_batches"] == 3
    assert summary["final_val_loss"] == record["val_loss"]

    config = dict(T.CONFIGS["smoke"])
    config.update({"n_steps": 1, "eval_every": 1, "val_batches": 3})
    payload = torch.load(tmp_path / "run" / "checkpoint.pt", map_location="cpu")
    model = T.build_model(config)
    model.load_state_dict(payload["model"])
    _, _, held_in, held_tg = T._split_windows(tokens, config["ctx"])
    n_val = min(len(held_in), config["batch_size"] * 3)
    total, count = 0.0, 0
    with torch.no_grad():
        for offset in range(0, n_val, config["batch_size"]):
            batch = slice(offset, offset + config["batch_size"])
            ids = torch.from_numpy(held_in[batch])
            targets = torch.from_numpy(held_tg[batch])
            logits = model(ids)
            total += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )
            )
            count += int(targets.numel())
    assert record["val_targets"] == count
    assert record["val_loss"] == pytest.approx(total / count, rel=1e-12)


def test_non_finite_metrics_are_written_as_null(tmp_path: Path) -> None:
    """A diverged run's record must still be readable by a strict parser.

    Mutation verdict: without ``_json_safe`` the file contains bare ``NaN``
    and ``Infinity``, and the ``parse_constant`` hook below raises on them.
    """
    run = tmp_path / "run"
    writer = T.MetricsWriter(run)
    writer.log(
        {
            "event": "eval",
            "loss": float("nan"),
            "positive": float("inf"),
            "negative": float("-inf"),
            "finite": 1.5,
        }
    )
    writer.finalize({"event": "summary", "loss": float("nan")})

    def reject(token: str):
        raise ValueError(f"non-finite literal {token!r} reached the file")

    record = json.loads(
        (run / "metrics.jsonl").read_text(encoding="utf-8"),
        parse_constant=reject,
    )
    assert record["loss"] is None
    assert record["positive"] is None
    assert record["negative"] is None
    assert record["finite"] == 1.5
    summary = json.loads(
        (run / "summary.json").read_text(encoding="utf-8"),
        parse_constant=reject,
    )
    assert summary["loss"] is None


def test_a_run_writes_a_manifest_with_its_provenance(tmp_path: Path) -> None:
    """The metrics answer what happened; the manifest answers where from.

    Mutation verdict: removing the ``_write_manifest`` call leaves no file
    and no ``manifest`` field in the summary or the run_start record.
    """
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    tokens = np.random.default_rng(23).integers(0, 512, size=4096)
    summary = T.train("smoke", tmp_path / "run", tokens, device="cpu", max_steps=1)
    path = tmp_path / "run" / "manifest.json"
    assert path.exists()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["format"] == "rope_attribution/train-manifest/v1"
    assert manifest["config_name"] == "smoke"
    assert manifest["n_params"] == T.count_params(T.CONFIGS["smoke"])
    assert manifest["world_size"] == 1
    assert manifest["tokens_total"] == int(tokens.size)
    for key in ("python", "torch", "numpy", "platform", "created_utc"):
        assert manifest[key], key
    assert "git_commit" in manifest
    assert summary["manifest"] == "manifest.json"
    start = json.loads(
        (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert start["manifest"] == "manifest.json"


def test_the_probe_can_read_bfloat16_tensors() -> None:
    """bfloat16 has no numpy dtype; the probe must widen before ``.numpy()``.

    This is the one failure a CPU-only suite cannot reach through a training
    run, because CPU bf16 is refused by policy. Mutation verdict: removing
    the ``.float()`` cast raises ``TypeError`` on the line below.
    """
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    import torch

    tensor = torch.tensor([1.5, -2.25], dtype=torch.bfloat16)
    out = T._to_numpy_f64(tensor)
    assert out.dtype == np.float64
    assert np.allclose(out, [1.5, -2.25])


def test_the_probe_runs_inside_the_runs_autocast_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The diagnostic must describe the arithmetic the training used.

    Mutation verdict: moving the ``_probe_diagnostics`` call outside the
    ``with _autocast_context(...)`` block makes ``probe_saw`` false.
    """
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    import contextlib

    state: dict = {"inside": False, "probe_saw": None}

    @contextlib.contextmanager
    def spy(device: str, dtype: str):
        previous = state["inside"]
        state["inside"] = True
        try:
            yield
        finally:
            state["inside"] = previous

    def fake_probe(*args, **kwargs):
        state["probe_saw"] = state["inside"]
        return {
            "closed_form_resid_max": 0.0,
            "rope_probe_resid_max": 0.0,
            "sign_flip_rate": 0.0,
            "rel_err_median_probe": 0.0,
            "n_probe_cells": 0,
        }

    monkeypatch.setattr(T, "_autocast_context", spy)
    monkeypatch.setattr(T, "_probe_diagnostics", fake_probe)
    tokens = np.random.default_rng(29).integers(0, 512, size=4096)
    T.train(
        "smoke", tmp_path / "run", tokens, device="cpu", max_steps=1,
        override={"eval_every": 1},
    )
    assert state["probe_saw"] is True


def test_the_cli_passes_a_prepared_memmap_to_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training must read the prepared stream, not tokenize it again.

    Mutation verdict: deleting the ``args.data_dir`` branch falls back to the
    synthetic smoke ids, so the captured array is not a memmap and does not
    start with the prepared byte.
    """
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    from rope_attribution import data as D

    out = tmp_path / "tokens"
    D.prepare(
        lambda text: [ord(char) for char in text],
        iter(["abcdefgh"] * 100),
        out,
        name="toy",
        source="unit-test",
        tokenizer="ord",
        shard_tokens=64,
    )
    captured: dict = {}

    def fake_train(*args, **kwargs):
        captured["tokens"] = args[2]
        captured.update(kwargs)
        return {"event": "summary", "steps": 0}

    monkeypatch.setattr(T, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "smoke",
            "--run-name",
            "stub_data",
            "--data-dir",
            str(out),
        ],
    )
    T.main()
    assert isinstance(captured["tokens"], np.memmap)
    assert int(captured["tokens"][0]) == ord("a")
    assert int(captured["tokens"][-1]) == ord("h")


def test_two_smoke_runs_log_identical_series(tmp_path: Path) -> None:
    """Same seed, same machine: only the wall-clock field may differ."""
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
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
    """The named configs must not be launchable onto the owner's cores --
    and a refusal must leave the previous run's records untouched."""
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text('{"event": "previous run"}\n', encoding="utf-8")
    tokens = np.random.default_rng(3).integers(0, 16, size=4096)
    with pytest.raises(RuntimeError, match="GPU territory"):
        T.train("train_124m", run, tokens, device="cpu", max_steps=1)
    assert "previous run" in (run / "metrics.jsonl").read_text(encoding="utf-8")


def test_the_thread_override_is_respected(tmp_path: Path) -> None:
    """--threads must reach torch; a hardcoded default is not the promise."""
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    import torch

    tokens = np.random.default_rng(5).integers(0, 512, size=4096)
    before = torch.get_num_threads()
    try:
        T.train(
            "smoke", tmp_path / "run", tokens, device="cpu",
            max_steps=1, override={"eval_every": 1}, threads=2,
        )
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)


def test_main_refuses_cpu_and_hands_the_override_to_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI entry point: same refusal, and --force-cpu is really passed on."""
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    monkeypatch.setattr(
        sys, "argv",
        ["train", "--config", "train_124m", "--device", "cpu"],
    )
    with pytest.raises(SystemExit, match="GPU territory"):
        T.main()

    captured: dict = {}

    def fake_train(*args, **kwargs):
        captured.update(kwargs)
        return {"stub": True}

    monkeypatch.setattr(T, "train", fake_train)
    monkeypatch.setattr(T, "_load_wikitext_tokens", lambda: np.zeros(8, dtype=np.int64))
    if not T.TRANSFORMERS_AVAILABLE:
        monkeypatch.setattr(T, "TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(
        sys, "argv",
        ["train", "--config", "train_124m", "--device", "cpu",
         "--force-cpu", "--run-name", "stub_check"],
    )
    T.main()
    assert captured["force_cpu"] is True
    # The CLI hands the policy over; train() resolves None to 4 on CPU.
    assert captured["threads"] is None


def test_the_default_thread_count_depends_on_the_device() -> None:
    assert T._resolve_threads("cpu", None) == 4
    assert T._resolve_threads("cuda", None) == 8
    assert T._resolve_threads("cuda", 2) == 2


def test_cuda_determinism_pins_the_cublas_workspace(monkeypatch) -> None:
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    import os

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    T._set_determinism(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_xla_refuses_clearly_when_torch_xla_is_absent(monkeypatch) -> None:
    monkeypatch.setattr(T, "XLA_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="torch_xla"):
        T._require_xla()


def test_the_cli_passes_xla_through_to_train(monkeypatch) -> None:
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    captured: dict = {}

    def fake_train(*args, **kwargs):
        captured.update(kwargs)
        return {"event": "summary", "steps": 0}

    monkeypatch.setattr(T, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--config", "smoke", "--device", "xla", "--run-name", "stub_xla"],
    )
    T.main()
    assert captured["device"] == "xla"


def test_data_seed_changes_the_batch_stream_but_not_the_model_seed(tmp_path) -> None:
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    tokens = np.random.default_rng(3).integers(0, 512, size=4096)
    first = T.train("smoke", tmp_path / "a", tokens, device="cpu", max_steps=2, data_seed=11)
    second = T.train("smoke", tmp_path / "b", tokens, device="cpu", max_steps=2, data_seed=12)
    assert first["final_train_loss_ema"] != second["final_train_loss_ema"], (
        "different data_seed values must draw different batches"
    )
    start = json.loads(
        (tmp_path / "a" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert start["seed"] == T.CONFIGS["smoke"]["seed"]
    assert start["data_seed"] == 11


def test_bf16_is_a_real_choice_and_not_silently_ignored_on_cpu() -> None:
    """The dtype in a preset has to reach the arithmetic or be refused."""
    assert T._amp_dtype("cpu", "fp32") is None
    with pytest.raises(ValueError, match="bf16"):
        T._amp_dtype("cpu", "bf16")
    with pytest.raises(ValueError, match="fp32"):
        T._amp_dtype("cpu", "fp16")
    if T.TORCH_AVAILABLE:
        import torch

        assert T._amp_dtype("cuda", "bf16") is torch.bfloat16
        assert T._amp_dtype("xla", "bf16") is torch.bfloat16


def test_tpu_presets_pass_their_dtype_to_the_trainer() -> None:
    """A preset's bf16 label must be an override, not documentation."""
    from rope_attribution import tpu_configs as TPU

    for name, preset in TPU.TPU_PRESETS.items():
        override = TPU.launch_overrides(name)
        assert override["dtype"] == preset["dtype"] == "bf16", name


def test_xla_update_reduces_before_clipping_and_steps_once() -> None:
    """The data-parallel order: reduce, clip the reduced gradient, step once.

    The fake ``xm`` multiplies each local gradient by ten. If clipping ran
    first, the parameter update would have norm ten after the fake reduction;
    with the intended order it has norm one. This is the exact bug a
    ``xm.optimizer_step`` call after ``xm.reduce_gradients`` would hide.
    """
    if not T.TORCH_AVAILABLE:
        pytest.skip("torch is not installed")
    import torch

    class FakeXM:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def reduce_gradients(self, model) -> None:
            self.calls.append("reduce")
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(10.0)

        def mark_step(self) -> None:
            self.calls.append("mark")

    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    xm = FakeXM()
    grad_norm = T._reduce_update(model, opt, xm, grad_clip=1.0)
    assert xm.calls == ["reduce", "mark"], xm.calls
    assert float(grad_norm) == pytest.approx(50.0), "the norm must be pre-clip, post-reduce"
    assert float(model.weight.detach().norm()) == pytest.approx(1.0), (
        "the step must use the clipped, reduced gradient exactly once"
    )
