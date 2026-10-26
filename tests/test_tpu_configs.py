"""Tests for the TPU v5e-8 presets and the launcher's CPU-checkable surface."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from rope_attribution import tpu_configs as C
from rope_attribution import tpu_launch as L
from rope_attribution import train as T


def test_the_presets_are_internally_consistent() -> None:
    C.validate_all()
    assert set(C.TPU_PRESETS) == {
        "tpu_v5e8_102m",
        "tpu_v5e8_152m",
        "tpu_v5e8_253m",
        "tpu_v5e8_303m",
    }


def test_the_global_batch_is_eight_chips_times_the_per_chip_batch() -> None:
    for name, preset in C.TPU_PRESETS.items():
        assert preset["chips"] == 8, name
        assert preset["hosts"] == 1, name
        assert preset["dtype"] == "bf16", name
        assert preset["global_batch_size"] == (
            preset["per_chip_batch_size"] * 8 * preset["grad_accum"]
        ), name
        assert preset["tokens_per_step"] == preset["global_batch_size"] * preset["seq_len"], name
        assert preset["total_tokens"] == preset["tokens_per_step"] * preset["n_steps"], name


def test_every_preset_key_is_a_runtime_knob_or_declared_metadata() -> None:
    """No preset field may be silently dropped on the way to ``train()``.

    Mutation verdict: removing ``grad_clip`` or ``lr_floor_fraction`` from
    ``launch_overrides`` leaves it out of the override dict, so the equality
    check fails; a new preset key that is in neither table fails the first
    assertion, which is the "silently ignored knob" bug this test exists for.
    """
    assert not set(C.RUNTIME_KEYS) & C.METADATA_KEYS
    for name, preset in C.TPU_PRESETS.items():
        unknown = set(preset) - set(C.RUNTIME_KEYS) - C.METADATA_KEYS
        assert not unknown, f"{name}: {sorted(unknown)} are neither runtime nor metadata"
        override = C.launch_overrides(name)
        assert set(override) == set(C.RUNTIME_KEYS.values()), name
        for preset_key, config_key in C.RUNTIME_KEYS.items():
            assert override[config_key] == preset[preset_key], (name, preset_key)
            assert config_key in T.CONFIGS["smoke"], (name, config_key)


def test_the_override_carries_the_knobs_that_used_to_be_dropped() -> None:
    """The two hyperparameters the old override lost must reach the trainer."""
    for name, preset in C.TPU_PRESETS.items():
        override = C.launch_overrides(name)
        assert override["grad_clip"] == preset["grad_clip"], name
        assert override["lr_floor_fraction"] == preset["lr_floor_fraction"], name
        for key in (
            "checkpoint_every", "eval_every", "val_batches", "log_every",
            "grad_accum",
        ):
            assert override[key] == preset[key], (name, key)


def test_expected_params_match_the_train_configs() -> None:
    for name, preset in C.TPU_PRESETS.items():
        config_name = str(preset["train_config"])
        assert preset["expected_params"] == T.count_params(T.CONFIGS[config_name]), name


def test_the_token_budget_is_chinchilla_scale() -> None:
    for name, preset in C.TPU_PRESETS.items():
        ratio = int(preset["total_tokens"]) / int(preset["expected_params"])
        assert 15.0 <= ratio <= 25.0, f"{name}: {ratio:.1f} tokens per parameter"


def test_the_launch_overrides_are_keys_train_already_accepts() -> None:
    smoke_keys = set(T.CONFIGS["smoke"])
    for name, preset in C.TPU_PRESETS.items():
        override = C.launch_overrides(name)
        assert set(override) <= smoke_keys, name
        merged = {**T.CONFIGS[str(preset["train_config"])], **override}
        assert merged["batch_size"] == preset["per_chip_batch_size"], name
        assert merged["ctx"] == preset["seq_len"], name
        assert merged["n_steps"] == preset["n_steps"], name
        assert merged["warmup_steps"] == preset["warmup_steps"], name


def _broken(name: str = "tpu_v5e8_102m") -> dict:
    return copy.deepcopy(C.TPU_PRESETS[name])


def test_a_wrong_global_batch_is_refused() -> None:
    preset = _broken()
    preset["global_batch_size"] = 63
    with pytest.raises(ValueError, match="global_batch_size"):
        C.validate_preset("broken", preset)


def test_a_non_xla_sequence_length_is_refused() -> None:
    preset = _broken()
    preset["seq_len"] = 1000
    preset["tokens_per_step"] = int(preset["global_batch_size"]) * 1000
    preset["total_tokens"] = int(preset["tokens_per_step"]) * int(preset["n_steps"])
    with pytest.raises(ValueError, match="multiple of 128"):
        C.validate_preset("broken", preset)


def test_float16_is_refused() -> None:
    preset = _broken()
    preset["dtype"] = "fp16"
    with pytest.raises(ValueError, match="bf16 or fp32"):
        C.validate_preset("broken", preset)


def test_a_wrong_parameter_count_is_refused() -> None:
    preset = _broken()
    preset["expected_params"] = int(preset["expected_params"]) + 1
    with pytest.raises(ValueError, match="expected_params"):
        C.validate_preset("broken", preset)


def test_a_budget_outside_twenty_tokens_per_parameter_is_refused() -> None:
    preset = _broken()
    preset["n_steps"] = int(preset["n_steps"]) * 3
    preset["total_tokens"] = int(preset["tokens_per_step"]) * int(preset["n_steps"])
    with pytest.raises(ValueError, match="tokens per parameter"):
        C.validate_preset("broken", preset)


def test_a_global_batch_that_ignores_accumulation_is_refused() -> None:
    preset = _broken()
    preset["grad_accum"] = 2
    with pytest.raises(ValueError, match="grad_accum"):
        C.validate_preset("broken", preset)


def test_the_preset_factory_counts_accumulation_in_the_global_batch() -> None:
    """The derived batch must come from the parts, not from a copied number.

    Every shipped preset accumulates one micro-batch, so the identity is
    invisible there; this asks the factory for two and checks the arithmetic.

    Mutation verdict: dropping ``* grad_accum`` from ``_preset`` leaves the
    global batch at 64 instead of 128, and the first assertion fails.
    """
    doubled = C._preset(
        "train_102m", n_steps=8_000, lr=1.0e-3, warmup_steps=160, grad_accum=2
    )
    assert doubled["grad_accum"] == 2
    assert doubled["global_batch_size"] == (
        doubled["per_chip_batch_size"] * 8 * 2
    )
    assert doubled["tokens_per_step"] == doubled["global_batch_size"] * doubled["seq_len"]
    assert doubled["total_tokens"] == doubled["tokens_per_step"] * doubled["n_steps"]


def test_warmup_must_be_shorter_than_the_run() -> None:
    preset = _broken()
    preset["warmup_steps"] = int(preset["n_steps"])
    with pytest.raises(ValueError, match="warmup_steps"):
        C.validate_preset("broken", preset)


def test_the_cli_exports_valid_json(capsys) -> None:
    C.main(["--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["topology"]["chips"] == 8
    assert set(data["presets"]) == set(C.TPU_PRESETS)


def test_the_cli_check_reports_every_preset(capsys) -> None:
    C.main(["--check"])
    assert "OK: 4 TPU v5e-8 presets" in capsys.readouterr().out


def test_the_launcher_dry_run_needs_no_tpu(capsys) -> None:
    L.main(["--preset", "tpu_v5e8_102m", "--dry-run"])
    data = json.loads(capsys.readouterr().out)
    assert data["nprocs"] == 8
    assert data["global_batch_size"] == 64
    assert data["rank_data_seeds"] == [20261010 + rank for rank in range(8)]
    assert data["run_dirs"][0].replace("\\", "/") == "runs/tpu_v5e8_102m/rank-0"
    assert data["data_dir"] is None


def test_the_launcher_dry_run_carries_a_prepared_data_dir(capsys) -> None:
    """The launcher must show the stream every rank will mmap, not hide it."""
    L.main(
        [
            "--preset",
            "tpu_v5e8_102m",
            "--dry-run",
            "--data-dir",
            "data/wikitext103",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert data["data_dir"].replace("\\", "/") == "data/wikitext103"


def test_the_launcher_refuses_a_process_count_that_is_not_eight() -> None:
    with pytest.raises(SystemExit, match="8-chip"):
        L.main(["--preset", "tpu_v5e8_102m", "--nprocs", "4", "--dry-run"])


def test_the_launcher_refuses_without_torch_xla(monkeypatch) -> None:
    monkeypatch.setattr(L, "XLA_AVAILABLE", False)
    with pytest.raises(SystemExit, match="torch_xla"):
        L.main(["--preset", "tpu_v5e8_102m", "--run-name", "no_tpu"])


def test_the_launcher_dry_run_shows_resume_and_effective_batch(capsys) -> None:
    """A dry run must show the batch a resume of these flags would really run.

    Mutation verdict: ignoring the CLI overrides in the dry-run arithmetic
    leaves global_batch_size at 64 instead of 4 * 8 * 2; dropping ``resume``
    from the payload loses the flag the launcher would hand to the ranks.
    """
    L.main(
        [
            "--preset", "tpu_v5e8_102m", "--dry-run", "--resume",
            "--batch-size", "4", "--grad-accum", "2", "--log-every", "25",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert data["resume"] is True
    assert data["overrides"] == {
        "batch_size": 4, "grad_accum": 2, "log_every": 25,
    }
    assert data["per_chip_batch_size"] == 4
    assert data["grad_accum"] == 2
    assert data["global_batch_size"] == 4 * 8 * 2
    assert data["tokens_per_step"] == 4 * 8 * 2 * 2048


def test_the_launcher_refuses_an_invalid_tuning_flag() -> None:
    with pytest.raises(SystemExit, match="grad-accum"):
        L.main(["--preset", "tpu_v5e8_102m", "--dry-run", "--grad-accum", "0"])


def test_the_train_plan_gives_each_rank_its_own_state_and_the_shared_knobs() -> None:
    """Rank 0 alone writes the evaluation checkpoint; every rank can resume.

    Mutation verdict: dropping ``save_resume`` makes per-rank ``--resume``
    impossible on ranks 1-7; sharing rank 0's checkpoint instead would give
    every rank rank 0's data order, so the per-rank ``data_seed`` and
    ``run_dir`` assertions pin the layout.
    """
    preset = C.TPU_PRESETS["tpu_v5e8_102m"]
    rank0 = L.train_plan(
        "tpu_v5e8_102m", 0, run_name="v5e8_102m", resume=True, max_steps=100,
    )
    rank3 = L.train_plan(
        "tpu_v5e8_102m", 3, run_name="v5e8_102m",
        overrides={"grad_accum": 2, "log_every": 25}, resume=True,
    )
    assert rank0["config_name"] == "train_102m"
    assert rank0["run_dir"] == Path("runs") / "v5e8_102m" / "rank-0"
    assert rank3["run_dir"] == Path("runs") / "v5e8_102m" / "rank-3"
    assert rank3["device"] == "xla"
    assert rank3["data_seed"] == int(preset["seed"]) + 3
    assert rank3["world_size"] == 8
    assert rank3["resume"] is True
    assert rank3["save_checkpoint"] is False
    assert rank3["save_resume"] is True
    assert rank0["save_checkpoint"] is True
    assert rank0["max_steps"] == 100
    assert rank3["override"]["grad_accum"] == 2
    assert rank3["override"]["log_every"] == 25
    assert rank3["override"]["grad_clip"] == preset["grad_clip"]
    assert rank3["override"]["lr_floor_fraction"] == preset["lr_floor_fraction"]
    assert rank3["override"]["checkpoint_every"] == preset["checkpoint_every"]
    assert rank3["override"]["eval_every"] == preset["eval_every"]
    assert rank3["override"]["val_batches"] == preset["val_batches"]

