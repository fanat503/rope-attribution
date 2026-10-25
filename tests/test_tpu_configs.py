"""Tests for the TPU v5e-8 presets and the launcher's CPU-checkable surface."""

from __future__ import annotations

import copy
import json

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
        assert preset["global_batch_size"] == preset["per_chip_batch_size"] * 8, name
        assert preset["tokens_per_step"] == preset["global_batch_size"] * preset["seq_len"], name
        assert preset["total_tokens"] == preset["tokens_per_step"] * preset["n_steps"], name


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
