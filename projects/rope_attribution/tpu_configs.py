"""Training presets for one TPU v5e-8 slice, with their arithmetic checked.

What this is
------------
A v5e-8 slice is eight chips in one host. The trainer in ``train.py`` is
single-device PyTorch, so the topology enters through the launcher: eight
processes, one chip each, and a per-chip batch size. A preset records the
hyperparameters for that topology and the *derived* quantities that are easy to
get wrong by hand:

* ``global_batch_size = per_chip_batch_size * chips`` (sequences per step);
* ``tokens_per_step = global_batch_size * seq_len``;
* ``total_tokens = tokens_per_step * n_steps``;
* ``expected_params = train.count_params(train.CONFIGS[train_config])``.

The last one is the point of the module. A preset is a *configuration*, not a
result: no model in this repository has been trained on a TPU, and nothing here
is quoted as a measurement. What is verified locally is the arithmetic, the
parameter counts and the JSON export; what is not verified is an XLA run, which
needs the TPU host. ``docs/TPU_V5E8.md`` states that boundary.

The token budget is set to roughly 20 tokens per parameter (the Chinchilla
ratio), because the smallest and largest presets use the same token-per-step
budget and a reader should be able to see that the steps differ for a reason.
``validate_preset`` pins the ratio to a band rather than to a single number.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from rope_attribution.train import CONFIGS, TRAIN_PARAM_BANDS, count_params

#: The slice this module describes. One host, eight chips, bfloat16.
TPU_V5E8: dict[str, object] = {
    "device": "xla",
    "chips": 8,
    "hosts": 1,
    "dtype": "bf16",
}

#: Eight sequences per chip at 2048 tokens is 131,072 tokens per global step.
#: That is four times the CPU presets' 32,768, which is why the learning rates
#: are higher than the CPU ladder's 6e-4 rather than copied from it.
_PER_CHIP_BATCH = 8
_SEQ_LEN = 2048
_CHIPS = 8


def _preset(
    train_config: str,
    *,
    n_steps: int,
    lr: float,
    warmup_steps: int,
    seed: int = 20261010,
) -> dict[str, object]:
    params = count_params(CONFIGS[train_config])
    global_batch = _PER_CHIP_BATCH * _CHIPS
    tokens_per_step = global_batch * _SEQ_LEN
    return {
        "train_config": train_config,
        "chips": _CHIPS,
        "hosts": 1,
        "dtype": "bf16",
        "per_chip_batch_size": _PER_CHIP_BATCH,
        "seq_len": _SEQ_LEN,
        "global_batch_size": global_batch,
        "tokens_per_step": tokens_per_step,
        "n_steps": n_steps,
        "total_tokens": tokens_per_step * n_steps,
        "expected_params": params,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": seed,
    }


#: Four sizes spanning the paper's 100M-300M range. Steps are chosen so that
#: every preset lands near 20 tokens per parameter; they are not a claim that a
#: run has happened.
TPU_PRESETS: dict[str, dict[str, object]] = {
    "tpu_v5e8_102m": _preset("train_102m", n_steps=16_000, lr=1.0e-3, warmup_steps=320),
    "tpu_v5e8_152m": _preset("train_152m", n_steps=24_000, lr=8.0e-4, warmup_steps=480),
    "tpu_v5e8_253m": _preset("train_253m", n_steps=40_000, lr=6.0e-4, warmup_steps=800),
    "tpu_v5e8_303m": _preset("train_303m", n_steps=48_000, lr=5.0e-4, warmup_steps=960),
}

_REQUIRED = frozenset(
    {
        "train_config",
        "chips",
        "hosts",
        "dtype",
        "per_chip_batch_size",
        "seq_len",
        "global_batch_size",
        "tokens_per_step",
        "n_steps",
        "total_tokens",
        "expected_params",
        "lr",
        "warmup_steps",
        "lr_floor_fraction",
        "grad_clip",
        "seed",
    }
)


def validate_preset(name: str, preset: dict[str, object]) -> None:
    """Raise ``ValueError`` on any internally inconsistent field.

    These are the checks a reader would otherwise have to do with a calculator,
    and the ones a hand-edited preset gets wrong first: the global batch, the
    token budget and the parameter count.
    """
    missing = _REQUIRED - set(preset)
    if missing:
        raise ValueError(f"{name}: missing keys {sorted(missing)}")
    if preset["chips"] != _CHIPS or preset["hosts"] != 1:
        raise ValueError(f"{name}: the slice is {_CHIPS} chips on one host")
    if preset["dtype"] not in ("bf16", "fp32"):
        raise ValueError(f"{name}: dtype must be bf16 or fp32, not {preset['dtype']!r}")
    if int(preset["per_chip_batch_size"]) <= 0:
        raise ValueError(f"{name}: per_chip_batch_size must be positive")
    if int(preset["seq_len"]) <= 0 or int(preset["seq_len"]) % 128:
        raise ValueError(f"{name}: seq_len must be a positive multiple of 128 for XLA")
    if int(preset["global_batch_size"]) != int(preset["per_chip_batch_size"]) * _CHIPS:
        raise ValueError(
            f"{name}: global_batch_size must be per_chip_batch_size * {_CHIPS}"
        )
    if int(preset["tokens_per_step"]) != int(preset["global_batch_size"]) * int(
        preset["seq_len"]
    ):
        raise ValueError(f"{name}: tokens_per_step must be global_batch_size * seq_len")
    if int(preset["total_tokens"]) != int(preset["tokens_per_step"]) * int(
        preset["n_steps"]
    ):
        raise ValueError(f"{name}: total_tokens must be tokens_per_step * n_steps")
    if int(preset["n_steps"]) <= int(preset["warmup_steps"]) or int(
        preset["warmup_steps"]
    ) <= 0:
        raise ValueError(f"{name}: need 0 < warmup_steps < n_steps")
    if float(preset["lr"]) <= 0.0 or float(preset["lr"]) > 3.0e-3:
        raise ValueError(f"{name}: lr is outside the bf16-safe band")
    if not 0.0 < float(preset["lr_floor_fraction"]) < 1.0:
        raise ValueError(f"{name}: lr_floor_fraction must be in (0, 1)")
    if not 0.0 < float(preset["grad_clip"]) <= 10.0:
        raise ValueError(f"{name}: grad_clip must be in (0, 10]")
    if int(preset["seed"]) < 0:
        raise ValueError(f"{name}: seed must be non-negative")

    config_name = str(preset["train_config"])
    if config_name not in TRAIN_PARAM_BANDS:
        raise ValueError(f"{name}: unknown train config {config_name!r}")
    expected = count_params(CONFIGS[config_name])
    if int(preset["expected_params"]) != expected:
        raise ValueError(
            f"{name}: expected_params is {preset['expected_params']}, "
            f"but {config_name} has {expected}"
        )
    low, high = TRAIN_PARAM_BANDS[config_name]
    if not low <= expected <= high:
        raise ValueError(f"{name}: {config_name} leaves its parameter band")
    ratio = int(preset["total_tokens"]) / expected
    if not 15.0 <= ratio <= 25.0:
        raise ValueError(
            f"{name}: {ratio:.1f} tokens per parameter is outside the 15-25 band"
        )


def validate_all() -> None:
    for name, preset in TPU_PRESETS.items():
        validate_preset(name, preset)


validate_all()


def launch_overrides(name: str) -> dict[str, object]:
    """The ``override`` dict ``train.train`` needs for this preset.

    Only keys that ``train.CONFIGS`` already has are returned; the launcher adds
    the per-rank seed. Keeping this function here means the launch path and the
    validator read the same fields.
    """
    preset = TPU_PRESETS[name]
    return {
        "batch_size": preset["per_chip_batch_size"],
        "ctx": preset["seq_len"],
        "lr": preset["lr"],
        "n_steps": preset["n_steps"],
        "warmup_steps": preset["warmup_steps"],
        "seed": preset["seed"],
        "dtype": preset["dtype"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print the presets as JSON")
    parser.add_argument("--check", action="store_true", help="validate and print a one-line result")
    args = parser.parse_args(argv)
    if args.json:
        print(json.dumps({"topology": TPU_V5E8, "presets": TPU_PRESETS}, indent=2))
        return
    if args.check:
        validate_all()
        print(f"OK: {len(TPU_PRESETS)} TPU v5e-8 presets are internally consistent")
        return
    print(f"{'preset':<18} {'params':>12} {'tokens/step':>12} {'steps':>8} {'total tokens':>14} {'lr':>8}")
    for name, preset in TPU_PRESETS.items():
        print(
            f"{name:<18} {int(preset['expected_params']):>12,} "
            f"{int(preset['tokens_per_step']):>12,} {int(preset['n_steps']):>8,} "
            f"{int(preset['total_tokens']):>14,} {float(preset['lr']):>8.1e}"
        )


if __name__ == "__main__":
    main()
