# Training on one TPU v5e-8 slice

This document describes the four presets in
`projects/rope_attribution/tpu_configs.py` and the launcher in
`projects/rope_attribution/tpu_launch.py`.

**Status boundary.** The preset arithmetic, the parameter counts and the
launcher's command line are tested on CPU (`tests/test_tpu_configs.py`). An
actual XLA run is **not** tested in this repository: CI has no TPU. Nothing in
the paper is measured on a TPU, and no model in this repository has been trained
on one. The numbers below are a configuration, not a result.

## Topology

| Field | Value |
| --- | --- |
| Slice | one v5e-8 host |
| Chips | 8 |
| Processes | 8, one per chip |
| Device string | `xla` |
| Compute dtype | `bf16` |
| Per-chip batch | 8 sequences |
| Sequence length | 2048 |
| Global batch | 64 sequences = 131,072 tokens per step |

The trainer in `train.py` is single-device PyTorch. The launcher starts eight
processes; every process builds the model from the **same** seed, draws batches
with its own `data_seed = preset_seed + rank`, and calls
`train.train(..., device="xla")`. The XLA update is ordered explicitly:
`xm.reduce_gradients` once, clip the reduced gradient once, `opt.step()` once,
then `xm.mark_step()`. It deliberately does not call `xm.optimizer_step`, which
would all-reduce a second time and make the effective clip threshold depend on
that hidden reduction. The eight chips therefore train one synchronised model.
Rank 0's directory is the canonical metric series; the other ranks record their
local loss and throughput, which differ by construction.

## Presets

| Preset | Parameters | Tokens/step | Steps | Total tokens | LR | Warmup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tpu_v5e8_102m` | 102,312,960 | 131,072 | 16,000 | 2,097,152,000 | 1.0e-3 | 320 |
| `tpu_v5e8_152m` | 151,868,928 | 131,072 | 24,000 | 3,145,728,000 | 8.0e-4 | 480 |
| `tpu_v5e8_253m` | 252,823,552 | 131,072 | 40,000 | 5,242,880,000 | 6.0e-4 | 800 |
| `tpu_v5e8_303m` | 303,163,392 | 131,072 | 48,000 | 6,291,456,000 | 5.0e-4 | 960 |

The token budget is roughly 20 tokens per parameter (the Chinchilla ratio),
which is why the step counts grow with the model. The learning rates are
starting points for a 4× larger global batch than the CPU presets; they have
not been tuned on TPU and are not results.

Inspect or validate them without a TPU:

```sh
python -m projects.rope_attribution.tpu_configs          # table
python -m projects.rope_attribution.tpu_configs --json   # machine-readable
python -m projects.rope_attribution.tpu_configs --check  # arithmetic check
python -m projects.rope_attribution.tpu_launch \
    --preset tpu_v5e8_102m --dry-run                     # no torch_xla needed
```

## Running

On the TPU VM, install a `torch_xla` build that matches the installed `torch`
and the TPU VM image; this repository does not pin a wheel, because the correct
pair depends on the image. Check the import before launching:

```sh
python -c "import torch_xla; print(torch_xla.__version__)"
export PJRT_DEVICE=TPU
python -m projects.rope_attribution.tpu_launch \
    --preset tpu_v5e8_102m --run-name v5e8_102m
```

The launcher uses `torch_xla.distributed.xla_multiprocessing.spawn` with
`nprocs=8`. Each rank writes `runs/v5e8_102m/rank-<i>/`; rank 0 is canonical.
Read a run with the same viewer as a CPU run:

```sh
python -m projects.rope_attribution.report_run runs/v5e8_102m/rank-0
```

The first run downloads WikiText-103 through `datasets`; each process uses the
same Hugging Face cache, and the cache's file locks make the concurrent first
download safe. Later runs read the cache. `train.py` refuses a CPU run at this
scale; the refusal is why the smoke config exists.

## If a preset does not fit

Lower `per_chip_batch_size` in `tpu_configs.py` and update the derived fields
(`global_batch_size`, `tokens_per_step`, `total_tokens`) until
`--check` passes. The validator refuses a preset whose global batch, token
budget or parameter count disagrees with its parts, so a rushed edit cannot
silently produce a different effective batch than the one written down. Retune
the learning rate for the new batch; the validator cannot do that for you.

## What is and is not verified

| Claim | Evidence |
| --- | --- |
| Preset arithmetic (8 chips, 64 sequences, 131,072 tokens/step) | `tests/test_tpu_configs.py` |
| Parameter counts match `train.CONFIGS` | `tests/test_tpu_configs.py` |
| Launcher CLI, dry run, process-count refusal | `tests/test_tpu_configs.py` |
| XLA import refusal when `torch_xla` is absent | `tests/test_train.py` |
| An actual 8-chip XLA training step | **Not verified here** |
| Any model quality or loss from these presets | **No such claim** |
