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
| Micro-batches per step (`grad_accum`) | 1 |
| Global batch | 8 chips × 8 sequences × 1 = 64 sequences = 131,072 tokens per step |

`grad_accum` is the memory knob: a preset with `grad_accum: 2` runs two
micro-batches of `per_chip_batch_size` through the model before one optimizer
step and scales each loss by `1/2`, so the gradient is the mean over the larger
global batch. The effective batch is
`per_chip_batch_size * chips * grad_accum`; `tpu_configs._preset` derives it and
`validate_preset` refuses a preset whose parts disagree.

`logit_chunk` is the second memory knob, and the one that decides whether the
large presets run at all. The tied head produces `positions * vocab` logits:
8 sequences of 2048 tokens is 16,384 positions, which is 3.3 GB of float32
logits per chip and as much again for their gradient. `train._lm_loss` applies
the head to `logit_chunk` positions at a time under `torch.utils.checkpoint`,
so only one chunk is saved and the head is recomputed during the backward pass;
the loss and its gradient are unchanged, because every target's cross-entropy is
independent of its neighbours. The presets use 1024 positions (206 MB); `0`
means one matrix for the whole batch and is what the `smoke` config keeps.

The trainer in `train.py` is single-device PyTorch. The launcher starts eight
processes; every process builds the model from the **same** seed, draws batches
with its own `data_seed = preset_seed + rank`, and calls
`train.train(..., device="xla", world_size=8)`. The XLA update is ordered explicitly:
`xm.reduce_gradients` once, clip the reduced gradient once, `opt.step()` once,
then `xm.mark_step()`. It deliberately does not call `xm.optimizer_step`, which
would all-reduce a second time and make the effective clip threshold depend on
that hidden reduction. The eight chips therefore train one synchronised model.
Rank 0's directory is the canonical metric series; the other ranks record their
local loss, which differ by construction.

`world_size=8` enters the token accounting, not the model. Each chip consumes
its own per-chip batch, so a step sees `8 * per_chip_batch_size * seq_len *
grad_accum = 131,072` tokens globally at the shipped presets, and every rank logs
that global count. Logging the per-rank count would under-report the run by 8x
and make the published tokens-per-parameter ratio wrong.

**Gradient reduction semantics (verified).** `xm.reduce_gradients` all-reduces
with SUM and then scales by `1 / world_size`. In
[`torch_xla/core/xla_model.py`](https://github.com/pytorch/xla/blob/master/torch_xla/core/xla_model.py),
`reduce_gradients` computes `count = runtime.world_size()` and calls
`all_reduce(REDUCE_SUM, gradients, scale=1.0 / count)`; `all_reduce` applies
`scale` after the reduce. The reduction is therefore a **mean**: the effective
learning rate is the preset's, and the `grad_norm` logged before clipping is the
norm of that mean gradient. The order (reduce once, clip the reduced gradient,
step once) is pinned on CPU with a fake `xm` in `tests/test_train.py`; the
sum-vs-mean question is settled by the source above, not by a TPU run.

The function takes the **optimizer**, not the model: it fetches the gradients
through `optimizer.__getstate__()["param_groups"]`. `train._reduce_update` passes
`opt`, and the fake-`xm` test asserts that it does.

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
python -m projects.rope_attribution.tpu_launch \
    --preset tpu_v5e8_102m --dry-run \
    --grad-accum 2 --log-every 25 --resume                # effective batch shown
```

The launcher accepts the same tuning flags as `train.py --help` (`--max-steps`,
`--dtype`, `--batch-size`, `--grad-accum`, `--logit-chunk`, `--log-every`,
`--eval-every`, `--checkpoint-every`) plus `--resume`. They are validated by the
shared `train.cli_overrides` and layered on top of the preset, so a flag means
the same thing on both entry points.

## Running

Prepare the token stream **once**, on the TPU VM or on a machine that shares
its filesystem. This step tokenizes, shards, checksums, and writes the single
`tokens.u16` every rank will mmap:

```sh
python -m projects.rope_attribution.data prepare \
    --source wikitext-103 --out data/wikitext103
python -m projects.rope_attribution.data check --out data/wikitext103
```

The check re-hashes every shard and the merged stream. Skipping the prepare
step is supported --- the launcher then calls the in-memory tokenizer in every
rank --- but that pays the tokenization cost eight times and leaves no
checksummed artefact.

On the TPU VM, install a `torch_xla` build that matches the installed `torch`
and the TPU VM image; this repository does not pin a wheel, because the correct
pair depends on the image. Check the import before launching:

```sh
python -c "import torch_xla; print(torch_xla.__version__)"
export PJRT_DEVICE=TPU
python -m projects.rope_attribution.tpu_launch \
    --preset tpu_v5e8_102m --run-name v5e8_102m \
    --data-dir data/wikitext103
```

The launcher uses `torch_xla.distributed.xla_multiprocessing.spawn` with
`nprocs=8`. Each rank writes `runs/v5e8_102m/rank-<i>/`; rank 0 is canonical.
Read a run with the same viewer as a CPU run:

```sh
python -m projects.rope_attribution.report_run runs/v5e8_102m/rank-0
```

The viewer writes `curves.csv` and `report.md` from the evaluation records and
`steps.csv` from the per-step telemetry. Step and checkpoint records are not
evaluation rows: `val_loss` exists only on eval records, so mixing the two
series would print a column of blanks.

The first `data prepare` downloads WikiText-103 through `datasets` and writes
the staged stream; later launches only mmap it. `train.py` refuses a CPU run at
this scale; the refusal is why the smoke config exists.

## Checkpoints and resuming

The evaluation checkpoint is written by rank 0 only; the resume state is written
by every rank:

| File | Writer | Contents | Purpose |
| --- | --- | --- | --- |
| `checkpoint.pt` | rank 0 | model weights, config, step | evaluation and fine-tuning |
| `checkpoint_resume.pt` | every rank | weights + AdamW moments + numpy/torch RNG states + counters | continuing that rank's run |

`checkpoint_resume.pt` is replaced atomically every `checkpoint_every` steps and
at the end. Every preset sets `checkpoint_every: 2000` explicitly in
`tpu_configs._preset`, and `launch_overrides` passes it to `train()`; it used to
be inherited from `train.CONFIGS[train_config]`, which made a preset's cadence
invisible. A crash therefore costs the steps since the last save, not the run.

Every rank writes its own resume file because the numpy data-order generator is
per rank; rank 0's file cannot stand in for rank 3's stream. The model and
optimizer contents in the eight files are copies of the same synchronised state,
so the disk cost is the price of not needing a broadcast. That is roughly three
fp32 copies of the parameter count per rank (weights, AdamW's first and second
moments), about 3.6 GB per rank for the 303M preset.

Resume is wired for one process and for the launcher: `train.py --resume` reads
`runs/<name>/checkpoint_resume.pt` and checks the config, parameter count, token
count, world size and data seed against the run that wrote it;
`tpu_launch.py --resume` passes `resume=True` to every rank, and each rank
restores its own file. The eight-process path is unit-tested on CPU through
`train_plan`; it has not been run on a TPU, so the launch itself remains
unverified.

## If a preset does not fit

`global_batch_size`, `tokens_per_step` and `total_tokens` are **derived** by
`tpu_configs._preset`, not hand-edited. To lower the memory per chip, edit the
module constant `_PER_CHIP_BATCH` (the factory takes no per-chip batch
argument); if the global batch should stay the same, raise `grad_accum` in the
same `_preset(...)` call. Then re-run
`python -m projects.rope_attribution.tpu_configs --check`.

The validator does not recompute anything: it checks the derived fields against
the parts. That is exactly what catches a hand-edited `global_batch_size` that
was patched until `--check` passed after the per-chip batch changed. Change the
inputs and let the factory derive the rest; do not patch the derived numbers.
Retune the learning rate for the new effective batch; the validator cannot do
that for you.

## What is and is not verified

| Claim | Evidence |
| --- | --- |
| Preset arithmetic (8 chips, 64 sequences, 131,072 tokens/step) | `tests/test_tpu_configs.py` |
| Preset arithmetic includes `grad_accum` and the token budget | `tests/test_tpu_configs.py` |
| Every preset knob reaches `train()` or is declared metadata | `tests/test_tpu_configs.py` |
| Parameter counts match `train.CONFIGS` | `tests/test_tpu_configs.py` |
| Launcher CLI, dry run, per-rank plan, process-count refusal | `tests/test_tpu_configs.py` |
| XLA import refusal when `torch_xla` is absent | `tests/test_train.py` |
| XLA backend refusal unless `PJRT_DEVICE=TPU` | `tests/test_train.py` |
| `xm.reduce_gradients` averages (SUM then `1/world_size`) | `torch_xla` source, cited above; call signature pinned by `tests/test_train.py` |
| Token accounting is global over ranks and accumulations | `tests/test_train.py` |
| Resume restores optimizer and data order, one summary, one `run_start` | `tests/test_train.py` |
| Per-step and checkpoint telemetry, and their report section | `tests/test_train.py`, `tests/test_report_run.py` |
| Multi-rank resume | Wired per rank (`train_plan`); unit-tested on CPU, not run on a TPU |
| An actual 8-chip XLA training step | **Not verified here** |
| Any model quality or loss from these presets | **No such claim** |
