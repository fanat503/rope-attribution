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
`train.train(..., device="xla", world_size=8)`. The XLA update is ordered explicitly:
`xm.reduce_gradients` once, clip the reduced gradient once, `opt.step()` once,
then `xm.mark_step()`. It deliberately does not call `xm.optimizer_step`, which
would all-reduce a second time and make the effective clip threshold depend on
that hidden reduction. The eight chips therefore train one synchronised model.
Rank 0's directory is the canonical metric series; the other ranks record their
local loss, which differ by construction.

`world_size=8` enters the token accounting, not the model. Each chip consumes
its own per-chip batch, so a step sees `8 * per_chip_batch_size * seq_len =
131,072` tokens globally, and every rank logs that global count. Logging the
per-rank count would under-report the run by 8x and make the published
tokens-per-parameter ratio wrong.

**Unverified:** whether `xm.reduce_gradients` sums or averages the eight
gradients. The order of the operations is tested with a fake `xm` on CPU, but
the reduction semantics need the TPU host (or a pin of the `torch_xla` source).
If it averages, the effective learning rate is the preset's; if it sums, it is
eight times larger. A first TPU run should log `grad_norm` and compare it
against a one-chip run before trusting the loss curve.

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

The first `data prepare` downloads WikiText-103 through `datasets` and writes
the staged stream; later launches only mmap it. `train.py` refuses a CPU run at
this scale; the refusal is why the smoke config exists.

## Checkpoints and resuming

Rank 0 writes two files into `runs/<name>/rank-0/`:

| File | Contents | Purpose |
| --- | --- | --- |
| `checkpoint.pt` | model weights, config, step | evaluation and fine-tuning |
| `checkpoint_resume.pt` | weights + AdamW moments + numpy/torch RNG states + counters | continuing the run |

`checkpoint_resume.pt` is replaced atomically every `checkpoint_every` steps
(2,000 in the presets) and at the end. A crash therefore costs the steps since
the last save, not the run. The file is written only by rank 0 because every
rank holds the same synchronised weights after the all-reduce.

Single-process resume is wired: `python -m ...train --resume` reads
`runs/<name>/checkpoint_resume.pt` and checks the config, parameter count,
token count, world size and data seed against the run that wrote it. **Multi-rank
resume is not wired:** it would need rank 0's state broadcast to the other
chips before the first step, and this repository does not pretend that path is
tested. Restart from scratch, or add the broadcast and test it on a TPU host.

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
| Token accounting is global over ranks | `tests/test_train.py` |
| Resume restores optimizer and data order | `tests/test_train.py` (single process, CPU) |
| `xm.reduce_gradients` sums or averages | **Not verified here**; needs the TPU host |
| Multi-rank resume | **Not implemented** |
| An actual 8-chip XLA training step | **Not verified here** |
| Any model quality or loss from these presets | **No such claim** |
