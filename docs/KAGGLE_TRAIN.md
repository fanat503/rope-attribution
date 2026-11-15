# Running the training measurement on Kaggle

This is the runbook for the experiment `projects/rope_attribution/train.py`
exists to make: watch the paper's attribution quantities *while a model this
repository owns is learning*. It is written for the owner to execute; nothing
here runs on the development machine, and no Kaggle kernel has been executed as
part of this repository's evidence.

**Status boundary.** The code paths are tested on CPU (`tests/test_train.py`,
`tests/test_tpu_configs.py`, `tests/test_data.py`). A Kaggle GPU session, a
Kaggle TPU VM and the `gigatoken` lane are **not** tested here. The step counts
and time estimates below are arithmetic, not a throughput benchmark. Run the
20-step timing job in §2 before committing a session to a budget.

## 0. What one session can and cannot do

A Kaggle GPU session is bounded (about 12 hours per run and about 30 GPU-hours
per week on the standard tier); a TPU VM session is bounded too. Pretraining a
124M model to GPT-2 quality takes hundreds of billions of tokens and does not
fit. What fits is a **training trajectory**: a few thousand optimizer steps,
enough for the diagnostics to move, with the loss falling and the checkpoints
saved. The scientific object is the curve, not a final accuracy number.

Do not describe a 3,000-step run as "a trained 124M model" in the paper. It is
"a 124M model observed for N tokens"; the claim it can support is about the
stability of the attribution quantities during training, and only if the curve
is actually stable.

## 1. Prepare the token stream once

Use the gigatoken lane (`docs/KAGGLE.md` §Lane C) for a corpus of tens of GB, or
Lane A/B for a small one. A 100M-token stream is 200 MB as `uint16`; a 2B-token
stream is 4 GB. Keep the stream under `/kaggle/working/data/<name>/` and verify
it:

```sh
python -m projects.rope_attribution.gigatoken_prep \
    --text /kaggle/input/<corpus>/corpus.txt \
    --out /kaggle/working/data/corpus \
    --name corpus \
    --max-tokens 500000000

python -m projects.rope_attribution.data check \
    --out /kaggle/working/data/corpus
```

`check` re-hashes the merged file. If it fails, do not train on the stream.

## 2. Measure throughput before choosing a budget

A 20-step run writes real telemetry and costs minutes:

```sh
python -m projects.rope_attribution.train \
    --config train_102m --device cuda \
    --data-dir /kaggle/working/data/corpus \
    --run-name timing --max-steps 20 --no-checkpoint \
    --batch-size 8 --grad-accum 4 --log-every 5
```

Read `runs/timing/metrics.jsonl`: the first `run_start` record carries
`global_batch_size` and the resolved `config` (including `ctx`), and each later
record carries `tok_per_s`. One optimizer step consumes
`global_batch_size * config.ctx` tokens, so a step budget is

```
steps = session_seconds * tok_per_s / (global_batch_size * ctx)
```

Leave 20% headroom for evaluation and checkpoint writes. If the projected run
does not reach at least 1,000 steps, lower `--batch-size`/`--grad-accum` or use
`train_102m` instead of `train_124m`; do not silently shrink the model and call
it the same experiment.

## 3. GPU lane

The named presets default to `fp32` because the reference device is a T4, which
has no `bf16`. On an A100/L4/H100 session add `--dtype bf16`.

```sh
python -m projects.rope_attribution.train \
    --config train_102m --device cuda \
    --data-dir /kaggle/working/data/corpus \
    --run-name kaggle_102m \
    --batch-size 8 --grad-accum 4 \
    --max-steps 3000 --checkpoint-every 500 \
    --eval-every 250 --log-every 25
```

At `batch_size=8`, `grad_accum=4`, `ctx=1024` one step is 32,768 tokens, so
3,000 steps is 98.3M tokens. The rolling `checkpoint_resume.pt` is written every
500 steps; after a session ends, copy `runs/kaggle_102m/` into a Kaggle dataset
and continue in the next session with the same command plus `--resume`.

## 4. TPU lane

`docs/TPU_V5E8.md` has the topology, the presets and the `torch_xla` caveat. On
a v5e-8 VM:

```sh
export PJRT_DEVICE=TPU
python -m projects.rope_attribution.tpu_launch \
    --preset tpu_v5e8_102m --run-name v5e8_102m \
    --data-dir /kaggle/working/data/corpus \
    --max-steps 4000 --checkpoint-every 500
```

One step is 131,072 tokens at the shipped presets, so 4,000 steps is 524M
tokens. The XLA path has never been run by this repository; if the first step
fails, record the traceback and use the GPU lane instead of debugging XLA inside
a timed session.

## 5. What to bring back

`runs/<name>/` contains:

| File | Size at 102M | Purpose |
| --- | ---: | --- |
| `metrics.jsonl` | a few MB | per-step loss, LR, grad norm, throughput, and the attribution probe |
| `summary.json` | KB | closing record: steps, tokens, final losses |
| `manifest.json` | KB | config, dtype, device, git commit, platform |
| `checkpoint.pt` | ~0.4 GB | weights only; evaluation and fine-tuning |
| `checkpoint_resume.pt` | ~1.2 GB | weights + AdamW moments + RNG; resuming |

Commit **only** `metrics.jsonl`, `summary.json` and `manifest.json` (for
example under `results/training/<run>/`). Never commit `checkpoint*.pt` or the
token stream; `runs/` and `data/` are already in `.gitignore`.

## 6. How to read the run

```sh
python -m projects.rope_attribution.report_run runs/kaggle_102m
```

writes `curves.csv` and `report.md`. The fields that matter for the paper are
`sign_flip_rate`, `rel_err_median_probe` and `rope_probe_resid_max`. The first
two are the position-free cost on the current weights; the third is the
rotation-fidelity guard and should stay at float noise.

Pre-register the read before looking at the curve:

* **Stable** if every evaluation after warmup is inside the 18.3–32.0% band
  `results/sensitivity.json` reports for the three published checkpoints.
* **Moving** if the curve leaves that band in one direction. A moving curve is
  a result about training, not a failure; report it with the step at which it
  left.
* **Not interpretable** if `rope_probe_resid_max` is not at float noise, or if
  the loss is flat. Fix the run, do not report the probe.

Do not select the evaluation step that looks best. The curve is the result.

## 7. What this run does not establish

* It is not a 1B–8B downstream result; that protocol is
  `docs/DOWNSTREAM_PROTOCOL.md` and needs a separate harness.
* It does not show that a position-free *attribution* changes a circuit. It
  shows whether the quantity the attribution summarizes stays put during
  training on weights this repository owns.
* It does not validate the TPU launcher, the gigatoken lane, or Kaggle itself.
  Those are infrastructure, and the honest label is "run by the owner on
  Kaggle", with the kernel log kept beside the artifacts.
