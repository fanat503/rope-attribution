"""Launch one TPU v5e-8 preset: eight processes, one chip each.

Status boundary
---------------
The preset arithmetic and the launcher's command line are tested on CPU. An
actual XLA run is **not** tested in this repository: CI has no TPU, and this
file refuses to pretend otherwise. On a TPU host, ``torch_xla`` provides the
``xla`` device and ``xla_multiprocessing.spawn`` starts one process per chip.

Each process runs the ordinary ``train.train`` loop on its own chip. Every rank
initialises from the preset's seed; only the data stream differs, through
``train(..., data_seed=seed + rank)``. ``train.py`` all-reduces the gradients
with ``xm.reduce_gradients`` before each update, so the eight chips train one
synchronised model. Rank 0's ``runs/<run-name>/rank-0/`` is the canonical metric
series; the other ranks record their local loss and throughput, which differ by
construction. The collective path needs the TPU host and is not exercised here.

Checkpoints and resume
----------------------
Rank 0 writes the evaluation checkpoint (``checkpoint.pt``), because every rank
holds the same synchronised weights after the all-reduce. Every rank writes its
own ``checkpoint_resume.pt``: the model and optimizer state in it are copies of
the same synchronised state, but the data-order generator is per rank, so rank 0's
file cannot stand in for rank 3's. ``--resume`` is therefore wired: each process
restores its own state, and no broadcast between ranks is needed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

from rope_attribution.tpu_configs import TPU_PRESETS, TPU_V5E8, launch_overrides
from rope_attribution.train import cli_overrides

XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None


def run_dir_for(root: Path, rank: int) -> Path:
    """The per-rank run directory. Pure, so the layout is testable."""
    return Path(root) / f"rank-{int(rank)}"


def train_plan(
    preset_name: str,
    index: int,
    *,
    run_name: str,
    overrides: dict | None = None,
    resume: bool = False,
    max_steps: int | None = None,
    run_root: Path = Path("runs"),
) -> dict:
    """The ``train()`` arguments for one rank, without touching torch.

    Pure so the launch layout is testable on a CPU-only machine: the per-rank
    run directory, the per-rank data seed, the global world size, and which
    files the rank writes. ``overrides`` are command-line flags layered on top
    of :func:`launch_overrides`, so a flag means the same thing here as it does
    in ``train.main``.
    """
    preset = TPU_PRESETS[preset_name]
    override = launch_overrides(preset_name)
    override.update(overrides or {})
    return {
        "config_name": str(preset["train_config"]),
        "run_dir": run_dir_for(Path(run_root) / run_name, index),
        "device": "xla",
        "override": override,
        "data_seed": int(preset["seed"]) + int(index),
        # Each chip consumes its own batch; the logged token count is the
        # global one, so eight chips report eight batches per step.
        "world_size": int(preset["chips"]),
        # Rank 0 is the canonical metric series and the synchronized model is
        # identical on every rank after the all-reduce, so one evaluation
        # checkpoint is enough. The resume state is per rank: its data order
        # is not.
        "save_checkpoint": int(index) == 0,
        "save_resume": True,
        "resume": bool(resume),
        "max_steps": max_steps,
    }


def _mp_fn(
    index: int,
    preset_name: str,
    run_name: str,
    data_dir: str | None,
    overrides: dict | None = None,
    resume: bool = False,
    max_steps: int | None = None,
) -> None:
    import torch_xla.core.xla_model as xm

    from rope_attribution.train import _load_wikitext_tokens, train

    if data_dir is None:
        # Every rank tokenizes the corpus in this fallback. Correct, but the
        # cost is paid eight times; ``--data-dir`` maps one prepared stream.
        tokens = _load_wikitext_tokens()
    else:
        from rope_attribution.data import load_tokens

        tokens = load_tokens(Path(data_dir))
    # The model seed is shared, so the replicas start identical; the data seed
    # is per rank, so they see different batches. Gradients are all-reduced in
    # train.py, which keeps the replicas synchronised.
    plan = train_plan(
        preset_name,
        index,
        run_name=run_name,
        overrides=overrides,
        resume=resume,
        max_steps=max_steps,
    )
    config_name = plan.pop("config_name")
    run_dir = plan.pop("run_dir")
    summary = train(config_name, run_dir, tokens, **plan)
    xm.master_print(f"[rank {index}] {json.dumps(summary)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(TPU_PRESETS), required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--nprocs", type=int, default=int(TPU_V5E8["chips"]))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="prepared token stream; without it every rank tokenizes the corpus",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue each rank's runs/<name>/rank-<i>/checkpoint_resume.pt",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="override the preset's n_steps"
    )
    parser.add_argument(
        "--dtype", choices=("fp32", "bf16"), default=None,
        help="override the preset's dtype",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="override the preset's per-chip batch size",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=None,
        help="micro-batches per optimizer step (saves memory, same global batch "
        "only when the per-chip batch is lowered with it)",
    )
    parser.add_argument(
        "--logit-chunk", type=int, default=None,
        help="positions per chunk when the output head is applied (0 = one matrix)",
    )
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the preset, the per-rank seeds and the effective batch; "
        "do not require torch_xla",
    )
    args = parser.parse_args(argv)

    preset = TPU_PRESETS[args.preset]
    chips = int(TPU_V5E8["chips"])
    if args.nprocs != chips:
        raise SystemExit(
            f"this preset describes a {chips}-chip slice; --nprocs must be {chips}, "
            f"not {args.nprocs}"
        )
    overrides = cli_overrides(
        dtype=args.dtype,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        logit_chunk=args.logit_chunk,
        log_every=args.log_every,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
    )
    run_name = args.run_name or args.preset
    if args.dry_run:
        per_chip = int(overrides.get("batch_size", preset["per_chip_batch_size"]))
        grad_accum = int(overrides.get("grad_accum", preset["grad_accum"]))
        seq_len = int(preset["seq_len"])
        global_batch = per_chip * args.nprocs * grad_accum
        n_steps = (
            args.max_steps if args.max_steps is not None else int(preset["n_steps"])
        )
        print(json.dumps(
            {
                "preset": args.preset,
                "run_name": run_name,
                "nprocs": args.nprocs,
                "per_chip_batch_size": per_chip,
                "seq_len": seq_len,
                "grad_accum": grad_accum,
                "global_batch_size": global_batch,
                "tokens_per_step": global_batch * seq_len,
                "n_steps": n_steps,
                "total_tokens": global_batch * seq_len * n_steps,
                "data_dir": None if args.data_dir is None else str(args.data_dir),
                "overrides": overrides,
                "resume": bool(args.resume),
                "max_steps": args.max_steps,
                "rank_data_seeds": [
                    int(preset["seed"]) + rank for rank in range(args.nprocs)
                ],
                "run_dirs": [
                    str(run_dir_for(Path("runs") / run_name, rank))
                    for rank in range(args.nprocs)
                ],
            },
            indent=2,
        ))
        return

    if not XLA_AVAILABLE:
        raise SystemExit(
            "torch_xla is not installed; install the TPU build on the TPU host "
            "(see docs/TPU_V5E8.md) or use --dry-run to inspect the preset"
        )
    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(
        _mp_fn,
        args=(
            args.preset,
            run_name,
            None if args.data_dir is None else str(args.data_dir),
            overrides,
            bool(args.resume),
            args.max_steps,
        ),
        nprocs=args.nprocs,
        start_method="spawn",
    )


if __name__ == "__main__":
    main()
