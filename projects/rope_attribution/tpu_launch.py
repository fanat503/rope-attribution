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
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

from rope_attribution.tpu_configs import TPU_PRESETS, TPU_V5E8, launch_overrides

XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None


def run_dir_for(root: Path, rank: int) -> Path:
    """The per-rank run directory. Pure, so the layout is testable."""
    return Path(root) / f"rank-{int(rank)}"


def _mp_fn(index: int, preset_name: str, run_name: str) -> None:
    import torch_xla.core.xla_model as xm

    from rope_attribution.train import _load_wikitext_tokens, train

    preset = TPU_PRESETS[preset_name]
    override = launch_overrides(preset_name)
    # The model seed is shared, so the replicas start identical; the data seed
    # is per rank, so they see different batches. Gradients are all-reduced in
    # train.py, which keeps the replicas synchronised.
    data_seed = int(preset["seed"]) + int(index)
    summary = train(
        str(preset["train_config"]),
        run_dir_for(Path("runs") / run_name, index),
        _load_wikitext_tokens(),
        device="xla",
        override=override,
        data_seed=data_seed,
        # Rank 0 is the canonical metric series and the synchronized model is
        # identical on every rank after the all-reduce, so one checkpoint is
        # enough. Writing eight copies of a multi-gigabyte file is wasteful.
        save_checkpoint=(int(index) == 0),
    )
    xm.master_print(f"[rank {index}] {json.dumps(summary)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(TPU_PRESETS), required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--nprocs", type=int, default=int(TPU_V5E8["chips"]))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the preset and the per-rank seeds; do not require torch_xla",
    )
    args = parser.parse_args(argv)

    preset = TPU_PRESETS[args.preset]
    chips = int(TPU_V5E8["chips"])
    if args.nprocs != chips:
        raise SystemExit(
            f"this preset describes a {chips}-chip slice; --nprocs must be {chips}, "
            f"not {args.nprocs}"
        )
    run_name = args.run_name or args.preset
    if args.dry_run:
        print(json.dumps(
            {
                "preset": args.preset,
                "run_name": run_name,
                "nprocs": args.nprocs,
                "per_chip_batch_size": preset["per_chip_batch_size"],
                "seq_len": preset["seq_len"],
                "global_batch_size": preset["global_batch_size"],
                "tokens_per_step": preset["tokens_per_step"],
                "n_steps": preset["n_steps"],
                "total_tokens": preset["total_tokens"],
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

    xmp.spawn(_mp_fn, args=(args.preset, run_name), nprocs=args.nprocs, start_method="spawn")


if __name__ == "__main__":
    main()
