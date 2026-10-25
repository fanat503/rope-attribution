"""Train the model family the paper's claims are about, with every metric.

Why this module exists
----------------------
Every number in the paper comes from checkpoints other people trained. The
attribution quantities this repository measures -- the exactness of the
additive decomposition, the position-free cost, the rotation fidelity --
have never been observed *during* training, on weights this repository owns.
A reviewer can reasonably ask whether the trained checkpoints are special:
did post-training wander into a configuration where the decomposition is
less exact than measured? This module turns the training loop into a
measurement: every evaluation period it logs the usual training metrics
(learning rate, gradient norm, losses, throughput) *and* the attribution
diagnostics on one fixed probe batch, so a run's ``metrics.jsonl`` answers
"what happened to the paper's quantities while the model learned".

The named configurations span 100M-300M parameters, on shapes the literature
already knows (GPT-2 tokenizer, tied embeddings, bias-free RMSNorm blocks,
full RoPE at head dimension 64):

* ``train_124m`` -- 12 layers, d=768, 12 heads, FFN 3072 (GPT-2 small shape);
* ``train_208m`` -- 24 layers, d=768, 12 heads, FFN 3072 (depth-scaled);
* ``train_303m`` -- 20 layers, d=1024, 16 heads, FFN 4096 (width-scaled).

Availability and safety
-----------------------
torch and transformers are optional dependencies; they are imported inside
the functions that use them and the test suite skips cleanly without them
(the same pattern real_model.py uses). The named 100M-300M configurations
are for a GPU machine: a CPU device refuses them unless ``--force-cpu`` is
passed, because a training run pegs every core of the machine this
repository is developed on. The tiny ``smoke`` configuration exists for the
test suite and for one-command verification that the whole loop works;
nothing in the repository runs anything larger automatically.

Determinism contract
--------------------
One seed drives the initialization, the data order and the probe sampling.
Two runs of the same command on the same torch version, thread count and
machine produce identical eval records on CPU: every field of every record
is byte-identical except the single throughput field ``tok_per_s``, which
is wall-clock by definition. On CUDA the cuBLAS workspace is fixed to
``:4096:8`` and deterministic algorithms are requested with ``warn_only``:
a kernel that cannot comply is reported, not silently assumed to be
reproducible. On TPU (``--device xla``) the trainer is a thin adapter over the
same loop: it imports ``torch_xla``, builds on the XLA device, all-reduces
gradients once, clips the reduced gradient, and steps the optimizer once. The
XLA path is not exercised by CI, which has no TPU; ``docs/TPU_V5E8.md`` says so
explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.util
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rope_attribution.rope import inv_freq
from rope_attribution.usefulness import DISTANCE_GRID

__all__ = [
    "CONFIGS",
    "TRAIN_PARAM_BANDS",
    "TORCH_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
    "TRAINING_AVAILABLE",
    "count_params",
    "lr_at",
    "MetricsWriter",
    "train",
    "main",
]

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
TRAINING_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE
XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None

# The GPT-2 vocabulary, pinning the tokenizer to one every reviewer has used.
VOCAB_SIZE = 50257
#: Head dimension everywhere, matching the paper's measured configuration.
HEAD_DIM = 64
#: Rotary base everywhere, ditto.
ROPE_BASE = 10000.0

# RMSNorm scale vectors are the only parameters besides the matrices, which
# are bias-free, and the embeddings, which are tied to the output head. RoPE
# adds nothing; the head is tied, so it adds nothing either.
_PARAM_FORMULA_DOC = (
    "embeddings V*d + per layer (4*d^2 attention + 2*d*f FFN + 2*d norms) "
    "+ final norm d; no biases, no positional parameters (RoPE), tied head"
)


def count_params(config: dict) -> int:
    """Exact parameter count in closed form.

    ``build_model`` realizes exactly these tensors on every block, and
    ``test_train.py`` checks model.parameters() against this function, so
    the formula cannot drift from the module.
    """
    dim, ffn, n_layers = config["dim"], config["ffn"], config["n_layers"]
    per_layer = 4 * dim * dim + 2 * dim * ffn + 2 * dim
    return config["vocab"] * dim + n_layers * per_layer + dim


CONFIGS: dict[str, dict] = {
    # A configuration the test suite can run end to end in seconds on CPU.
    # It is not a paper object; it exists so the whole loop -- forward,
    # backward, metrics, diagnostics, summary -- is verifiable cheaply.
    "smoke": {
        "n_layers": 2,
        "n_heads": 2,
        "head_dim": 32,
        "dim": 64,
        "ffn": 256,
        "vocab": 512,
        "ctx": 64,
        "lr": 3e-3,
        "n_steps": 12,
        "batch_size": 4,
        "eval_every": 4,
        "checkpoint_every": 4,
        "val_batches": 2,
        "warmup_steps": 4,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
        "dtype": "fp32",
    },
    "train_102m": {
        "n_layers": 9,
        "n_heads": 12,
        "head_dim": HEAD_DIM,
        "dim": 768,
        "ffn": 3072,
        "vocab": VOCAB_SIZE,
        "ctx": 1024,
        "lr": 6e-4,
        "n_steps": 80_000,
        "batch_size": 32,
        "eval_every": 500,
        "checkpoint_every": 2_000,
        "val_batches": 8,
        "warmup_steps": 2_000,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
        "dtype": "fp32",
    },
    "train_124m": {
        "n_layers": 12,
        "n_heads": 12,
        "head_dim": HEAD_DIM,
        "dim": 768,
        "ffn": 3072,
        "vocab": VOCAB_SIZE,
        "ctx": 1024,
        "lr": 6e-4,
        "n_steps": 100_000,
        "batch_size": 32,
        "eval_every": 500,
        "checkpoint_every": 2_000,
        "val_batches": 8,
        "warmup_steps": 2_000,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
        "dtype": "fp32",
    },
    "train_152m": {
        "n_layers": 16,
        "n_heads": 12,
        "head_dim": HEAD_DIM,
        "dim": 768,
        "ffn": 3072,
        "vocab": VOCAB_SIZE,
        "ctx": 1024,
        "lr": 6e-4,
        "n_steps": 110_000,
        "batch_size": 32,
        "eval_every": 500,
        "checkpoint_every": 2_000,
        "val_batches": 8,
        "warmup_steps": 2_000,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
        "dtype": "fp32",
    },
    "train_208m": {
        "n_layers": 24,
        "n_heads": 12,
        "head_dim": HEAD_DIM,
        "dim": 768,
        "ffn": 3072,
        "vocab": VOCAB_SIZE,
        "ctx": 1024,
        "lr": 6e-4,
        "n_steps": 120_000,
        "batch_size": 32,
        "eval_every": 500,
        "checkpoint_every": 2_000,
        "val_batches": 8,
        "warmup_steps": 2_000,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
        "dtype": "fp32",
    },
    "train_253m": {
        "n_layers": 16,
        "n_heads": 16,
        "head_dim": HEAD_DIM,
        "dim": 1024,
        "ffn": 4096,
        "vocab": VOCAB_SIZE,
        "ctx": 1024,
        "lr": 6e-4,
        "n_steps": 130_000,
        "batch_size": 32,
        "eval_every": 500,
        "checkpoint_every": 2_000,
        "val_batches": 8,
        "warmup_steps": 2_000,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
        "dtype": "fp32",
    },
    "train_303m": {
        "n_layers": 20,
        "n_heads": 16,
        "head_dim": HEAD_DIM,
        "dim": 1024,
        "ffn": 4096,
        "vocab": VOCAB_SIZE,
        "ctx": 1024,
        "lr": 6e-4,
        "n_steps": 140_000,
        "batch_size": 32,
        "eval_every": 500,
        "checkpoint_every": 2_000,
        "val_batches": 8,
        "warmup_steps": 2_000,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
        "dtype": "fp32",
    },
}

#: The named configurations must land inside these bands; a config that edits
#: its widths out of its band fails the suite instead of shipping.
TRAIN_PARAM_BANDS = {
    "train_102m": (95_000_000, 120_000_000),
    "train_124m": (100_000_000, 130_000_000),
    "train_152m": (140_000_000, 170_000_000),
    "train_208m": (190_000_000, 220_000_000),
    "train_253m": (240_000_000, 270_000_000),
    "train_303m": (290_000_000, 320_000_000),
}

# Weight decay applies to the model's matrices, not to norms or embeddings.
_WD_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out")

#: A CPU run above this parameter count refuses to start without an explicit
#: override: the machine this repository is developed on has six cores.
CPU_PARAM_LIMIT = 20_000_000


def _cpu_refusal(
    config_name: str, n_params: int, device: str, force_cpu: bool
) -> str | None:
    """The refusal message for a named config asked to run on CPU, or None.

    A string rather than an exception so both callers -- ``train()`` raising
    RuntimeError, ``main()`` raising SystemExit -- share one wording, and so
    the torch-free test suite can exercise the whole matrix without a model.
    """
    if device != "cpu" or force_cpu or n_params <= CPU_PARAM_LIMIT:
        return None
    return (
        f"{config_name} realizes {n_params:,} parameters; the named "
        "configurations are GPU territory and this machine must not be "
        "pegged by them. Pass --force-cpu to override deliberately, or "
        "call train() with device='cuda'."
    )


def lr_at(step: int, config: dict) -> float:
    """The schedule: linear warmup to ``lr``, then cosine decay to the floor.

    Pure arithmetic on the config, so the test suite checks the shape of the
    schedule without torch: zero at step 0, the peak exactly at the warmup
    boundary, and never below the floor afterwards.
    """
    peak = config["lr"]
    warmup = config["warmup_steps"]
    if step <= 0:
        return 0.0
    if step < warmup:
        return peak * step / warmup
    floor = peak * config["lr_floor_fraction"]
    progress = (step - warmup) / max(1, config["n_steps"] - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (peak - floor) * cosine


class MetricsWriter:
    """One JSONL line per record, plus a closing ``summary.json``.

    Keeps the byte-identical promise cheap: keys are written in insertion
    order, floats in repr, one line per record -- two identical runs produce
    identical files on the same torch version, thread count and machine,
    except the wall-clock throughput field.
    """

    def __init__(self, run_dir: Path, *, append: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        if append:
            # A resumed run continues the same metric series. The previous
            # summary described a run that had not finished, so it must not
            # outlive the record it belongs to; it is rewritten at finalize.
            if self.summary_path.exists():
                self.summary_path.unlink()
            lines = (
                self.path.read_text(encoding="utf-8").splitlines()
                if self.path.exists()
                else []
            )
            self._lines: list[str] = [line for line in lines if line.strip()]
            return
        # A run directory is a fresh run: files from a previous run of the
        # same name must not survive next to this run's records -- a stale
        # metrics.jsonl would be appended to, and a stale summary.json from
        # a crashed run would outlive the run it describes.
        for path in (self.path, self.summary_path):
            if path.exists():
                path.unlink()
        self._lines: list[str] = []

    def log(self, record: dict) -> None:
        # Append rather than rewrite: a crash mid-run then costs the tail
        # line, not the history. Non-finite floats become null here, so the
        # file is strict JSON even when the run diverged.
        line = json.dumps(_json_safe(record), allow_nan=False)
        self._lines.append(line)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def finalize(self, summary: dict) -> None:
        self.summary_path.write_text(
            json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @property
    def records(self) -> list:
        return [json.loads(line) for line in self._lines]


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a torch payload to a sibling temporary file, then replace.

    A crash during a multi-gigabyte write cannot leave the destination
    truncated. ``os.replace`` is atomic on the same filesystem.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cpu_tree(value):
    """Move every tensor in a nested optimizer/RNG structure to CPU."""
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _save_checkpoint(
    model,
    config: dict,
    path: Path,
    *,
    step: int,
    n_params: int,
) -> str:
    """Save the trained weights, not a resumable training state.

    The checkpoint is for evaluation and for the open-model path the README
    describes: it contains the CPU-side ``state_dict`` plus the config needed
    to rebuild the module. Optimizer and RNG state are deliberately omitted.
    A file that looks resumable but silently restarts AdamW from zero is worse
    than a file that says what it is; the resumable file is
    ``checkpoint_resume.pt``, written by :func:`_save_training_state`.
    """
    state = _cpu_tree(model.state_dict())
    payload = {
        "format": "rope_attribution/train-checkpoint/v1",
        "config": dict(config),
        "step": int(step),
        "n_params": int(n_params),
        "model": state,
    }
    path = Path(path)
    _atomic_torch_save(payload, path)
    return path.name


def _save_training_state(
    model,
    opt,
    rng: np.random.Generator,
    path: Path,
    *,
    config: dict,
    step: int,
    n_params: int,
    world_size: int,
    loss_ema: float | None,
    best_val_loss: float | None,
    best_step: int | None,
    final_val_loss: float | None,
    tokens_seen: int,
    elapsed_s: float,
    data_seed: int | None,
    tokens_total: int,
) -> str:
    """The state a run needs to continue, as against a state to evaluate.

    AdamW's moments and the data-order generator are part of the experiment:
    resuming without them is a different run wearing the same name. The
    numpy generator state is stored as the dict its own API returns, so
    restoring it is exact rather than a reseed.
    """
    import torch

    payload = {
        "format": "rope_attribution/train-state/v1",
        "config": dict(config),
        "step": int(step),
        "n_params": int(n_params),
        "world_size": int(world_size),
        "data_seed": None if data_seed is None else int(data_seed),
        "tokens_total": int(tokens_total),
        "tokens_seen": int(tokens_seen),
        "elapsed_s": float(elapsed_s),
        "loss_ema": loss_ema,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "final_val_loss": final_val_loss,
        "model": _cpu_tree(model.state_dict()),
        "optimizer": _cpu_tree(opt.state_dict()),
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state().cpu(),
    }
    path = Path(path)
    _atomic_torch_save(payload, path)
    return path.name


def _load_training_state(path: Path) -> dict:
    """Load and validate a ``checkpoint_resume.pt``, refusing other formats."""
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"resume requested but {path} does not exist; a fresh run must be "
            "started without --resume"
        )
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "rope_attribution/train-state/v1":
        raise ValueError(
            f"{path} is {payload.get('format')!r}, not a resumable training "
            "state; checkpoint.pt holds weights only and cannot resume a run"
        )
    return payload


def _git_commit() -> str | None:
    """The checked-out commit, or ``None`` when git is unavailable.

    A run's numbers depend on the code that produced them. Recording the
    commit in the manifest is cheap; reconstructing it after the fact is
    usually impossible.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _write_manifest(
    run_dir: Path,
    *,
    config_name: str,
    config: dict,
    n_params: int,
    device: str,
    dtype: str,
    world_size: int,
    tokens_total: int,
    data_seed: int | None,
    resumed_from_step: int | None = None,
) -> str:
    """The provenance a reviewer needs to reproduce or disbelieve a run.

    Separate from ``metrics.jsonl`` on purpose: a wall-clock timestamp there
    would break the byte-identical promise between two seeded runs, while in
    the manifest it is the point.
    """
    import torch

    payload = {
        "format": "rope_attribution/train-manifest/v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "config_name": config_name,
        "config": _json_safe(config),
        "n_params": int(n_params),
        "device": device,
        "dtype": dtype,
        "world_size": int(world_size),
        "tokens_total": int(tokens_total),
        "data_seed": None if data_seed is None else int(data_seed),
        "resumed_from_step": resumed_from_step,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    name = (
        "manifest.json"
        if resumed_from_step is None
        else f"manifest_resume_{int(resumed_from_step)}.json"
    )
    path = Path(run_dir) / name
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return name


def _set_determinism(seed: int) -> None:
    import torch

    # cuBLAS needs a fixed workspace for deterministic reductions. The variable
    # must be set before the CUDA context is initialised, which is why this
    # happens before the model is built. ``setdefault`` leaves an explicit
    # caller choice intact.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_threads(device: str, threads: int | None) -> int:
    """The torch thread count: an explicit value wins, else CPU 4 / GPU 8."""
    if threads is not None:
        return int(threads)
    return 4 if device == "cpu" else 8


def _require_xla():
    """Import torch_xla and return its XLA model module, or refuse clearly.

    Importing ``torch_xla`` is what registers the ``xla`` device with torch, so
    this must run before the first ``torch.device("xla")`` / ``.to("xla")``.
    """
    if not XLA_AVAILABLE:
        raise RuntimeError(
            "device 'xla' requested but torch_xla is not installed; "
            "install the TPU build of torch_xla on the TPU host "
            "(see docs/TPU_V5E8.md)"
        )
    import torch_xla.core.xla_model as xm

    return xm


def _amp_dtype(device: str, dtype: str):
    """The autocast dtype, or ``None`` for a full-precision run.

    ``dtype`` is an explicit training choice, never a label: a ``bf16`` run on
    CPU is refused rather than silently trained in fp32, because a metrics file
    that says ``bf16`` must describe the arithmetic that produced it. CUDA and
    XLA both support bfloat16 autocast; the master weights and optimizer states
    stay fp32, which is what makes the run reproducible rather than a
    lower-precision copy of AdamW.
    """
    if dtype not in ("fp32", "bf16"):
        raise ValueError(f"dtype must be 'fp32' or 'bf16', not {dtype!r}")
    if dtype == "fp32":
        return None
    if device not in ("cuda", "xla"):
        raise ValueError(
            f"dtype='bf16' is supported on cuda and xla, not {device!r}; "
            "use dtype='fp32' for a CPU run"
        )
    import torch

    return torch.bfloat16


def _autocast_context(device: str, dtype: str):
    """Autocast only around the forward pass; the loss stays in fp32."""
    amp_dtype = _amp_dtype(device, dtype)
    if amp_dtype is None:
        return contextlib.nullcontext()
    import torch

    return torch.autocast(device_type=device, dtype=amp_dtype)


def _reduce_update(model, opt, xm, grad_clip: float):
    """All-reduce once, clip the reduced gradient, then step once.

    The order is the point. Clipping each rank's local gradient before the
    all-reduce would make the effective clip threshold depend on how many
    chips happened to see a large gradient; calling ``xm.optimizer_step`` after
    an explicit ``reduce_gradients`` would all-reduce a second time. This helper
    is deliberately torch_xla-free at import time so the order can be tested on
    CPU with a fake ``xm``.
    """
    import torch

    if xm is not None:
        xm.reduce_gradients(model)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    if xm is not None:
        xm.mark_step()
    return grad_norm


def _to_numpy_f64(tensor):
    """Transfer to CPU first, then widen: never double the device transfer.

    bfloat16 has no numpy dtype, so a bf16 tensor is widened to float32 on the
    CPU before ``.numpy()``. Without that cast the probe raises ``TypeError``
    on the one device the bf16 presets target, which is exactly where it
    cannot be caught by a CPU test run.
    """
    import torch

    cpu = tensor.detach().cpu()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.float()
    return cpu.numpy().astype(np.float64)


def _json_safe(value):
    """A JSON value with every non-finite float replaced by ``None``.

    ``json.dumps`` writes bare ``NaN``/``Infinity`` by default. Those are not
    JSON, so a metrics file containing one is unreadable by any strict parser
    -- and a validation loss that went non-finite is precisely the run whose
    records a reader needs. ``None`` says "not a number" in a way every parser
    accepts; ``allow_nan=False`` then turns any missed case into an exception
    at write time rather than a corrupt file at read time.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _cos_sin_torch(positions: np.ndarray, head_dim: int, base: float, device):
    """The cos/sin tables, built from the *reference* frequency ladder.

    ``rope.inv_freq`` is the ladder the paper's algebra is about, so the
    training model rotates at exactly those frequencies; the probe then
    measures how faithfully the float32 torch rotation follows the float64
    reference on the same weights.
    """
    import torch

    freqs = inv_freq(head_dim, base)
    angles = positions[:, None].astype(np.float64) * freqs[None, :]
    angles = np.concatenate([angles, angles], axis=-1)
    cos = torch.from_numpy(np.cos(angles)).to(device=device, dtype=torch.float32)
    sin = torch.from_numpy(np.sin(angles)).to(device=device, dtype=torch.float32)
    return cos, sin


def _rotate_half_torch(x):
    """``cat((-x2, x1))`` over the last axis, the transformers convention."""
    import torch

    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_np(x: np.ndarray) -> np.ndarray:
    """The float64 twin of :func:`_rotate_half_torch`."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def _apply_rope_ref_4d(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """``rope.apply_rope`` for a tensor that also carries a head axis.

    ``rope.apply_rope`` broadcasts ``cos`` of shape ``(seq, head_dim)``
    against the last two axes, which fits ``(seq, dim)`` layouts; the probe's
    captured tensors are ``(batch, seq, heads, head_dim)``, so the same
    arithmetic dodges the head axis here. ``tests/test_train.py`` pins this
    helper against ``rope.apply_rope``'s own output on the overlapping
    shapes, so the two implementations cannot drift apart.
    """
    return x * cos[None, :, None, :] + _rotate_half_np(x) * sin[None, :, None, :]


def build_model(config: dict, device: str = "cpu"):
    """Build the decoder. Torch classes are defined here, lazily.

    Pre-norm RMSNorm blocks, bias-free attention with the split-half RoPE
    convention of ``rope.py``, GELU MLP, no dropout (determinism), and an
    output head tied to the embedding. The per-pair algebra the paper proves
    applies to exactly the ``q_proj``/``k_proj`` outputs of this shape.
    """
    import torch
    import torch.nn as nn

    if config["dim"] % config["n_heads"] != 0:
        raise ValueError("dim must be divisible by n_heads")
    if config["n_heads"] * config["head_dim"] != config["dim"]:
        raise ValueError(
            f"n_heads*head_dim = {config['n_heads'] * config['head_dim']} "
            f"must equal dim = {config['dim']}"
        )

    class RMSNorm(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + 1e-6) * self.weight

    class Block(nn.Module):
        def __init__(self, cfg: dict):
            super().__init__()
            dim, ffn = cfg["dim"], cfg["ffn"]
            self.attn_norm = RMSNorm(dim)
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.o_proj = nn.Linear(dim, dim, bias=False)
            self.mlp_norm = RMSNorm(dim)
            self.fc_in = nn.Linear(dim, ffn, bias=False)
            self.fc_out = nn.Linear(ffn, dim, bias=False)
            self.n_heads = cfg["n_heads"]
            self.head_dim = cfg["head_dim"]

        def forward(self, x, cos_buf, sin_buf, capture, index):
            import torch.nn.functional as F

            residual = x
            h = self.attn_norm(x)
            B, T, _ = h.shape
            q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim)
            k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim)
            v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim)
            if capture is not None:
                capture.setdefault("pre_rope", {})[index] = (q, k)
            cos4 = cos_buf[:T][None, :, None, :]  # (1, T, 1, head_dim)
            sin4 = sin_buf[:T][None, :, None, :]
            q = q * cos4 + _rotate_half_torch(q) * sin4
            k = k * cos4 + _rotate_half_torch(k) * sin4
            if capture is not None:
                capture.setdefault("post_rope", {})[index] = (q, k)
            q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.permute(0, 2, 1, 3).reshape(B, T, -1)
            x = residual + self.o_proj(out)
            residual = x
            h = self.mlp_norm(x)
            return residual + self.fc_out(F.gelu(self.fc_in(h)))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = dict(config)
            self.embed = nn.Embedding(config["vocab"], config["dim"])
            self.blocks = nn.ModuleList(Block(config) for _ in range(config["n_layers"]))
            self.final_norm = RMSNorm(config["dim"])
            self.base = ROPE_BASE
            positions = np.arange(config["ctx"], dtype=np.float64)
            # Registered as non-persistent buffers so ``.to(device)`` moves
            # the tables with the weights -- a caller doing
            # ``build_model(cfg).to("cuda")`` must not device-mismatch on
            # the first forward -- without inflating ``state_dict()``.
            cos, sin = _cos_sin_torch(positions, config["head_dim"], self.base, "cpu")
            self.register_buffer("cos_buf", cos, persistent=False)
            self.register_buffer("sin_buf", sin, persistent=False)

        def forward(self, ids, capture=None):
            x = self.embed(ids)
            for index, block in enumerate(self.blocks):
                x = block(x, self.cos_buf, self.sin_buf, capture, index)
            return torch.nn.functional.linear(self.final_norm(x), self.embed.weight)

    model = Model()
    # GPT-2-style initialization: small-variance normal weights, with the
    # output projection of each residual branch scaled by depth. The default
    # nn.Embedding normal(0, 1) init would place logits at O(d) scale, and
    # the loss would sit far above ln(vocab) for thousands of steps.
    with torch.no_grad():
        model.embed.weight.normal_(0.0, 0.02)
        scaled = 0.02 / math.sqrt(2.0 * max(1, config["n_layers"]))
        for block in model.blocks:
            for name in ("q_proj", "k_proj", "v_proj", "fc_in"):
                getattr(block, name).weight.normal_(0.0, 0.02)
            for name in ("o_proj", "fc_out"):
                getattr(block, name).weight.normal_(0.0, scaled)
    return model.to(device)


def _probe_diagnostics(model, ids, deltas, n_pairs: int, seed: int) -> dict:
    """The attribution diagnostics, in float64, on weights-as-they-are.

    Runs the model once with ``capture`` on, then:

    * ``closed_form_resid_max`` -- |sum_k closed-form pair contribution -
      brute-force relative score|, the exactness the paper's Table reports,
      measured on this model's own weights right now; machine-epsilon scale;
    * ``rope_probe_resid_max`` -- |torch rotation - rope.py rotation| after
      float64 casts, the implementation-fidelity floor;
    * ``sign_flip_rate`` / ``rel_err_median_probe`` -- the position-free cost
      statistic computed by the very ``measure_pair_cost`` the paper's
      per-cell statistic comes from. The logged rate is the mean over the
      sampled cells -- a micro-average over heads and pairs, not the
      paper's head-pooled rate -- and ``n_probe_cells`` is logged beside it.

    The probe is one fixed batch and the per-pair sampling is seeded, so the
    diagnostic series across a run's evaluations are comparable.
    """
    import torch

    from rope_attribution.experiments import pair_terms
    from rope_attribution.usefulness import measure_pair_cost

    rng = np.random.default_rng(seed)
    capture: dict = {}
    with torch.no_grad():
        model(torch.from_numpy(ids).to(next(model.parameters()).device), capture=capture)
    resid = 0.0
    rope_resid = 0.0
    n_cells = 0
    n_flips = 0
    rels = []
    T = int(ids.shape[1])
    positions = np.arange(T, dtype=np.float64)
    ref_cos, ref_sin = reference_cos_sin(positions, model.config["head_dim"])
    freqs = inv_freq(model.config["head_dim"])
    for index in sorted(capture["pre_rope"]):
        q_pre, k_pre = capture["pre_rope"][index]
        q_post, k_post = capture["post_rope"][index]
        q_np = _to_numpy_f64(q_pre)
        k_np = _to_numpy_f64(k_pre)
        ref_q = _apply_rope_ref_4d(q_np, ref_cos, ref_sin)
        ref_k = _apply_rope_ref_4d(k_np, ref_cos, ref_sin)
        rope_resid = max(
            rope_resid,
            float(np.max(np.abs(_to_numpy_f64(q_post) - ref_q))),
            float(np.max(np.abs(_to_numpy_f64(k_post) - ref_k))),
        )
        for head in range(q_np.shape[2]):
            for _pair in range(n_pairs):
                m = int(rng.integers(0, T - 1))
                n = (m + 1 + int(rng.integers(0, T - 1))) % T
                # n != m on every path: the offset (m + 1 + r) with r < T - 1
                # cannot reach m again modulo T, so no equality guard here --
                # an earlier revision carried one that could never fire. On a
                # wrap n lands below m, and the swap below restores the m < n
                # order the probe assumes.
                if n < m:
                    m, n = n, m
                qv, kv = q_np[0, m, head], k_np[0, n, head]
                cost = measure_pair_cost(qv, kv, freqs, deltas)
                n_cells += int(cost["n_material"].sum())
                n_flips += int(cost["sign_flip_fraction"].sum())
                live = cost["live"]
                if live.any():
                    rels.append(cost["rel_err"][live])
                # Brute force: rotate the key into the query's frame with the
                # same reference arithmetic at float64, then dot; the per-pair
                # closed form must reproduce the score to machine epsilon.
                # (delta > 0 by construction above.)
                delta = float(n - m)
                k_rel = kv * _cos_at(delta, freqs) + _rotate_half_np(kv) * _sin_at(
                    delta, freqs
                )
                score = float(qv @ k_rel)
                terms = pair_terms(qv, kv, freqs, int(delta))
                closed = float(
                    np.sum(
                        terms.aligned * np.cos(delta * freqs)
                        + terms.crossed * np.sin(delta * freqs)
                    )
                )
                resid = max(resid, abs(score - closed))
    rel = np.concatenate(rels) if rels else np.array([])
    return {
        "closed_form_resid_max": resid,
        "rope_probe_resid_max": rope_resid,
        "sign_flip_rate": n_flips / n_cells if n_cells else float("nan"),
        "rel_err_median_probe": float(np.median(rel)) if rel.size else float("nan"),
        "n_probe_cells": n_cells,
    }


def _cos_at(delta: float, freqs: np.ndarray) -> np.ndarray:
    """cos(delta*omega) duplicated to the split-half table of length head_dim."""
    half = np.cos(delta * freqs)
    return np.concatenate([half, half])


def _sin_at(delta: float, freqs: np.ndarray) -> np.ndarray:
    half = np.sin(delta * freqs)
    return np.concatenate([half, half])


def reference_cos_sin(positions: np.ndarray, head_dim: int) -> tuple:
    from rope_attribution.rope import rope_cos_sin

    return rope_cos_sin(positions, inv_freq(head_dim))


def _split_windows(tokens: np.ndarray, ctx: int) -> tuple:
    """Next-token windows: inputs, targets, and the held-out decile.

    Window ``k`` of a segment covers inputs ``s[k*ctx : (k+1)*ctx]`` and
    targets ``s[k*ctx+1 : (k+1)*ctx+1]``, so every target is the token that
    actually follows its input -- a ``torch.roll`` construction would point
    the window's last position at its own first token. The split is taken
    on the flat stream before windowing, so no target of a training window
    lies inside the held-out block (the final decile of the stream, which is
    the fixed validation set of the run).

    Pure numpy and torch-free so the suite checks the construction without
    a model. The dtype is preserved: a ``uint16`` memmap of token ids must not
    be widened to ``int64`` here, because that would materialize the whole
    corpus -- the training loop casts one batch at a time instead.
    """
    stream = np.asarray(tokens).reshape(-1)
    split = len(stream) - max(1, len(stream) // 10)

    def windows(segment: np.ndarray) -> tuple:
        n = (len(segment) - 1) // ctx
        if n <= 0:
            empty = segment[:0].reshape(0, ctx)
            return empty, empty
        return (
            segment[: n * ctx].reshape(n, ctx),
            segment[1 : n * ctx + 1].reshape(n, ctx),
        )

    train_in, train_tg = windows(stream[:split])
    held_in, held_tg = windows(stream[split:])
    return train_in, train_tg, held_in, held_tg


def _id_batch(rows: np.ndarray):
    """One batch of token ids as a contiguous ``int64`` tensor.

    The windows may be a view into a ``uint16`` memmap; casting the whole
    stream would defeat the memory map, so the cast happens per batch and
    nowhere else.
    """
    import torch

    return torch.from_numpy(np.ascontiguousarray(rows, dtype=np.int64))


def train(
    config_name: str,
    run_dir: Path,
    tokens: np.ndarray,
    device: str = "auto",
    max_steps: int | None = None,
    override: dict | None = None,
    force_cpu: bool = False,
    threads: int | None = None,
    data_seed: int | None = None,
    save_checkpoint: bool = True,
    world_size: int = 1,
    resume: bool = False,
) -> dict:
    """One training run. Returns the summary record it also writes.

    ``tokens`` is an int array of token ids (any shape; cut into next-token
    windows, with the final decile of the stream held out as the fixed
    validation block). The test suite passes synthetic ids directly, which
    keeps the whole data path out of the network. ``force_cpu`` is the
    deliberate override of the named-config CPU refusal; ``threads``
    overrides the default thread count (4 on CPU, 8 otherwise).

    ``world_size`` is the number of data-parallel ranks. It enters the token
    accounting, not the model: each rank consumes its own batch, so the run
    has seen ``batch_size * ctx * world_size`` tokens per step. Logging the
    per-rank count would under-report an eight-chip run by 8x.

    ``resume`` continues from ``run_dir/checkpoint_resume.pt``, restoring the
    model, the AdamW moments and the data-order generator. Resuming without
    those is a different run, so the file carries all three and the config is
    checked against the one that wrote it.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed; see train_requirements.txt")
    import torch
    import torch.nn.functional as F

    if int(world_size) < 1:
        raise ValueError(f"world_size must be at least 1, not {world_size!r}")
    world_size = int(world_size)
    run_dir = Path(run_dir)
    config = dict(CONFIGS[config_name])
    config.update(override or {})
    if max_steps is not None:
        config["n_steps"] = int(max_steps)
    params = count_params(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    xm = _require_xla() if device == "xla" else None
    dtype = str(config.get("dtype", "fp32"))
    # Validate before building the model or creating the writer: a bad dtype
    # must not leave a half-initialised run directory behind.
    _amp_dtype(device, dtype)
    refusal = _cpu_refusal(config_name, params, device, force_cpu)
    if refusal:
        raise RuntimeError(refusal)
    _set_determinism(config["seed"])
    if xm is None:
        torch.set_num_threads(_resolve_threads(device, threads))
    model = build_model(config, device=device)
    decayed, decayless = [], []
    for name, parameter in model.named_parameters():
        (decayed if any(k in name for k in _WD_KEYS) else decayless).append(parameter)
    opt = torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": 0.1},
            {"params": decayless, "weight_decay": 0.0},
        ],
        lr=config["lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    rng = np.random.default_rng(config["seed"] if data_seed is None else data_seed)
    ctx = config["ctx"]
    train_in, train_tg, held_in, held_tg = _split_windows(tokens, ctx)
    if len(train_in) < config["batch_size"] or len(held_in) < 1:
        raise ValueError(
            f"{int(np.asarray(tokens).size)} tokens make {len(train_in)} training windows of "
            f"{ctx}; not enough for a batch of {config['batch_size']} plus "
            "a validation block"
        )
    tokens_total = int(np.asarray(tokens).size)
    resume_name = "checkpoint_resume.pt"
    resumed_from: int | None = None
    if resume:
        state = _load_training_state(run_dir / resume_name)
        saved = dict(state["config"])
        differing = sorted(
            key
            for key in (set(saved) | set(config)) - {"n_steps"}
            if saved.get(key) != config.get(key)
        )
        if differing:
            raise ValueError(
                "cannot resume: the saved config differs from this call in "
                f"{differing}; a resumed run must use the configuration that "
                "wrote the checkpoint"
            )
        if int(state["n_params"]) != params:
            raise ValueError(
                f"cannot resume: saved n_params {state['n_params']} != {params}"
            )
        if int(state["tokens_total"]) != tokens_total:
            raise ValueError(
                f"cannot resume: the saved run used {state['tokens_total']} tokens, "
                f"this call passes {tokens_total}; the data stream must be the "
                "same one the checkpoint was trained on"
            )
        if int(state["world_size"]) != world_size:
            raise ValueError(
                f"cannot resume: the saved run used world_size "
                f"{state['world_size']}, this call passes {world_size}"
            )
        saved_data_seed = state.get("data_seed")
        if data_seed is None:
            data_seed = saved_data_seed
        elif saved_data_seed is None or int(data_seed) != int(saved_data_seed):
            raise ValueError(
                f"cannot resume with data_seed {data_seed}; the saved run used "
                f"{saved_data_seed}"
            )
        if int(config["n_steps"]) < int(state["step"]):
            raise ValueError(
                f"cannot resume: the checkpoint is at step {state['step']}, "
                f"beyond this call's n_steps={config['n_steps']}"
            )
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        step = int(state["step"])
        resumed_from = step
        loss_ema = state["loss_ema"]
        best_val_loss = state["best_val_loss"]
        best_step = state["best_step"]
        final_val_loss = state["final_val_loss"]
        tokens_seen = int(state["tokens_seen"])
        # Throughput is a property of the whole run, not of the segment since
        # the last restart, so the clock starts where the saved run left it.
        start = time.perf_counter() - float(state["elapsed_s"])
        # The writer is created only after every refusal check: a refused run
        # must not have wiped the previous run's records. A resumed run
        # continues the series instead of starting a second one.
        writer = MetricsWriter(run_dir, append=True)
        manifest = _write_manifest(
            run_dir,
            config_name=config_name,
            config=config,
            n_params=params,
            device=device,
            dtype=dtype,
            world_size=world_size,
            tokens_total=tokens_total,
            data_seed=data_seed,
            resumed_from_step=resumed_from,
        )
    else:
        writer = MetricsWriter(run_dir)
        # A fresh run must not inherit a checkpoint from an earlier run of
        # the same name: a stale file would be mistaken for this run's
        # weights, and with --no-checkpoint it would be the only file left.
        for stale in sorted(run_dir.glob("checkpoint*.pt")):
            stale.unlink()
        for stale in sorted(run_dir.glob("manifest*.json")):
            stale.unlink()
        step = 0
        loss_ema = None
        best_val_loss = None
        best_step = None
        final_val_loss = None
        tokens_seen = 0
        start = time.perf_counter()
        manifest = _write_manifest(
            run_dir,
            config_name=config_name,
            config=config,
            n_params=params,
            device=device,
            dtype=dtype,
            world_size=world_size,
            tokens_total=tokens_total,
            data_seed=data_seed,
        )
    writer.log(
        {
            "event": "run_start",
            "config_name": config_name,
            "config": config,
            "param_formula": _PARAM_FORMULA_DOC,
            "n_params": params,
            "device": device,
            "dtype": dtype,
            "tokens_total": tokens_total,
            "world_size": world_size,
            "token_accounting": "global: batch tokens summed over all ranks",
            "resumed_from_step": resumed_from,
            "manifest": manifest,
            "seed": config["seed"],
            "data_seed": None if data_seed is None else int(data_seed),
        }
    )
    deltas = np.array(DISTANCE_GRID, dtype=np.int64)
    probe_seed = config["seed"]
    checkpoint_every = config.get("checkpoint_every")
    val_batches = int(config.get("val_batches", 1))
    if val_batches < 1:
        raise ValueError(f"val_batches must be at least 1, not {val_batches}")
    resume_checkpoint_name = None
    while step < config["n_steps"]:
        picks = rng.integers(0, len(train_in), size=config["batch_size"])
        ids = _id_batch(train_in[picks]).to(device)
        targets = _id_batch(train_tg[picks]).to(device)
        # Forward in bf16 when requested; the cross-entropy and the master
        # weights stay fp32. A bf16 label in the metrics therefore describes
        # the arithmetic that actually ran.
        with _autocast_context(device, dtype):
            logits = model(ids)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # The update that follows is the (step + 1)-th, so the schedule is
        # read at step + 1: without this the first update would run at
        # lr_at(0) == 0 and waste a step of the budget.
        current_lr = lr_at(step + 1, config)
        for group in opt.param_groups:
            group["lr"] = current_lr
        # Data-parallel replicas: reduce once, then clip the reduced gradient,
        # then step once. Every rank initialises from the same config["seed"];
        # only data_seed differs, so the replicas remain one synchronised
        # model rather than eight independent runs.
        grad_norm = _reduce_update(model, opt, xm, config["grad_clip"])
        loss_value = float(loss.item())
        loss_ema = loss_value if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_value
        # Global tokens: every rank consumed its own batch this step.
        tokens_seen += int(ids.numel()) * world_size
        step += 1
        if step % config["eval_every"] == 0 or step == config["n_steps"]:
            # Several consecutive validation batches, not one: a single batch
            # of 32 sequences is a noisy estimate and the noise does not
            # shrink with training. The average below is target-weighted, so
            # a shorter final batch cannot tilt it.
            n_val = min(len(held_in), config["batch_size"] * val_batches)
            val_loss_sum = 0.0
            val_target_count = 0
            val_batch_count = 0
            with torch.no_grad(), _autocast_context(device, dtype):
                for offset in range(0, n_val, config["batch_size"]):
                    batch = slice(offset, offset + config["batch_size"])
                    val_ids = _id_batch(held_in[batch]).to(device)
                    val_targets = _id_batch(held_tg[batch]).to(device)
                    val_logits = model(val_ids)
                    val_loss_sum += float(
                        F.cross_entropy(
                            val_logits.float().reshape(-1, val_logits.shape[-1]),
                            val_targets.reshape(-1),
                            reduction="sum",
                        ).item()
                    )
                    val_target_count += int(val_targets.numel())
                    val_batch_count += 1
            val_loss = val_loss_sum / max(1, val_target_count)
            final_val_loss = float(val_loss)
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_step = step
            probe = train_in[0][None, :]
            # The probe inherits the run's arithmetic. Measuring a bf16 run's
            # rotation in fp32 would report a fidelity floor the trained model
            # does not have; the probe is supposed to describe the run.
            with _autocast_context(device, dtype):
                diagnostics = _probe_diagnostics(
                    model, probe, deltas, n_pairs=12, seed=probe_seed
                )
            writer.log(
                {
                    "event": "eval",
                    "step": step,
                    "lr": current_lr,
                    "grad_norm": float(grad_norm),
                    "train_loss": loss_value,
                    "train_loss_ema": loss_ema,
                    "val_loss": val_loss,
                    "val_batches": val_batch_count,
                    "val_targets": val_target_count,
                    # Kept as a convenience field for a reader who thinks in
                    # perplexity; it is exp(val_loss), not a separate model.
                    "val_perplexity": (
                        float(math.exp(min(50.0, val_loss)))
                        if math.isfinite(val_loss)
                        else None
                    ),
                    "tokens_seen": tokens_seen,
                    "tok_per_s": tokens_seen / max(1e-9, time.perf_counter() - start),
                    "probe_dtype": dtype,
                    **diagnostics,
                }
            )
        if (
            save_checkpoint
            and checkpoint_every
            and step % int(checkpoint_every) == 0
            and step < config["n_steps"]
        ):
            # One rolling resume file, replaced atomically: a crash costs the
            # steps since the last save, never the whole run, and the disk
            # does not grow with the number of saves.
            resume_checkpoint_name = _save_training_state(
                model,
                opt,
                rng,
                run_dir / resume_name,
                config=config,
                step=step,
                n_params=params,
                world_size=world_size,
                loss_ema=loss_ema,
                best_val_loss=best_val_loss,
                best_step=best_step,
                final_val_loss=final_val_loss,
                tokens_seen=tokens_seen,
                elapsed_s=time.perf_counter() - start,
                data_seed=data_seed,
                tokens_total=tokens_total,
            )
    checkpoint_name = None
    if save_checkpoint:
        checkpoint_name = _save_checkpoint(
            model,
            config,
            run_dir / "checkpoint.pt",
            step=step,
            n_params=params,
        )
        # The final state is resumable too: continuing a finished run with a
        # larger n_steps is a legitimate way to extend a budget.
        resume_checkpoint_name = _save_training_state(
            model,
            opt,
            rng,
            run_dir / resume_name,
            config=config,
            step=step,
            n_params=params,
            world_size=world_size,
            loss_ema=loss_ema,
            best_val_loss=best_val_loss,
            best_step=best_step,
            final_val_loss=final_val_loss,
            tokens_seen=tokens_seen,
            elapsed_s=time.perf_counter() - start,
            data_seed=data_seed,
            tokens_total=tokens_total,
        )
    summary = {
        "event": "summary",
        "config_name": config_name,
        "n_params": params,
        "steps": step,
        "resumed_from_step": resumed_from,
        "final_train_loss_ema": loss_ema,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "checkpoint": checkpoint_name,
        "resume_checkpoint": resume_checkpoint_name,
        "tokens_seen": tokens_seen,
        "tokens_total": tokens_total,
        "world_size": world_size,
        "device": device,
        "manifest": manifest,
    }
    writer.log(summary)
    writer.finalize(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "xla"), default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="tokenized stream from `python -m ...data prepare`; mmapped once per rank",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="do not write run_dir/checkpoint.pt at the end of the run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue run_dir/checkpoint_resume.pt instead of starting fresh",
    )
    args = parser.parse_args()
    if not TORCH_AVAILABLE:
        raise SystemExit("torch is not installed; see train_requirements.txt")
    if args.device in ("auto", "cpu"):
        device = args.device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        params = count_params(CONFIGS[args.config])
        refusal = _cpu_refusal(args.config, params, device, args.force_cpu)
        if refusal:
            raise SystemExit(refusal)
    run_dir = Path("runs") / (args.run_name or args.config)
    if args.data_dir is not None:
        from rope_attribution.data import load_tokens

        tokens = load_tokens(args.data_dir)
    elif args.config == "smoke":
        tokens = np.random.default_rng(7).integers(0, 512, size=4096)
    else:
        if not TRANSFORMERS_AVAILABLE:
            raise SystemExit(
                "transformers and datasets are not installed; the wikitext "
                "path needs train_requirements.txt"
            )
        tokens = _load_wikitext_tokens()
    summary = train(
        args.config,
        run_dir,
        tokens,
        device=args.device,
        max_steps=args.max_steps,
        force_cpu=args.force_cpu,
        threads=args.threads,
        save_checkpoint=not args.no_checkpoint,
        resume=args.resume,
    )
    print("wrote", run_dir, json.dumps(summary))


def _load_wikitext_tokens() -> np.ndarray:
    """The convenience corpus: wikitext-103 raw, GPT-2 tokenized, deterministic.

    This path exists for a single-process CPU run. It tokenizes the whole
    corpus in memory, so eight TPU ranks calling it would each pay that cost;
    a real multi-chip run should prepare the stream once with
    ``python -m projects.rope_attribution.data`` and pass ``--data-dir``.
    Requires train_requirements.txt; the smoke config needs no network at
    all. Streamed through ``np.fromiter`` so the ~10^8-token corpus is never
    materialized as a Python int list, which would cost gigabytes before
    training starts.
    """
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    try:
        from datasets import load_dataset

        split = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        stream = itertools.chain.from_iterable(
            tokenizer.encode(example["text"]) for example in split
        )
        tokens = np.fromiter(stream, dtype=np.int64)
    except Exception as exc:  # pragma: no cover - network/cache path
        raise SystemExit(
            f"could not load wikitext-103 ({exc}); download it first or pass "
            "your own token stream to train(..., tokens=...)"
        ) from exc
    return tokens


if __name__ == "__main__":
    main()
