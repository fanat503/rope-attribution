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
is wall-clock by definition. CUDA kernels are pinned to deterministic
algorithms with ``warn_only`` -- the flag records any kernel
that cannot comply rather than aborting mid-run.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
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
        "warmup_steps": 4,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
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
        "warmup_steps": 2_000,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
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
        "warmup_steps": 2_000,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
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
        "warmup_steps": 2_000,
        "lr_floor_fraction": 0.1,
        "grad_clip": 1.0,
        "seed": 20261010,
    },
}

#: The named configurations must land inside these bands; a config that edits
#: its widths out of its band fails the suite instead of shipping.
TRAIN_PARAM_BANDS = {
    "train_124m": (100_000_000, 130_000_000),
    "train_208m": (190_000_000, 220_000_000),
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

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
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
        # line, not the history.
        line = json.dumps(record)
        self._lines.append(line)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def finalize(self, summary: dict) -> None:
        self.summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    @property
    def records(self) -> list:
        return [json.loads(line) for line in self._lines]


def _set_determinism(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    * ``sign_flip_rate`` / ``rel_err_median`` -- the position-free cost
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
        q_np = q_pre.to(torch.float64).cpu().numpy()
        k_np = k_pre.to(torch.float64).cpu().numpy()
        ref_q = _apply_rope_ref_4d(q_np, ref_cos, ref_sin)
        ref_k = _apply_rope_ref_4d(k_np, ref_cos, ref_sin)
        rope_resid = max(
            rope_resid,
            float(np.max(np.abs(q_post.to(torch.float64).cpu().numpy() - ref_q))),
            float(np.max(np.abs(k_post.to(torch.float64).cpu().numpy() - ref_k))),
        )
        for head in range(q_np.shape[2]):
            for _pair in range(n_pairs):
                m = int(rng.integers(0, T - 1))
                n = (m + 1 + int(rng.integers(0, T - 1))) % T
                if n == m:
                    n = (m + 1) % T
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
    a model.
    """
    stream = np.asarray(tokens, dtype=np.int64).reshape(-1)
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


def train(
    config_name: str,
    run_dir: Path,
    tokens: np.ndarray,
    device: str = "auto",
    max_steps: int | None = None,
    override: dict | None = None,
    force_cpu: bool = False,
    threads: int | None = None,
) -> dict:
    """One training run. Returns the summary record it also writes.

    ``tokens`` is an int array of token ids (any shape; cut into next-token
    windows, with the final decile of the stream held out as the fixed
    validation block). The test suite passes synthetic ids directly, which
    keeps the whole data path out of the network. ``force_cpu`` is the
    deliberate override of the named-config CPU refusal; ``threads``
    overrides the default thread count (4 on CPU, 8 otherwise).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed; see train_requirements.txt")
    import torch
    import torch.nn.functional as F

    config = dict(CONFIGS[config_name])
    config.update(override or {})
    if max_steps is not None:
        config["n_steps"] = int(max_steps)
    params = count_params(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    refusal = _cpu_refusal(config_name, params, device, force_cpu)
    if refusal:
        raise RuntimeError(refusal)
    _set_determinism(config["seed"])
    torch.set_num_threads(
        threads if threads is not None else (4 if device == "cpu" else 8)
    )
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
    rng = np.random.default_rng(config["seed"])
    ctx = config["ctx"]
    train_in, train_tg, held_in, held_tg = _split_windows(tokens, ctx)
    if len(train_in) < config["batch_size"] or len(held_in) < 1:
        raise ValueError(
            f"{len(tokens)} tokens make {len(train_in)} training windows of "
            f"{ctx}; not enough for a batch of {config['batch_size']} plus "
            "a validation block"
        )
    # The writer is created only after every refusal check: a refused run
    # must not have wiped the previous run's records.
    writer = MetricsWriter(run_dir)
    step = 0
    loss_ema: float | None = None
    start = time.perf_counter()
    tokens_seen = 0
    writer.log(
        {
            "event": "run_start",
            "config_name": config_name,
            "config": config,
            "param_formula": _PARAM_FORMULA_DOC,
            "n_params": params,
            "device": device,
            "tokens_total": int(len(tokens)),
            "seed": config["seed"],
        }
    )
    deltas = np.array(DISTANCE_GRID, dtype=np.int64)
    probe_seed = config["seed"]
    while step < config["n_steps"]:
        picks = rng.integers(0, len(train_in), size=config["batch_size"])
        ids = torch.from_numpy(train_in[picks]).to(device)
        targets = torch.from_numpy(train_tg[picks]).to(device)
        logits = model(ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["grad_clip"]
        )
        # The update that follows is the (step + 1)-th, so the schedule is
        # read at step + 1: without this the first update would run at
        # lr_at(0) == 0 and waste a step of the budget.
        current_lr = lr_at(step + 1, config)
        for group in opt.param_groups:
            group["lr"] = current_lr
        opt.step()
        loss_value = float(loss.item())
        loss_ema = loss_value if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_value
        tokens_seen += int(ids.numel())
        step += 1
        if step % config["eval_every"] == 0 or step == config["n_steps"]:
            val_ids = torch.from_numpy(held_in[: config["batch_size"]]).to(device)
            val_targets = torch.from_numpy(held_tg[: config["batch_size"]]).to(device)
            with torch.no_grad():
                val_logits = model(val_ids)
                val_loss = F.cross_entropy(
                    val_logits.reshape(-1, val_logits.shape[-1]).float(),
                    val_targets.reshape(-1),
                ).item()
            probe = train_in[0][None, :]
            diagnostics = _probe_diagnostics(
                model, probe, deltas, n_pairs=12, seed=probe_seed
            )
            writer.log(
                {
                    "event": "eval",
                    "step": step,
                    "lr": current_lr,
                    "grad_norm": float(grad_norm),
                    "train_loss_ema": loss_ema,
                    "val_loss": val_loss,
                    "tokens_seen": tokens_seen,
                    "tok_per_s": tokens_seen / max(1e-9, time.perf_counter() - start),
                    **diagnostics,
                }
            )
    summary = {
        "event": "summary",
        "config_name": config_name,
        "n_params": params,
        "steps": step,
        "final_train_loss_ema": loss_ema,
        "device": device,
    }
    writer.log(summary)
    writer.finalize(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
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
    if args.config == "smoke":
        tokens = np.random.default_rng(7).integers(0, 512, size=4096)
    else:
        if not TRANSFORMERS_AVAILABLE:
            raise SystemExit(
                "transformers and datasets are not installed; the wikitext "
                "path needs train_requirements.txt"
            )
        tokens = _load_wikitext_tokens()
    summary = train(args.config, run_dir, tokens, device=args.device,
                    max_steps=args.max_steps, force_cpu=args.force_cpu,
                    threads=args.threads)
    print("wrote", run_dir, json.dumps(summary))


def _load_wikitext_tokens() -> np.ndarray:
    """The default corpus: wikitext-103 raw, GPT-2 tokenized, deterministic.

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
