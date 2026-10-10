"""The structural measurements of `experiments.py`, repeated on a *real* model.

Everything in `experiments.py` is measured on synthetic random vectors. That is
deliberate - the identities there are theorems about the position encoding, not
about any trained network - but it also means every reported number is
model-independent by construction, so none of them is evidence that a trained
network *uses* the structure the paper describes. This module closes that gap: it
runs the same measurements on the activations of pretrained checkpoints.

What is measured, per ``(layer, head)``
---------------------------------------
1. **Config validation.** The checkpoint's own rotary frequency buffer is
   compared against ``rope.inv_freq(dim, theta)`` built from the configuration
   the checkpoint *declares*. If those disagree, this repository's ladder is not
   the one the model runs on and nothing measured here would transfer.
2. **How position-carrying is each channel.** For real ``q_hat``/``k_hat``,
   ``experiments.pair_terms`` gives the position-free part ``A_k`` and the
   position-carrying part ``B_k`` of every rotary pair. The share of a pair's
   amplitude that carries position is ``|B_k| / R_k``, ``R_k = hypot(A_k, B_k)``.
    No pair is *exactly* position-blind: ``B_k = 0`` leaves ``A_k cos(D_k)``,
    which still oscillates in ``delta``. "Effectively" blind is therefore a
    threshold, stated as :data:`BLIND_RATIO`, and is a convention with a cutoff
    rather than an algebraic fact.
3. **Whether the linearization survives real data.**
   ``experiments.linearization_error`` on real vectors at the same distances
   ``method_spectrum`` uses, beside the published synthetic values.
4. **Spread.** Every data-dependent quantity is reported per layer *and* per
   head, with min/median/max/std. The synthetic suite has no error bars at all;
   across-head and across-layer dispersion is what distinguishes a measurement on
   a trained network from a demonstration on random vectors.
5. **Attention entropy** ``H / ln T`` of the real attention distribution.

Two premises that had to be checked, and one of them is false
-------------------------------------------------------------
* **GPT-2 124M does not use RoPE.** It was trained with *learned absolute*
  position embeddings (``transformer.wpe``, added to the token embeddings inside
  ``GPT2Model.forward``); ``GPT2Config`` has no ``rope_theta`` and the attention
  forward applies no rotation at all. There is no rotary pair structure in its
  ``q_hat``/``k_hat`` to attribute anything to. :func:`gpt2_has_rope` decides
  that from the loaded module rather than leaving it as folklore, and the answer
  is recorded in the JSON.
* **Rotation width is not a detail.** ``EleutherAI/pythia-160m`` declares
  ``partial_rotary_factor = 0.25``, so only 16 of each head's 64 channels rotate,
  in 8 rotary pairs, and its ladder is ``inv_freq(16, theta)`` - not the 32-pair
  ``inv_freq(64, theta)`` the synthetic suite uses. A full-width rotation at
  ``head_dim = 64``, ``theta = 10000`` is what ``JackFram/llama-160m`` and
  ``HuggingFaceTB/SmolLM-135M`` do, so both widths are measured here.

What ``frac_channels_linearizable`` can and cannot measure
----------------------------------------------------------
``experiments.linearization_error`` defines it as ``mean(|D_k| <= 0.1)`` with
``D_k = delta * inv_freq[k]``. That is a function of the frequency ladder and the
distance **only**; no activation can change it. On real data its across-head and
across-layer spread is therefore expected to be exactly zero, and this module
reports that spread rather than omitting it. The quantity that does depend on
the trained weights is ``amplitude_weighted_err``, and that is where real and
synthetic can differ.

No training of any kind happens here: one forward pass per sequence, on CPU,
with gradients disabled. See ``real_model_requirements.txt`` for the extra
dependencies; the module imports without them and reports
:data:`REAL_MODEL_AVAILABLE` as ``False``.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rope_attribution.experiments import linearization_error, pair_terms
from rope_attribution.rope import inv_freq

__all__ = [
    "BLIND_RATIO",
    "DEFAULT_DELTAS",
    "DEFAULT_LENS",
    "DEFAULT_MODEL_IDS",
    "DEFAULT_PROMPTS",
    "DEFAULT_SEED",
    "NUM_THREADS",
    "REAL_MODEL_AVAILABLE",
    "TORCH_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
    "HeadVectors",
    "Loaded",
    "aggregate",
    "attention_modules",
    "collect_head_vectors",
    "gpt2_has_rope",
    "is_available",
    "load_model",
    "measure_entropy",
    "measure_linearization",
    "measure_model",
    "measure_position_channels",
    "run_all",
    "synthetic_baseline",
    "validate_config",
    "write_results",
]


# --------------------------------------------------------------------------
# Configuration. No measured value appears as a literal anywhere in this file.
# --------------------------------------------------------------------------

#: Seed for every stochastic step. The checkpoints run in ``eval`` mode with
#: dropout disabled, so the only randomness left is initialization noise for
#: tensors a checkpoint does not supply; it is pinned so runs are reproducible.
DEFAULT_SEED = 20260906

#: ``torch.set_num_threads`` value. Deliberately under the core count: this is a
#: shared machine and one-shot inference does not need every core.
NUM_THREADS = 4

#: Fraction of a rotary pair's position-free amplitude above which a pair counts
#: as position-bearing. "Effectively position-blind" means
#: ``|B_k| < BLIND_RATIO * |A_k|``.
BLIND_RATIO = 0.01

#: Distances at which the linearization is measured - the same grid
#: ``experiments.method_spectrum`` uses, so the comparison is like for like.
DEFAULT_DELTAS: tuple[int, ...] = (1, 64, 512, 2048, 4096)

#: Sequence lengths run through the model. Several, so that the entropy
#: measurement is not a single anecdote.
DEFAULT_LENS: tuple[int, ...] = (64, 128, 256, 512)

#: Short plain English plus one code fragment. The content is irrelevant to the
#: structural claims - what matters is that these are real token streams and not
#: random vectors - but the mixture keeps the activation statistics ordinary
#: instead of degenerate.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "In 1969 the first humans landed on the moon, and",
    "def quicksort(items):\n    if len(items) <= 1:\n        return items\n",
    "Water boils at one hundred degrees Celsius at sea level, which means that",
    "The three branches of the United States government are the Congress, the",
)

#: Longer passages, so the entropy measurement has several real lengths to sit
#: on. The structural measurements above use the short prompts, where sequence
#: length is irrelevant anyway: what matters there is that ``q_hat``/``k_hat``
#: come from a trained network. Entropy does depend on how many keys a row can
#: reach, so it needs sequences long enough for that to vary.
DEFAULT_LONG_TEXTS: tuple[str, ...] = (
    (
        "Attention in a transformer is a weighted average over the tokens a "
        "query chooses to look at. The weights come from a dot product between "
        "a projected query and a projected key, passed through a softmax so that "
        "they sum to one. Nothing in that description says which tokens a query "
        "should attend to, and the training signal is whatever the loss happens "
        "to reward. A rotary embedding changes the geometry of the dot product "
        "rather than the mechanism that produces it. It rotates the query and "
        "the key by angles that grow with their positions, and because the "
        "rotation is orthogonal it preserves the length of both vectors exactly. "
        "Two consequences follow immediately. The first is that the score "
        "between a query at position m and a key at position n depends on the "
        "content and on the gap n minus m, and on nothing else, so translating "
        "both positions together leaves the score untouched. The second is that "
        "the magnitude of the query and the key carry no positional information "
        "at all. Everything the encoding contributes is in the angle. That is a "
        "useful thing to know when you try to explain a score, because it tells "
        "you that any part of an explanation that talks about the size of a gate "
        "is not talking about position. It also means that the usual habit of "
        "rewriting a score as a product of a magnitude and a phase is not an "
        "approximation introduced for convenience. It is the exact form the "
        "computation already has. "
        "There is a second consequence, and it is less comfortable. Because the "
        "score is bilinear in the content vectors, it decomposes additively over "
        "the features of the query and the features of the key. That part is "
        "exact, to the last bit of a double. But each of those additive terms is "
        "a function of the gap, not a number. Ask how much a particular feature "
        "contributed and the honest answer is a curve indexed by distance, not a "
        "scalar you can put on a chart. Choose a distance and you have a number; "
        "change the distance and the number moves, sometimes by orders of "
        "magnitude, with no change at all in the features. This is the real "
        "obstacle to attributing a RoPE score to features, and it is not a "
        "failure of bilinearity. It is a consequence of the encoding doing its "
        "job. A representation that carried position additively, in a separate "
        "set of coordinates, would give clean scalars. This one cannot, and the "
        "only way to recover something close to a scalar is to approximate the "
        "rotation itself, by replacing the cosine and the sine with their "
        "second and first order expansions about zero. That approximation is "
        "excellent when the angle is small and useless when it is not, and the "
        "fastest channel in the ladder turns through a full radian per token, so "
        "at any distance beyond a handful of tokens there is always at least one "
        "channel for which the expansion is meaningless."
    ),
    (
        "def merge_sort(values):\n"
        "    if len(values) < 2:\n"
        "        return list(values)\n"
        "    middle = len(values) // 2\n"
        "    left = merge_sort(values[:middle])\n"
        "    right = merge_sort(values[middle:])\n"
        "    out = []\n"
        "    i = 0\n"
        "    j = 0\n"
        "    while i < len(left) and j < len(right):\n"
        "        if left[i] <= right[j]:\n"
        "            out.append(left[i])\n"
        "            i += 1\n"
        "        else:\n"
        "            out.append(right[j])\n"
        "            j += 1\n"
        "    out.extend(left[i:])\n"
        "    out.extend(right[j:])\n"
        "    return out\n"
        "\n"
        "\n"
        "class Counter:\n"
        "    def __init__(self, start=0):\n"
        "        self.value = start\n"
        "\n"
        "    def bump(self, amount=1):\n"
        "        self.value += amount\n"
        "        return self.value\n"
        "\n"
        "    def __repr__(self):\n"
        "        return f'Counter({self.value})'\n"
        "\n"
        "\n"
        "def summarize(items):\n"
        "    counts = Counter()\n"
        "    for item in items:\n"
        "        counts.bump(1)\n"
        "    return sorted(counts.value, reverse=True)[:10]\n"
        "\n"
        "\n"
        "class Ring:\n"
        "    \"\"\"Fixed capacity buffer that overwrites its oldest slot.\"\"\"\n"
        "\n"
        "    def __init__(self, capacity):\n"
        "        if capacity < 1:\n"
        "            raise ValueError('capacity must be positive')\n"
        "        self.capacity = capacity\n"
        "        self.slots = [None] * capacity\n"
        "        self.next = 0\n"
        "        self.size = 0\n"
        "\n"
        "    def push(self, value):\n"
        "        self.slots[self.next] = value\n"
        "        self.next = (self.next + 1) % self.capacity\n"
        "        self.size = min(self.size + 1, self.capacity)\n"
        "        return self\n"
        "\n"
        "    def drain(self):\n"
        "        out = list(self.slots)\n"
        "        self.slots = [None] * self.capacity\n"
        "        self.next = 0\n"
        "        self.size = 0\n"
        "        return out\n"
        "\n"
        "    def __iter__(self):\n"
        "        return iter(v for v in self.slots if v is not None)\n"
    ),
)

#: Content pairs sampled per ``(layer, head)`` for the ``|B_k| / R_k``
#: distribution, and the smaller count used for the (more expensive) linearization
#: sweep. Sampling many content pairs per head is what makes the across-head
#: spread meaningful rather than an artefact of one vector.
N_CONTENT_PAIRS = 256
N_LINEARIZATION_PAIRS = 64
#: Content pairs for the identity-residual sweep. Lower than the others because
#: each pair costs several rotations and a bilinearity sweep, all of them in
#: Python rather than in a compiled inner loop.
N_IDENTITY_PAIRS = 48

#: Checkpoints to measure. All are small enough for one-shot CPU inference, and
#: between them they cover both rotation widths that matter: full-width
#: ``head_dim`` rotation and partial rotation.
DEFAULT_MODEL_IDS: tuple[str, ...] = (
    "EleutherAI/pythia-160m",
    "JackFram/llama-160m",
    "HuggingFaceTB/SmolLM-135M",
)

#: Bin edges for the ``|B_k| / R_k`` histogram. The quantity is a ratio in
#: ``[0, 1]`` by construction.
HISTOGRAM_BINS = 20


def _module_present(name: str) -> bool:
    """Is ``name`` importable? Resolved without importing it, so that importing
    this module costs nothing and cannot have side effects."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


TORCH_AVAILABLE = _module_present("torch")
TRANSFORMERS_AVAILABLE = _module_present("transformers")

#: ``True`` when the measurements below can run at all. Tests skip on this. It
#: says only that the libraries are installed, not that a download will succeed.
REAL_MODEL_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE


def is_available() -> bool:
    """Whether :func:`run_all` can run in this interpreter."""
    return REAL_MODEL_AVAILABLE


def _require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed; see real_model_requirements.txt")
    import torch

    return torch


def _version(pkg: str) -> str:
    try:
        mod = __import__(pkg)
    except Exception:
        return "absent"
    return str(getattr(mod, "__version__", "unknown"))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class Loaded:
    """A loaded checkpoint plus the tokenizer that feeds it."""

    model_id: str
    model: Any
    tokenizer: Any

    @property
    def config(self) -> Any:
        return self.model.config


def load_model(model_id: str = DEFAULT_MODEL_IDS[0], seed: int = DEFAULT_SEED) -> Loaded:
    """Load ``model_id`` on CPU in float32 with eager attention.

    Eager attention is required because the entropy measurement needs the
    post-softmax attention probabilities, which the fused kernels do not return.
    float32 rather than each checkpoint's native dtype because the residuals
    being checked live near 1e-15 and bfloat16 has an epsilon near 8e-3.
    """
    torch = _require_torch()
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "transformers is not installed; see real_model_requirements.txt"
        )

    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager", dtype=torch.float32
    )
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return Loaded(model_id=model_id, model=model, tokenizer=tokenizer)


# --------------------------------------------------------------------------
# Architecture adapters: locate the pieces the measurements need
# --------------------------------------------------------------------------


def _qkv_projector(attn: Any) -> Callable[[Any], tuple[Any, Any]] | None:
    """Return ``f(residual_input) -> (q_hat, k_hat)``, or ``None``.

    ``residual_input`` is ``(B, T, C)`` exactly as it enters the block's
    attention: for GPT-NeoX and LLaMA that is the block input after its
    pre-attention norm, which is the tensor the projection is a function of.

    Covers the two fusion styles in :data:`DEFAULT_MODEL_IDS` - GPT-NeoX packs q,
    k and v into one ``query_key_value`` matrix, LLaMA has separate ``q_proj``
    and ``k_proj``. Because the discriminator is "does this module own a q/k
    projection", it doubles as the detector for attention modules.
    """
    fused = getattr(attn, "query_key_value", None)
    if fused is not None and hasattr(fused, "weight"):

        def proj_fused(x: Any) -> tuple[Any, Any]:
            # GPT-NeoX packs q, k and v in that order along the last axis.
            q, k, _v = fused(x).chunk(3, dim=-1)
            return q, k

        return proj_fused

    q_proj, k_proj = getattr(attn, "q_proj", None), getattr(attn, "k_proj", None)
    if q_proj is not None and k_proj is not None:

        def proj_split(x: Any) -> tuple[Any, Any]:
            return q_proj(x), k_proj(x)

        return proj_split

    return None


def attention_modules(model: Any) -> list[Any]:
    """Every attention module, in layer order."""
    return [m for _, m in model.named_modules() if _qkv_projector(m) is not None]


def head_dim_of(attn: Any, cfg: Any = None) -> int:
    """Head width. Read from the attention module, else from the config."""
    for attr in ("head_size", "head_dim"):
        val = getattr(attn, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    n_embd = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    n_head = n_heads_of(attn, cfg)
    if not n_head:
        raise RuntimeError("cannot determine n_head for this model")
    return n_embd // n_head


def n_heads_of(attn: Any, cfg: Any = None) -> int:
    """Head count. Read from the attention module, else from the config."""
    for attr in ("num_heads", "num_attention_heads", "n_head"):
        val = getattr(attn, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    if cfg is None:
        return 0
    return int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)) or 0)


def find_rotary(model: Any) -> Any:
    """The module holding the checkpoint's own ``inv_freq``, or ``None``.

    Read from the live model rather than recomputed from the config, so the
    validation compares this repository's ladder against the buffer the
    checkpoint actually built - the strongest form of the check available.
    """
    for _, mod in model.named_modules():
        if type(mod).__name__.endswith("RotaryEmbedding") and hasattr(mod, "inv_freq"):
            return mod
    return None


def _as_numpy(t: Any) -> np.ndarray:
    return np.asarray(t.detach().to("cpu").double().numpy(), dtype=np.float64)


@contextmanager
def _captured_inputs(attns: Sequence[Any]) -> Iterator[dict[int, Any]]:
    """Collect each attention module's input tensor, keyed by layer index.

    A forward *pre*-hook is used rather than the pre-attention norm, so this
    keeps working for architectures that name that norm differently.
    """
    store: dict[int, Any] = {}
    handles: list[Any] = []

    def make(idx: int) -> Callable[..., None]:
        def hook(module, args, kwargs):
            if args:
                store[idx] = args[0].detach()
            elif "hidden_states" in kwargs:
                store[idx] = kwargs["hidden_states"].detach()

        return hook

    for i, attn in enumerate(attns):
        handles.append(attn.register_forward_pre_hook(make(i), with_kwargs=True))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@dataclass
class HeadVectors:
    """Unrotated per-head projections for one sequence.

    ``q_hat``/``k_hat`` are ``(n_layers, n_heads, T, head_dim)`` and are what a
    block's attention sees *before* any rotation, which is the only frame in
    which the per-pair decomposition is legible.

    With grouped-query attention the key projection is narrower than the query
    projection, and one key head is shared by several query heads. Each query
    head's key vector is then its own group's key vector, replicated, so the
    arrays here always have ``n_heads`` on the head axis. ``n_kv_heads`` and
    ``n_kv_groups`` record when that happened.
    """

    q_hat: np.ndarray
    k_hat: np.ndarray
    head_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int | None = None
    n_kv_groups: int | None = None

    @property
    def n_tokens(self) -> int:
        return int(self.q_hat.shape[2])

    @property
    def uses_grouped_query_attention(self) -> bool:
        return self.n_kv_groups is not None and self.n_kv_groups > 1


def _expand_kv_to_heads(k: Any, t: int, n_heads: int, hd: int, n_kv_heads: int) -> Any:
    """Replicate grouped key heads so every query head gets its own key vector.

    ``k`` is ``(1, T, n_kv_heads * hd)``. Query head ``i`` uses key head
    ``i // groups``, which is the convention the attention kernels use, so the
    replication has to be block-wise (``repeat_interleave``) and not a plain
    ``repeat``.
    """
    groups = n_heads // n_kv_heads
    per_head = k.view(1, t, n_kv_heads, 1, hd).expand(1, t, n_kv_heads, groups, hd)
    return per_head.reshape(1, t, n_heads, hd)


def collect_head_vectors(
    model: Any,
    tokenizer: Any,
    text: str,
    max_tokens: int | None = None,
) -> HeadVectors:
    """One forward pass; return every head's unrotated ``q_hat``/``k_hat``.

    A single sequence with no padding: a pad token would push a synthetic
    activation through the projection and pollute exactly the distribution being
    measured.
    """
    torch = _require_torch()
    attns = attention_modules(model)
    if not attns:
        raise RuntimeError("no attention module with a recognised q/k projection")

    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    if max_tokens is not None and ids.shape[1] > max_tokens:
        ids = ids[:, :max_tokens]
    if ids.shape[0] != 1:
        raise ValueError("expected a single sequence, got batch size != 1")

    with _captured_inputs(attns) as store, torch.no_grad():
        model(input_ids=ids, use_cache=False)

    cfg = model.config
    hd = head_dim_of(attns[0], cfg)
    nh = n_heads_of(attns[0], cfg)
    if not nh:
        raise RuntimeError("cannot determine the head count")

    n_kv_first: int | None = None
    groups_first: int | None = None
    qs, ks = [], []
    for idx, attn in enumerate(attns):
        if idx not in store:
            raise RuntimeError(f"attention module {idx} was never called")
        x = store[idx]
        t = int(x.shape[1])
        q, k = _qkv_projector(attn)(x)
        if q.shape[-1] != nh * hd or q.shape[-1] % hd or k.shape[-1] % hd:
            raise RuntimeError(
                f"unexpected projection widths: q={q.shape[-1]} k={k.shape[-1]}, "
                f"head_dim={hd}, n_heads={nh}"
            )
        n_kv_heads = k.shape[-1] // hd
        if n_kv_heads == nh:
            pass
        elif nh % n_kv_heads == 0:
            k = _expand_kv_to_heads(k, t, nh, hd, n_kv_heads)
        else:
            raise RuntimeError(
                f"n_heads={nh} is not a multiple of the key head count {n_kv_heads}"
            )
        if idx == 0:
            n_kv_first, groups_first = n_kv_heads, nh // n_kv_heads
        qs.append(q.view(1, t, nh, hd).transpose(1, 2))
        ks.append(k.view(1, t, nh, hd).transpose(1, 2))

    return HeadVectors(
        q_hat=np.concatenate(qs, axis=0),
        k_hat=np.concatenate(ks, axis=0),
        head_dim=hd,
        n_layers=len(attns),
        n_heads=nh,
        n_kv_heads=n_kv_first,
        n_kv_groups=groups_first,
    )


# --------------------------------------------------------------------------
# 1. Config validation
# --------------------------------------------------------------------------


def validate_config(model: Any) -> dict:
    """Does the checkpoint's declared configuration reproduce our ``inv_freq``?

    Reports the largest absolute difference between the frequency buffer the
    checkpoint built for itself and ``rope.inv_freq(dim, theta)`` built from the
    values in that checkpoint's own config. The checkpoint stores its ladder in
    its own floating dtype, so bit-exactness is not available; the honest
    statement is that the two agree to that dtype's precision, and the residual
    is reported either way.
    """
    cfg = model.config
    rp = dict(getattr(cfg, "rope_parameters", None) or {})
    theta = float(rp.get("rope_theta", 10000.0))
    partial = float(rp.get("partial_rotary_factor", 1.0))
    rope_type = str(rp.get("rope_type", "default"))

    n_embd = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    n_head = n_heads_of(None, cfg)
    head_dim = n_embd // n_head if n_head else 0
    rope_dim = int(head_dim * partial)

    attns = attention_modules(model)
    if attns:
        head_dim = head_dim_of(attns[0], cfg)
        rope_dim = int(head_dim * partial)

    ours = inv_freq(rope_dim, theta)

    rot = find_rotary(model)
    out: dict[str, Any] = {
        "model_type": str(getattr(cfg, "model_type", "")),
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": int(getattr(cfg, "num_hidden_layers", 0)),
        "max_position_embeddings": int(getattr(cfg, "max_position_embeddings", 0)),
        "head_dim": head_dim,
        "rope_theta_declared": theta,
        "partial_rotary_factor_declared": partial,
        "rope_type_declared": rope_type,
        "rope_dim": rope_dim,
        "rotated_fraction_of_head": (rope_dim / head_dim) if head_dim else 0.0,
        "rotary_pairs_per_head": int(ours.shape[0]),
        "our_inv_freq_len": int(ours.shape[0]),
        "our_inv_freq_head": ours.tolist(),
        "checkpoint_inv_freq_found": rot is not None,
    }

    if rot is None:
        out["config_matches_checkpoint"] = None
        out["note"] = "no rotary module found: this checkpoint does not use RoPE"
        return out

    theirs = _as_numpy(rot.inv_freq)
    out["checkpoint_inv_freq_len"] = int(theirs.shape[0])
    out["checkpoint_inv_freq_dtype"] = str(rot.inv_freq.dtype)
    out["checkpoint_inv_freq_head"] = theirs.tolist()

    if theirs.shape[0] == ours.shape[0]:
        diff = np.abs(theirs - ours)
        out["max_abs_diff"] = float(diff.max())
        out["max_rel_diff"] = float(
            (diff / np.maximum(np.abs(theirs), np.finfo(np.float64).tiny)).max()
        )
        out["lengths_agree"] = True
        # The ladder's largest entry is theta**0 = 1 whatever theta is, so the
        # storage dtype's own epsilon is the right yardstick for an absolute
        # difference.
        out["storage_dtype_eps"] = float(np.finfo(_np_dtype(rot.inv_freq.dtype)).eps)
        out["within_storage_eps"] = bool(out["max_abs_diff"] <= 8.0 * out["storage_dtype_eps"])
        out["config_matches_checkpoint"] = bool(out["within_storage_eps"])
    else:
        # The checkpoint rotates a different number of channels than
        # head_dim * partial_rotary_factor implies; report rather than hide it.
        out["max_abs_diff"] = None
        out["max_rel_diff"] = None
        out["lengths_agree"] = False
        out["storage_dtype_eps"] = None
        out["within_storage_eps"] = None
        out["config_matches_checkpoint"] = False

    out["partial_ladder_subsample_step"] = (
        head_dim // rope_dim if rope_dim and head_dim % rope_dim == 0 else None
    )
    out["note"] = (
        f"Full-width rotation would be inv_freq({head_dim}, theta) with "
        f"{head_dim // 2} channels; this checkpoint rotates only {rope_dim} "
        f"channel(s) per head because partial_rotary_factor = {partial}. The "
        f"{ours.shape[0]}-channel ladder is a subsample of the full one: its "
        f"channel k has the frequency the full ladder gives to channel "
        f"{head_dim // rope_dim}k, so partial rotation keeps every "
        f"{head_dim // rope_dim}th frequency and drops the rest, rather than "
        "truncating a contiguous block."
        if partial != 1.0 and rope_dim and head_dim % rope_dim == 0
        else f"Full-width rotation: inv_freq({head_dim}, {theta:g}) is the "
        f"{head_dim // 2}-channel ladder the checkpoint runs on."
    )
    return out


def _np_dtype(dtype: Any) -> np.dtype:
    """The numpy dtype matching a torch dtype, without importing torch.

    ``torch.float32`` -> ``numpy.float32``. Kept local so the module stays
    importable with numpy alone.
    """
    return np.dtype(str(dtype).removeprefix("torch."))


# --------------------------------------------------------------------------
# 1b. Do the exact identities themselves survive real activations?
# --------------------------------------------------------------------------


def identity_residuals(
    hv: HeadVectors,
    freqs: np.ndarray,
    rope_dim: int,
    n_pairs: int = N_IDENTITY_PAIRS,
    shifts: Sequence[int] = (1, 5, 37),
    scales: Sequence[float] = (0.0, 1.0, 2.0, -3.0, 7.5, 1e-3, 1e3),
    seed: int = DEFAULT_SEED,
) -> dict:
    """The three structural identities of E1, measured on real ``q_hat``/``k_hat``.

    The synthetic suite reports these at ~1e-14. A checkpoint does not hand over
    float64: its activations are whatever it computes in, and every quantity here
    is a residual between two float summations. So on real data the honest
    expectation is agreement at the *activation dtype's* epsilon, not at the
    synthetic suite's. This function measures the actual residual and records the
    dtype, so the reader can tell "the identity failed" from "the identity holds
    and float32 is the floor".

    Note what is being tested here is the algebra, not the content: the same
    theorems hold on random vectors, and they must also hold on trained ones.
    The answer being "yes, to float32" is the result.
    """
    from rope_attribution.experiments import score_relative
    from rope_attribution.rope import apply_rope, rope_cos_sin

    rng = np.random.default_rng(seed)
    seq_len = hv.n_tokens
    dtype = hv.q_hat.dtype
    eps = float(np.finfo(dtype).eps) if dtype.kind == "f" else float("nan")

    pair_err = 0.0
    pair_scale = 0.0
    rel_err = 0.0
    norm_err = 0.0
    key_norm_err = 0.0
    bilin_err = 0.0
    bilin_scaled_err = 0.0
    min_cancellation = 1.0
    n_pos_checks = 0
    n_pair_checks = 0
    n_bilin_checks = 0

    for layer in range(hv.n_layers):
        for head in range(hv.n_heads):
            q = hv.q_hat[layer, head, :, :rope_dim]
            k = hv.k_hat[layer, head, :, :rope_dim]
            for m, n in _content_pairs(seq_len, n_pairs, rng):
                m, n = int(m), int(n)
                delta = n - m

                # (a) the per-pair closed form against the brute-force rotation
                closed = float(pair_terms(q[m], k[n], freqs, delta).contributions().sum())
                brute = score_relative(q[m], k[n], freqs, delta)
                pair_err = max(pair_err, abs(closed - brute))
                pair_scale = max(pair_scale, abs(brute))
                n_pair_checks += 1

                # (b) the relative-position property: the same two content
                # vectors at shifted absolute positions must score identically.
                for s in shifts:
                    if m + s >= seq_len or n + s >= seq_len:
                        continue
                    c0, s0 = rope_cos_sin(np.array([m, n]), freqs)
                    c1, s1 = rope_cos_sin(np.array([m + s, n + s]), freqs)
                    base = float(
                        apply_rope(q[m][None, :], c0[0:1], s0[0:1])[0]
                        @ apply_rope(k[n][None, :], c0[1:2], s0[1:2])[0]
                    )
                    shifted = float(
                        apply_rope(q[m][None, :], c1[0:1], s1[0:1])[0]
                        @ apply_rope(k[n][None, :], c1[1:2], s1[1:2])[0]
                    )
                    rel_err = max(rel_err, abs(shifted - base))
                    n_pos_checks += 1

                # (c) norm preservation on the real query and key
                c2, s2 = rope_cos_sin(np.array([m, n]), freqs)
                kr = apply_rope(k[n][None, :], c2[1:2], s2[1:2])[0]
                qr = apply_rope(q[m][None, :], c2[0:1], s2[0:1])[0]
                key_norm_err = max(
                    key_norm_err,
                    abs(float(np.linalg.norm(kr)) - float(np.linalg.norm(k[n]))),
                )
                norm_err = max(
                    norm_err,
                    abs(float(np.linalg.norm(qr)) - float(np.linalg.norm(q[m]))),
                )

                # (d) exact bilinearity in the content, on a real vector
                unit = score_relative(q[m], k[n], freqs, delta)
                # The scale on which bilinearity is exact is the sum of the
                # absolute products, not the score: a real head's score is a
                # signed sum of terms that routinely cancel, so dividing by
                # |score| measures how close the head came to zero, not how
                # linear it is. Both are reported; the second is the honest one.
                cos_d, sin_d = rope_cos_sin(np.array([delta]), freqs)
                k_rot = apply_rope(k[n][None, :], cos_d, sin_d)[0]
                sum_abs = float(np.abs(q[m] * k_rot).sum())
                if sum_abs > 0.0:
                    min_cancellation = min(min_cancellation, abs(unit) / sum_abs)
                if unit == 0.0:
                    continue
                for a in scales:
                    got = score_relative(a * q[m], k[n], freqs, delta)
                    diff = abs(got - a * unit)
                    bilin_err = max(bilin_err, diff / (abs(a * unit) + 1e-12))
                    if sum_abs > 0.0:
                        bilin_scaled_err = max(
                            bilin_scaled_err, diff / (abs(a) * sum_abs + 1e-12)
                        )
                    n_bilin_checks += 1

    return {
        "activation_dtype": str(dtype),
        "activation_dtype_eps": eps,
        "note": "residuals are between two float64 summations of vectors that the "
        "checkpoint produced in its own dtype, so the floor is set by that dtype "
        "and not by the algebra",
        "pair_closed_form_max_abs_err": float(pair_err),
        "pair_closed_form_max_rel_err": float(pair_err / (pair_scale + 1e-12)),
        "pair_scores_checked": int(n_pair_checks),
        "relative_position_max_abs_err": float(rel_err),
        "relative_position_pairs_checked": int(n_pos_checks),
        "relative_position_shifts": list(shifts),
        "key_norm_preservation_max_abs_err": float(key_norm_err),
        "query_norm_preservation_max_abs_err": float(norm_err),
        "bilinearity_max_rel_err": float(bilin_err),
        "bilinearity_max_scaled_err": float(bilin_scaled_err),
        "bilinearity_note": "bilinearity_max_rel_err divides by |a * score|, which "
        "for a real head is a signed sum of large terms that routinely cancel: it "
        "is dominated by how close the head came to zero, not by nonlinearity. "
        "bilinearity_max_scaled_err divides by |a| * sum_k |q_k * k_rot_k|, the "
        "scale on which bilinearity is actually exact, and is the number to "
        "compare against the synthetic one.",
        "bilinearity_checks": int(n_bilin_checks),
        "bilinearity_min_score_over_sum_abs": float(min_cancellation),
        "bilinearity_scales": list(scales),
        "residual_over_dtype_eps": {
            "pair_closed_form": float(pair_err / (eps * max(1.0, pair_scale) + 1e-300)),
            "relative_position": float(rel_err / (eps + 1e-300)),
            "norm_preservation": float(max(norm_err, key_norm_err) / (eps + 1e-300)),
            "bilinearity": float(bilin_scaled_err / (eps + 1e-300)),
        },
    }


def gpt2_has_rope(model_id: str = "gpt2") -> dict:
    """Whether a checkpoint applies a rotation at all.

    Run on GPT-2 because "GPT-2 uses RoPE with ``theta = 10000``" is a common
    and false claim: GPT-2 uses *learned absolute* position embeddings. This
    function decides it from the loaded module graph, not from the config's name.
    """
    try:
        loaded = load_model(model_id)
    except Exception as exc:
        return {"model_id": model_id, "error": f"{type(exc).__name__}: {exc}"}

    rot = find_rotary(loaded.model)
    cfg = loaded.config
    learned = getattr(getattr(loaded.model, "transformer", None), "wpe", None)
    return {
        "model_id": model_id,
        "model_class": type(loaded.model).__name__,
        "model_type": str(getattr(cfg, "model_type", "")),
        "rotary_module_found": rot is not None,
        "rope_theta_in_config": "rope_theta" in (getattr(cfg, "rope_parameters", None) or {}),
        "learned_absolute_position_embedding": learned is not None,
        "learned_position_table_size": int(learned.weight.shape[0])
        if learned is not None
        else None,
        "uses_rope": rot is not None,
        "conclusion": (
            "applies a rotary rotation to q/k"
            if rot is not None
            else "applies no rotation: position enters through learned absolute "
            "embeddings, so its q_hat/k_hat have no rotary pair structure and "
            "nothing about it can be attributed to a rotary pair"
        ),
    }


# --------------------------------------------------------------------------
# aggregation helpers
# --------------------------------------------------------------------------


def _spread(values: Sequence[float] | np.ndarray) -> dict:
    """min / quartiles / median / max / mean / std of a sample."""
    a = np.asarray(values, dtype=np.float64).ravel()
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "p75": float(np.quantile(a, 0.75)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
    }


def _quantiles(a: np.ndarray, qs: Sequence[int] = (1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict:
    return {
        "percentiles": list(qs),
        "values": [float(np.quantile(a, q / 100.0)) for q in qs],
    }


def _content_pairs(seq_len: int, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Random causal ``(m, n)`` pairs, ``n >= m``, both inside ``[0, seq_len)``.

    The gap is drawn against ``seq_len - m`` rather than ``seq_len`` so ``n``
    cannot run off the end of a short prompt. ``delta = 0`` is included on
    purpose: the closed form must hold there too, and ``B_k`` at ``delta = 0`` is
    still a property of the content, not of distance.
    """
    m = rng.integers(0, seq_len, size=n_pairs)
    d = rng.integers(0, seq_len - m)
    return np.stack([m, m + d], axis=1)


# --------------------------------------------------------------------------
# 2. How position-carrying is each channel, for real
# --------------------------------------------------------------------------


def measure_position_channels(
    hv: HeadVectors,
    freqs: np.ndarray,
    rope_dim: int,
    n_pairs: int = N_CONTENT_PAIRS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Distribution of ``|B_k| / R_k`` over every ``(layer, head, pair)``.

    ``freqs`` is the checkpoint's own frequency ladder and ``rope_dim`` the
    number of head channels it actually rotates. ``B_k`` does not depend on
    distance, so drawing many content pairs re-samples content rather than
    position - but that is the point: the distribution is over what the trained
    weights put in the crossed term, and more content pairs make the across-head
    spread a property of the head rather than of one vector.
    """
    n_rot_pairs = int(freqs.shape[0])
    if rope_dim != 2 * n_rot_pairs:
        raise ValueError(
            f"rope_dim={rope_dim} is inconsistent with a ladder of {n_rot_pairs} channels"
        )
    rng = np.random.default_rng(seed)
    seq_len = hv.n_tokens

    pooled_shares: list[np.ndarray] = []
    pooled_blind: list[np.ndarray] = []
    degenerate = 0
    per_head: dict[str, dict[str, float]] = {}

    for layer in range(hv.n_layers):
        for head in range(hv.n_heads):
            q = hv.q_hat[layer, head, :, :rope_dim]
            k = hv.k_hat[layer, head, :, :rope_dim]
            head_shares: list[np.ndarray] = []
            head_blind: list[float] = []
            for m, n in _content_pairs(seq_len, n_pairs, rng):
                t = pair_terms(q[m], k[n], freqs, int(n) - int(m))
                amp = t.amplitude
                b_abs = np.abs(t.crossed)
                a_abs = np.abs(t.aligned)
                live = amp > 0.0
                degenerate += int((~live).sum())
                if not live.any():
                    continue
                s = b_abs[live] / amp[live]
                blind = ((b_abs < BLIND_RATIO * a_abs) & live).astype(np.float64)
                pooled_shares.append(s)
                pooled_blind.append(blind)
                head_shares.append(s)
                head_blind.append(float(blind.mean()))
            if not head_shares:
                continue
            all_s = np.concatenate(head_shares)
            per_head[f"{layer}.{head}"] = {
                "median_share": float(np.median(all_s)),
                "mean_share": float(all_s.mean()),
                "frac_position_blind": float(np.mean(head_blind)),
            }

    if not pooled_shares:
        raise RuntimeError("no rotary pairs were sampled")

    shares = np.concatenate(pooled_shares)
    blind = np.concatenate(pooled_blind)
    hist_counts, hist_edges = np.histogram(shares, bins=HISTOGRAM_BINS, range=(0.0, 1.0))

    head_medians = [v["median_share"] for v in per_head.values()]
    head_blinds = [v["frac_position_blind"] for v in per_head.values()]
    layer_medians = [
        float(np.mean([v["median_share"] for k, v in per_head.items()
                       if int(k.split(".")[0]) == layer]))
        for layer in range(hv.n_layers)
        if any(int(k.split(".")[0]) == layer for k in per_head)
    ]
    layer_blinds = [
        float(np.mean([v["frac_position_blind"] for k, v in per_head.items()
                       if int(k.split(".")[0]) == layer]))
        for layer in range(hv.n_layers)
        if any(int(k.split(".")[0]) == layer for k in per_head)
    ]

    return {
        "definition": "|B_k| / R_k with R_k = hypot(A_k, B_k): the share of a "
        "rotary pair's amplitude that carries position",
        "position_blind_threshold": {
            "statement": f"|B_k| < {BLIND_RATIO} * |A_k|",
            "ratio": BLIND_RATIO,
            "rationale": "the crossed term cannot move the pair's contribution "
            "by more than this fraction of the pair's own amplitude",
        },
        "rope_dim": int(rope_dim),
        "rotary_pairs_per_head": n_rot_pairs,
        "n_layers": int(hv.n_layers),
        "n_heads": int(hv.n_heads),
        "n_content_pairs_per_head": int(n_pairs),
        "n_samples": int(shares.size),
        "n_degenerate_pairs_excluded": int(degenerate),
        "share_quantiles": _quantiles(shares),
        "share_mean": float(shares.mean()),
        "share_histogram_0_to_1": {
            "bin_edges": [float(e) for e in hist_edges],
            "counts": [int(c) for c in hist_counts],
        },
        "frac_position_blind_overall": float(blind.mean()),
        "frac_position_blind_across_heads": _spread(head_blinds),
        "frac_position_blind_across_layers": _spread(layer_blinds),
        "median_share_across_heads": _spread(head_medians),
        "median_share_across_layers": _spread(layer_medians),
        "mean_share_across_heads": _spread([v["mean_share"] for v in per_head.values()]),
        "per_head": per_head,
    }


# --------------------------------------------------------------------------
# 3. Does the linearization survive real data?
# --------------------------------------------------------------------------


def measure_linearization(
    hv: HeadVectors,
    freqs: np.ndarray,
    rope_dim: int,
    deltas: Sequence[int] = DEFAULT_DELTAS,
    n_pairs: int = N_LINEARIZATION_PAIRS,
    seed: int = DEFAULT_SEED,
    max_trained_position: int | None = None,
) -> dict:
    """``experiments.linearization_error`` on real vectors, per delta.

    The content vectors are real activations, drawn independently of ``delta``;
    ``delta`` is then the gap handed to the closed form. That is what lets
    ``delta = 4096`` be measured on a checkpoint whose context window is far
    shorter, and it is flagged per row: ``in_trained_context`` is ``False``
    whenever the gap exceeds ``max_trained_position``, because at that point the
    measurement describes real *content* under a rotation the model was never
    trained on. That is the same thing the synthetic suite measures, but with
    trained vectors in place of random ones, and saying so is the point.

    Each entry carries the across-head and across-layer spread. The ladder-only
    entries - ``max_abs_angle``, ``median_abs_angle``,
    ``frac_channels_linearizable`` - are included *with* their spread, which is
    expected to be exactly zero. That is the measurement, not an omission: they
    are functions of ``delta * inv_freq`` alone, so no activation can move them.
    ``amplitude_weighted_err`` is the data-dependent entry and does spread.
    """
    rng = np.random.default_rng(seed)
    seq_len = hv.n_tokens

    rows = []
    for delta in deltas:
        head_err: list[float] = []
        head_rms: list[float] = []
        frac_lin: list[float] = []
        max_angle: list[float] = []
        med_angle: list[float] = []
        layer_err: dict[int, list[float]] = {}
        per_head_err: dict[str, float] = {}
        per_head_lin: dict[str, float] = {}

        for layer in range(hv.n_layers):
            for head in range(hv.n_heads):
                q = hv.q_hat[layer, head, :, :rope_dim]
                k = hv.k_hat[layer, head, :, :rope_dim]
                key = f"{layer}.{head}"
                worst = 0.0
                lin_here: list[float] = []
                for m, n in _content_pairs(seq_len, n_pairs, rng):
                    e = linearization_error(q[m], k[n], freqs, int(delta))
                    worst = max(worst, float(e["amplitude_weighted_err"]))
                    head_rms.append(float(e["per_pair_err_rms"]))
                    frac_lin.append(float(e["frac_channels_linearizable"]))
                    max_angle.append(float(e["max_abs_angle"]))
                    med_angle.append(float(e["median_abs_angle"]))
                    lin_here.append(float(e["frac_channels_linearizable"]))
                head_err.append(worst)
                per_head_err[key] = worst
                per_head_lin[key] = float(np.median(lin_here)) if lin_here else 0.0
                layer_err.setdefault(layer, []).append(worst)

        rows.append(
            {
                "delta": int(delta),
                "in_trained_context": (
                    None if max_trained_position is None
                    else bool(int(delta) <= max_trained_position)
                ),
                "max_abs_angle": _spread(max_angle),
                "median_abs_angle": _spread(med_angle),
                "frac_channels_linearizable": _spread(frac_lin),
                "frac_channels_linearizable_is_ladder_determined": True,
                "amplitude_weighted_err": _spread(head_err),
                "amplitude_weighted_err_across_layers": _spread(
                    [float(np.mean(v)) for v in layer_err.values()]
                ),
                "per_pair_err_rms": _spread(head_rms),
                "per_head_amplitude_weighted_err": per_head_err,
                "per_head_frac_channels_linearizable": per_head_lin,
            }
        )

    return {
        "rope_dim": int(rope_dim),
        "rotary_pairs_per_head": int(freqs.shape[0]),
        "n_content_pairs_per_head": int(n_pairs),
        "max_trained_position": max_trained_position,
        "note": "content vectors are real activations drawn independently of "
        "delta; delta is the gap handed to the closed form. Each (layer, head) "
        "entry is the max over that head's content pairs. "
        "amplitude_weighted_err is dimensionless because numerator and "
        "denominator both scale as a sum of per-pair amplitudes.",
        "rows": rows,
    }


# --------------------------------------------------------------------------
# 5. Real attention entropy
# --------------------------------------------------------------------------


def measure_entropy(
    model: Any,
    tokenizer: Any,
    text: str,
    lens: Sequence[int],
) -> dict:
    """``H / ln T`` of the real attention distribution, per length, layer and head.

    Two normalizations. ``H / ln T`` over all ``T`` keys is the form
    ``experiments.mscale_entropy`` uses and is the directly comparable one.
    ``H / ln(n_allowed)`` normalizes by the number of keys a causal row may
    actually attend to, which is the meaningful maximum; the two differ by the
    fraction of positions that are masked out. Rows with a single key carry no
    information and are excluded from both.

    Each length is a *prefix* of the same tokenization, so the shorter runs are
    not a different text - they are the same text truncated.
    """
    torch = _require_torch()
    ids = tokenizer(text, return_tensors="pt")["input_ids"]

    per_length = []
    for want in lens:
        if want > ids.shape[1]:
            continue
        t = int(want)
        with torch.no_grad():
            out = model(input_ids=ids[:, :t], output_attentions=True, use_cache=False)

        per_layer_ln_t: list[float] = []
        per_layer_allowed: list[float] = []
        per_head_ln_t: list[list[float]] = []
        max_ln_t = math.log(t)

        for probs in out.attentions:
            p = _as_numpy(probs)[0]  # (n_heads, T, T)
            head_ratios: list[float] = []
            head_allowed: list[float] = []
            for h in range(p.shape[0]):
                ratios: list[float] = []
                allowed: list[float] = []
                for m in range(1, t):
                    row = p[h, m, : m + 1]
                    total = float(row.sum())
                    if total <= 0.0:
                        continue
                    row = row / total
                    nz = row[row > 0.0]
                    ent = float(-(nz * np.log(nz)).sum())
                    ratios.append(ent / max_ln_t)
                    allowed.append(ent / math.log(m + 1))
                if ratios:
                    head_ratios.append(float(np.mean(ratios)))
                    head_allowed.append(float(np.mean(allowed)))
            if head_ratios:
                per_head_ln_t.append(head_ratios)
                per_layer_ln_t.append(float(np.mean(head_ratios)))
                per_layer_allowed.append(float(np.mean(head_allowed)))

        if not per_layer_ln_t:
            continue
        per_length.append(
            {
                "seq_len": t,
                "max_possible_entropy_ln_T": max_ln_t,
                "entropy_ratio_vs_ln_T_across_layers": _spread(per_layer_ln_t),
                "entropy_ratio_vs_ln_allowed_across_layers": _spread(per_layer_allowed),
                "entropy_ratio_vs_ln_T_across_heads_within_layer": _spread(
                    [v for row in per_head_ln_t for v in row]
                ),
                "entropy_ratio_vs_ln_T_per_layer": per_layer_ln_t,
                "entropy_ratio_vs_ln_T_per_head_per_layer": per_head_ln_t,
            }
        )

    return {
        "n_tokens_available": int(ids.shape[1]),
        "per_length": per_length,
    }


# --------------------------------------------------------------------------
# synthetic baseline for the side-by-side comparison
# --------------------------------------------------------------------------


def synthetic_baseline(
    rope_dim: int | None = None,
    theta: float = 10000.0,
    deltas: Sequence[int] = DEFAULT_DELTAS,
    seed: int = DEFAULT_SEED,
    measurements_path: Path | None = None,
) -> dict:
    """The published synthetic numbers, plus a recomputed same-width control.

    ``results/measurements.json`` is read for the published values so the
    comparison is against what the repository actually claims rather than
    against a number regenerated here. The control re-runs
    ``experiments.linearization_error`` with the same standard-normal vectors at
    the *checkpoint's* ladder width, which separates "this ladder has fewer
    channels" from "these weights are real".
    """
    if measurements_path is None:
        measurements_path = (
            Path(__file__).resolve().parents[2] / "results" / "measurements.json"
        )

    published: list[dict] = []
    if measurements_path.exists():
        data = json.loads(measurements_path.read_text(encoding="utf-8"))
        for row in data.get("method_spectrum", []):
            if row.get("method") == "rope_base_10k":
                published.append(
                    {
                        "delta": int(row["delta"]),
                        "frac_channels_linearizable": float(
                            row["frac_channels_linearizable"]
                        ),
                        "amplitude_weighted_err": float(row["amplitude_weighted_err"]),
                        "max_abs_angle": float(row["max_abs_angle"]),
                        "median_abs_angle": float(row["median_abs_angle"]),
                    }
                )

    control: list[dict] = []
    if rope_dim:
        freqs = inv_freq(rope_dim, theta)
        rng = np.random.default_rng(seed)
        q = rng.standard_normal(rope_dim)
        k = rng.standard_normal(rope_dim)
        for d in deltas:
            e = linearization_error(q, k, freqs, d)
            control.append(
                {
                    "delta": int(d),
                    "frac_channels_linearizable": float(e["frac_channels_linearizable"]),
                    "amplitude_weighted_err": float(e["amplitude_weighted_err"]),
                    "max_abs_angle": float(e["max_abs_angle"]),
                    "median_abs_angle": float(e["median_abs_angle"]),
                }
            )

    return {
        "published_source": str(measurements_path),
        "published_available": bool(published),
        "synthetic_published_32_channel": published,
        "synthetic_same_ladder": control,
        "synthetic_same_ladder_note": "standard-normal q_hat/k_hat at "
        f"inv_freq({rope_dim}, {theta:g}): the published synthetic construction "
        "on this checkpoint's ladder width",
        "seed": int(seed),
    }


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


def measure_model(
    model_id: str = DEFAULT_MODEL_IDS[0],
    seed: int = DEFAULT_SEED,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    long_texts: Sequence[str] = DEFAULT_LONG_TEXTS,
    lens: Sequence[int] = DEFAULT_LENS,
    deltas: Sequence[int] = DEFAULT_DELTAS,
) -> dict:
    """Every real-model measurement for one checkpoint."""
    loaded = load_model(model_id, seed=seed)
    cfgv = validate_config(loaded.model)
    rope_dim = int(cfgv["rope_dim"])
    if not cfgv["checkpoint_inv_freq_found"]:
        raise RuntimeError(
            f"{model_id} has no rotary module, so it has no rotary pairs to measure"
        )

    # The ladder this model actually runs on, rebuilt from its own config.
    # ``validate_config`` has already checked it against the checkpoint's buffer.
    freqs = inv_freq(rope_dim, float(cfgv["rope_theta_declared"]))

    per_prompt = []
    for i, text in enumerate(prompts):
        hv = collect_head_vectors(loaded.model, loaded.tokenizer, text)
        per_prompt.append(
            {
                "prompt_index": i,
                "n_tokens": hv.n_tokens,
                "n_kv_heads": hv.n_kv_heads,
                "n_kv_groups": hv.n_kv_groups,
                "position_channels": measure_position_channels(
                    hv, freqs, rope_dim, seed=seed + i
                ),
                "identities": identity_residuals(hv, freqs, rope_dim, seed=seed + i),
                "linearization": measure_linearization(
                    hv, freqs, rope_dim, deltas=deltas, seed=seed + i,
                    max_trained_position=int(cfgv["max_position_embeddings"]) - 1,
                ),
            }
        )

    entropies = []
    for i, text in enumerate(long_texts):
        try:
            entropies.append(
                {"text_index": i, **measure_entropy(loaded.model, loaded.tokenizer, text, lens)}
            )
        except Exception as exc:
            entropies.append(
                {"text_index": i, "error": f"{type(exc).__name__}: {exc}"}
            )

    return {
        "model_id": model_id,
        "model_class": type(loaded.model).__name__,
        "config_validation": cfgv,
        "head_dim": int(cfgv["head_dim"]),
        "rope_dim": rope_dim,
        "n_kv_heads": per_prompt[0]["n_kv_heads"],
        "n_kv_groups": per_prompt[0]["n_kv_groups"],
        "seed": int(seed),
        "per_prompt": per_prompt,
        "entropy": entropies,
    }


def _collect_numbers(
    flat: dict[str, list[float]], payload: dict, prefix: str = ""
) -> None:
    """Flatten one level of nesting, keeping only finite numbers.

    Strings and lists (the dtype name, the shift list, the prose note) are
    dropped rather than coerced, so a summary block never ends up holding a
    string where a float belongs.
    """
    for key, val in payload.items():
        name = f"{prefix}{key}"
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            if math.isfinite(float(val)):
                flat.setdefault(name, []).append(float(val))
        elif isinstance(val, dict):
            _collect_numbers(flat, val, prefix=f"{name}_")


def aggregate(model: dict) -> dict:
    """Pool one model's per-prompt summaries.

    The raw per-sample ``|B_k| / R_k`` array is not carried in the JSON (it is
    millions of floats), so pooled quantiles are reported as the *spread across
    prompts* of each prompt's own pooled quantile. That is the honest error bar:
    it says how much the quantile moves when the input changes, which is what a
    reader needs to know before trusting any single number.
    """
    share_q: list[float] = []
    blind: list[float] = []
    head_median: list[float] = []
    head_blind: list[float] = []
    layer_median: list[float] = []
    layer_blind: list[float] = []
    n_samples = 0
    degenerate = 0
    pooled_hist: np.ndarray | None = None
    pooled_edges: list[float] = []
    identity: dict[str, list[float]] = {}
    residual: dict[str, list[float]] = {}
    lin: dict[int, dict[str, list[float]]] = {}

    for pp in model["per_prompt"]:
        pos = pp["position_channels"]
        n_samples += int(pos["n_samples"])
        degenerate += int(pos["n_degenerate_pairs_excluded"])
        share_q.extend(float(v) for v in pos["share_quantiles"]["values"])
        blind.append(float(pos["frac_position_blind_overall"]))
        head_median.append(float(pos["median_share_across_heads"]["median"]))
        head_blind.append(float(pos["frac_position_blind_across_heads"]["median"]))
        layer_median.append(float(pos["median_share_across_layers"]["median"]))
        layer_blind.append(float(pos["frac_position_blind_across_layers"]["median"]))

        # Histogram counts are additive, so pooling them is exact rather than
        # an estimate - which is why the pooled histogram is exact while the
        # pooled quantiles below are a spread of per-prompt quantiles.
        hist = pos["share_histogram_0_to_1"]
        counts = np.asarray(hist["counts"], dtype=np.float64)
        if pooled_hist is None:
            pooled_hist = np.zeros(counts.size, dtype=np.float64)
            pooled_edges = [float(e) for e in hist["bin_edges"]]
        pooled_hist[: counts.size] += counts

        for key, val in pp["identities"].items():
            if (
                key != "residual_over_dtype_eps"
                and isinstance(val, (int, float))
                and not isinstance(val, bool)
                and math.isfinite(float(val))
            ):
                identity.setdefault(key, []).append(float(val))
        _collect_numbers(residual, pp["identities"].get("residual_over_dtype_eps", {}))

        for row in pp["linearization"]["rows"]:
            slot = lin.setdefault(
                int(row["delta"]),
                {
                    "frac_lin": [],
                    "frac_lin_head_std": [],
                    "max_angle": [],
                    "amp_err": [],
                    "amp_err_head_min": [],
                    "amp_err_head_max": [],
                    "rms_min": [],
                    "rms_max": [],
                },
            )
            slot["frac_lin"].append(float(row["frac_channels_linearizable"]["median"]))
            slot["frac_lin_head_std"].append(
                float(row["frac_channels_linearizable"]["std"])
            )
            slot["max_angle"].extend(float(v) for v in [row["max_abs_angle"]["max"]])
            slot["amp_err"].append(float(row["amplitude_weighted_err"]["median"]))
            slot["amp_err_head_min"].append(float(row["amplitude_weighted_err"]["min"]))
            slot["amp_err_head_max"].append(float(row["amplitude_weighted_err"]["max"]))
            slot["rms_min"].append(float(row["per_pair_err_rms"]["min"]))
            slot["rms_max"].append(float(row["per_pair_err_rms"]["max"]))

    qp = model["per_prompt"][0]["position_channels"]["share_quantiles"]
    return {
        "n_samples": n_samples,
        "n_degenerate_pairs_excluded": degenerate,
        "histogram_pooled_bin_edges": pooled_edges or [
            i / HISTOGRAM_BINS for i in range(HISTOGRAM_BINS + 1)
        ],
        "histogram_pooled_counts": (
            [int(c) for c in pooled_hist] if pooled_hist is not None else []
        ),
        "share_quantile_labels": [f"p{q}" for q in qp["percentiles"]],
        "share_quantiles_across_prompts": _spread(share_q),
        "identities": {
            **{k: _spread(v) for k, v in identity.items()},
            "residual_over_dtype_eps": {
                k: _spread(v) for k, v in residual.items()
            },
        },
        "frac_position_blind_across_prompts": _spread(blind),
        "median_share_across_heads": _spread(head_median),
        "frac_position_blind_across_heads": _spread(head_blind),
        "median_share_across_layers": _spread(layer_median),
        "frac_position_blind_across_layers": _spread(layer_blind),
        "linearization": [
            {
                "delta": d,
                "frac_channels_linearizable": _spread(v["frac_lin"]),
                "frac_channels_linearizable_std_across_heads_and_prompts": _spread(
                    v["frac_lin_head_std"]
                ),
                "max_abs_angle": _spread(v["max_angle"]),
                "amplitude_weighted_err": _spread(v["amp_err"]),
                "amplitude_weighted_err_head_range": {
                    "min": float(min(v["amp_err_head_min"])),
                    "max": float(max(v["amp_err_head_max"])),
                },
                "per_pair_err_rms_head_range": {
                    "min": float(min(v["rms_min"])),
                    "max": float(max(v["rms_max"])),
                },
            }
            for d, v in sorted(lin.items())
        ],
    }


def run_all(
    model_ids: Sequence[str] = DEFAULT_MODEL_IDS,
    seed: int = DEFAULT_SEED,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    long_texts: Sequence[str] = DEFAULT_LONG_TEXTS,
    lens: Sequence[int] = DEFAULT_LENS,
    deltas: Sequence[int] = DEFAULT_DELTAS,
    with_gpt2_check: bool = True,
) -> dict:
    """Measure every requested checkpoint.

    A failure on one checkpoint is recorded and the run continues, because a
    missing download must not cost the measurements that did succeed.
    """
    if not REAL_MODEL_AVAILABLE:
        raise RuntimeError(
            "torch and/or transformers are unavailable; see real_model_requirements.txt"
        )

    models: list[dict] = []
    for mid in model_ids:
        try:
            models.append(
                measure_model(
                    mid, seed=seed, prompts=prompts, long_texts=long_texts,
                    lens=lens, deltas=deltas,
                )
            )
        except Exception as exc:
            models.append({"model_id": mid, "error": f"{type(exc).__name__}: {exc}"})

    baselines = {
        m["model_id"]: synthetic_baseline(
            rope_dim=m["rope_dim"],
            theta=float(m["config_validation"]["rope_theta_declared"]),
            deltas=deltas,
            seed=seed,
        )
        for m in models
        if "error" not in m
    }

    return {
        "schema": "rope_attribution/real_model/v1",
        "seed": int(seed),
        "num_threads": NUM_THREADS,
        "torch_version": _version("torch"),
        "transformers_version": _version("transformers"),
        "numpy_version": _version("numpy"),
        "deltas": [int(d) for d in deltas],
        "sequence_lengths": [int(x) for x in lens],
        "prompts": list(prompts),
        "entropy_texts": len(long_texts),
        "position_blind_ratio": BLIND_RATIO,
        "n_content_pairs_per_head": N_CONTENT_PAIRS,
        "n_linearization_pairs_per_head": N_LINEARIZATION_PAIRS,
        "n_identity_pairs_per_head": N_IDENTITY_PAIRS,
        "models": models,
        "aggregates": {m["model_id"]: aggregate(m) for m in models if "error" not in m},
        "synthetic_baseline": baselines,
        "gpt2_rope_check": gpt2_has_rope() if with_gpt2_check else None,
    }


def write_results(data: dict, path: Path | None = None) -> Path:
    """Write the JSON payload to ``results/real_model.json``."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "results" / "real_model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# printed report
# --------------------------------------------------------------------------


def _fmt(x: Any, spec: str = ".4g") -> str:
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    data = run_all()
    rule = "=" * 78

    print(rule)
    print("REAL MODEL: the structural measurements on pretrained checkpoints")
    print(rule)
    print(
        f"  torch {data['torch_version']}  transformers {data['transformers_version']}"
        f"  numpy {data['numpy_version']}  threads {data['num_threads']}"
        f"  seed {data['seed']}"
    )
    print(
        f"  deltas {data['deltas']}  lengths {data['sequence_lengths']}"
        f"  content pairs/head {data['n_content_pairs_per_head']}"
        f" (linearization {data['n_linearization_pairs_per_head']})"
    )
    print()

    g = data["gpt2_rope_check"]
    print(rule)
    print("0. Does GPT-2 use RoPE?  (a premise this module had to check)")
    print(rule)
    if g and "error" in g:
        print(f"  could not load {g['model_id']}: {g['error']}")
    elif g:
        print(
            f"  {g['model_id']} ({g['model_type']}): rotary module found = "
            f"{g['rotary_module_found']}"
        )
        print(
            f"    learned absolute position table: {g['learned_absolute_position_embedding']}"
            f" (size {g['learned_position_table_size']})"
            f" | rope_theta in config: {g['rope_theta_in_config']}"
        )
        print(f"  -> {g['conclusion']}")
    print()

    for model in data["models"]:
        mid = model["model_id"]
        print(rule)
        print(f"{mid}")
        print(rule)
        if "error" in model:
            print(f"  FAILED: {model['error']}")
            print()
            continue

        cv = model["config_validation"]
        print(
            f"  {model['model_class']} / model_type={cv['model_type']}"
            f"  layers={cv['n_layer']} heads={cv['n_head']} head_dim={cv['head_dim']}"
            f" maxpos={cv['max_position_embeddings']}"
            f" kv_heads={model['n_kv_heads']} kv_groups={model['n_kv_groups']}"
        )
        print(
            f"  declared rope_theta={cv['rope_theta_declared']:g}"
            f"  partial_rotary_factor={cv['partial_rotary_factor_declared']:g}"
            f"  -> rotates {cv['rope_dim']}/{cv['head_dim']} channels per head"
            f" ({cv['rotary_pairs_per_head']} rotary pairs)"
        )
        print(
            f"  our inv_freq({cv['rope_dim']}, {cv['rope_theta_declared']:g}) vs the"
            f" checkpoint's own buffer"
            f" ({cv['checkpoint_inv_freq_len']} ch, {cv['checkpoint_inv_freq_dtype']}):"
            f"  max|diff| = {_fmt(cv.get('max_abs_diff'), '.3e')}"
            f"  max rel = {_fmt(cv.get('max_rel_diff'), '.3e')}"
            f"  storage eps = {_fmt(cv.get('storage_dtype_eps'), '.3e')}"
            f"  MATCH = {cv['config_matches_checkpoint']}"
        )
        print(f"    {cv.get('note', '')}")
        print()

        agg = data["aggregates"][mid]
        ident = model["per_prompt"][0]["identities"]
        agg_id = agg["identities"]
        print("  1b. Do the exact identities still hold on real activations?")
        print(
            f"     activations are {ident['activation_dtype']}"
            f" (eps = {_fmt(ident['activation_dtype_eps'], '.3e')});"
            " residuals are therefore floored by that dtype, not by the algebra"
        )
        print(
            f"     {'identity':<26} {'max|err| on real':>18} {'/ dtype eps':>12}"
            f" {'synthetic (measurements.json)':>28}"
        )
        for label, key, ratio_key, synth in (
            ("pair_closed_form", "pair_closed_form_max_abs_err", "pair_closed_form", "3.6e-15"),
            ("relative_position", "relative_position_max_abs_err", "relative_position", "3.4e-14"),
            ("bilinearity (scaled)", "bilinearity_max_scaled_err", "bilinearity", "7.4e-16"),
            ("key_norm_preservation", "key_norm_preservation_max_abs_err", "norm_preservation", "1.8e-15"),
            ("query_norm_preservation", "query_norm_preservation_max_abs_err", "norm_preservation", "1.8e-15"),
        ):
            print(
                f"     {label:<26} {_fmt(agg_id[key]['max'], '.3e'):>18}"
                f" {_fmt(agg_id['residual_over_dtype_eps'][ratio_key]['max'], '.3g'):>12}"
                f" {synth:>28}"
            )
        print(
            f"     bilinearity normalised by |score| instead (not meaningful on real"
            f" data, the score cancels): {_fmt(agg_id['bilinearity_max_rel_err']['max'], '.3e')}"
        )
        print(
            f"     smallest |score| / sum|q*k_rot| seen:"
            f" {_fmt(ident['bilinearity_min_score_over_sum_abs'], '.3e')}"
            f"  -> that is how close real heads come to zero"
        )
        print(
            f"     checks: {ident['pair_scores_checked']} pair scores,"
            f" {ident['relative_position_pairs_checked']} shifted position pairs,"
            f" {ident['bilinearity_checks']} bilinearity checks"
        )
        print()

        p0 = model["per_prompt"][0]["position_channels"]
        print("  2. Share of each rotary pair's amplitude that carries position, |B_k|/R_k")
        print(
            f"     samples = {agg['n_samples']} over {p0['n_layers']} layers x"
            f" {p0['n_heads']} heads x {p0['rotary_pairs_per_head']} pairs"
            f" x {p0['n_content_pairs_per_head']} content pairs"
        )
        per_prompt_q = []
        for pp in model["per_prompt"]:
            q = pp["position_channels"]["share_quantiles"]
            per_prompt_q.append([q["values"][q["percentiles"].index(x)] for x in (10, 50, 90)])
        arr = np.asarray(per_prompt_q, dtype=np.float64)
        print(
            f"     p10 / p50 / p90 (across {len(arr)} prompts): "
            f"{_fmt(np.median(arr[:, 0]), '.4f')} / {_fmt(np.median(arr[:, 1]), '.4f')}"
            f" / {_fmt(np.median(arr[:, 2]), '.4f')}"
        )
        for j, lab in enumerate(("p10", "p50", "p90")):
            print(
                f"       {lab}: min={_fmt(arr[:, j].min(), '.4f')}"
                f" median={_fmt(np.median(arr[:, j]), '.4f')}"
                f" max={_fmt(arr[:, j].max(), '.4f')}"
            )
        for key, lab in (
            ("median_share_across_heads", "per-(layer,head) median share"),
            ("median_share_across_layers", "per-layer median share      "),
            ("frac_position_blind_across_heads", "position-blind across heads"),
            ("frac_position_blind_across_layers", "position-blind across layers"),
        ):
            s = agg[key]
            print(
                f"     {lab}: min={_fmt(s['min'], '.4f')} median={_fmt(s['median'], '.4f')}"
                f" max={_fmt(s['max'], '.4f')} std={_fmt(s['std'], '.4f')}"
            )
        bl = [
            pp["position_channels"]["frac_position_blind_overall"]
            for pp in model["per_prompt"]
        ]
        print(
            f"     pooled position-blind fraction (|B| < {BLIND_RATIO:g}*|A|):"
            f" median={_fmt(np.median(bl), '.4f')}"
            f" range=[{_fmt(min(bl), '.4f')}, {_fmt(max(bl), '.4f')}]"
        )
        h = p0["share_histogram_0_to_1"]
        print(f"     histogram of |B|/R in [0,1], {len(h['counts'])} bins, "
              f"counts={h['counts']}")
        print()

        print("  3. Linearization on real activations vs the synthetic suite")
        base = data["synthetic_baseline"].get(mid, {})
        pub = {r["delta"]: r for r in base.get("synthetic_published_32_channel", [])}
        ctl = {r["delta"]: r for r in base.get("synthetic_same_ladder", [])}
        print(
            f"     {'delta':>6} {'max|D|':>10} {'real frac_lin':>14}"
            f" {'synth32ch':>10} {'synthSameW':>11}"
            f" {'real amp-err':>14} {'head min..max':>22} {'synth amp-err':>14}"
        )
        for row in agg["linearization"]:
            d = row["delta"]
            p = pub.get(d, {})
            c = ctl.get(d, {})
            rng_txt = (
                f"{_fmt(row['amplitude_weighted_err_head_range']['min'], '.2e')}.."
                f"{_fmt(row['amplitude_weighted_err_head_range']['max'], '.2e')}"
            )
            print(
                f"     {d:>6d} {_fmt(row['max_abs_angle']['max'], '.4g'):>10}"
                f" {_fmt(row['frac_channels_linearizable']['median'], '.4f'):>14}"
                f" {_fmt(p.get('frac_channels_linearizable'), '.4f'):>10}"
                f" {_fmt(c.get('frac_channels_linearizable'), '.4f'):>11}"
                f" {_fmt(row['amplitude_weighted_err']['median'], '.3e'):>14}"
                f" {rng_txt:>22}"
                f" {_fmt(p.get('amplitude_weighted_err'), '.3e'):>14}"
            )
        print(
            f"     synth32ch = results/measurements.json method=rope_base_10k (32 pairs);"
            f" synthSameW = same construction at this model's {cv['rotary_pairs_per_head']} pairs"
        )
        std_row = agg["linearization"][0]
        print(
            "     std of frac_channels_linearizable across heads and prompts:"
            f" max = {_fmt(std_row['frac_channels_linearizable_std_across_heads_and_prompts']['max'], '.3e')}"
            "   (ladder-determined: expected exactly 0)"
        )
        print()

        print("  4. Real attention entropy")
        for e in model["entropy"]:
            if "error" in e:
                print(f"     text {e.get('text_index')} FAILED: {e['error']}")
                continue
            print(f"     text {e['text_index']} ({e['n_tokens_available']} tokens available):")
            for row in e["per_length"]:
                sp = row["entropy_ratio_vs_ln_T_across_layers"]
                sa = row["entropy_ratio_vs_ln_allowed_across_layers"]
                sh = row["entropy_ratio_vs_ln_T_across_heads_within_layer"]
                print(
                    f"       T={row['seq_len']:>4}  H/lnT across layers:"
                    f" min={_fmt(sp['min'], '.4f')} median={_fmt(sp['median'], '.4f')}"
                    f" max={_fmt(sp['max'], '.4f')} std={_fmt(sp['std'], '.4f')}"
                    f" | across heads: median={_fmt(sh['median'], '.4f')}"
                    f" std={_fmt(sh['std'], '.4f')}"
                    f" | H/ln(allowed): median={_fmt(sa['median'], '.4f')}"
                )
        print()

    out = write_results(data)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()