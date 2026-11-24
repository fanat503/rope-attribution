"""Which basis should carry the QK attribution: SAE features or neurons?

``qk_attribution`` is exact for any dictionary: the decomposition
``x = base + sum_i f_i d_i`` holds by construction, and the leftover lands in the
base bucket. Exactness therefore says nothing about *which* ``d_i`` to use. This
module holds the checkpoint, the layer, the heads, the token pairs and the
measurement fixed, and varies only the basis.

Four arms, all expressible as ``(f, W_dec)``:

* ``sae``         -- a trained sparse dictionary, the incumbent.
* ``neurons``     -- the MLP write directions of every block that contributed
                     to the residual, with the activations those blocks actually
                     computed. The output projection's bias is not a direction
                     and stays in the base.
* ``random``      -- Gaussian unit directions, the "any basis works" control.
                     It is not a candidate: a sufficiently wide random basis can
                     match a candidate's fidelity slope because the probe rewards
                     any basis that moves the score in the right direction, so
                     the decision rule is restricted to the SAE-versus-neurons
                     pair and the random arm is reported as a control.
* ``coordinates`` -- the standard basis, the harness sanity check. At full
                     width it reconstructs the residual exactly, so the feature
                     terms sum to the small epsilon term in LayerNorm's
                     Jacobian and the base carries almost the whole score;
                     that is LayerNorm's near-scale-invariance, not a coverage
                     failure. It is run at full width and is not a competitor.

The comparison is reported as a table, not collapsed into one score. A basis
can have a larger base term and still be the better object to read, so the
questions are kept apart:

* ``base_share``   -- the magnitude of the base term relative to the score,
                     averaged over pairs. This is the paper's accounting ratio,
                     not a coverage fraction: it can exceed one because the
                     feature terms cancel, and it is not comparable across
                     distance grids or head sets. The arms here share both, so
                     the comparison is paired.
* ``top_k_share`` -- ``sum |query_features| / |score|``, the magnitude the
                     selected features carry before cancellation. It can exceed
                     one and is not a fraction of the score; it says how much
                     absolute movement the decomposition attributes, not how
                     much of the net score it explains.
* ``fidelity``     -- ablate the top query features and compare the predicted
                     score change with the actual one. This is the only causal
                     check; the rest are accounting. The slope is a calibration
                     coefficient through the origin, not a bounded fraction:
                     1.0 means the linear prediction has the right scale on
                     average, and a value above 1 means the ablation moves the
                     score more than the linear share predicts.
* ``stability``    -- whether the *feature-level* sign-flip rate survives the
                     basis change. The paper's 27.0% headline is a sign-flip
                     rate over the raw q/k rotary channels and the distance
                     grid; the basis cannot change those channels, so the
                     headline is basis-agnostic by construction and is not
                     re-measured here. What the basis can change is the
                     feature-level sweep, which is the object of this column.
* ``cost``         -- a dictionary has to be trained, neurons do not.

The arms do not cover the same objects. The SAE reconstructs the residual;
the neuron basis covers only the MLP writes, so its base necessarily carries
the embedding, every attention write and the MLP output biases. ``base_share``
is therefore an accounting ratio for each arm, not a common coverage scale.
The base-share condition in ``DECISION`` is consequently a conservative guard,
not a like-for-like coverage test: the neuron arm is held to the SAE's base
share even though its base necessarily carries objects the SAE's features
cover. Passing it says the MLP writes carry enough of the score to be read
without pushing the rest into the base; failing it does not by itself show the
neuron basis is worse to read.

Fidelity has a ceiling: LayerNorm renormalises after an ablation, so even a
perfect basis cannot match the linear prediction exactly. The ceiling is the
same test with *every* selected feature ablated, and the graded slope is only
interpretable next to the ceiling slope. For a full basis the full-ablation
prediction is near zero, so the ceiling slope and ratio are undefined; that is
expected.

The decision rule is fixed before the run, in ``KILL`` and ``DECISION`` below,
so the result cannot be read backwards into a threshold that happens to fit.
Stage 1 stops on a fidelity slope below half the SAE's; Stage 2 replaces the
SAE only on at least 0.8 of its slope with no base-share inflation. The ratio
of sums is reported next to the slope but is not the decision variable: it is
a total-effect ratio, and a per-pair offset can make it negative or huge while
the slope through the origin stays near one.

The run is staged. Stage 1 is the same command with ``--heads 0,1 --pairs 8
--no-sweep``; read ``verdict.stage1_kill`` before paying for the sweep. Stage 2
runs the full heads and sweep only if Stage 1 did not kill the neuron arm.

Only the basis changes. ``token_buckets`` is reused unchanged, including its
exact LayerNorm Jacobian; ``Basis`` is duck-typed on ``W_dec``, which is the
only attribute that function reads. The forward capture and the layer
addressing are reused from ``qk_attribution`` rather than copied, so the two
harnesses cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import qk_attribution as Q
from . import real_model as RM
from . import rope as R

__all__ = [
    "BASIS_NAMES",
    "DECISION",
    "DEFAULT_ABLATION",
    "KILL",
    "SCHEMA",
    "SWEEP_DELTAS",
    "Basis",
    "Geometry",
    "NeuronBlock",
    "bakeoff",
    "coordinate_basis",
    "main",
    "measure_basis",
    "neuron_basis",
    "random_basis",
    "sae_basis",
    "verdict",
    "write_results",
]

SCHEMA = "basis-bakeoff/v1"

#: Every arm this module can build. ``sae`` needs a dictionary file; the rest
#: are derived from the checkpoint or from a seed.
BASIS_NAMES: tuple[str, ...] = ("sae", "neurons", "random", "coordinates")

#: The 13-point log grid the paper's sign-flip rate is quoted on. The
#: bake-off reuses the distances so the feature-level sweep has the same support
#: as the headline; it does not reproduce the headline itself. The paper's 27.0%
#: is measured by ``usefulness.py`` over the raw q/k channels and its own prompt
#: set, while this module sweeps the features of one passage. The sweep column
#: answers whether the feature-level sign behaviour is basis-dependent, not
#: whether 27.0% repeats.
SWEEP_DELTAS: tuple[int, ...] = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
)

#: Query features ablated for the graded fidelity test. The ceiling ablates
#: every selected feature, so this number only sets the resolution between the
#: two ends.
DEFAULT_ABLATION = 8

#: Stage 1 stops the sweep if neurons lose more than half of the SAE's
#: calibration. Fixed before the run, like ``DECISION``.
KILL = "stop if neurons.fidelity_slope < 0.5 * sae.fidelity_slope"

#: Stage 2 replacement rule. The slope, not the ratio of sums, is the primary
#: fidelity: the ratio is a total-effect ratio, and a per-pair offset can make
#: it negative or huge while the slope stays near one. ``base_share`` is the
#: paper's accounting ratio, so ``+ 0.10`` is an allowance on that scale, not a
#: coverage fraction.
DECISION = (
    "neurons win if fidelity_slope > 0 and fidelity_slope >= 0.8 * "
    "sae.fidelity_slope and base_share <= sae.base_share + 0.10; otherwise "
    "the SAE stays"
)

_ERF = np.frompyfunc(math.erf, 1, 1)


def _gelu(x: np.ndarray) -> np.ndarray:
    """The exact GELU the checkpoint's MLP applies, without a SciPy dependency."""
    scaled = np.asarray(x, dtype=np.float64) / math.sqrt(2.0)
    return 0.5 * np.asarray(x, dtype=np.float64) * (
        1.0 + _ERF(scaled).astype(np.float64)
    )


def _to_numpy(tensor: Any) -> np.ndarray:
    """A torch tensor or a plain array, as ``float64``."""
    if hasattr(tensor, "detach"):
        return tensor.detach().numpy().astype(np.float64)
    return np.asarray(tensor, dtype=np.float64)


def _unit_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows scaled to unit length, plus the norms that were divided out.

    Folding the norm into the activation leaves ``f_i d_i`` unchanged but makes
    ``top-k`` select by contribution rather than by raw activation, which is the
    convention the SAE already follows because its decoder rows are unit-norm.
    """
    norms = np.linalg.norm(matrix, axis=1)
    return matrix / np.maximum(norms, 1e-12)[:, None], norms


@dataclass(frozen=True)
class Basis:
    """A named decomposition of the residual into directions plus activations.

    ``encoder`` maps one residual vector, shape ``(d_model,)`` or
    ``(n_tokens, d_model)``, to that basis's activations. For the neuron arm the
    encoder is bound to the captured forward pass and validates the residual's
    shape rather than recomputing the MLP; see ``neuron_basis``. ``W_dec`` holds
    the unit-norm directions, one per row, in the same order.
    """

    name: str
    W_dec: np.ndarray
    encoder: Callable[[np.ndarray], np.ndarray]
    provenance: Mapping[str, Any]

    def encode(self, x: np.ndarray) -> np.ndarray:
        activations = np.asarray(self.encoder(x), dtype=np.float64)
        if activations.shape[-1] != self.n_features:
            raise ValueError(
                f"{self.name}: encoder returned {activations.shape[-1]} values "
                f"for {self.n_features} directions"
            )
        return activations

    @property
    def n_features(self) -> int:
        return int(self.W_dec.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_dec.shape[1])


def sae_basis(
    dictionary: Q.SparseDictionary, *, source: str | None = None, name: str = "sae"
) -> Basis:
    """The trained dictionary, with its decoder rows made unit-norm.

    The encoder is unchanged; the row norm is folded into its output, so the
    reconstruction ``f W_dec`` is identical to the one ``qk_attribution`` uses.
    """
    unit, norms = _unit_rows(dictionary.W_dec)

    def encoder(x: np.ndarray) -> np.ndarray:
        return dictionary.encode(x) * norms

    return Basis(
        name=name,
        W_dec=unit,
        encoder=encoder,
        provenance={
            "kind": "sae",
            "source": source,
            "activation": dictionary.activation,
            "top_k": None if dictionary.top_k is None else int(dictionary.top_k),
            "n_features": dictionary.n_features,
            "format": dictionary.format,
        },
    )


@dataclass(frozen=True)
class NeuronBlock:
    """One block's MLP write, with the input that block actually saw.

    ``inputs`` is the post-LayerNorm input to ``dense_h_to_4h``, captured from
    the forward pass that produced the residual. It is not a function of the
    residual: the residual is the block's *output*, so recomputing the MLP from
    it would use the wrong vector and skip the block's own LayerNorm.
    """

    block: int
    inputs: np.ndarray  # (n_tokens, d_model), post-MLP LayerNorm
    w_in: np.ndarray  # (d_ff, d_model)
    b_in: np.ndarray  # (d_ff,)
    w_out: np.ndarray  # (d_model, d_ff)
    b_out: np.ndarray | None  # (d_model,), not covered by the directions

    def __post_init__(self) -> None:
        if self.inputs.ndim != 2 or self.inputs.shape[1] != self.w_in.shape[1]:
            raise ValueError(
                f"block {self.block}: inputs have shape {self.inputs.shape}, "
                f"expected (n_tokens, {self.w_in.shape[1]})"
            )
        if self.w_out.shape[1] != self.w_in.shape[0]:
            raise ValueError(
                f"block {self.block}: dense_h_to_4h has {self.w_in.shape[0]} "
                f"outputs but dense_4h_to_h has {self.w_out.shape[1]} inputs"
            )
        if self.b_in.shape != (self.w_in.shape[0],):
            raise ValueError(f"block {self.block}: b_in has shape {self.b_in.shape}")
        if self.b_out is not None and self.b_out.shape != (self.w_out.shape[0],):
            raise ValueError(f"block {self.block}: b_out has shape {self.b_out.shape}")


def neuron_basis(blocks: Sequence[NeuronBlock], *, name: str = "neurons") -> Basis:
    """The MLP write directions of every block that contributed to ``x``.

    The directions are the columns of each block's ``dense_4h_to_h`` weight,
    made unit-norm with the norm folded into the activation. The activation is
    the exact GELU the block applied to its captured input, so the covered part
    is the sum of the MLP writes, not a projection of the final residual.

    The output projection's bias is not a direction and is left in the base;
    the same is true of the embedding and every attention write. That is a
    property of the object, not a defect of the decomposition.

    The encoder is bound to the forward pass that produced the captured inputs:
    the residual alone does not determine what each MLP read, because the
    residual is the block's output. It therefore accepts only the full
    ``(n_tokens, d_model)`` matrix with the captured number of rows.
    """
    ordered = sorted(blocks, key=lambda block: int(block.block))
    if not ordered:
        raise ValueError("neuron_basis needs at least one block")
    if [int(block.block) for block in ordered] != list(range(len(ordered))):
        raise ValueError(
            "neuron blocks must be contiguous from block 0; got "
            f"{[int(block.block) for block in ordered]}"
        )

    units: list[np.ndarray] = []
    activations: list[np.ndarray] = []
    for block in ordered:
        unit, norms = _unit_rows(block.w_out.T)
        units.append(unit)
        activations.append(
            _gelu(block.inputs @ block.w_in.T + block.b_in) * norms
        )
    W_dec = np.concatenate(units, axis=0)
    captured = np.concatenate(activations, axis=1)
    n_tokens = int(captured.shape[0])

    def encoder(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[0] != n_tokens
            or values.shape[1] != W_dec.shape[1]
        ):
            raise ValueError(
                "the neuron basis is bound to the forward pass that produced "
                f"the residual; expected ({n_tokens}, {W_dec.shape[1]}), got "
                f"{values.shape}"
            )
        return captured

    return Basis(
        name=name,
        W_dec=W_dec,
        encoder=encoder,
        provenance={
            "kind": "neurons",
            "activation": "gelu",
            "blocks": [int(block.block) for block in ordered],
            "n_features": int(W_dec.shape[0]),
            "d_model": int(W_dec.shape[1]),
            "output_bias_in_base": any(block.b_out is not None for block in ordered),
        },
    )


def _zeros_if_none(tensor: Any, size: int) -> np.ndarray:
    """A bias tensor, or zeros when the projection has no bias."""
    if tensor is None:
        return np.zeros(int(size), dtype=np.float64)
    return _to_numpy(tensor)


def _require_exact_gelu(model: Any) -> None:
    """Refuse a checkpoint whose MLP does not use exact GELU.

    The neuron basis recomputes the block's activation from its captured input;
    if the block applied a different nonlinearity, the recomputation would not
    be the write the residual actually contains.
    """
    activation = getattr(getattr(model, "config", None), "hidden_act", None)
    if activation != "gelu":
        raise NotImplementedError(
            "the neuron basis implements exact GELU, but this checkpoint uses "
            f"hidden_act={activation!r}; add that activation before measuring"
        )


def _capture_neuron_inputs(
    model: Any, sae_layer: int
) -> tuple[dict[int, np.ndarray], list[Any]]:
    """Hook every contributing block's first MLP projection.

    The hook records the tensor the projection was called with, which is the
    post-LayerNorm input the block actually read. Recomputing the MLP from the
    residual would use the block's output instead of its input and skip the
    LayerNorm, so the capture is not optional.

    Returns the captures and the handles, so the caller can remove them in a
    ``finally`` even if the forward pass raises.
    """
    captured: dict[int, np.ndarray] = {}
    handles: list[Any] = []
    try:
        for layer in range(int(sae_layer) + 1):
            mlp = model.gpt_neox.layers[layer].mlp
            for attribute in ("dense_h_to_4h", "dense_4h_to_h"):
                if not hasattr(mlp, attribute):
                    raise NotImplementedError(
                        f"{type(mlp).__name__} has no {attribute}; this adapter is "
                        "GPT-NeoX specific"
                    )

            def hook(
                module: Any, args: Any, output: Any, layer: int = layer
            ) -> None:
                captured[layer] = _to_numpy(args[0])[0]

            handles.append(mlp.dense_h_to_4h.register_forward_hook(hook))
    except Exception:
        # A partially registered set of hooks would keep writing into a capture
        # the caller never receives; remove them before re-raising.
        for handle in handles:
            handle.remove()
        raise
    return captured, handles


def _neuron_blocks(
    model: Any, captured: Mapping[int, np.ndarray], sae_layer: int
) -> list[NeuronBlock]:
    """Assemble the captured inputs and the projection weights into blocks."""
    blocks: list[NeuronBlock] = []
    for layer in range(int(sae_layer) + 1):
        if layer not in captured:
            raise ValueError(f"no MLP input was captured for block {layer}")
        mlp = model.gpt_neox.layers[layer].mlp
        w_in = _to_numpy(mlp.dense_h_to_4h.weight)
        w_out = _to_numpy(mlp.dense_4h_to_h.weight)
        b_out = (
            None
            if mlp.dense_4h_to_h.bias is None
            else _to_numpy(mlp.dense_4h_to_h.bias)
        )
        blocks.append(
            NeuronBlock(
                block=layer,
                inputs=captured[layer],
                w_in=w_in,
                b_in=_zeros_if_none(mlp.dense_h_to_4h.bias, w_in.shape[0]),
                w_out=w_out,
                b_out=b_out,
            )
        )
    return blocks


def random_basis(
    d_model: int, n_features: int, *, seed: int = 0, name: str = "random"
) -> Basis:
    """Gaussian unit directions, used as a control rather than a candidate.

    The coefficients are the inner products ``f_i = <d_i, x>``. They are not
    least-squares coefficients, because independent Gaussian directions are not
    orthogonal; the decomposition is still exact because the base absorbs
    ``x - sum_i f_i d_i``. The arm tests how much fidelity any sufficiently wide
    set of directions can buy, not whether those directions are readable.
    """
    if n_features < 1 or d_model < 1:
        raise ValueError("random_basis needs positive sizes")
    rng = np.random.default_rng(seed)
    unit, _ = _unit_rows(rng.standard_normal((int(n_features), int(d_model))))

    def encoder(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ unit.T

    return Basis(
        name=name,
        W_dec=unit,
        encoder=encoder,
        provenance={
            "kind": "random",
            "seed": int(seed),
            "n_features": int(n_features),
        },
    )


def coordinate_basis(d_model: int, *, name: str = "coordinates") -> Basis:
    """The standard basis, used as the harness sanity check.

    At full width it reconstructs ``x`` exactly. LayerNorm's Jacobian nearly
    annihilates ``x`` (the norm is scale-invariant up to ``eps``), so the
    feature terms sum to a small epsilon term and the base carries almost the
    whole score. That is the expected outcome, not a failure: it checks that
    ``base + sum(features) == score`` closes and that the full-ablation
    prediction is near zero. The arm is run at full width and is not a top-k
    competitor.
    """
    identity = np.eye(int(d_model), dtype=np.float64)

    def encoder(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    return Basis(
        name=name,
        W_dec=identity,
        encoder=encoder,
        provenance={"kind": "coordinates", "n_features": int(d_model)},
    )


def _mask_top_k(activations: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` largest-magnitude entries and exclude the rest.

    ``token_buckets`` selects by descending value, not by magnitude, so a signed
    basis would spend its budget on the largest positives. Marking every other
    entry ``-inf`` makes that selection agree with magnitude without changing
    the function: exactly ``k`` entries stay finite, and they are the top-k.
    """
    values = np.asarray(activations, dtype=np.float64)
    count = min(int(k), values.size)
    if count < 1:
        raise ValueError("top-k must keep at least one feature")
    chosen = np.argpartition(-np.abs(values), count - 1)[:count]
    masked = np.full(values.size, -np.inf)
    masked[chosen] = values[chosen]
    return masked


@dataclass(frozen=True)
class Geometry:
    """The fixed model tensors every arm shares."""

    ln_weight: np.ndarray
    ln_bias: np.ndarray
    ln_eps: float
    w_q: np.ndarray  # (n_heads, head_dim, d_model)
    b_q: np.ndarray  # (n_heads, head_dim)
    w_k: np.ndarray
    b_k: np.ndarray
    freqs: np.ndarray
    rotary_dim: int

    @property
    def n_heads(self) -> int:
        return int(self.w_q.shape[0])


def _score(q: np.ndarray, k: np.ndarray, delta: int, geometry: Geometry) -> float:
    return float(
        Q.rotated_dot(
            q[None, :], k[None, :], int(delta), geometry.freqs, geometry.rotary_dim
        )[0, 0]
    )


def _ablation_score(
    x_m: np.ndarray,
    buckets_m: Q.TokenBuckets,
    basis: Basis,
    geometry: Geometry,
    head: int,
    k_n: np.ndarray,
    delta: int,
    subset: np.ndarray,
) -> float:
    """The query score after removing ``subset`` features from the residual.

    The features are removed from ``x``, not from the projected vector, so the
    LayerNorm is recomputed on the ablated input. That is the counterfactual the
    linear prediction is compared against.
    """
    removed = (
        buckets_m.activations[subset, None] * basis.W_dec[buckets_m.indices[subset]]
    ).sum(axis=0)
    h, _ = Q.layernorm_affine(
        x_m - removed, geometry.ln_weight, geometry.ln_bias, geometry.ln_eps
    )
    q_ablated = h @ geometry.w_q[head].T + geometry.b_q[head]
    return _score(q_ablated, k_n, delta, geometry)


def _ratio(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    scale: float | None = None,
    min_fraction: float = 1e-3,
) -> float | None:
    """``sum(actual) / sum(predicted)``, or ``None`` when it is not identified.

    This is a total-effect ratio, not a per-pair fidelity. It is meaningful only
    when the denominator is not the residue of cancellation; ``min_fraction``
    rejects a prediction whose net is below a thousandth of ``scale`` (the score
    magnitude the prediction is supposed to explain). Even then a nonzero mean
    offset can make the ratio negative while the slope through the origin is
    near one, so the slope is the fidelity statistic and this ratio is kept for
    continuity with the paper's accounting. A non-finite input is refused rather
    than propagated into the JSON writer.
    """
    values = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(observed))):
        return None
    total = float(values.sum())
    magnitude = float(np.abs(values).sum())
    if magnitude < 1e-12:
        return None
    if scale is not None and float(scale) <= 0.0:
        return None
    if abs(total) < 1e-9 * magnitude:
        return None
    if scale is not None and abs(total) < float(min_fraction) * abs(scale):
        return None
    return float(observed.sum() / total)


def _identified(predicted_sq: float, n_pairs: int, scale: float) -> bool:
    """Whether a prediction varies enough to fit a slope or a correlation.

    ``scale`` is the summed score magnitude over the same pairs. A prediction
    whose RMS is below a thousandth of the mean score magnitude is effectively
    zero: a slope through the origin would divide by that noise, and a
    correlation would describe it. The full-ablation prediction of a complete
    basis is exactly this case. A non-finite prediction or scale is refused for
    the same reason.
    """
    if predicted_sq <= 0.0 or n_pairs < 1 or scale <= 0.0:
        return False
    if not (math.isfinite(predicted_sq) and math.isfinite(scale)):
        return False
    return math.sqrt(predicted_sq / n_pairs) >= 1e-3 * (scale / n_pairs)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` with fewer than three pairs.

    With two points the correlation is plus or minus one by construction and
    says nothing about agreement. Non-finite input is refused for the same
    reason as in ``_ratio``.
    """
    if len(left) < 3:
        return None
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _fidelity(
    x: np.ndarray,
    buckets: Sequence[Q.TokenBuckets],
    basis: Basis,
    geometry: Geometry,
    head: int,
    deltas: Sequence[int],
    n_pairs: int,
    ablation: int,
    seed: int,
) -> dict[str, Any]:
    """Ablate query features and compare the predicted change with the actual.

    ``query_features`` is the additive share of the observed score carried by
    each feature, so removing a set should lower the score by the sum of those
    shares if the LayerNorm were affine. It is not, and the ceiling run measures
    how much that costs: the graded set is the ``ablation`` largest-magnitude
    shares, while the ceiling set is every selected query feature. The graded
    slope is only interpretable next to the ceiling slope, and a complete basis
    makes the ceiling prediction near zero, so both ceiling statistics are
    undefined there.

    ``slope`` is a calibration coefficient through the origin, not a fraction
    in ``[0, 1]``: 1.0 means the linear prediction has the right scale on
    average, and a value above 1 means the ablation moves the score more than
    the linear share predicts.
    """
    rng = np.random.default_rng(seed)
    n_tokens = int(x.shape[0])
    predicted: list[float] = []
    actual: list[float] = []
    ceiling_predicted: list[float] = []
    ceiling_actual: list[float] = []
    score_scale = 0.0
    graded_features = 0
    ceiling_features = 0

    for delta in deltas:
        pairs = Q.sample_pairs(n_tokens, int(delta), int(n_pairs), rng)
        for m, n in pairs:
            m, n = int(m), int(n)
            bm = buckets[m]
            bn = buckets[n]
            terms = Q.attribution_terms(
                bm, bn, n - m, geometry.freqs, geometry.rotary_dim
            )
            contribution = terms.query_features
            if contribution.size == 0:
                continue
            score_scale += abs(terms.score)
            k_n = bn.base_k + bn.feature_k.sum(axis=0)
            order = np.argsort(-np.abs(contribution))
            graded = order[: min(int(ablation), order.size)]
            graded_features = int(graded.size)
            ceiling_features = int(order.size)

            graded_score = _ablation_score(
                x[m], bm, basis, geometry, head, k_n, n - m, graded
            )
            predicted.append(-float(contribution[graded].sum()))
            actual.append(float(graded_score - terms.score))

            ceiling_score = _ablation_score(
                x[m], bm, basis, geometry, head, k_n, n - m, order
            )
            ceiling_predicted.append(-float(contribution[order].sum()))
            ceiling_actual.append(float(ceiling_score - terms.score))

    p = np.asarray(predicted, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    cp = np.asarray(ceiling_predicted, dtype=np.float64)
    ca = np.asarray(ceiling_actual, dtype=np.float64)
    p_sq = float((p**2).sum())
    a_sq = float((a**2).sum())
    pa = float((p * a).sum())
    cp_sq = float((cp**2).sum())
    ca_sq = float((ca**2).sum())
    cpa = float((cp * ca).sum())
    graded_identified = _identified(p_sq, int(p.size), score_scale)
    ceiling_identified = _identified(cp_sq, int(cp.size), score_scale)
    return {
        "n_pairs": int(p.size),
        "graded_features": graded_features,
        "ceiling_features": ceiling_features,
        "score_scale": float(score_scale),
        "predicted_sum": float(p.sum()),
        "actual_sum": float(a.sum()),
        "predicted_sq_sum": p_sq,
        "actual_sq_sum": a_sq,
        "predicted_actual_sum": pa,
        "ratio": _ratio(actual, predicted, scale=score_scale),
        "slope": pa / p_sq if graded_identified else None,
        "correlation": (
            _correlation(predicted, actual) if graded_identified else None
        ),
        "ceiling_predicted_sum": float(cp.sum()),
        "ceiling_actual_sum": float(ca.sum()),
        "ceiling_predicted_sq_sum": cp_sq,
        "ceiling_actual_sq_sum": ca_sq,
        "ceiling_predicted_actual_sum": cpa,
        "ceiling_ratio": _ratio(ceiling_actual, ceiling_predicted, scale=score_scale),
        "ceiling_slope": cpa / cp_sq if ceiling_identified else None,
        "ceiling_correlation": (
            _correlation(ceiling_predicted, ceiling_actual)
            if ceiling_identified
            else None
        ),
    }


def measure_basis(
    basis: Basis,
    x: np.ndarray,
    geometry: Geometry,
    deltas: Sequence[int],
    n_pairs: int,
    seed: int,
    top_k: int,
    heads: Sequence[int] | None = None,
    sweep_deltas: Sequence[int] | None = SWEEP_DELTAS,
    ablation: int = DEFAULT_ABLATION,
) -> list[dict[str, Any]]:
    """Run the attribution and the fidelity probe for every requested head.

    The realized-distance grid is clipped to the passage: a distance larger
    than ``n_tokens - 1`` cannot be sampled, and keeping it would publish an
    empty bin as if it had been measured. The content-fixed sweep is not
    clipped, because it evaluates the same content pair at artificial
    distances rather than sampling tokens that far apart.
    """
    if int(top_k) < 1:
        raise ValueError("top-k must be positive")
    if int(n_pairs) < 1:
        raise ValueError("pairs per distance must be positive")
    if int(ablation) < 1:
        raise ValueError("ablation must select at least one feature")
    if x.ndim != 2 or x.shape[1] != basis.d_model:
        raise ValueError(
            f"{basis.name}: residual has shape {x.shape}, expected "
            f"(n_tokens, {basis.d_model})"
        )
    activations = basis.encode(x)
    masked = np.stack(
        [_mask_top_k(activations[token], top_k) for token in range(x.shape[0])]
    )
    grid = Q.distance_grid(int(x.shape[0]) - 1, deltas)
    if grid.size == 0:
        raise ValueError(
            f"{basis.name}: none of the requested distances fits a passage of "
            f"{x.shape[0]} tokens"
        )
    sweep = (
        None
        if sweep_deltas is None
        else np.unique(np.array([int(d) for d in sweep_deltas], dtype=np.int64))
    )
    if sweep is not None and sweep.size and np.any(sweep <= 0):
        raise ValueError("sweep distances must be positive")
    selected = list(range(geometry.n_heads)) if heads is None else [int(h) for h in heads]
    if not selected:
        raise ValueError("at least one head is required")
    if len(set(selected)) != len(selected):
        raise ValueError("head indices must be unique")

    rows: list[dict[str, Any]] = []
    for head in selected:
        if head < 0 or head >= geometry.n_heads:
            raise ValueError(f"head {head} is outside [0, {geometry.n_heads})")
        buckets = [
            Q.token_buckets(
                x[token],
                masked[token],
                basis,
                geometry.ln_weight,
                geometry.ln_bias,
                geometry.ln_eps,
                geometry.w_q[head],
                geometry.b_q[head],
                geometry.w_k[head],
                geometry.b_k[head],
                top_k,
            )
            for token in range(x.shape[0])
        ]
        measured = Q.measure_head(
            buckets,
            geometry.freqs,
            geometry.rotary_dim,
            grid,
            int(n_pairs),
            np.random.default_rng(seed + head),
            basis.n_features,
            sweep_deltas=sweep,
        )
        feature_metrics = (
            Q.profile_metrics(measured["sweep"]["feature"][1])
            if "feature" in measured["sweep"]
            else {}
        )
        pair_metrics = (
            Q.profile_metrics(measured["sweep"]["pair"][1])
            if "pair" in measured["sweep"]
            else {}
        )
        rows.append(
            {
                "head": int(head),
                "deltas": [int(d) for d in grid],
                "sweep_deltas": (
                    None if sweep is None else [int(d) for d in sweep]
                ),
                "base_share_mean": _optional_float(measured["base_share_mean"]),
                "top_k_share_mean": _optional_float(measured["top_k_share_mean"]),
                "max_relative_exactness_error": float(
                    measured["max_relative_exactness_error"]
                ),
                "sweep_feature_sign_flip_fraction": _optional_float(
                    feature_metrics.get("sign_flip_fraction")
                ),
                "sweep_pair_sign_flip_fraction": _optional_float(
                    pair_metrics.get("sign_flip_fraction")
                ),
                "fidelity": _fidelity(
                    x,
                    buckets,
                    basis,
                    geometry,
                    head,
                    grid,
                    int(n_pairs),
                    int(ablation),
                    seed + 10_000 + head,
                ),
            }
        )
    return rows


def _optional_float(value: Any) -> float | None:
    """A finite float, or ``None`` for a missing or non-finite measurement."""
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool the per-head rows; ``None`` stays ``None`` rather than becoming NaN.

    Fidelity is pooled as a slope through the origin over all pairs, not as a
    mean of per-head ratios: a head whose predictions cancel has a huge ratio
    and would otherwise dominate the average. The ratio of sums and the pooled
    correlation are reported alongside it because the slope alone does not say
    whether the prediction tracks the actual change or only its scale.
    """

    def mean_of(getter: Callable[[Mapping[str, Any]], Any]) -> float | None:
        values = [getter(row) for row in rows]
        finite = [float(value) for value in values if value is not None]
        return float(np.mean(finite)) if finite else None

    def total(getter: Callable[[Mapping[str, Any]], Any]) -> float:
        return float(sum(float(getter(row)) for row in rows))

    n_pairs = int(sum(int(row["fidelity"]["n_pairs"]) for row in rows))
    p_sum = total(lambda row: row["fidelity"]["predicted_sum"])
    a_sum = total(lambda row: row["fidelity"]["actual_sum"])
    p_sq = total(lambda row: row["fidelity"]["predicted_sq_sum"])
    a_sq = total(lambda row: row["fidelity"]["actual_sq_sum"])
    pa = total(lambda row: row["fidelity"]["predicted_actual_sum"])
    cp_sum = total(lambda row: row["fidelity"]["ceiling_predicted_sum"])
    ca_sum = total(lambda row: row["fidelity"]["ceiling_actual_sum"])
    cp_sq = total(lambda row: row["fidelity"]["ceiling_predicted_sq_sum"])
    ca_sq = total(lambda row: row["fidelity"]["ceiling_actual_sq_sum"])
    cpa = total(lambda row: row["fidelity"]["ceiling_predicted_actual_sum"])
    scale = total(lambda row: row["fidelity"]["score_scale"])

    def pooled_ratio(actual_sum: float, predicted_sum: float, predicted_sq: float) -> float | None:
        if predicted_sq <= 0.0 or scale <= 0.0:
            return None
        if abs(predicted_sum) < 1e-9 * math.sqrt(n_pairs * predicted_sq):
            return None
        if abs(predicted_sum) < 1e-3 * scale:
            return None
        return actual_sum / predicted_sum

    def pooled_slope(cross: float, predicted_sq: float) -> float | None:
        if not _identified(predicted_sq, n_pairs, scale):
            return None
        return cross / predicted_sq

    def pooled_correlation(
        predicted_sum: float,
        actual_sum: float,
        predicted_sq: float,
        actual_sq: float,
        cross: float,
    ) -> float | None:
        if not _identified(predicted_sq, n_pairs, scale):
            return None
        if n_pairs < 3:
            return None
        var_p = n_pairs * predicted_sq - predicted_sum**2
        var_a = n_pairs * actual_sq - actual_sum**2
        if var_p <= 0.0 or var_a <= 0.0:
            return None
        cov = n_pairs * cross - predicted_sum * actual_sum
        return cov / math.sqrt(var_p * var_a)

    return {
        "n_heads": len(rows),
        "n_pairs": n_pairs,
        "base_share_mean": mean_of(lambda row: row["base_share_mean"]),
        "top_k_share_mean": mean_of(lambda row: row["top_k_share_mean"]),
        "max_relative_exactness_error": max(
            (float(row["max_relative_exactness_error"]) for row in rows),
            default=0.0,
        ),
        "sweep_feature_sign_flip_fraction": mean_of(
            lambda row: row["sweep_feature_sign_flip_fraction"]
        ),
        "sweep_pair_sign_flip_fraction": mean_of(
            lambda row: row["sweep_pair_sign_flip_fraction"]
        ),
        "fidelity_ratio": pooled_ratio(a_sum, p_sum, p_sq),
        "fidelity_slope": pooled_slope(pa, p_sq),
        "fidelity_correlation": pooled_correlation(p_sum, a_sum, p_sq, a_sq, pa),
        "ceiling_ratio": pooled_ratio(ca_sum, cp_sum, cp_sq),
        "ceiling_slope": pooled_slope(cpa, cp_sq),
        "ceiling_correlation": pooled_correlation(
            cp_sum, ca_sum, cp_sq, ca_sq, cpa
        ),
    }


def verdict(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply ``DECISION`` to the measured arms, without re-reading the numbers.

    The rule is evaluated in one place so the result cannot be assembled from a
    more flattering subset of the table. ``None`` is a refusal to decide, not a
    pass: a missing fidelity or base share means the run cannot support the
    replacement.
    """
    by_name = {str(arm["name"]): arm for arm in arms}
    if "sae" not in by_name or "neurons" not in by_name:
        return {
            "applicable": False,
            "reason": "both the sae and neurons arms are required",
            "rule": DECISION,
            "kill": KILL,
        }
    sae = by_name["sae"]["aggregate"]
    neurons = by_name["neurons"]["aggregate"]
    fidelity_sae = sae.get("fidelity_slope")
    fidelity_neurons = neurons.get("fidelity_slope")
    base_sae = sae.get("base_share_mean")
    base_neurons = neurons.get("base_share_mean")
    required = (fidelity_sae, fidelity_neurons, base_sae, base_neurons)
    if any(
        value is None or not math.isfinite(float(value)) for value in required
    ):
        return {
            "applicable": False,
            "reason": "a required fidelity or base-share metric is undefined",
            "rule": DECISION,
            "kill": KILL,
        }
    if float(fidelity_sae) <= 0.0:
        return {
            "applicable": False,
            "reason": "the SAE fidelity slope is not positive; the rule is undefined",
            "rule": DECISION,
            "kill": KILL,
        }
    kill = float(fidelity_neurons) < 0.5 * float(fidelity_sae)
    fidelity_ok = float(fidelity_neurons) > 0.0 and float(
        fidelity_neurons
    ) >= 0.8 * float(fidelity_sae)
    base_ok = float(base_neurons) <= float(base_sae) + 0.10
    if kill:
        reason = "stage-1 kill: fidelity slope below half the SAE's"
    elif not fidelity_ok:
        reason = "fidelity slope below 0.8x the SAE's"
    elif not base_ok:
        reason = "base share exceeds the SAE's by more than 0.10"
    else:
        reason = "replacement criterion met"
    return {
        "applicable": True,
        "neurons_win": bool(fidelity_ok and base_ok and not kill),
        "stage1_kill": bool(kill),
        "fidelity_ok": bool(fidelity_ok),
        "base_ok": bool(base_ok),
        "reason": reason,
        "rule": DECISION,
        "kill": KILL,
    }


def bakeoff(
    model_id: str = Q.DEFAULT_MODEL_ID,
    sae_path: str | Path | None = None,
    sae_layer: int = Q.DEFAULT_SAE_LAYER,
    text: str = Q.DEFAULT_TEXT,
    top_k: int = Q.DEFAULT_TOP_K,
    n_pairs: int = Q.DEFAULT_PAIRS_PER_DELTA,
    deltas: Sequence[int] = Q.DEFAULT_DELTAS,
    seed: int = 0,
    heads: Sequence[int] | None = None,
    bases: Sequence[str] = BASIS_NAMES,
    random_features: int | None = None,
    sweep_deltas: Sequence[int] | None = SWEEP_DELTAS,
    ablation: int = DEFAULT_ABLATION,
) -> dict[str, Any]:
    """Measure every requested basis on one checkpoint, layer and passage.

    The model is loaded once and the residual is captured once; every arm sees
    the same ``x``, the same heads and the same sampled token pairs. The
    realized-distance grid is clipped to the passage, and both the requested
    and the realized grids are recorded so a short passage cannot be read as a
    full sweep.

    The random control's width follows the SAE when the SAE arm is present,
    otherwise the neuron arm, otherwise the model. It is therefore not
    width-matched to the neuron candidate: its job is to test how much fidelity
    the incumbent's width can buy, not to be a fair alternative to neurons.
    """
    requested = [str(name) for name in bases]
    if not requested:
        raise ValueError("at least one basis is required")
    if len(set(requested)) != len(requested):
        raise ValueError(f"bases must be unique; got {requested}")
    unknown = sorted(set(requested) - set(BASIS_NAMES))
    if unknown:
        raise ValueError(f"unknown bases {unknown}; choose from {list(BASIS_NAMES)}")
    if "sae" in requested and sae_path is None:
        raise ValueError(
            "the sae arm needs --sae; drop it from --bases to run the baselines "
            "on their own"
        )
    if "sae" in requested and not Path(str(sae_path)).is_file():
        raise FileNotFoundError(
            f"the sae dictionary {sae_path!s} does not exist; the arm needs a "
            "local safetensors file"
        )
    if int(top_k) < 1:
        raise ValueError("top-k must be positive")
    if int(n_pairs) < 1:
        raise ValueError("pairs per distance must be positive")
    if int(ablation) < 1:
        raise ValueError("ablation must select at least one feature")
    if int(sae_layer) < 0:
        raise ValueError("sae layer must be non-negative")
    if not any(int(distance) > 0 for distance in deltas):
        raise ValueError("at least one positive distance is required")
    if heads is not None:
        requested_heads = [int(head) for head in heads]
        if not requested_heads:
            raise ValueError("at least one head is required")
        if len(set(requested_heads)) != len(requested_heads):
            raise ValueError("head indices must be unique")
        if any(head < 0 for head in requested_heads):
            raise ValueError("head indices must be non-negative")
    if random_features is not None and int(random_features) < 1:
        raise ValueError("random features must be positive")
    if sweep_deltas is not None and any(
        int(distance) <= 0 for distance in sweep_deltas
    ):
        raise ValueError("sweep distances must be positive")

    loaded = RM.load_model(model_id, seed=seed)
    model, tokenizer = loaded.model, loaded.tokenizer
    if not hasattr(model, "gpt_neox"):
        raise NotImplementedError(
            f"{type(model).__name__} is not a GPT-NeoX model; the layer "
            "addressing and the MLP write directions here are GPT-NeoX specific"
        )
    n_blocks = len(model.gpt_neox.layers)
    attn_index = Q._attention_index(n_blocks, sae_layer)
    captured: dict[int, np.ndarray] = {}
    handles: list[Any] = []
    if "neurons" in requested:
        _require_exact_gelu(model)
        captured, handles = _capture_neuron_inputs(model, sae_layer)
    try:
        run = Q._forward(model, tokenizer, text, attn_index)
    finally:
        for handle in handles:
            handle.remove()

    x = run["hidden_states"][int(sae_layer) + 1][0].numpy().astype(np.float64)
    n_tokens = int(x.shape[0])
    if n_tokens < 8:
        raise ValueError(f"the passage tokenized to {n_tokens} tokens, too few to bin")
    grid = Q.distance_grid(n_tokens - 1, deltas)
    if grid.size == 0:
        raise ValueError(
            f"none of the requested distances fits a passage of {n_tokens} tokens"
        )

    attn = RM.attention_modules(model)[attn_index]
    ln = model.gpt_neox.layers[attn_index].input_layernorm
    ln_weight = _to_numpy(ln.weight)
    ln_bias = _to_numpy(ln.bias)
    attn_in = run["attn_in"][0].numpy().astype(np.float64)
    ln_residual = float(
        np.abs(Q.layernorm(x, ln_weight, ln_bias, float(ln.eps)) - attn_in).max()
    )
    # Relative, because the model computed the norm in float32: an absolute
    # threshold would refuse a correct run on a wide residual.
    ln_scale = max(float(np.abs(attn_in).max()), 1e-12)
    if ln_residual / ln_scale > 1e-3:
        raise ValueError(
            f"the captured residual differs from the attention input by "
            f"{ln_residual / ln_scale:.2e} relative; refusing to attribute a "
            "different vector"
        )

    hidden_size = int(x.shape[1])
    n_heads = int(model.config.num_attention_heads)
    head_dim = hidden_size // n_heads
    w_q, b_q, w_k, b_k = Q.head_projections(attn, hidden_size, n_heads, head_dim)
    cfgv = RM.validate_config(model)
    rotary_dim = int(cfgv["rope_dim"])
    freqs = R.inv_freq(rotary_dim, float(cfgv["rope_theta_declared"]))
    geometry = Geometry(
        ln_weight=ln_weight,
        ln_bias=ln_bias,
        ln_eps=float(ln.eps),
        w_q=w_q,
        b_q=b_q,
        w_k=w_k,
        b_k=b_k,
        freqs=freqs,
        rotary_dim=rotary_dim,
    )

    dictionary: Q.SparseDictionary | None = None
    if "sae" in requested:
        dictionary = Q.SparseDictionary.load_safetensors(sae_path)
        if dictionary.d_model != hidden_size:
            raise ValueError(
                f"the dictionary has width {dictionary.d_model} but the checkpoint "
                f"has width {hidden_size}"
            )

    neuron_width = (int(sae_layer) + 1) * int(
        getattr(model.config, "intermediate_size", hidden_size)
    )
    built: list[tuple[Basis, int, str, str | None]] = []
    for name in requested:
        if name == "sae":
            assert dictionary is not None
            built.append(
                (sae_basis(dictionary, source=str(sae_path)), int(top_k), "candidate", None)
            )
        elif name == "neurons":
            built.append(
                (
                    neuron_basis(_neuron_blocks(model, captured, sae_layer)),
                    int(top_k),
                    "candidate",
                    None,
                )
            )
        elif name == "random":
            if random_features is not None:
                width = int(random_features)
            elif dictionary is not None:
                width = dictionary.n_features
            elif "neurons" in requested:
                width = neuron_width
            else:
                width = hidden_size
            built.append(
                (random_basis(hidden_size, width, seed=seed), int(top_k), "control", None)
            )
        elif name == "coordinates":
            built.append(
                (
                    coordinate_basis(hidden_size),
                    hidden_size,
                    "sanity",
                    "full width; base carries the score by LayerNorm scale "
                    "invariance, not a coverage failure",
                )
            )

    arms = []
    for basis, arm_top_k, role, note in built:
        rows = measure_basis(
            basis,
            x,
            geometry,
            grid,
            n_pairs,
            seed,
            arm_top_k,
            heads=heads,
            sweep_deltas=sweep_deltas,
            ablation=ablation,
        )
        arm: dict[str, Any] = {
            "name": basis.name,
            "role": role,
            "top_k": int(min(arm_top_k, basis.n_features)),
            "n_features": basis.n_features,
            "d_model": basis.d_model,
            "provenance": dict(basis.provenance),
            "heads": rows,
            "aggregate": _aggregate(rows),
        }
        if note is not None:
            arm["note"] = note
        arms.append(arm)

    return {
        "schema": SCHEMA,
        "model_id": model_id,
        "sae_layer": int(sae_layer),
        "attention_layer": attn_index,
        "text": text,
        "n_tokens": n_tokens,
        "top_k": int(top_k),
        "pairs_per_delta": int(n_pairs),
        "seed": int(seed),
        "deltas": [int(d) for d in grid],
        "deltas_requested": [
            int(d) for d in np.unique(np.array(list(deltas), dtype=np.int64))
        ],
        "sweep_deltas": (
            None
            if sweep_deltas is None
            else [int(d) for d in np.unique(np.array(list(sweep_deltas), dtype=np.int64))]
        ),
        "rotary_dim": rotary_dim,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "measured_heads": (
            [int(row["head"]) for row in arms[0]["heads"]] if arms else []
        ),
        "ln_residual": ln_residual,
        "ablation_features": int(ablation),
        "decision": DECISION,
        "kill": KILL,
        "verdict": verdict(arms),
        "bases": arms,
    }


def write_results(data: dict[str, Any], path: Path | None = None) -> Path:
    out = (
        path
        or Path(__file__).resolve().parent.parent.parent
        / "results"
        / "basis_bakeoff.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Basis bake-off: compare the SAE, neuron, random and coordinate "
            "decompositions on one checkpoint and passage."
        ),
        epilog=(
            "Stage 1 is --heads 0,1 --pairs 8 --no-sweep; read "
            "verdict.stage1_kill before paying for the full sweep."
        ),
    )
    parser.add_argument("--model", default=Q.DEFAULT_MODEL_ID)
    parser.add_argument(
        "--sae",
        type=Path,
        default=None,
        help="safetensors dictionary for the sae arm; not vendored",
    )
    parser.add_argument("--sae-layer", type=int, default=Q.DEFAULT_SAE_LAYER)
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help=(
            "UTF-8 passage to attribute; default is qk_attribution.DEFAULT_TEXT, "
            "which is 320 tokens, so the realized grid stops there"
        ),
    )
    parser.add_argument(
        "--bases",
        default=",".join(BASIS_NAMES),
        help="comma-separated arms: sae,neurons,random,coordinates (default all four)",
    )
    parser.add_argument(
        "--heads", default=None, help="comma-separated head indices (default all)"
    )
    parser.add_argument("--top-k", type=int, default=Q.DEFAULT_TOP_K)
    parser.add_argument("--pairs", type=int, default=Q.DEFAULT_PAIRS_PER_DELTA)
    parser.add_argument(
        "--random-features",
        type=int,
        default=None,
        help=(
            "width of the random arm (default: the SAE width, the neuron width, "
            "or the model width)"
        ),
    )
    parser.add_argument("--ablation", type=int, default=DEFAULT_ABLATION)
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="skip the content-fixed sweep; the sign-flip column is then null",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    bases = [part.strip() for part in str(args.bases).split(",") if part.strip()]
    heads = (
        None
        if args.heads is None
        else [int(part) for part in str(args.heads).split(",") if part.strip()]
    )
    text = (
        Q.DEFAULT_TEXT
        if args.text_file is None
        else args.text_file.read_text(encoding="utf-8")
    )
    data = bakeoff(
        model_id=args.model,
        sae_path=args.sae,
        sae_layer=args.sae_layer,
        text=text,
        top_k=args.top_k,
        n_pairs=args.pairs,
        seed=args.seed,
        heads=heads,
        bases=bases,
        random_features=args.random_features,
        sweep_deltas=None if args.no_sweep else SWEEP_DELTAS,
        ablation=args.ablation,
    )
    path = write_results(data, args.out)
    print(
        f"{'basis':<12} {'role':<10} {'features':>8} {'k':>5} {'pairs':>6} "
        f"{'base':>6} {'top-k':>6} {'slope':>7} {'ratio':>7} {'cslope':>7} "
        f"{'flips':>6}"
    )
    for arm in data["bases"]:
        aggregate = arm["aggregate"]

        def show(value: Any) -> str:
            return "  --  " if value is None else f"{float(value):.3f}"

        print(
            f"{arm['name']:<12} {arm['role']:<10} {arm['n_features']:>8} "
            f"{arm['top_k']:>5} {aggregate['n_pairs']:>6} "
            f"{show(aggregate['base_share_mean'])} "
            f"{show(aggregate['top_k_share_mean'])} "
            f"{show(aggregate['fidelity_slope'])} "
            f"{show(aggregate['fidelity_ratio'])} "
            f"{show(aggregate['ceiling_slope'])} "
            f"{show(aggregate['sweep_feature_sign_flip_fraction'])}"
        )
    result = data["verdict"]
    if result.get("applicable"):
        print(f"verdict: {'neurons win' if result['neurons_win'] else 'SAE stays'}")
        print(f"reason: {result['reason']}")
    else:
        print(f"verdict: not applicable ({result['reason']})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
