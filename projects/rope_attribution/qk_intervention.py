"""Does the attribution predict what happens when you remove the thing it names?

The gap this closes
-------------------
Every number in the paper up to here is a property of a *decomposition*. A
skeptical reader is entitled to ask the question that separates an attribution
from a description: if you take the unit the attribution ranks highest and remove
it from the computation, does the model do what the attribution said it would?

This module answers that for the one intervention that is exact under RoPE.
Scaling the two channels of rotary pair ``j`` in the **query projection** by
``alpha`` scales that pair's contribution to *every* score by ``alpha``:

```text
c_j(delta) = alpha * (A_j cos(delta theta_j) + B_j sin(delta theta_j)).
```

Nothing else changes. The value projection, the residual stream, the MLP, the
other pairs and the unrotated tail are untouched, and the edit is applied before
RoPE, so the rotation itself is not disturbed. ``alpha = 0`` is an exact ablation
of that pair and of nothing else.

The experiment
--------------
For a chosen ``(layer, head, query)`` the head attends most strongly to some key
at distance ``delta*``. That cell is selected from the baseline forward pass
alone, with the diagonal excluded and with saturated rows rejected --- a head
whose attention is already ``1`` cannot respond to a score edit, so measuring it
would only report the saturation. Two attributions propose which pair to remove:

* **exact** --- the pair with the largest contribution ``c_j(delta*)`` at the
  distance actually in use;
* **surrogate** --- the pair with the largest least-squares constant ``c_hat_j``
  over the distances the query can see, which is what a position-free attribution
  table contains.

Each is applied by a hook on the query projection and the model is run. Three
numbers are then reported, and they answer different questions.

*Suppression.* ``achieved / intended``, where *intended* is what the attribution
believed it was removing and *achieved* is the true contribution of the pairs it
removed, both at ``delta*``. It is the currency of the intervention: ``1`` is a
faithful summary, ``0`` removes nothing that mattered, and **negative means the
edit amplified the score it was meant to suppress**. The exact rule scores ``1``
by construction --- it ranks by the very quantity it removes --- so it is an
oracle here and is reported as one, not as a competitor.

*Prediction.* The post-softmax attention change is predicted in closed form and
compared with the real forward pass. This is a harness identity: it must hold to
float32 precision for whichever pairs were removed. It validates that the
per-pair decomposition plus softmax is the whole mechanism, and it is not a
comparison between the two rules.

*Movement.* The measured ``|delta attention|`` at the target key, so that a small
effect is distinguishable from a saturated cell that could not move.

So the reported quantity is a **measured error in the currency of an
intervention**, not a win over a baseline.

Why the ablation is not a strawman
----------------------------------
It is the smallest intervention the object supports, applied at the same
magnitude for both rules (one pair, ``alpha = 0``), on cells the model actually
uses (the head's own strongest attention). The comparison is between *which pair
each attribution names*, with everything else held fixed. A reviewer can rerun it
with any budget.

What this module does not claim
-------------------------------
It does not claim a language-model accuracy effect: single-head, single-pair
edits leave the logits nearly unchanged, and the module reports that rather than
dressing it up. It does not claim the attention pattern *should* be explained by
one pair; the measure is achieved-versus-intended suppression in the units the
intervention spends. It does not address rerouting: this is one forward pass,
not a fine-tune, so the claim is immediate causal contribution, not necessity
after adaptation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from rope_attribution import qk_attribution as QA
from rope_attribution import real_model as RM
from rope_attribution.experiments import pair_terms
from rope_attribution.qk_attribution import rotated_dot
from rope_attribution.rope import inv_freq

__all__ = [
    "BUDGETS",
    "DEFAULT_MODEL_IDS",
    "pair_channels",
    "scale_query_pairs",
    "score_matrix",
    "attention_probs",
    "pair_contribution_matrix",
    "predict_attention",
    "select_pairs",
    "measure_cell",
    "measure_model",
    "run_all",
    "write_results",
    "main",
]

DEFAULT_MODEL_IDS: tuple[str, ...] = RM.DEFAULT_MODEL_IDS

# How many pairs an intervention may remove. One is the smallest non-trivial
# edit; the artefact records it so a reader cannot mistake it for a sweep.
BUDGETS: tuple[int, ...] = (1,)

# Number of texts and cells. One 320-token passage, and one cell per head at
# most, so the cells describe many heads rather than many queries of one head.
CELLS_PER_PROMPT = 32
TEXTS: tuple[str, ...] = (QA.DEFAULT_TEXT,)

# A cell is skipped when the strongest attention is below this, because a flat
# head has no pattern for an attribution to explain.
MIN_PEAK_ATTENTION = 1e-3

# And when it is above this, because a saturated head has no room to respond: an
# ablation there changes the score and leaves the attention at ``1 - epsilon``,
# which measures nothing. Both bounds are read from the baseline forward pass
# only, so cell selection never sees the outcome of an intervention.
MAX_CELL_PEAK = 0.9

# A query with fewer visible keys than this is skipped, because the surrogate is
# fitted over the distances that query can see: below a few dozen the window is
# too short for a full period at most frequencies and agreement is forced.
MIN_VISIBLE_KEYS = 32


def _q_projection(attn: object, cfg: object = None) -> tuple[object, str, int, int]:
    """The module whose output carries the query, plus layout facts.

    ``fused`` means GPT-NeoX's per-head ``(q, k, v)`` interleave; ``split`` means
    a plain ``q_proj`` whose rows are ``n_heads * head_dim`` before reshaping.
    Guessing the layout would silently edit the wrong head, which is why the two
    cases are distinguished by module presence rather than by width, why the head
    geometry comes from the config-aware helpers in ``real_model``, and why the
    resulting width is asserted against the module's own weight shape.
    """
    n_heads = RM.n_heads_of(attn, cfg)
    head_dim = RM.head_dim_of(attn, cfg)
    if not n_heads or not head_dim:
        raise RuntimeError(
            f"cannot determine head geometry for {type(attn).__name__}: "
            f"n_heads={n_heads}, head_dim={head_dim}"
        )
    if hasattr(attn, "query_key_value"):
        module = attn.query_key_value
        expected = 3 * n_heads * head_dim
        if int(module.weight.shape[0]) != expected:
            raise RuntimeError(
                f"{type(attn).__name__}.query_key_value has {module.weight.shape[0]} "
                f"rows, but 3 * {n_heads} heads * {head_dim} = {expected}"
            )
        return module, "fused", head_dim, n_heads
    if hasattr(attn, "q_proj"):
        module = attn.q_proj
        expected = n_heads * head_dim
        if int(module.weight.shape[0]) != expected:
            raise RuntimeError(
                f"{type(attn).__name__}.q_proj has {module.weight.shape[0]} rows, "
                f"but {n_heads} heads * {head_dim} = {expected}"
            )
        return module, "split", head_dim, n_heads
    raise NotImplementedError(
        f"{type(attn).__name__} has neither query_key_value nor q_proj"
    )


def pair_channels(
    head: int, pairs: Sequence[int], head_dim: int, rotary_dim: int, n_heads: int, layout: str
) -> list[int]:
    """Output-channel indices of the given rotary pairs, for one head's query.

    RoPE pairs channel ``j`` with ``j + rotary_dim/2`` inside the head's own
    rotation block (the split-half convention, GPT-NeoX layout). The offset of
    the head's block differs between the fused and split layouts and that is the
    only difference.
    """
    half = rotary_dim // 2
    base = head * 3 * head_dim if layout == "fused" else head * head_dim
    channels: list[int] = []
    for pair in pairs:
        if not 0 <= pair < half:
            raise ValueError(f"rotary pair {pair} outside [0, {half})")
        channels.append(base + int(pair))
        channels.append(base + int(pair) + half)
    return channels


@contextlib.contextmanager
def scale_query_pairs(
    model: object,
    layer: int,
    head: int,
    pairs: Sequence[int],
    scale: float,
    rope_dim: int,
) -> Iterator[None]:
    """Multiply the given rotary pairs of one head's query by ``scale``.

    The hook sits on the query projection, which is applied *before* RoPE, so
    scaling a pair's two channels scales the rotated pair by the same factor and
    leaves the rotation angle alone. Every score for this head at this layer is
    changed by exactly ``(scale - 1) * c_j(delta)`` per selected pair.
    """
    attns = RM.attention_modules(model)
    attn = attns[int(layer)]
    module, layout, head_dim, n_heads = _q_projection(attn, getattr(model, "config", None))
    channels = pair_channels(head, pairs, head_dim, rope_dim, n_heads, layout)
    torch = RM._require_torch()

    def hook(_module, _inputs, output):
        index = torch.as_tensor(channels, dtype=torch.long, device=output.device)
        edited = output.clone()
        flat = edited.reshape(-1, edited.shape[-1])
        flat[:, index] = flat[:, index] * scale
        return edited

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def pair_contribution_matrix(
    q_hat: np.ndarray, k_hat: np.ndarray, pair: int, theta: float, rotary_dim: int
) -> np.ndarray:
    """``c_j(t - s)`` for every causal ``(t, s)``, shape ``(T, T)``, zero elsewhere.

    ``A_j`` and ``B_j`` belong to the *pair of tokens*, not to a token, so the
    matrix is built from outer products of the two channels rather than from one
    scalar coefficient. Using a single ``(A_j, B_j)`` --- the value at the query
    token --- is the natural mistake here and silently describes a sequence of
    different content pairs.

    A key sits behind its query, so the angle is ``(s - t) * theta``, i.e. the
    negative of the distance. Getting this sign wrong mirrors every sine term and
    is the exact error corrected in the QK artefacts; it is pinned by a test.
    """
    half = rotary_dim // 2
    q1 = q_hat[:, pair]
    q2 = q_hat[:, pair + half]
    k1 = k_hat[:, pair]
    k2 = k_hat[:, pair + half]
    aligned = np.outer(q1, k1) + np.outer(q2, k2)
    crossed = np.outer(q2, k1) - np.outer(q1, k2)

    n_tokens = q_hat.shape[0]
    distance = np.subtract.outer(np.arange(n_tokens), np.arange(n_tokens))
    angle = -distance.astype(np.float64) * theta
    out = aligned * np.cos(angle) + crossed * np.sin(angle)
    return np.tril(out)


def score_matrix(
    q_hat: np.ndarray, k_hat: np.ndarray, freqs: np.ndarray, rotary_dim: int
) -> np.ndarray:
    """Pre-softmax scores for one head, ``(T, T)``, ``-inf`` above the diagonal.

    The rotary pairs contribute their closed form and the unrotated tail its
    ordinary inner product; the two are added in the same expression, which is
    the identity of Section 3 rather than an approximation. ``rotated_dot``
    provides an independent construction and the test suite checks the two agree.
    """
    n_tokens = q_hat.shape[0]
    scores = np.zeros((n_tokens, n_tokens), dtype=np.float64)
    half = rotary_dim // 2
    for pair in range(half):
        scores = scores + pair_contribution_matrix(
            q_hat, k_hat, pair, float(freqs[pair]), rotary_dim
        )
    if rotary_dim < q_hat.shape[1]:
        scores = scores + np.tril(q_hat[:, rotary_dim:] @ k_hat[:, rotary_dim:].T)
    out = np.full((n_tokens, n_tokens), -np.inf, dtype=np.float64)
    out[np.tril_indices(n_tokens)] = scores[np.tril_indices(n_tokens)]
    return out


def attention_probs(scores: np.ndarray, head_dim: int) -> np.ndarray:
    """Row-wise softmax of ``scores / sqrt(head_dim)`` with ``-inf`` masked out."""
    scaled = scores / np.sqrt(float(head_dim))
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def predict_attention(
    q_hat: np.ndarray,
    k_hat: np.ndarray,
    freqs: np.ndarray,
    rotary_dim: int,
    head_dim: int,
    ablate: Sequence[int] = (),
) -> np.ndarray:
    """Attention after removing the given rotary pairs, from the closed form."""
    scores = score_matrix(q_hat, k_hat, freqs, rotary_dim)
    if ablate:
        for pair in ablate:
            scores = scores - pair_contribution_matrix(
                q_hat, k_hat, int(pair), float(freqs[pair]), rotary_dim
            )
    return attention_probs(scores, head_dim)


def select_pairs(values: np.ndarray, budget: int) -> list[int]:
    """The ``budget`` largest positive entries of ``values``, largest first.

    Positive only: the intervention *suppresses* a contribution, and suppressing
    a pair that is already negative at this distance would raise the score, which
    is the opposite of what the attribution recommends. A cell with no positive
    entry has nothing to suppress and the caller skips it.
    """
    order = np.argsort(-values)
    chosen = [int(i) for i in order[:budget] if values[i] > 0.0]
    return chosen


def _surrogate_vector(
    aligned: np.ndarray, crossed: np.ndarray, freqs: np.ndarray, n_visible: int
) -> np.ndarray:
    """The best position-free constant per pair, over this query's causal window.

    The window is every distance the query can actually see, ``1 .. n_visible``,
    because that is the table a practitioner would fit. Fitting over the full
    sequence would let a query at position 5 be summarised by distances it cannot
    attend to, which is not the object under test.

    ``aligned``/``crossed`` are the *cell's* pair coefficients, i.e. for the one
    (query token, key token) pair whose attention is being explained. A
    token-averaged coefficient vector would describe a different content pair and
    would silently turn this into a different experiment.
    """
    deltas = np.arange(1, int(n_visible) + 1, dtype=np.float64)
    angles = np.outer(freqs, deltas)
    cos_mean = np.cos(angles).mean(axis=1)
    sin_mean = np.sin(angles).mean(axis=1)
    return aligned * cos_mean + crossed * sin_mean


def _forward_attention(model: object, tokenizer: object, text: str) -> dict:
    torch = RM._require_torch()
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_attentions=True)
    return {
        "ids": ids,
        "attentions": [a[0].numpy().astype(np.float64) for a in out.attentions],
        "logits": out.logits[0].numpy().astype(np.float64),
    }


def measure_cell(
    model: object,
    tokenizer: object,
    text: str,
    hv: RM.HeadVectors,
    freqs: np.ndarray,
    rope_dim: int,
    layer: int,
    head: int,
    query: int,
    budget: int,
    baseline: dict,
) -> dict | None:
    """One (layer, head, query) cell: two selections, their predictions, two runs.

    Three quantities are reported and they answer different questions.

    ``suppression_ratio`` is ``achieved / intended``, where *intended* is what the
    attribution believed it was removing and *achieved* is the true contribution
    of the pairs it actually removed, both at the distance the head uses. It is
    the currency of the intervention: ``1`` means the suppression did what it was
    told, ``0`` means it removed nothing that mattered, and **negative means the
    edit amplified the score it was supposed to suppress**. The exact rule scores
    ``1`` by construction --- it selects by the very quantity it then removes ---
    so it is an oracle here, not a competitor, and is reported as such.

    ``prediction_error`` is ``max |predicted - measured|`` over the query row, a
    float32 identity check on the harness: the closed form plus softmax must
    reproduce the real forward pass. It is *not* a comparison between the rules,
    because the prediction is made for whichever pairs were actually removed.

    ``delta_attention`` is the measured post-softmax change at the target key, so
    a reader can see whether the intervention moved anything at all.
    """
    torch = RM._require_torch()
    del torch
    q = hv.q_hat[layer, head, :, :rope_dim]
    k = hv.k_hat[layer, head, :, :rope_dim]
    q_full = hv.q_hat[layer, head]
    k_full = hv.k_hat[layer, head]
    n_tokens = q.shape[0]

    base_attn = baseline["attentions"][layer][head]
    # Mask the diagonal before taking the argmax: a head that attends to itself
    # has no distance to attribute, and skipping it here rather than after the
    # argmax keeps the candidate list and the measurement consistent.
    row = base_attn[query].copy()
    row[query:] = 0.0
    target = int(np.argmax(row))
    peak = float(row[target])
    if peak < MIN_PEAK_ATTENTION:
        return None
    delta_star = int(query - target)

    terms = pair_terms(q[query], k[target], freqs, -delta_star)
    c_exact = np.asarray(terms.aligned * np.cos(terms.angle) + terms.crossed * np.sin(terms.angle))
    c_hat = _surrogate_vector(
        np.asarray(terms.aligned), np.asarray(terms.crossed), freqs, query
    )

    exact_pairs = select_pairs(c_exact, budget)
    surrogate_pairs = select_pairs(c_hat, budget)
    if not exact_pairs or not surrogate_pairs:
        return None

    head_dim = int(hv.head_dim)
    pred_base = predict_attention(q_full, k_full, freqs, rope_dim, head_dim)
    base_err = float(np.abs(pred_base[query] - base_attn[query]).max())

    records = []
    for rule, pairs, estimate in (
        ("exact", exact_pairs, c_exact),
        ("surrogate", surrogate_pairs, c_hat),
    ):
        predicted = predict_attention(q_full, k_full, freqs, rope_dim, head_dim, pairs)
        with scale_query_pairs(model, layer, head, pairs, 0.0, rope_dim):
            measured_run = _forward_attention(model, tokenizer, text)
        measured = measured_run["attentions"][layer][head]
        intended = float(np.asarray(estimate)[pairs].sum())
        achieved = float(c_exact[pairs].sum())
        records.append(
            {
                "rule": rule,
                "pairs": pairs,
                "intended_suppression": intended,
                "achieved_suppression": achieved,
                "suppression_ratio": achieved / intended if intended != 0.0 else None,
                "prediction_error": float(np.abs(predicted[query] - measured[query]).max()),
                "predicted_at_target": float(predicted[query, target]),
                "measured_at_target": float(measured[query, target]),
                "delta_attention": float(measured[query, target] - base_attn[query, target]),
                "predicted_delta_attention": float(
                    predicted[query, target] - base_attn[query, target]
                ),
            }
        )

    exact = next(r for r in records if r["rule"] == "exact")
    surrogate = next(r for r in records if r["rule"] == "surrogate")
    return {
        "layer": int(layer),
        "head": int(head),
        "query": int(query),
        "target": target,
        "delta": int(delta_star),
        "n_visible": int(query),
        "baseline_attention": peak,
        "baseline_prediction_error": base_err,
        "exact": exact,
        "surrogate": surrogate,
        "same_pair": exact["pairs"] == surrogate["pairs"],
        "ratio": (
            surrogate["achieved_suppression"] / exact["achieved_suppression"]
            if exact["achieved_suppression"] != 0.0
            else None
        ),
    }


def measure_model(
    model_id: str = DEFAULT_MODEL_IDS[0],
    seed: int = RM.DEFAULT_SEED,
    texts: Sequence[str] = TEXTS,
    budget: int = 1,
    cells_per_prompt: int = CELLS_PER_PROMPT,
) -> dict:
    """The intervention experiment on one checkpoint."""
    loaded = RM.load_model(model_id, seed=seed)
    cfgv = RM.validate_config(loaded.model)
    rope_dim = int(cfgv["rope_dim"])
    freqs = inv_freq(rope_dim, float(cfgv["rope_theta_declared"]))

    cells = []
    n_heads = 0
    for i, text in enumerate(texts):
        hv = RM.collect_head_vectors(loaded.model, loaded.tokenizer, text)
        n_heads = max(n_heads, hv.n_heads)
        baseline = _forward_attention(loaded.model, loaded.tokenizer, text)
        n_tokens = hv.n_tokens

        # One candidate per head: its strongest attention row that is neither
        # saturated nor too short to fit a window over. Taking the per-head best
        # before ranking keeps the cell set spread over heads instead of being
        # many queries of whichever head happens to be sharpest. Every quantity
        # here is read from the baseline forward pass, before any intervention.
        candidates = []
        for layer in range(hv.n_layers):
            for head in range(hv.n_heads):
                attn = baseline["attentions"][layer][head]
                best: tuple[float, int, int, int] | None = None
                for query in range(MIN_VISIBLE_KEYS, n_tokens):
                    row = attn[query].copy()
                    row[query:] = 0.0
                    peak = float(row.max())
                    if MIN_PEAK_ATTENTION <= peak <= MAX_CELL_PEAK:
                        if best is None or peak > best[0]:
                            best = (peak, layer, head, query)
                if best is not None:
                    candidates.append(best)
        candidates.sort(reverse=True)
        taken = 0
        for peak, layer, head, query in candidates:
            if taken >= cells_per_prompt:
                break
            record = measure_cell(
                loaded.model, loaded.tokenizer, text, hv, freqs, rope_dim,
                layer, head, query, budget, baseline,
            )
            if record is None:
                continue
            record["prompt_index"] = i
            record["n_tokens"] = n_tokens
            cells.append(record)
            taken += 1

    if not cells:
        raise RuntimeError(f"{model_id}: no cell produced a measurable intervention")

    exact_err = np.array([c["exact"]["prediction_error"] for c in cells])
    surrogate_err = np.array([c["surrogate"]["prediction_error"] for c in cells])
    exact_delta = np.abs(np.array([c["exact"]["delta_attention"] for c in cells]))
    surrogate_delta = np.abs(np.array([c["surrogate"]["delta_attention"] for c in cells]))
    differs = np.array([not c["same_pair"] for c in cells])
    exact_ratio = np.array(
        [c["exact"]["suppression_ratio"] for c in cells if c["exact"]["suppression_ratio"] is not None]
    )
    surrogate_ratio = np.array(
        [
            c["surrogate"]["suppression_ratio"]
            for c in cells
            if c["surrogate"]["suppression_ratio"] is not None
        ]
    )
    differing_ratio = np.array(
        [
            c["surrogate"]["suppression_ratio"]
            for c in cells
            if not c["same_pair"] and c["surrogate"]["suppression_ratio"] is not None
        ]
    )
    return {
        "model_id": model_id,
        "rope_dim": rope_dim,
        "rope_theta": float(cfgv["rope_theta_declared"]),
        "n_heads": n_heads,
        "n_cells": len(cells),
        "n_selection_differs": int(differs.sum()),
        "budget": int(budget),
        "cell_selection": {
            "min_peak": MIN_PEAK_ATTENTION,
            "max_peak": MAX_CELL_PEAK,
            "min_visible_keys": MIN_VISIBLE_KEYS,
            "rule": "one candidate per head: its strongest baseline attention row "
            "that is neither saturated nor shorter than min_visible_keys, "
            "diagonal excluded, chosen before any intervention",
        },
        "schema_note": "suppression_ratio = achieved / intended at the head's own "
        "strongest unsaturated key; prediction_error is a max over the query row "
        "of |predicted - measured| post-softmax attention from a float32 forward pass",
        "pooled": {
            # The harness identity: closed form + softmax against the real pass.
            "baseline_prediction_error_median": float(
                np.median([c["baseline_prediction_error"] for c in cells])
            ),
            "exact_prediction_error_median": float(np.median(exact_err)),
            "exact_prediction_error_max": float(exact_err.max()),
            "surrogate_prediction_error_median": float(np.median(surrogate_err)),
            "surrogate_prediction_error_max": float(surrogate_err.max()),
            # The oracle: 1 by construction, reported to show the algebra closes.
            "exact_suppression_ratio_min": float(exact_ratio.min()),
            "exact_suppression_ratio_max": float(exact_ratio.max()),
            # The measured cost of the position-free summary.
            "surrogate_suppression_ratio_median": float(np.median(surrogate_ratio)),
            "surrogate_suppression_ratio_min": float(surrogate_ratio.min()),
            "surrogate_sign_reversal_fraction": float(np.mean(surrogate_ratio < 0.0)),
            "differing_suppression_ratio_median": float(np.median(differing_ratio))
            if differing_ratio.size
            else None,
            "differing_sign_reversal_fraction": float(np.mean(differing_ratio < 0.0))
            if differing_ratio.size
            else None,
            # How much the intervention moved, so a null is not mistaken for a
            # saturated cell.
            "exact_abs_delta_attention_median": float(np.median(exact_delta)),
            "surrogate_abs_delta_attention_median": float(np.median(surrogate_delta)),
            "fraction_selection_differs": float(differs.mean()),
        },
        "cells": cells,
    }


def run_all(
    model_ids: Sequence[str] = DEFAULT_MODEL_IDS,
    seed: int = RM.DEFAULT_SEED,
    texts: Sequence[str] = TEXTS,
    budgets: Sequence[int] = BUDGETS,
    cells_per_prompt: int = CELLS_PER_PROMPT,
) -> dict:
    return {
        "schema": "rope-intervention/v1",
        "seed": int(seed),
        "budgets": [int(b) for b in budgets],
        "cells_per_prompt": int(cells_per_prompt),
        "estimand": "achieved / intended suppression of the score at the head's "
        "own strongest unsaturated key, after removing the pair each attribution "
        "ranks first from the query projection only; the post-softmax change is "
        "predicted in closed form and confirmed against the real forward pass",
        "models": [
            measure_model(m, seed=seed, texts=texts, budget=int(budgets[0]),
                          cells_per_prompt=cells_per_prompt)
            for m in model_ids
        ],
    }


def write_results(data: dict, path: Path | None = None) -> Path:
    path = path or Path("results/intervention.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODEL_IDS))
    parser.add_argument("--seed", type=int, default=RM.DEFAULT_SEED)
    parser.add_argument("--budget", type=int, default=BUDGETS[0])
    parser.add_argument("--cells", type=int, default=CELLS_PER_PROMPT)
    parser.add_argument("--out", type=Path, default=Path("results/intervention.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    data = run_all(
        tuple(args.models), seed=args.seed, budgets=(args.budget,),
        cells_per_prompt=args.cells,
    )
    path = write_results(data, args.out)
    print(f"wrote {path}")
    for model in data["models"]:
        p = model["pooled"]
        print(
            f"{model['model_id']}: cells={model['n_cells']} "
            f"differ={p['fraction_selection_differs']:.2f} "
            f"exact ratio [{p['exact_suppression_ratio_min']:.6f}, "
            f"{p['exact_suppression_ratio_max']:.6f}] "
            f"surrogate ratio median={p['surrogate_suppression_ratio_median']:.3f} "
            f"sign-reversal={p['surrogate_sign_reversal_fraction']:.2f} "
            f"identity err median={p['exact_prediction_error_median']:.2e} "
            f"|d attn| median={p['exact_abs_delta_attention_median']:.2e}"
        )


if __name__ == "__main__":
    main()
