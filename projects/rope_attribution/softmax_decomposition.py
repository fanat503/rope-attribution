"""What the softmax does to a per-pair attribution, exactly.

The question
------------
The paper's attribution object lives on the score ``s_ij``.  A reviewer can
accept every score-level identity and still ask the only question that reaches a
deployed model: the model does not use scores, it uses ``p = softmax(s)``.  Does
the per-pair decomposition survive that map?

There are three different answers depending on which object is being
decomposed, and keeping them apart is the whole point of this module.

1. **Probabilities do not admit an additive per-pair decomposition.**  This is
   not a limitation of the algebra; it is a no-go result.  With two keys and two
   pairs, write ``s_1 = a + b`` and ``s_2 = 0``.  Then
   ``p_1 = sigmoid(a + b)`` and ``d2 p_1 / da db = sigmoid''(a + b)``, which is
   nonzero.  If ``p_1`` were ``f(a) + g(b)`` the mixed derivative would vanish.
   No choice of per-pair terms can reproduce the softmax.

2. **Log-odds are additive, exactly.**  For any two keys ``j`` and ``k``,
   ``log(p_j / p_k) = s_j - s_k``: the log-normaliser cancels.  Since ``s`` is a
   sum over rotary pairs plus the unrotated tail, the log-odds are a sum over
   pairs plus one tail term.  Nothing is approximated and nothing is dropped.

3. **The odds ratio factorises, exactly.**
   ``p_j / p_k = prod_p exp(c_{j,p} - c_{k,p}) * exp(tail_j - tail_k)``.
   Additive in log space, multiplicative in probability space.

The positional contrast, in closed form
---------------------------------------
Fix one key content vector and place it at two distances ``d_j`` and ``d_k``.
For one rotary pair with ``A cos(theta d) + B sin(theta d) = rho cos(theta d -
psi)``, the pair's contribution to the log-odds of the two placements is

    Delta(d_j, d_k) = -2 rho sin(theta (d_j + d_k)/2 - psi) sin(theta (d_j - d_k)/2).

Two consequences are exact and testable:

* ``Delta = 0`` whenever ``theta (d_j - d_k)`` is a multiple of ``2*pi``.  The
  pair is **blind** to two placements separated by an integer number of its own
  periods, whatever the content phase.
* ``|Delta|`` is largest at half a period, ``theta (d_j - d_k) = pi``, where it
  reaches ``2 rho |sin(theta (d_j + d_k)/2 - psi)|``.

The characteristic length is therefore the same ``L* = 2*pi / theta`` that
appears in the window law: a window of one period makes every constant wrong
(relative error 1), and a separation of one period makes the pair blind in
log-odds (contrast 0).  Softmax does not introduce a second scale.

What does *not* survive
-----------------------
For two *different* key vectors the pair coefficients ``A`` and ``B`` differ, so
the closed form above describes the positional part only.  The additive
log-odds decomposition is still exact, but its per-pair terms are
``c_{j,p} - c_{k,p}`` with different content, and there is no formula in the
midpoint and separation alone.  This module measures both objects and never
conflates them.

The gradient identity
---------------------
For a scalar ``w`` that enters every score through pair ``p`` with coefficient
``c_{j,p}``, the softmax Jacobian gives

    d p_j / d w = p_j (c_{j,p} - sum_k p_k c_{k,p}).

The pair's effect on the attention distribution is its contribution **centred
by its own attention-weighted mean**.  The subtracted term is a constant across
``j`` for a fixed query and pair, so it is absorbed by the best constant
surrogate: the window law's relative error is invariant under this centering.
That is why the score-level collapse result applies to the softmax gradient
without renormalisation.

A monotone softcap (Gemma 2) composes one more scalar map after the score.  The
score identities still hold; the log-odds of the *logits* are not additive in
pairs, and the residual is measured rather than assumed away.

What this module does not claim
-------------------------------
It does not claim a per-pair decomposition of the attention output: given the
probabilities that map is exactly linear in the values, but the probabilities
themselves are the non-additive object.  It does not claim the no-go result is
new; it is the elementary mixed-partial argument, stated here because the
paper's position depends on saying precisely where additivity stops.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rope_attribution import real_model as RM
from rope_attribution.experiments import pair_terms
from rope_attribution.rope import inv_freq

__all__ = [
    "SCHEMA",
    "DEFAULT_MAX_TOKENS",
    "N_QUERIES",
    "pair_phase",
    "contribution",
    "contrast_closed_form",
    "contrast_direct",
    "pair_coefficients",
    "score_matrix",
    "log_odds_matrix",
    "odds_factors",
    "centered_profile",
    "softmax_weights",
    "softmax_mixed_partial",
    "measure_head",
    "measure_model",
    "run_all",
    "write_results",
    "main",
]

SCHEMA = "rope_attribution/softmax_decomposition/v1"

# Enough positions to include many periods of the slowest pairs and thousands
# of periods of the fastest, without making the per-head cost noticeable.
DEFAULT_MAX_TOKENS = 256

# Queries sampled per head.  The last token is always included because it sees
# the full context; the others are evenly spaced so the measurement is not a
# single position's accident.
N_QUERIES = 4


def pair_phase(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rho, psi)`` with ``a cos(t) + b sin(t) = rho cos(t - psi)``."""
    return np.hypot(a, b), np.arctan2(b, a)


def contribution(
    a: np.ndarray, b: np.ndarray, theta: np.ndarray, delta: float
) -> np.ndarray:
    """Exact pair contribution ``a cos(theta delta) + b sin(theta delta)``."""
    return a * np.cos(theta * delta) + b * np.sin(theta * delta)


def contrast_closed_form(
    a: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    delta_j: float,
    delta_k: float,
) -> np.ndarray:
    """The positional log-odds contribution of one pair, in closed form.

    ``a`` and ``b`` are the pair's content coefficients for a *single* key
    vector, so the two placements share content and only the distance changes.
    """
    rho, psi = pair_phase(a, b)
    mid = 0.5 * (delta_j + delta_k)
    half = 0.5 * (delta_j - delta_k)
    return -2.0 * rho * np.sin(theta * mid - psi) * np.sin(theta * half)


def contrast_direct(
    a: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    delta_j: float,
    delta_k: float,
) -> np.ndarray:
    """The same contrast as ``contribution(delta_j) - contribution(delta_k)``.

    Kept separate from :func:`contrast_closed_form` so the tests compare two
    independent expressions rather than a function with itself.
    """
    return contribution(a, b, theta, delta_j) - contribution(
        a, b, theta, delta_k
    )


def pair_coefficients(
    q_hat: np.ndarray, k_hat: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(A, B)`` for one head vector pair, shape ``(n_pairs,)``.

    ``A`` and ``B`` do not depend on distance; calling ``pair_terms`` at
    ``delta = 0`` is the cheapest way to obtain them and is explicit about that.
    """
    terms = pair_terms(q_hat, k_hat, freqs, 0)
    return terms.aligned, terms.crossed


def score_matrix(
    q_hat: np.ndarray,
    k_hats: np.ndarray,
    freqs: np.ndarray,
    rope_dim: int,
    query_index: int,
) -> dict[str, np.ndarray]:
    """Exact scores and per-pair coefficients for one query against all keys.

    ``k_hats`` is ``(n_keys, head_dim)``.  Only causal keys ``n <= query_index``
    are returned; the distances are ``query_index - n``.  The score is the sum
    of the rotary pairs plus the unrotated tail, which is distance independent.
    """
    n_keys = int(query_index) + 1
    if n_keys < 1:
        raise ValueError("query_index must be non-negative")
    q = np.asarray(q_hat, dtype=np.float64)
    k = np.asarray(k_hats[:n_keys], dtype=np.float64)
    deltas = (int(query_index) - np.arange(n_keys)).astype(np.float64)

    n_pairs = int(freqs.size)
    q_rot = q[: 2 * n_pairs]
    a = np.empty((n_pairs, n_keys), dtype=np.float64)
    b = np.empty((n_pairs, n_keys), dtype=np.float64)
    for n in range(n_keys):
        a[:, n], b[:, n] = pair_coefficients(q_rot, k[n, : 2 * n_pairs], freqs)

    angles = np.outer(freqs, deltas)
    pair_scores = a * np.cos(angles) + b * np.sin(angles)
    tail = np.zeros(n_keys, dtype=np.float64)
    if rope_dim < q.size:
        tail = k[:, rope_dim:] @ q[rope_dim:]
    scores = pair_scores.sum(axis=0) + tail
    return {
        "deltas": deltas,
        "a": a,
        "b": b,
        "pair_scores": pair_scores,
        "tail": tail,
        "scores": scores,
    }


def log_odds_matrix(
    pair_scores: np.ndarray, anchor: int = 0
) -> np.ndarray:
    """Per-pair contribution to ``s_j - s_anchor``, shape ``(n_pairs, n_keys)``."""
    if not 0 <= int(anchor) < pair_scores.shape[1]:
        raise ValueError("anchor is outside the key range")
    return pair_scores - pair_scores[:, [int(anchor)]]


def odds_factors(
    pair_scores: np.ndarray, tail: np.ndarray, anchor: int = 0
) -> np.ndarray:
    """Per-pair factors whose product is ``p_j / p_anchor``.

    Returned in log space as ``(n_pairs, n_keys)``; the tail is a separate
    vector.  A caller exponentiates only after subtracting its own maximum.
    """
    delta = log_odds_matrix(pair_scores, anchor)
    return delta


def softmax_weights(scores: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    logits = np.asarray(scores, dtype=np.float64) * float(scale)
    if logits.size == 0:
        raise ValueError("softmax needs at least one score")
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()


def centered_profile(
    pair_scores: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """``c_{j,p} - sum_k p_k c_{k,p}``, shape ``(n_pairs, n_keys)``.

    This is the exact factor multiplying ``p_j`` in ``d p_j / d w`` for a
    parameter that enters every score through the pair.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (pair_scores.shape[1],):
        raise ValueError("weights must have one entry per key")
    return pair_scores - pair_scores @ w


def softmax_mixed_partial(a: float = 0.5, b: float = 0.5) -> float:
    """``d2 sigmoid(a + b) / da db`` -- nonzero off the symmetric point.

    With two keys and two pairs, ``p_1 = sigmoid(a + b)``.  Any additive
    decomposition ``f(a) + g(b)`` has mixed partial zero everywhere; this
    function is the one-line obstruction.  The default ``(0.5, 0.5)`` is chosen
    away from zero, where the mixed partial of the sigmoid vanishes by symmetry.
    """
    s = float(a) + float(b)
    e = np.exp(-abs(s))
    sig = 1.0 / (1.0 + e) if s >= 0 else e / (1.0 + e)
    return float(sig * (1.0 - sig) * (1.0 - 2.0 * sig))


def _score_scale(attn: object, head_dim: int) -> float:
    """The positive scalar the model multiplies scores by before softmax."""
    value = getattr(attn, "scaling", None)
    if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
        return float(value)
    return float(head_dim) ** -0.5


def _softcap(attn: object) -> float | None:
    value = getattr(attn, "attn_logit_softcapping", None)
    if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
        return float(value)
    return None


def measure_head(
    hv: RM.HeadVectors,
    freqs: np.ndarray,
    rope_dim: int,
    layer: int,
    head: int,
    query_indices: Sequence[int],
    *,
    score_scale: float = 1.0,
    softcap: float | None = None,
) -> dict:
    """Exactness residuals and descriptive shares for one head.

    Every residual is a max-abs discrepancy between two independently computed
    objects, so a correct implementation reports ``~1e-15`` and a wrong ordering
    of the norm, a dropped tail or a wrong anchor reports ``O(1)``.
    """
    q_all = hv.q_hat[layer, head]
    k = hv.k_hat[layer, head]
    n_tokens = int(k.shape[0])

    log_odds_res = 0.0
    odds_res = 0.0
    contrast_res = 0.0
    null_res = 0.0
    center_res = 0.0
    share_top: list[float] = []
    participation: list[float] = []
    tail_fraction: list[float] = []
    softcap_res = 0.0

    for query_index in query_indices:
        if not 1 <= int(query_index) < n_tokens:
            continue
        q = q_all[int(query_index)]
        table = score_matrix(q, k, freqs, rope_dim, int(query_index))
        scores = table["scores"]
        deltas = table["deltas"]
        pair_scores = table["pair_scores"]
        tail = table["tail"]

        # 1. Additive log-odds: exact, anchor at the oldest key.
        delta = log_odds_matrix(pair_scores, anchor=0)
        rebuilt = delta.sum(axis=0) + (tail - tail[0])
        log_odds_res = max(
            log_odds_res,
            float(np.max(np.abs(rebuilt - (scores - scores[0])))),
        )

        # 2. Odds ratio: product of per-pair factors, exact in log space.
        p = softmax_weights(scores, score_scale)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(np.maximum(p, np.finfo(float).tiny)) - np.log(
                np.maximum(p[0], np.finfo(float).tiny)
            )
        odds_res = max(
            odds_res, float(np.max(np.abs(log_ratio - score_scale * rebuilt)))
        )

        # 3. Closed-form positional contrast against the direct difference,
        #    for every pair and every key, using key 0's content at both
        #    distances.  This is the object the closed form is about.
        a0, b0 = table["a"][:, 0], table["b"][:, 0]
        for n in range(1, len(deltas)):
            closed = contrast_closed_form(a0, b0, freqs, deltas[0], deltas[n])
            direct = contrast_direct(a0, b0, freqs, deltas[0], deltas[n])
            contrast_res = max(
                contrast_res, float(np.max(np.abs(closed - direct)))
            )
        # Blindness at one full period of each pair.  The separation
        # 2*pi/theta is generally not on the token grid, so the closed form is
        # evaluated at that exact real separation instead of snapping it to an
        # integer and calling the snap exact.
        period = 2.0 * np.pi / freqs
        blind = contrast_closed_form(a0, b0, freqs, 0.0, period)
        null_res = max(null_res, float(np.max(np.abs(blind))))

        # 4. Centered gradient: finite difference of the softmax under a
        #    perturbation that enters every score through one pair.
        eps = 1e-6
        for pair in (0, len(freqs) // 2, len(freqs) - 1):
            perturbed = scores + eps * pair_scores[pair]
            p_eps = softmax_weights(perturbed, score_scale)
            finite = (p_eps - p) / eps
            exact = score_scale * p * centered_profile(
                pair_scores[pair : pair + 1], p
            )[0]
            center_res = max(
                center_res, float(np.max(np.abs(finite - exact)))
            )

        # 5. Descriptive: how concentrated is the log-odds across pairs?
        #    Shares include the unrotated tail as one component so they sum to
        #    one exactly even under partial RoPE.  The key with the largest
        #    absolute log-odds is used, not an arbitrary one.
        total = scores - scores[0]
        if total.size > 1:
            idx = int(np.argmax(np.abs(total[1:]))) + 1
            if abs(total[idx]) > 1e-12:
                shares = np.append(
                    delta[:, idx] / total[idx],
                    (tail[idx] - tail[0]) / total[idx],
                )
                share_top.append(float(np.max(np.abs(shares))))
                participation.append(
                    float(1.0 / np.sum(shares**2)) if np.any(shares) else 0.0
                )
                tail_fraction.append(float(abs(shares[-1])))

        # 6. Softcap: the score identities hold, but the logits are a monotone
        #    nonlinear map of the score.  Measure the residual it introduces.
        if softcap is not None:
            logits = softcap * np.tanh(score_scale * scores / softcap)
            p_cap = softmax_weights_from_logits(logits)
            with np.errstate(divide="ignore", invalid="ignore"):
                cap_ratio = np.log(
                    np.maximum(p_cap, np.finfo(float).tiny)
                ) - np.log(np.maximum(p_cap[0], np.finfo(float).tiny))
            softcap_res = max(
                softcap_res,
                float(np.max(np.abs(cap_ratio - score_scale * rebuilt))),
            )

    return {
        "layer": int(layer),
        "head": int(head),
        "n_queries": len(share_top),
        "log_odds_residual": log_odds_res,
        "odds_residual": odds_res,
        "contrast_residual": contrast_res,
        "blind_at_period_residual": null_res,
        "centered_gradient_residual": center_res,
        "softcap_log_odds_residual": softcap_res if softcap is not None else None,
        "max_single_pair_share": (
            float(np.mean(share_top)) if share_top else None
        ),
        "participation_ratio": (
            float(np.mean(participation)) if participation else None
        ),
        "tail_fraction": (
            float(np.mean(tail_fraction)) if tail_fraction else None
        ),
    }


def softmax_weights_from_logits(logits: np.ndarray) -> np.ndarray:
    """Softmax of already-scaled logits (used by the softcap path)."""
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()


def _sample_queries(n_tokens: int, n_queries: int) -> list[int]:
    if n_tokens < 2:
        return []
    last = n_tokens - 1
    if n_queries <= 1:
        return [last]
    picks = np.linspace(max(1, n_tokens // 4), last, n_queries)
    return sorted({int(p) for p in np.round(picks)})


def measure_model(
    model_id: str = RM.DEFAULT_MODEL_IDS[0],
    seed: int = RM.DEFAULT_SEED,
    prompts: Sequence[str] = ("The quick brown fox jumps over the lazy dog.",),
    max_tokens: int = DEFAULT_MAX_TOKENS,
    n_queries: int = N_QUERIES,
    loaded: RM.Loaded | None = None,
) -> dict:
    """Run the decomposition check on every head of one checkpoint."""
    if loaded is None:
        loaded = RM.load_model(model_id, seed=seed)
    cfgv = RM.validate_config(loaded.model)
    if not cfgv["checkpoint_inv_freq_found"]:
        raise RuntimeError(f"{model_id} has no rotary module")
    rope_dim = int(cfgv["rope_dim"])
    theta = float(cfgv["rope_theta_declared"])
    freqs = inv_freq(rope_dim, theta)
    attns = RM.attention_modules(loaded.model)
    if not attns:
        raise RuntimeError(f"{model_id} exposes no attention modules")

    rows: list[dict] = []
    for text in prompts:
        hv = RM.collect_head_vectors(
            loaded.model, loaded.tokenizer, text, max_tokens=max_tokens
        )
        queries = _sample_queries(hv.n_tokens, n_queries)
        for layer in range(hv.n_layers):
            attn = attns[layer]
            scale = _score_scale(attn, hv.head_dim)
            cap = _softcap(attn)
            for head in range(hv.n_heads):
                if hv.n_kv_groups and hv.n_kv_heads:
                    # Replicated key heads are the same tensor; measuring every
                    # replica would multiply one number by the group size and
                    # make the head pool look more precise than it is.
                    if head % hv.n_kv_groups:
                        continue
                rows.append(
                    measure_head(
                        hv,
                        freqs,
                        rope_dim,
                        layer,
                        head,
                        queries,
                        score_scale=scale,
                        softcap=cap,
                    )
                )

    def _max(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.max(vals)) if vals else None

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "schema": SCHEMA,
        "model_id": model_id,
        "seed": int(seed),
        "rope_dim": rope_dim,
        "rope_theta": theta,
        "max_tokens": int(max_tokens),
        "n_heads_measured": len(rows),
        "qk_norm": bool(getattr(hv, "qk_norm", False)),
        "head_rows": rows,
        "log_odds_residual_max": _max("log_odds_residual"),
        "odds_residual_max": _max("odds_residual"),
        "contrast_residual_max": _max("contrast_residual"),
        "blind_at_period_residual_max": _max("blind_at_period_residual"),
        "centered_gradient_residual_max": _max("centered_gradient_residual"),
        "softcap_log_odds_residual_max": _max("softcap_log_odds_residual"),
        "max_single_pair_share_mean": _mean("max_single_pair_share"),
        "participation_ratio_mean": _mean("participation_ratio"),
        "tail_fraction_mean": _mean("tail_fraction"),
    }


def run_all(
    model_ids: Sequence[str] = RM.DEFAULT_MODEL_IDS,
    seed: int = RM.DEFAULT_SEED,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Measure every requested checkpoint, keeping per-model provenance."""
    models = []
    for model_id in model_ids:
        payload = measure_model(model_id, seed=seed, max_tokens=max_tokens)
        models.append(payload)
    return {
        "schema": SCHEMA,
        "seed": int(seed),
        "max_tokens": int(max_tokens),
        "models": models,
    }


def write_results(payload: dict, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(RM.DEFAULT_MODEL_IDS))
    parser.add_argument("--seed", type=int, default=RM.DEFAULT_SEED)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = run_all(args.models, seed=args.seed, max_tokens=args.max_tokens)
    if args.out is not None:
        write_results(payload, args.out)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
