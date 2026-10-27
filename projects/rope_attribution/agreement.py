"""What a position-free scalar gets wrong that a reader would act on.

Why this module exists
----------------------
The paper's headline is a *sign* error rate: the best position-free scalar has the
wrong sign on roughly a quarter of per-pair, per-distance cells. A reviewer's
first objection is that a sign error in a small term does not matter, and the
paper's answer so far is that sign is what an explanation commits to.

This module adds the answer that does not lean on that defence. What a reader
actually does with an attribution is *rank* the feature pairs and act on the top
one. So: rank the pairs by exact contribution at a given distance, rank them by
the position-free surrogate, and measure how often the two put the same pair
first. That is a downstream consequence with no downstream task - it needs no
sparse autoencoder, no benchmark, and no model beyond the activations the rest of
the repository already loads.

What is measured
----------------
``top1_agreement``
    Fraction of (query, key, distance) triples where the two rankings put the same
    pair first.
``rank_correlation``
    Spearman rho between the two orderings over all pairs, per triple. A library
    statistic: the committed checkpoint artifact reports no rank correlation and
    the paper does not quote one.
``top1_agreement_top_decile``
    The rate at which the surrogate's pick lands inside the exact ranking's top
    decile - the pairs a reader is most likely to act on, which is where a
    disagreement is most expensive. On an 8-pair head the decile holds one pair,
    so there the statistic equals the top-1 agreement.
``top1_agreement_material``
    A library variant over arrays - the same restricted to cells whose exact term
    carries at least a stated fraction of its own peak. It is not part of the
    committed artifact and the paper does not quote it.

Scope, stated once
------------------
The checkpoint measurement (the lower half of this module) runs on the same
three trained checkpoints and the same content-pair sampling the rest of the
repository uses, and commits ``results/agreement.json``. It is not a claim about
any downstream task's accuracy; it is a claim about how often the position-free
object answers a question about the exact one incorrectly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rope_attribution import real_model as RM
from rope_attribution.experiments import pair_terms
from rope_attribution.rope import inv_freq
from rope_attribution.usefulness import DISTANCE_GRID, cos_sin_means

__all__ = [
    "TOP_DECILE",
    "average_ranks",
    "spearman",
    "top1_agreement",
    "rank_correlation",
    "agreement_profile",
    "ranking_votes",
    "measure_head_agreement",
    "measure_model",
    "run_all",
    "main",
]

# The decile the exact ranking must place a pair in for ``top_decile`` to count it.
# Fixed in advance rather than chosen after seeing the result.
TOP_DECILE = 0.10


def average_ranks(x: Sequence[float]) -> np.ndarray:
    """Ranks 1..n with ties sharing their average rank.

    Ties must be averaged rather than broken arbitrarily: a repeated value that
    silently became a unique rank would change the correlation in a way that
    depends on the sort order.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 2:
        raise ValueError(f"need at least 2 values, got {arr.size}")
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(arr.size, dtype=np.float64)
    sorted_arr = arr[order]
    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rho between two equal-length orderings."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.size != right.size:
        raise ValueError(f"length mismatch: {left.size} vs {right.size}")
    if left.size < 2:
        raise ValueError(f"need at least 2 values, got {left.size}")
    rx = average_ranks(left)
    ry = average_ranks(right)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom == 0.0:
        # A constant ordering carries no information about the other. Returning
        # 0 rather than NaN keeps the aggregate statistics well-defined, and the
        # tests pin the cases where the answer must be 0.
        return 0.0
    return float((rx * ry).sum() / denom)


def top1_agreement(
    exact: Sequence[Sequence[float]],
    surrogate: Sequence[Sequence[float]],
) -> float:
    """Fraction of rows whose argmax is the same index in both.

    ``exact`` and ``surrogate`` are the per-pair contributions at each distance,
    as (n_distances, n_pairs). Ties in the argmax count as agreement only if the
    same index wins in both, which is what a reader taking a top-1 would see.
    """
    e = np.asarray(exact, dtype=np.float64)
    s = np.asarray(surrogate, dtype=np.float64)
    if e.shape != s.shape:
        raise ValueError(f"shape mismatch: {e.shape} vs {s.shape}")
    if e.ndim != 2:
        raise ValueError(f"expected a 2-D (distance, pair) array, got {e.ndim}D")
    return float(np.mean(np.argmax(e, axis=1) == np.argmax(s, axis=1)))


def rank_correlation(
    exact: Sequence[Sequence[float]],
    surrogate: Sequence[Sequence[float]],
) -> dict:
    """Spearman rho between the two orderings, per row and overall."""
    e = np.asarray(exact, dtype=np.float64)
    s = np.asarray(surrogate, dtype=np.float64)
    if e.shape != s.shape:
        raise ValueError(f"shape mismatch: {e.shape} vs {s.shape}")
    per_row = [spearman(e[i], s[i]) for i in range(e.shape[0])]
    return {
        "per_row": [float(v) for v in per_row],
        "mean": float(np.mean(per_row)),
        "sd": float(np.std(per_row, ddof=1)) if len(per_row) > 1 else 0.0,
        "min": float(np.min(per_row)),
        "max": float(np.max(per_row)),
    }


def agreement_profile(
    exact: Sequence[Sequence[float]],
    surrogate: Sequence[Sequence[float]],
    materiality: float = 0.05,
) -> dict:
    """All of the agreement statistics in one call.

    ``exact`` and ``surrogate`` are (n_distances, n_pairs) arrays of the exact
    per-pair contribution magnitude and of the position-free surrogate. Rows are
    distances; pairs are ranked within a row.

    The three conditioned variants are *row subsets*, not masks. Masking a row's
    values before the argmax is a no-op for the top decile, because the row
    maximum is in the top decile by construction - the first version of this
    function did exactly that and reported three identical numbers, which is what
    gave the bug away.
    """
    e = np.asarray(exact, dtype=np.float64)
    s = np.asarray(surrogate, dtype=np.float64)
    if e.shape != s.shape:
        raise ValueError(f"shape mismatch: {e.shape} vs {s.shape}")
    if e.ndim != 2:
        raise ValueError(f"expected a 2-D (distance, pair) array, got {e.ndim}D")

    argmax_e = np.argmax(e, axis=1)
    argmax_s = np.argmax(s, axis=1)
    agree = argmax_e == argmax_s

    # Does the surrogate's chosen pair lie inside the exact top decile? A weaker
    # but meaningful question: is the surrogate at least picking from the region
    # the exact ranking counts as important?
    k = max(1, int(round(TOP_DECILE * e.shape[1])))
    threshold = np.partition(e, -k, axis=1)[:, -k][:, None]
    in_decile = e >= threshold
    surrogate_in_decile = np.array(
        [in_decile[i, argmax_s[i]] for i in range(e.shape[0])], dtype=bool
    )

    # Restrict to the rows whose own peak is at least `materiality` of the
    # largest peak across rows - that is, distances where the head says something
    # at all. Conditioning within a row is a no-op, because the argmax is the row
    # maximum by construction; the first version did that and every row qualified.
    peak = np.abs(e).max(axis=1)
    scale = float(peak.max()) if peak.size else 0.0
    top_is_material = peak >= materiality * scale

    def rate(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        return float(np.mean(agree[mask]))

    return {
        "n_distances": int(e.shape[0]),
        "n_pairs": int(e.shape[1]),
        "top1_agreement": float(np.mean(agree)),
        "top1_agreement_top_decile": float(np.mean(surrogate_in_decile)),
        "top1_agreement_material": rate(top_is_material),
        "n_material_rows": int(top_is_material.sum()),
        "rank_correlation": rank_correlation(e, s),
    }


# --------------------------------------------------------------------------
# Measurement on trained checkpoints
# --------------------------------------------------------------------------
#
# The library above operates on arrays; this half drives it over the same
# three trained checkpoints and the same content-pair sampling the rest of
# the repository uses, in the shape of ``usefulness.py``: pre-registered
# distance grid, deterministic per-head seeds, exclusions counted rather
# than hidden, and one committed results file the paper's numbers come from.

_VOTE_KEYS = (
    "n_rows",
    "n_live",
    "n_agree",
    "n_decile",
    "n_excluded_exact_zero",
    "n_excluded_surrogate_zero",
)


def ranking_votes(exact: np.ndarray, surrogate: np.ndarray) -> dict:
    """One content pair's vote counts.

    ``exact`` is ``(n_deltas, n_pairs)`` -- the exact per-rotary-pair
    contribution of one ``(query, key)`` pair, one row per distance on the
    pre-registered grid. ``surrogate`` is ``(n_pairs,)`` -- the best
    position-free constant, the same values for every distance. A live row
    votes "agree" when both rankings put the same pair first, and votes
    "decile" when the surrogate's pick lies inside that row's exact top
    decile.

    Rows that are not votes are counted, never dropped silently:

    * an all-zero exact row would hand argmax to pair 0 in *both* rankings
      and record a false agreement -- the same degenerate-head trap that
      produced the false 100% on llama before it was fixed;
    * an all-zero surrogate does the same by a second route, and it is
      recorded per row because one content pair can exclude the whole grid.

    The two exclusion counters overlap when both degeneracies hold at once,
    so neither counter, nor their sum, is claimed to be the number of
    excluded rows; a row casts at most one vote because ``live`` conjoins
    both negations.
    """
    e = np.asarray(exact, dtype=np.float64)
    s = np.asarray(surrogate, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(f"exact must be 2-D (deltas, pairs), got {e.ndim}D")
    if s.ndim != 1 or s.shape[0] != e.shape[1]:
        raise ValueError(
            f"surrogate must be 1-D of length {e.shape[1]}, got shape {s.shape}"
        )
    if e.shape[1] < 2:
        raise ValueError(f"a ranking needs at least 2 pairs, got {e.shape[1]}")

    n_rows = int(e.shape[0])
    argmax_s = int(np.argmax(s))
    rows_zero = np.all(e == 0.0, axis=1)
    surrogate_zero = bool(np.all(s == 0.0))
    live = ~rows_zero & ~surrogate_zero
    argmax_e = np.argmax(e, axis=1)
    agree = (argmax_e == argmax_s) & live
    k = max(1, int(round(TOP_DECILE * e.shape[1])))
    threshold = np.partition(e, -k, axis=1)[:, -k]
    in_decile = e >= threshold[:, None]
    decile = in_decile[np.arange(n_rows), argmax_s] & live
    return {
        "n_rows": n_rows,
        "n_live": int(live.sum()),
        "n_agree": int(agree.sum()),
        "n_decile": int(decile.sum()),
        "n_excluded_exact_zero": int(rows_zero.sum()),
        "n_excluded_surrogate_zero": n_rows if surrogate_zero else 0,
    }


def _zero_counts() -> dict:
    return dict.fromkeys(_VOTE_KEYS, 0)


def _add_counts(acc: dict, add: dict) -> None:
    for key in _VOTE_KEYS:
        acc[key] += add[key]


def measure_head_agreement(
    hv: RM.HeadVectors,
    freqs: np.ndarray,
    rope_dim: int,
    deltas: np.ndarray,
    n_pairs: int,
    seed: int,
    layer: int,
    head: int,
) -> dict:
    """Pooled vote counts over one head's content pairs."""
    rng = np.random.default_rng(seed)
    q = hv.q_hat[layer, head, :, :rope_dim]
    k = hv.k_hat[layer, head, :, :rope_dim]
    cos_mean, sin_mean = cos_sin_means(deltas, freqs)
    acc = _zero_counts()
    n_used = 0
    for m, n in RM._content_pairs(hv.n_tokens, n_pairs, rng):
        terms = pair_terms(q[m], k[n], freqs, int(deltas[0]))
        a_k, b_k = terms.aligned, terms.crossed
        angles = np.outer(freqs, deltas.astype(np.float64))
        exact = a_k[:, None] * np.cos(angles) + b_k[:, None] * np.sin(angles)
        surrogate = a_k * cos_mean + b_k * sin_mean
        _add_counts(acc, ranking_votes(exact.T, surrogate))
        n_used += 1
    if not n_used:
        return {}
    return {"layer": int(layer), "head": int(head), "n_content_pairs": n_used, **acc}


def measure_model(
    model_id: str = RM.DEFAULT_MODEL_IDS[0],
    seed: int = RM.DEFAULT_SEED,
    prompts: Sequence[str] = RM.DEFAULT_PROMPTS,
    deltas: np.ndarray | None = None,
    n_pairs: int = RM.N_CONTENT_PAIRS,
) -> dict:
    """The ranking-agreement measurement, on one trained checkpoint."""
    loaded = RM.load_model(model_id, seed=seed)
    cfgv = RM.validate_config(loaded.model)
    if not cfgv["checkpoint_inv_freq_found"]:
        raise RuntimeError(f"{model_id} has no rotary module, so it has no rotary pairs")
    rope_dim = int(cfgv["rope_dim"])
    freqs = inv_freq(rope_dim, float(cfgv["rope_theta_declared"]))

    if deltas is None:
        deltas = np.array(DISTANCE_GRID, dtype=np.int64)

    per_prompt = []
    for i, text in enumerate(prompts):
        hv = RM.collect_head_vectors(loaded.model, loaded.tokenizer, text)
        heads = []
        for layer in range(hv.n_layers):
            for head in range(hv.n_heads):
                rec = measure_head_agreement(
                    hv, freqs, rope_dim, deltas, n_pairs,
                    seed + i * 1000 + layer * 31 + head,
                    layer, head,
                )
                if rec:
                    heads.append(rec)
        per_prompt.append({"prompt_index": i, "n_tokens": hv.n_tokens, "heads": heads})

    return _summarise_model(model_id, rope_dim, deltas, per_prompt)


def _summarise_model(
    model_id: str, rope_dim: int, deltas: np.ndarray, per_prompt: list
) -> dict:
    acc = _zero_counts()
    n_heads = 0
    for entry in per_prompt:
        for h in entry["heads"]:
            _add_counts(acc, h)
            n_heads += 1
    votes = acc["n_live"]
    if not votes:
        raise RuntimeError(f"{model_id}: no live votes were cast")
    n_pairs = rope_dim // 2
    return {
        "model_id": model_id,
        "rope_dim": rope_dim,
        "n_rotary_pairs": n_pairs,
        "chance_top1": 1.0 / n_pairs,
        "n_deltas": int(deltas.size),
        "deltas": [int(d) for d in deltas],
        "n_heads": n_heads,
        "n_votes": votes,
        "n_agree": acc["n_agree"],
        "n_decile": acc["n_decile"],
        "n_excluded_exact_zero": acc["n_excluded_exact_zero"],
        "n_excluded_surrogate_zero": acc["n_excluded_surrogate_zero"],
        "top1_agreement": acc["n_agree"] / votes,
        "top1_agreement_top_decile": acc["n_decile"] / votes,
        "per_prompt": per_prompt,
    }


def run_all(model_ids: Sequence[str] = RM.DEFAULT_MODEL_IDS) -> dict:
    models = [measure_model(m) for m in model_ids]
    # The per-model summaries report their own counts (``n_votes`` holds the
    # live-vote total); the pooled record keeps the same vocabulary the model
    # summary uses, and the fractions are recomputed from the summed counts.
    agree = int(sum(m["n_agree"] for m in models))
    decile = int(sum(m["n_decile"] for m in models))
    live = int(sum(m["n_votes"] for m in models))
    ex_zero = int(sum(m["n_excluded_exact_zero"] for m in models))
    ex_szero = int(sum(m["n_excluded_surrogate_zero"] for m in models))
    if not live:
        raise RuntimeError("no live votes were cast on any checkpoint")
    return {
        "schema": "rope_attribution/agreement/v1",
        "distance_grid": list(DISTANCE_GRID),
        "seed": RM.DEFAULT_SEED,
        "top_decile": TOP_DECILE,
        "models": models,
        "pooled": {
            "n_live": live,
            "n_agree": agree,
            "n_decile": decile,
            "n_excluded_exact_zero": ex_zero,
            "n_excluded_surrogate_zero": ex_szero,
            "top1_agreement": agree / live,
            "top1_agreement_top_decile": decile / live,
        },
    }


def main() -> None:
    data = run_all()
    for m in data["models"]:
        print(
            f"  {m['model_id']}: top1 {100.0 * m['top1_agreement']:.1f}% of "
            f"{m['n_votes']} votes (chance {100.0 * m['chance_top1']:.2f}%, "
            f"top decile {100.0 * m['top1_agreement_top_decile']:.1f}%), "
            f"excluded {m['n_excluded_exact_zero']} all-zero rows"
        )
    print(
        f"pooled: top1 {100.0 * data['pooled']['top1_agreement']:.1f}% of "
        f"{data['pooled']['n_live']} live votes"
    )
    out = Path(__file__).resolve().parents[2] / "results" / "agreement.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
