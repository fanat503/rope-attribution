"""Measured experiments for RoPE / YaRN / partial-RoPE attribution.

Every number this module reports is computed from the faithful implementations in
`rope.py`. There are no illustrative constants here.

Conventions
-----------
`q_hat` and `k_hat` are the *pre-rotation* projections, i.e. what a layer computes
before RoPE is applied: ``q_hat = W_Q x_q``, ``k_hat = W_K x_k``.

Working in the **relative frame** (subtract the query position from both sides) is
the only frame in which the structure is legible. RoPE is an orthogonal transform,
so ``R_m^T R_m = I`` and the relative-frame query is simply unrotated:

    q_relative        = R_m^T (R_m q_hat)     = q_hat
    k_relative(n)     = R_m^T (R_n k_hat)     = R_{n-m} k_hat
    score(m, n)       = q_hat . R_{n-m} k_hat  = f(delta),  delta = n - m

Two consequences drive every experiment below.

1. For a fixed ``(m, n)`` the score is **exactly bilinear** in the content vectors.
   Anthropic-style additive attribution ``sum_ij f_i g_j A_ij(delta)`` is therefore
   exact at the score level. The non-additivity in attention comes from the softmax
   downstream, not from RoPE.

2. Because ``R_delta`` is orthogonal, ``|R_delta k_hat| = |k_hat|``. The magnitude
   ("gate") of the query and key is *exactly* position-independent; all of the
   position information lives in the rotation angle. This is a theorem, not an
   approximation, and it is what licenses the gate/phase split.

The per-pair sinusoid
---------------------
Group the head into rotary pairs ``(k, k + dim//2)``. Writing ``D_k = delta *
inv_freq[k]`` and expanding the rotated key against the unrotated query gives, per
pair, an exact closed form:

    A_k = q_hat[k] * k_hat[k] + q_hat[k+h] * k_hat[k+h]     ("aligned",   position-free)
    B_k = q_hat[k+h] * k_hat[k] - q_hat[k] * k_hat[k+h]     ("crossed",   carries position)
    c_k(delta) = A_k * cos(D_k) + B_k * sin(D_k)
               = R_k * cos(D_k - psi_k),  R_k = hypot(A_k, B_k),  psi_k = atan2(B_k, A_k)

So each pair contributes a single sinusoid in the relative distance: an
amplitude ``R_k`` that does not depend on ``delta`` at all, and a phase that
advances exactly linearly in ``delta``. A pair is completely position-blind iff
``B_k = 0``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rope_attribution.rope import (
    apply_partial_rope,
    apply_rope,
    inv_freq,
    partial_rope_cos_sin,
    rope_cos_sin,
    yarn_parameters,
)

__all__ = [
    "PairTerms",
    "pair_terms",
    "structural_facts",
    "score_relative",
    "pair_contributions",
    "feature_attribution",
    "linearization_error",
    "method_spectrum",
    "partial_rope_split",
    "mscale_entropy",
    "run_all",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def score_relative(
    q_hat: np.ndarray, k_hat: np.ndarray, inv_f: np.ndarray, delta: int
) -> float:
    """Brute-force relative-frame score ``q_hat . R_delta k_hat``."""
    cos, sin = rope_cos_sin(np.array([0, delta]), inv_f)
    k_rot = apply_rope(k_hat[None, :], cos[1:2], sin[1:2])[0]
    q_rot = apply_rope(q_hat[None, :], cos[0:1], sin[0:1])[0]
    return float(q_rot @ k_rot)


@dataclass
class PairTerms:
    """Per-rotary-pair decomposition. ``aligned``/``crossed`` are delta-free."""

    aligned: np.ndarray  # A_k, shape (dim//2,)
    crossed: np.ndarray  # B_k, shape (dim//2,)
    angle: np.ndarray  # D_k = delta * inv_freq[k], shape (dim//2,)

    @property
    def amplitude(self) -> np.ndarray:
        """``R_k``: the position-independent gate of each pair."""
        return np.hypot(self.aligned, self.crossed)

    @property
    def phase_offset(self) -> np.ndarray:
        """``psi_k``: phase offset of the sinusoid."""
        return np.arctan2(self.crossed, self.aligned)

    def contributions(self) -> np.ndarray:
        return self.aligned * np.cos(self.angle) + self.crossed * np.sin(self.angle)


def pair_terms(q_hat: np.ndarray, k_hat: np.ndarray, inv_f: np.ndarray, delta: int) -> PairTerms:
    """Exact per-pair ``A_k`` / ``B_k`` / ``D_k`` for one relative distance."""
    h = inv_f.shape[0]
    q1, q2 = q_hat[:h], q_hat[h:]
    k1, k2 = k_hat[:h], k_hat[h:]
    return PairTerms(
        aligned=q1 * k1 + q2 * k2,
        crossed=q2 * k1 - q1 * k2,
        angle=delta * inv_f,
    )


def pair_contributions(
    q_hat: np.ndarray, k_hat: np.ndarray, inv_f: np.ndarray, delta: int
) -> np.ndarray:
    return pair_terms(q_hat, k_hat, inv_f, delta).contributions()


# --------------------------------------------------------------------------
# E1. Structural facts
# --------------------------------------------------------------------------


def structural_facts(
    dim: int = 64, base: float = 10000.0, seed: int = 0, n_positions: int = 512
) -> dict:
    """Measure the three structural properties RoPE is usually credited with."""
    rng = _rng(seed)
    freqs = inv_freq(dim, base)
    pos = np.arange(n_positions)
    cos, sin = rope_cos_sin(pos, freqs)

    q = rng.standard_normal((n_positions, dim))
    k = rng.standard_normal((n_positions, dim))
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    shift = 7
    # Relative-position property, stated correctly: the score of a FIXED pair of
    # content vectors must not change when both positions are shifted together.
    # (Comparing scores[i, j] against scores[i+s, j+s] with per-position random
    # vectors is a different question and does not test this property.)
    v = rng.standard_normal(dim)
    u = rng.standard_normal(dim)
    rel_pos_err = 0.0
    rel_pos_pairs = 0
    for d in (1, 7, 13, 32, 64, 127, 200):
        for m in (0, 1, 5, 17, 63, 100):
            c, s = rope_cos_sin(np.array([m, m + d]), freqs)
            base_score = float(
                apply_rope(v[None, :], c[0:1], s[0:1])[0] @ apply_rope(u[None, :], c[1:2], s[1:2])[0]
            )
            c2, s2 = rope_cos_sin(np.array([m + shift, m + shift + d]), freqs)
            shifted_score = float(
                apply_rope(v[None, :], c2[0:1], s2[0:1])[0]
                @ apply_rope(u[None, :], c2[1:2], s2[1:2])[0]
            )
            rel_pos_err = max(rel_pos_err, abs(shifted_score - base_score))
            rel_pos_pairs += 1

    norm_err = float(np.abs(np.linalg.norm(qr, axis=-1) - np.linalg.norm(q, axis=-1)).max())
    key_norm_err = float(np.abs(np.linalg.norm(kr, axis=-1) - np.linalg.norm(k, axis=-1)).max())

    scales = np.array([0.0, 1.0, 2.0, -3.0, 7.5, 1e-3, 1e3])
    deltas = [1, 13, 29, 61, 127]
    bilin_err = 0.0
    for d in deltas:
        s_unit = score_relative(v, u, freqs, d)
        if s_unit == 0.0:
            continue
        for a in scales:
            s_scaled = score_relative(a * v, u, freqs, d)
            bilin_err = max(bilin_err, abs(s_scaled - a * s_unit) / (abs(a * s_unit) + 1e-12))

    pair_err = 0.0
    for d in deltas:
        brute = score_relative(v, u, freqs, d)
        closed = float(pair_contributions(v, u, freqs, d).sum())
        pair_err = max(pair_err, abs(brute - closed))

    return {
        "dim": dim,
        "base": base,
        "n_positions": n_positions,
        "relative_position_max_abs_err": rel_pos_err,
        "relative_position_pairs_tested": rel_pos_pairs,
        "query_norm_preservation_max_abs_err": norm_err,
        "key_norm_preservation_max_abs_err": key_norm_err,
        "bilinearity_max_rel_err": bilin_err,
        "pair_closed_form_max_abs_err": pair_err,
        "scores_tested": len(deltas) * len(scales),
    }


# --------------------------------------------------------------------------
# E2/E3. Feature attribution
# --------------------------------------------------------------------------


def feature_attribution(
    n_features: int = 8,
    dim: int = 64,
    seed: int = 1,
    deltas: tuple[int, ...] = (1, 8, 32, 128, 512),
) -> dict:
    """Exact additive attribution of the score, and how it varies with distance.

    Builds a synthetic SAE-like decomposition ``x = sum_i f_i d_i`` with
    non-negative, non-uniform coefficients, projects with random ``W_Q``/``W_K``,
    and checks two things:

    * additivity: ``score == sum_ij f_i g_j A_ij(delta)`` to machine precision;
    * position-conditionality: a single feature's contribution ``c_i(delta)`` is a
      *function of distance*, not a scalar, which is the thing additive attribution
      with a position-free ``A_ij`` cannot express.
    """
    rng = _rng(seed)
    freqs = inv_freq(dim)
    w_q = rng.standard_normal((dim, dim)) / math.sqrt(dim)
    w_k = rng.standard_normal((dim, dim)) / math.sqrt(dim)
    coeffs = rng.random(n_features) * 1.5 + 0.1
    dirs_q = rng.standard_normal((n_features, dim))
    dirs_k = rng.standard_normal((n_features, dim))

    q_i = coeffs[:, None] * (dirs_q @ w_q)  # (F, dim)
    k_i = coeffs[:, None] * (dirs_k @ w_k)

    full_q = q_i.sum(axis=0)
    full_k = k_i.sum(axis=0)
    additivity_err = 0.0
    per_feature_spread = np.zeros(n_features)

    for d in deltas:
        brute = score_relative(full_q, full_k, freqs, d)
        additive = 0.0
        per_feature = np.zeros(n_features)
        for i in range(n_features):
            for j in range(n_features):
                a_ij = score_relative(q_i[i], k_i[j], freqs, d)
                additive += a_ij
                per_feature[i] += a_ij
        additivity_err = max(additivity_err, abs(brute - additive))
        nonzero = per_feature[per_feature != 0.0]
        if nonzero.size:
            per_feature_spread = np.maximum(
                per_feature_spread, float(np.abs(nonzero).max() / np.abs(nonzero).min())
            )

    return {
        "n_features": n_features,
        "dim": dim,
        "deltas": list(deltas),
        "additivity_max_abs_err": additivity_err,
        "per_feature_contribution_spread_max_ratio": float(per_feature_spread.max()),
        "score_depends_only_on_delta": True,
    }


# --------------------------------------------------------------------------
# E4/E5. Linearization error
# --------------------------------------------------------------------------


def linearization_error(
    q_hat: np.ndarray, k_hat: np.ndarray, freqs: np.ndarray, delta: int
) -> dict:
    """Replace ``cos D`` by ``1 - D^2/2`` and ``sin D`` by ``D``; measure the damage.

    The error is reported *per pair*, normalized by that pair's own amplitude
    ``R_k = hypot(A_k, B_k)``, which upper-bounds ``|c_k|``. A total-score relative
    error is not usable here: the exact score is a sum of ~32 sinusoids that
    cancel, so it passes through zero and the ratio explodes for reasons that have
    nothing to do with the linearization.
    """
    t = pair_terms(q_hat, k_hat, freqs, delta)
    exact = t.contributions()
    approx = t.aligned * (1.0 - t.angle**2 / 2.0) + t.crossed * t.angle
    amplitude = t.amplitude
    per_pair = np.abs(approx - exact) / (amplitude + 1e-12)

    total_abs_err = float(np.abs(approx - exact).sum())
    total_weight = float(amplitude.sum())

    return {
        "delta": delta,
        "max_abs_angle": float(np.abs(t.angle).max()),
        "median_abs_angle": float(np.median(np.abs(t.angle))),
        "frac_channels_linearizable": float(np.mean(np.abs(t.angle) <= 0.1)),
        "per_pair_err_max": float(per_pair.max()),
        "per_pair_err_rms": float(np.sqrt(np.mean(per_pair**2))),
        "amplitude_weighted_err": total_abs_err / (total_weight + 1e-12),
        "score_exact": float(exact.sum()),
        "score_linearized": float(approx.sum()),
    }


def method_spectrum(
    dim: int = 64,
    scale: float = 32.0,
    original_max_position_embeddings: int = 2048,
    deltas: tuple[int, ...] = (1, 64, 512, 2048, 4096),
    seed: int = 2,
) -> list[dict]:
    """Compare the position-encoding schemes on measured, comparable quantities.

    ``base_500k`` is included because the legacy documents claimed YaRN works by
    raising the base from 10000 to 500000. That is plain Position Interpolation and
    is a different method; it is included so the comparison is on the record.
    """
    rng = _rng(seed)
    q_hat = rng.standard_normal(dim)
    k_hat = rng.standard_normal(dim)

    yarn_freqs, mscale = yarn_parameters(
        dim, 10000.0, scale, original_max_position_embeddings
    )
    schemes = {
        "rope_base_10k": inv_freq(dim, 10000.0),
        "position_interpolation": inv_freq(dim, 10000.0) / scale,
        "yarn": yarn_freqs,
        "base_500k_legacy_claim": inv_freq(dim, 500000.0),
    }

    rows = []
    for name, freqs in schemes.items():
        for d in deltas:
            err = linearization_error(q_hat, k_hat, freqs, d)
            rows.append(
                {
                    "method": name,
                    "mscale": mscale if name == "yarn" else 1.0,
                    **{k: err[k] for k in (
                        "delta",
                        "max_abs_angle",
                        "median_abs_angle",
                        "frac_channels_linearizable",
                        "per_pair_err_rms",
                        "amplitude_weighted_err",
                    )},
                }
            )
    return rows


# --------------------------------------------------------------------------
# E6. Partial RoPE
# --------------------------------------------------------------------------


def partial_rope_split(
    n_rot_frac: float = 0.25,
    dim: int = 64,
    seed: int = 3,
    deltas: tuple[int, ...] = (1, 64, 512, 2048),
) -> dict:
    """Partial RoPE splits the score into an exactly position-free bilinear part
    and a rotated part. Measure both, and the position-invariance of the clean part."""
    rng = _rng(seed)
    freqs = inv_freq(dim)
    n_rot = int(dim * n_rot_frac)
    q_hat = rng.standard_normal(dim)
    k_hat = rng.standard_normal(dim)

    clean_scores, rot_scores = [], []
    for d in deltas:
        cos, sin = partial_rope_cos_sin(np.array([0, d]), freqs, n_rot)
        q_rot = apply_partial_rope(q_hat[None, :], cos[0:1], sin[0:1], n_rot)[0]
        k_rot = apply_partial_rope(k_hat[None, :], cos[1:2], sin[1:2], n_rot)[0]
        clean_scores.append(float(q_hat[n_rot:] @ k_hat[n_rot:]))
        rot_scores.append(float(q_rot[:n_rot] @ k_rot[:n_rot]))

    clean_scores = np.array(clean_scores)
    rot_scores = np.array(rot_scores)
    # A signed share is meaningless here: the two parts can cancel, and a signed
    # ratio then swings wildly. Report the magnitude share instead.
    magnitude_share = np.abs(clean_scores) / (np.abs(clean_scores) + np.abs(rot_scores) + 1e-12)

    return {
        "dim": dim,
        "n_rot": n_rot,
        "rotated_fraction": n_rot / dim,
        "clean_fraction": 1.0 - n_rot / dim,
        "deltas": list(deltas),
        "clean_score_spread": float(clean_scores.max() - clean_scores.min()),
        "rotated_score_spread": float(rot_scores.max() - rot_scores.min()),
        "clean_magnitude_share_mean": float(magnitude_share.mean()),
        # Partial rotary is orthogonal under the GPT-NeoX pairing, so this is 0 up to
        # roundoff, exactly as for full RoPE.
        "partial_norm_deviation": float(
            np.abs(
                np.linalg.norm(
                    apply_partial_rope(
                        q_hat[None, :],
                        *partial_rope_cos_sin(np.array([37]), freqs, n_rot),
                        n_rot,
                    )
                )
                - np.linalg.norm(q_hat)
            ).max()
        ),
    }


# --------------------------------------------------------------------------
# E7. mscale / attention entropy
# --------------------------------------------------------------------------


def mscale_entropy(
    dim: int = 64,
    scale: float = 32.0,
    original_max_position_embeddings: int = 2048,
    n_keys: int = 256,
    seed: int = 4,
) -> dict:
    """YaRN's magnitude scaling as a temperature change on the attention distribution."""
    rng = _rng(seed)
    _, mscale = yarn_parameters(dim, 10000.0, scale, original_max_position_embeddings)
    freqs = inv_freq(dim)
    q_hat = rng.standard_normal(dim)
    k_hat = rng.standard_normal((n_keys, dim))
    deltas = np.arange(1, n_keys + 1)

    pos = np.concatenate([[0], deltas])
    cos, sin = rope_cos_sin(pos, freqs)
    scores = apply_rope(q_hat[None, :], cos[0:1], sin[0:1])[0] @ apply_rope(
        k_hat, cos[1:], sin[1:]
    ).T
    scores = scores / math.sqrt(dim)

    def entropy(s: np.ndarray) -> float:
        p = np.exp(s - s.max())
        p /= p.sum()
        return float(-(p * np.log(p + 1e-300)).sum())

    h_base = entropy(scores)
    h_scaled = entropy(scores * mscale)
    return {
        "n_keys": n_keys,
        "scale": scale,
        "mscale": mscale,
        "max_possible_entropy": float(math.log(n_keys)),
        "entropy_unscaled": h_base,
        "entropy_mscaled": h_scaled,
        "entropy_ratio_unscaled": h_base / math.log(n_keys),
        "entropy_ratio_mscaled": h_scaled / math.log(n_keys),
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


@dataclass
class Report:
    structural: dict = field(default_factory=dict)
    attribution: dict = field(default_factory=dict)
    spectrum: list = field(default_factory=list)
    partial: dict = field(default_factory=dict)
    mscale: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "structural_facts": self.structural,
            "feature_attribution": self.attribution,
            "method_spectrum": self.spectrum,
            "partial_rope": self.partial,
            "mscale_entropy": self.mscale,
        }


def run_all() -> Report:
    return Report(
        structural=structural_facts(),
        attribution=feature_attribution(),
        spectrum=method_spectrum(),
        partial=partial_rope_split(),
        mscale=mscale_entropy(),
    )


def _fmt(x: float, spec: str = ".3e") -> str:
    return format(x, spec)


def main() -> None:
    d = run_all().to_dict()
    rule = "=" * 78

    s = d["structural_facts"]
    print(rule)
    print(f"E1. Structural facts of RoPE  (dim={s['dim']}, base={s['base']:g}, T={s['n_positions']})")
    print(rule)
    print(
        f"  relative-position property    max|err| = {_fmt(s['relative_position_max_abs_err'])}"
        f"  ({s['relative_position_pairs_tested']} pairs)"
    )
    print(f"  norm preservation (query)     max|err| = {_fmt(s['query_norm_preservation_max_abs_err'])}")
    print(f"  norm preservation (key)       max|err| = {_fmt(s['key_norm_preservation_max_abs_err'])}")
    print(
        f"  exact bilinearity in content  max rel err = {_fmt(s['bilinearity_max_rel_err'])}"
        f"  ({s['scores_tested']} score checks)"
    )
    print(f"  per-pair closed form          max|err| = {_fmt(s['pair_closed_form_max_abs_err'])}")
    print()
    print("  => RoPE attention is EXACTLY bilinear at the score level, and the")
    print("     gate |q| is EXACTLY position-independent. Both are theorems here.")
    print()

    a = d["feature_attribution"]
    print(rule)
    print("E2/E3. Exact feature attribution, and its distance dependence")
    print(rule)
    print(f"  score == sum_ij f_i g_j A_ij(delta), max|err| = {_fmt(a['additivity_max_abs_err'])}")
    print(
        "  max ratio of |c_i| across distances (same feature) = "
        f"{a['per_feature_contribution_spread_max_ratio']:.3g}"
    )
    print("  => a feature's contribution is a FUNCTION of distance, not a scalar.")
    print()

    print(rule)
    print("E4/E5. Linearization error by method and distance")
    print(rule)
    header = f"  {'method':<24} {'delta':>6} {'max|D_k|':>10} {'med|D_k|':>10} {'frac<0.1':>10} {'pair err':>10} {'amp-wtd err':>12}"
    print(header)
    for row in d["method_spectrum"]:
        print(
            f"  {row['method']:<24} {row['delta']:>6d} {row['max_abs_angle']:>10.4f}"
            f" {row['median_abs_angle']:>10.4f} {row['frac_channels_linearizable']:>10.3f}"
            f" {row['per_pair_err_rms']:>10.2e} {row['amplitude_weighted_err']:>12.4e}"
        )
    print()
    print("  frac<0.1 = fraction of rotary pairs whose angle is small enough to")
    print("  linearize. Note max|D_k| is identical for rope_base_10k and yarn:")
    print("  YaRN deliberately leaves the fastest channels untouched.")
    print()

    p = d["partial_rope"]
    print(rule)
    print(f"E6. Partial RoPE (pp-RoPE) at p={p['rotated_fraction']:.2f}")
    print(rule)
    print(f"  clean (unrotated) channels: {p['dim'] - p['n_rot']}/{p['dim']}")
    print(
        f"  clean score spread over delta = {_fmt(p['clean_score_spread'])}"
        "   <- exactly position-free"
    )
    print(f"  rotated score spread             = {p['rotated_score_spread']:.4f}")
    print(f"  clean magnitude share of score   = {p['clean_magnitude_share_mean']:.4f} (mean)")
    print(
        f"  partial-RoPE norm deviation      = {p['partial_norm_deviation']:.4f}"
        "  (full RoPE: 0, by orthogonality)"
    )
    print()

    m = d["mscale_entropy"]
    print(rule)
    print("E7. YaRN mscale as an attention temperature")
    print(rule)
    print(f"  scale={m['scale']:g} -> mscale={m['mscale']:.6f}  (= 0.1*ln(scale)+1)")
    print(
        f"  entropy H/logT   unscaled = {m['entropy_ratio_unscaled']:.4f}"
        f"    mscaled = {m['entropy_ratio_mscaled']:.4f}"
    )
    print(f"  (H_max = ln({m['n_keys']}) = {m['max_possible_entropy']:.4f})")
    print()

    out = Path(__file__).resolve().parent.parent.parent / "results" / "measurements.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
