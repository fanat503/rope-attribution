# Theory and proofs for *Attributing RoPE*

Author: Slyatski Ilya

This document is the self-contained mathematical companion to `paper/main.tex`.
It states the exact identities, proves the position-conditionality result, and
derives the least-squares position-free surrogate used by the measurements. The
numbering is independent of the paper; the corresponding paper objects are named
in each section.

Notation follows the paper and the implementation:

- `d` is an even head dimension, `h = d/2`;
- `R_m` is the block-diagonal RoPE rotation at absolute position `m`;
- `invf_k = base^(-2k/d)` for `k = 0, ..., h-1`;
- `q = W_Q x_m` and `k = W_K x_n` are the projections **before** rotation;
- `delta = n - m` is the relative distance.

The implementation uses the split-half convention: channel `k` is paired with
channel `k + h`. All identities below are for that convention. The code that
instantiates them is `projects/rope_attribution/experiments.py::pair_terms` and
`projects/rope_attribution/rope.py`.

## 1. The relative-frame identity

**Proposition 1 (relative frame).** For every `m, n`,

```text
(R_m q)^T (R_n k) = q^T R_{n-m} k.
```

**Proof.** Each `R_m` is block diagonal with `2 x 2` blocks

```text
[ cos(m invf_k)  -sin(m invf_k) ]
[ sin(m invf_k)   cos(m invf_k) ],
```

and each block is orthogonal, so `R_m^T R_m = I`. Therefore

```text
(R_m q)^T (R_n k)
  = q^T R_m^T R_n k
  = q^T R_{n-m} k.
```

The last equality uses the angle addition formula: rotations about the same axis
compose by adding angles, and `R_a^T = R_{-a}`. QED.

This is exact for every finite `m, n`. It is not an approximation and it does not
depend on the content vectors.

## 2. The per-pair closed form

Partition `q = (q_1, q_2)` and `k = (k_1, k_2)` into their first and second
halves in `R^h`. Let `D_k = delta * invf_k`. The `k`-th rotation block acts on
the pair `(q[k], q[k+h])` and `(k[k], k[k+h])`. Direct expansion gives

```text
A_k = q[k] k[k] + q[k+h] k[k+h],
B_k = q[k+h] k[k] - q[k] k[k+h],
c_k(delta) = A_k cos(D_k) + B_k sin(D_k).
```

Equivalently,

```text
R_k = hypot(A_k, B_k),
psi_k = atan2(B_k, A_k),
c_k(delta) = R_k cos(D_k - psi_k),
```

and the full score is

```text
s(delta) = sum_{k=0}^{h-1} c_k(delta).
```

`A_k` and `B_k` are functions of the content vectors alone. `R_k` is the
**gate** of the pair; `psi_k` is its content-determined phase offset; the only
distance dependence is the angle `D_k = delta * invf_k`.

**Corollary 2 (norms are position-free).** `|R_m q| = |q|` and `|R_n k| = |k|`
for all positions. Hence the magnitudes carry no position information.

**Proof.** `R_m` is orthogonal, so `|R_m q|^2 = q^T R_m^T R_m q = q^T q`.
QED.

This is the norm-preservation half of the gate/phase split. It is stronger than
saying the magnitude is approximately unchanged: under RoPE it is exactly
invariant. The per-pair closed form above supplies the other half: `R_k` and
`psi_k` depend on the content alone, while `D_k` depends only on distance. A
pair can still have a non-constant contribution when `B_k = 0`; then

```text
c_k(delta) = A_k cos(delta invf_k),
```

which oscillates through the full range. `B_k = 0` is therefore **not** a
position-blind case. The threshold used in `real_model.BLIND_RATIO` is a
separate, explicitly approximate convention.

## 3. Exact additive attribution

Let the query be a non-negative combination of feature directions,

```text
q = sum_i f_i d_i^(q),     f_i >= 0,
```

and similarly `k = sum_j g_j d_j^(k)`. Bilinearity of the inner product gives,
for every fixed `delta`,

```text
s(delta)
  = (sum_i f_i d_i^(q))^T R_delta (sum_j g_j d_j^(k))
  = sum_{i,j} f_i g_j A_ij(delta),
```

where

```text
A_ij(delta) = (d_i^(q))^T R_delta d_j^(k).
```

This is exact. The residual reported in the paper is the floating-point residual
of evaluating the two equal expressions, not a modelling error. The softmax is
outside this identity: it normalizes scores across keys and is where
non-additivity of attention **weights** enters. The paper deliberately does not
claim additive attribution of the post-softmax weights.

## 4. Position-free coefficients are impossible

**Theorem 3 (conditionality).** Let `I` be a finite set of rotary pairs with
distinct positive frequencies `invf_k`. Define

```text
g(delta) = sum_{k in I} (A_k cos(delta invf_k) + B_k sin(delta invf_k)).
```

Then `g` is independent of `delta` if and only if `A_k = B_k = 0` for every
`k in I`.

**Proof.** (`=>`) Suppose `g(delta) = c` for all `delta`. Then `g - c = 0` on an
interval. The functions

```text
{1} union {cos(alpha delta), sin(alpha delta) : alpha in S}
```

with `S` a finite set of distinct positive frequencies are linearly independent.
One way to see this is to write each cosine and sine as a sum of complex
exponentials `exp(i alpha delta)` and `exp(-i alpha delta)`. The frequencies
`{0} union {+alpha, -alpha : alpha in S}` are distinct, and distinct complex
exponentials are linearly independent (their Wronskian is a nonzero
Vandermonde-type determinant, or equivalently a nontrivial finite combination
has a unique almost-periodic expansion). Hence every coefficient of `g - c` is
zero: `c = 0` and `A_k = B_k = 0` for all `k`.

(`<=`) If every `A_k = B_k = 0`, then `g` is identically zero, hence independent
of `delta`. QED.

**Corollary 4 (a nonzero contribution changes sign).** If `g` is not identically
zero, it has zero almost-periodic mean:

```text
lim_{T -> infinity} (1/T) integral_0^T g(delta) d delta = 0,
```

because every `cos` and `sin` term has zero mean. A continuous, not-identically-
zero function with zero mean takes both positive and negative values: if
`g >= 0` everywhere, continuity and the fact that `g` is positive somewhere
would make its mean positive; the case `g <= 0` is symmetric. Therefore no
constant `c` has the same sign as `g` everywhere: at every point where
`g(delta) > 0` and `c <= 0`, or where `g(delta) < 0` and `c >= 0`, the sign is
wrong (with `c = 0` counted as having no sign). QED.

When the frequencies are commensurate, the same argument is a mean over a
common period. When they are not, the almost-periodic mean is the correct
replacement; the conclusion is unchanged.

This is the precise sense in which a position-free scalar is not merely
inaccurate: for a nonzero feature contribution there is no constant summary that
is sign-correct on the whole distance range.

The theorem is about a feature's contribution across **all** distances. On a
finite grid a contribution can happen to be sign-constant over the sampled
points; the paper's 54-point grid was chosen before looking and every one of the
eight features changes sign on it. The empirical statement is therefore stronger
than "there exists an unlucky grid".

## 5. The best position-free scalar, and its error

For one rotary pair, let `c_n = c(delta_n)` on a finite grid
`delta_1, ..., delta_N`. The best constant in the least-squares sense is

```text
c* = argmin_c sum_n (c - c_n)^2 = (1/N) sum_n c_n.
```

**Proof.** The objective is a convex quadratic in `c`; its derivative is
`2 sum_n (c - c_n)`, which vanishes exactly at the mean. QED.

Substituting the closed form,

```text
c* = A_k * mean_n cos(delta_n invf_k)
   + B_k * mean_n sin(delta_n invf_k).
```

This is exactly the surrogate implemented in
`projects/rope_attribution/usefulness.py::measure_pair_cost`. It is the best
constant for that pair on that grid, not a strawman.

For one pair the largest possible deviation is bounded by the peak-to-peak
range:

```text
|c_n - c*| <= max_n c_n - min_n c_n <= 2 R_k,
```

because `c*` is a convex combination of the `c_n` and therefore lies inside
`[-R_k, R_k]`. The bound `2 R_k` is attained in the limit by a grid that puts
almost all its mass near one extreme and one point near the other. The important
consequence is that the **per-pair** error is bounded by twice the gate. A
position-free scalar can therefore look reasonable pair by pair and still fail
badly for a feature, because a feature is a sum of pairs and the sum can cancel.

For a feature,

```text
g_i(delta) = sum_{k in I_i} c_k(delta),
c_i* = sum_{k in I_i} c_k*,
```

and

```text
|g_i(delta) - c_i*|
  <= sum_{k in I_i} |c_k(delta) - c_k*|
  <= 2 sum_{k in I_i} R_k.
```

The denominator that matters is the feature's own peak,
`max_delta |g_i(delta)|`, which can be much smaller than `sum_k R_k` through
cancellation. Hence the relative error

```text
max_delta |g_i(delta) - c_i*| / max_delta |g_i(delta)|
```

has no universal bound of `2`; cancellation can make it arbitrarily large. This
is the conditioning issue behind the measured median relative error of `1.13`
in `results/usefulness.json`: the numerator is a sum of per-pair errors, while
the denominator is the signed total, and the total can be small exactly where the
pairs disagree.

## 6. Position-free summaries collapse at `theta * L = 2*pi`

Section 5 prices the best position-free scalar on a *fixed* grid and reports one
number. This section replaces the number with the function it is a value of, so
that the rate can be predicted rather than tabulated.

Fix one rotary pair. Its contribution is

```text
c(delta) = rho cos(theta delta - psi),   rho > 0, theta > 0,
```

with `rho` and `psi` fixed by the content (Section 2). Let `W = [a, a + L]` be a
window of distances and put `x = theta L`, the number of radians the pair sweeps
across it. Define the minimax error of a position-free summary as

```text
E(W)   = min_{c0 in R} max_{delta in W} |c(delta) - c0|,
E_rel  = E(W) / rho.
```

`E` is the error of the *best possible* constant, so every particular choice of
summary --- the per-feature mean, the least-squares fit, the value at one chosen
distance --- is at least this wrong.

The window is treated as an interval of distances. The model realizes integer
distances only; the interval is the range the summary is fitted over, which is
what a practitioner chooses. This is stated because it is a modelling choice, not
an identity.

**Proposition 5 (threshold).**

1. If `x >= 2 pi`, then `E_rel = 1` exactly.
2. If `x < 2 pi`, then `E_rel >= (1 - cos(x/2)) / 2`, with equality when `W` is
   centred on a maximum or a minimum of `c`.
3. As the placement of `W` varies, the achievable values of `E_rel` fill exactly
   the interval

   ```text
   [ (1 - cos(x/2)) / 2 ,  sin(min(x, pi)/2) ].
   ```

   So (1) and (2) are the two ends of one interval: for `x < 2 pi` the best
   placement is strictly better than the worst, and at `x = 2 pi` the interval
   collapses to the single point `1`, where *every* placement is equally bad.

**Proof of (1).** The phase map `delta -> theta delta - psi` sends `W` onto an
interval of length `x >= 2 pi`. Any interval of that length contains a phase
congruent to `0` and a phase congruent to `pi` modulo `2 pi`, so `c` attains both
`rho` and `-rho` on `W`. For any `c0`,

```text
max(|rho - c0|, |-rho - c0|) = rho + |c0| >= rho,
```

with equality at `c0 = 0`. Hence `E = rho`. QED.

**Proof of (2).** Scale `rho = 1`. Let `I` be the phase interval, of length
`x < 2 pi`, and let `R(I) = max_I cos - min_I cos`, so that `E = R(I)/2`. It
suffices to show `R(I) >= 1 - cos(x/2)`.

*Case (a): `I` contains a phase congruent to `0`.* Then `max_I cos = 1`. Let
`d` be the larger of the two distances from that phase to the endpoints of `I`;
since the two distances sum to `x`, `d >= x/2`. If `d <= pi`, the endpoint at
distance `d` has value `cos d <= cos(x/2)`, because `cos` is decreasing on
`[0, pi]` and `x/2 <= d <= pi`. If `d > pi`, then `I` also contains the phase at
distance `pi`, where `cos = -1 <= cos(x/2)`. Either way `min_I cos <= cos(x/2)`,
so `R(I) >= 1 - cos(x/2)`.

*Case (b): `I` contains a phase congruent to `pi`.* Then `min_I cos = -1`. With
`d >= x/2` the larger endpoint distance from that phase: if `d <= pi`, the
endpoint value is `-cos d >= -cos(x/2)`, so `max_I cos >= -cos(x/2)` and
`R(I) >= 1 - cos(x/2)`; if `d > pi`, `I` contains the adjacent peak, so
`max_I cos = 1` and `R(I) = 2 >= 1 - cos(x/2)`.

*Case (c): `I` contains neither.* Then `cos` is strictly monotone on `I`, so `I`
lies inside a monotone branch, whose length is `pi`; in particular `x < pi`. Write
`I = [u, u + x]` with `0 < u` and `u + x < pi`. Then

```text
R(I) = cos u - cos(u + x) = 2 sin(u + x/2) sin(x/2) >= 2 sin^2(x/2)
     = 1 - cos x >= 1 - cos(x/2),
```

using `u + x/2 >= x/2` and `sin` increasing on `[0, pi/2]`, then `x > x/2` and
`cos` decreasing on `[0, pi]`.

Equality in case (a) requires `d = x/2`, i.e. both endpoints equidistant from the
peak: `W` centred on a maximum. Symmetrically in case (b) for a minimum. QED.

**Proof of (3).** Scale `rho = 1` and let `I = [a, a + x]` be the phase interval,
so `E_rel = R(I)/2` with `R` the range of `cos` on `I`. The set of ranges
attained as `a` varies is an interval because `a -> R(I_a)` is continuous, so it
suffices to identify its endpoints.

*Lower endpoint.* Proposition 5(2) gives `R >= 1 - cos(x/2)` for every `a`, and
the centred placement attains it. Nothing more to show.

*Upper endpoint, `x <= pi`.* Three cases.
If `I` contains a phase congruent to `0`, then `max_I cos = 1`. Every point of
`I` lies within `x` of that phase, and `x <= pi`, so `cos >= cos(x)` throughout
and `R <= 1 - cos x = 2 sin^2(x/2) <= 2 sin(x/2)`.
If `I` contains a phase congruent to `pi`, the same argument with the sign
reversed gives `max_I cos <= -cos(x)` and again `R <= 1 - cos x`.
If `I` contains neither, `cos` has no interior extremum on `I` and is therefore
monotone there, so `R = |cos a - cos(a+x)| = 2 |sin(a + x/2)| sin(x/2) <= 2 sin(x/2)`.
The bound is attained by `a + x/2 = pi/2`, i.e. by centring `I` on an
inflection point of `cos`.

*Upper endpoint, `pi <= x < 2 pi`.* `R <= 2` always, and `R = 2` is attained by
any interval of length at least `pi`, which necessarily contains both a phase
congruent to `0` and one congruent to `pi`. Since `sin(min(x, pi)/2) = sin(pi/2)
= 1` in this range, the stated endpoint is `2` in units of `R`, i.e. `1` in
units of `E_rel`.

*`x >= 2 pi`.* Both endpoints are `1` by Proposition 5(1). QED.

Two consequences are worth naming because the paper uses them. First, the ceiling
is reached *only* at `x >= 2 pi` for the best placement, but the *worst* placement
already reaches it at `x = pi`; a number reported without its window placement is
therefore ambiguous below `2 pi` and unambiguous above it. Second, `E_rel` is
continuous and non-decreasing in `x` at fixed placement, and attains every value
in `[0, 1]` as `x` runs over `[0, 2 pi]`, so every intermediate rate is
achievable by choosing a window --- which is why a rate quoted without its window
is not a property of RoPE, only of the grid it was measured on.

**Corollary 6 (the least-squares surrogate inherits the ceiling).** The mean of
`c` over `W` is `c_bar = rho (sin b - sin a) / x`, where `[a, b]` is the phase
interval, so

```text
|c_bar| <= 2 rho / x.
```

For `x >= 2 pi` the minimax constant is `0` by Proposition 5, hence

```text
|c_bar - c*| <= 2 rho / x  ->  0.
```

So the least-squares surrogate used by `usefulness.py` is not a poor choice of
fitting criterion relative to the minimax one: past the threshold both are at the
ceiling, and the ceiling is "this pair contributes nothing". The gap between them
is the price of least squares, and it vanishes as the window grows.

**Corollary 7 (why partial RoPE hides the collapse at score level).** Let a head
have rotary pairs `K` and unrotated tail `t = q_tail . k_tail`, which does not
depend on `delta`. Then

```text
score(delta) = sum_{k in K} c_k(delta) + t,
error(delta) = sum_{k in K} (c_hat_k - c_k(delta)),
```

and the score-level relative error is `max_delta |error| / max_delta |score|`.
The tail appears in the denominator and nowhere in the numerator. Adding a
constant to the denominator leaves the numerator unchanged and therefore
*reduces* the relative error, whatever the pairs are doing. At `t = 0` (full
RoPE) there is no such constant: the denominator is the signed sum of the same
sinusoids that appear in the numerator, so no dilution is available, and the two
rates must agree up to cancellation between pairs.

This is the p-dependence in one line: the gap between the pair-level rate
(`21.4%` on `pythia-70m`) and the score-level rate (`0.017%` in the same
artefact) is a property of the architecture's rotary fraction (`rotary_pct =
0.25` there), not a contradiction between two measurements.
`results/collapse.json` measures the dilution per checkpoint.

**Status of these statements.** Parts (1) and (2) are the textbook fact that the
best uniform approximation of a continuous function by a constant is its
midrange, plus the elementary range of a cosine on an interval; see Rivlin,
*An Introduction to the Approximation of Functions*, and Powell,
*Approximation Theory and Methods*. They are **not** claimed as new results of
this paper. What is claimed is the consequence: the sign-error rate reported in
Section 5 is a value of this function, the threshold at `theta L = 2 pi` is
where it saturates, and the ratio between the rotary-pair rate and the
feature-level rate is the rotary fraction. Those are checkable statements about
this object, and `results/collapse.json` checks them on three checkpoints.

## 7. What the measurements check, and what they do not

The paper's numerical checks instantiate the statements above:

- `results/measurements.json` and `figures/fig09_position_conditional_attribution.csv`
  check the per-pair closed form, the relative-position identity, norm
  preservation, bilinearity, and the eight sign changes.
- `results/usefulness.json` checks the least-squares surrogate on three trained
  checkpoints and reports the sign-error rate; `results/usefulness_grid_independence.json`
  checks that the rate is not an artefact of grid density.
- `results/real_model.json` repeats the identities on trained activations and
  reports the dtype floor.
- `results/collapse.json` checks Section 6 on three checkpoints: the channel-level
  minimax error is exactly `1` for every pair past `theta L = 2 pi`, the
  sub-threshold floor `(1 - cos(x/2))/2` is respected by every measured pair,
  and the score-level rate is the channel-level rate diluted by the unrotated
  tail share.

What the theory does **not** establish:

1. It does not claim that a downstream language-model task is harmed by a
   position-free attribution; that is an empirical, model-specific question and
   is outside the paper's claims.
2. It does not claim that every feature in a real model changes sign on every
   grid; the theorem is about a nonzero contribution over a full period, and the
   measured 54-point grid is the pre-registered test.
3. It does not claim that the softmax preserves additivity. The opposite is
   true and is stated in the paper.
4. It does not turn a finite-grid measurement into a proof. The proofs are
   analytic; the numbers are measurements with their own artifacts and tests.

## 8. Numerical conventions

The implementation evaluates in float64 for the synthetic identities and on
float32 activations for the trained-checkpoint identities, widening to float64
only for the reference computation. The residuals are therefore bounded below by
the dtype of the activations, not by the algebra. The per-pair code uses
`hypot(A_k, B_k)` rather than `sqrt(A_k^2 + B_k^2)` to avoid overflow in
intermediate squares; this is a numerical detail, not a change to the
mathematics.
