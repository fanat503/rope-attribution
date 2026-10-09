# Attributing RoPE: exact score attribution, and what actually limits it

## TL;DR

Under rotary position embeddings an attention score is **exactly** bilinear in the
content vectors at fixed relative distance, so the additive feature decomposition
`score = Σ_ij f_i g_j A_ij(δ)` is exact — not approximate. But `A_ij` is a
function of the distance, and that function **reverses sign**: on the 54-point
grid all 8 of 8 features change sign somewhere in `δ ∈ [1, 8192]`. So a
position-free per-feature scalar is not merely inaccurate, it is *backwards*:
replacing `c_i(δ)` with its per-feature mean gets the sign wrong on **26.8%** of
per-pair, per-distance cells of three trained checkpoints, with a median relative
error of `1.13` against a peak of `1`. The well-posed object is per-feature and
per-distance — and it is closed-form, so it costs the same as the scalar it
replaces.

Three claims we explicitly do **not** make: the algebra is already published
(arXiv:2607.25507, §3–§4); RoPE internals are not under-explored in general; and
nothing here evaluates a downstream task.

## Dataflow

Every number in the paper traces to one of two generated artefacts. Nothing is
typed in by hand.

```
                       rope.py  (RoPE / YaRN / partial RoPE, numpy float64)
                          │
        ┌─────────────────┼──────────────────────┐
        │                 │                      │
  experiments.py    statistics.py          usefulness.py
  structural facts, grid-independent        what a position-free
  spectra, additivity,  │                  scalar costs, on real
  partial RoPE, mscale  │                  checkpoints
        │               │                      │
        ▼               ▼                      ▼
  results/           results/              results/
  measurements.json  statistics.json       usefulness.json
        │               │                      │
        │               └──────────┬───────────┘
        │                          │
        ▼                          ▼
   figures/*.png  ◄──── figures.py ────┘
   + CSV sidecars  (10 figures, every array plotted is shipped)
        │
        ▼
   tests/test_paper_claims.py   parses paper/main.tex, re-derives every literal
   tests/test_real_model.py     re-runs the checkpoint measurements
   tests/test_usefulness.py     re-prices the position-free alternative
```

The claim registry in `tests/test_paper_claims.py` is the audit: it locates each
number in `paper/main.tex` by pattern, compares it against the artefact, and
fails if a number in the paper has no claim behind it. Adding a number to the
paper without a test for it fails the suite.

## Reproduce everything

```bash
pip install -e . && pip install -r requirements-dev.txt
pytest -q                                        # 1298 tests
python -m projects.rope_attribution.experiments  # results/measurements.json
python -m projects.rope_attribution.statistics  # results/statistics.json
python -m projects.rope_attribution.figures      # figures/*.png + CSV sidecars
```

On a machine with GNU make, `make all` does all of the above and recompiles the
paper. The `make` targets are exercised by CI rather than only asserted here: the
workflow runs `make help` and `make results` and reports the diff against the
committed `results/`, so a broken target fails the build instead of this
README's promise failing quietly. `make help` lists every target.

The checkpoint measurements are opt-in because they need CPU torch, which is not
a test dependency:

```bash
pip install -r real_model_requirements.txt
python -m projects.rope_attribution.real_model    # results/real_model.json
python -m projects.rope_attribution.usefulness    # results/usefulness.json
```

Both committed artefacts are verified bit-identical across three independent
regenerations.

## Citation

```bibtex
@misc{slyatski2026attributingrope,
  title  = {Attributing Rotary Position Embeddings: Exact Score Attribution,
            and What Actually Limits It},
  author = {Slyatski, Ilya},
  year   = {2026},
  note   = {Preprint. Algebra standard: see arXiv:2607.25507 for the per-pair
             closed form this work attributes rather than derives.}
}
```

If you use this repository, also cite the paper it builds on — RoFormer
(arXiv:2104.09864) — and YaRN (arXiv:2309.00071), which it measures as a control.

## What is computed

```
projects/rope_attribution/
  rope.py         RoPE, YaRN, partial RoPE (numpy, float64)
  experiments.py  seven measured experiments
  statistics.py   grid-independence and seed-variance statistics
  usefulness.py   cost of the best position-free surrogate, on real checkpoints
  real_model.py   identities re-measured on three trained checkpoints
  figures.py      the figure set, every value computed at run time
tests/            1298 tests
figures/          ten figures, each with a CSV sidecar
results/          measurements.json, statistics.json, usefulness.json
```

There is an earlier body of exploratory work under `projects/frontier-01-*`.
**Some of those scripts print illustrative constants rather than computing
measurements, and their novelty claim is known to be false.** See
[Legacy material](#legacy-material) before citing anything from them. The code
under `projects/rope_attribution/` is the maintained implementation and is the
only part of this repository that should be relied on.

## The setting

Work in the relative frame: subtract the query position from both query and key.
Because the rotation is orthogonal, `R_m^T R_m = I`, so

```
q_relative      = R_m^T (R_m q_hat) = q_hat
k_relative(n)   = R_m^T (R_n k_hat) = R_{n-m} k_hat
score(m, n)     = q_hat . R_{n-m} k_hat          (depends only on delta = n - m)
```

Group the head into rotary pairs `(k, k + d/2)` and put `D_k = delta * inv_freq[k]`.
Expanding the rotated key against the unrotated query gives an exact closed form
per pair, with `A_k`, `B_k` fixed and only the angle depending on distance:

```
A_k = q_hat[k]  *k_hat[k]  + q_hat[k+d/2] *k_hat[k+d/2]      "aligned",  position-free
B_k = q_hat[k+d/2]*k_hat[k] - q_hat[k] *k_hat[k+d/2]          "crossed",  carries position
c_k(delta) = A_k cos(D_k) + B_k sin(D_k) = R_k cos(D_k - psi_k)
```

`R_k = hypot(A_k, B_k)` is a single sinusoid in distance: an amplitude that does
not depend on `delta` at all, and a phase that advances exactly linearly in it.
A pair is completely position-blind precisely when `B_k = 0`.

## Results

All measured at `dim = 64`, `base = 10000`, `scale = 32`. Full values in
`results/measurements.json`; the figures are in `figures/`.

**The structural properties hold exactly.**

| property | measured |
|---|---|
| relative-position property (42 position pairs) | `3.4e-14` |
| norm preservation, query and key | `1.8e-15` |
| bilinearity in content, 35 score checks | `7.4e-16` |
| per-pair closed form vs brute force | `3.6e-15` |
| feature additivity, `score == sum_ij f_i g_j A_ij(delta)` | `3.6e-14` |

**Attribution is exact, but position-conditional.** With `x = sum_i f_i d_i` and
random `W_Q`, `W_K`, the score decomposes additively into per-feature-pair terms
`A_ij(delta)` to `3.6e-14`. But `A_ij` depends on the distance, so a single
feature's contribution is a *function of distance*, not a scalar. Measured on the
54-point grid, **all eight features reverse sign** somewhere in
`delta in [1, 8192]`, while the magnitudes at the two ends of the range agree to
within a factor of `10^-0.08`. Summarising `c_i(delta)` by its per-feature mean
— the least-squares-optimal position-free choice — leaves a residual **15 orders
of magnitude** above the additivity residual, and that factor stays within
`15.0`–`15.2` as the grid is made 120x denser. This, not a failure of
bilinearity, is the real obstacle to position-free feature attribution under RoPE
(`figures/fig09_position_conditional_attribution.png`).

> The paper previously led with a worst-feature ratio of `920x`. That number was a
> sampling artefact — a `max/min` over a grid, so it grows without bound as the
> grid gets denser. `rope_attribution/statistics.py` reproduces the growth (a
> factor of 32 across six grid densities, non-monotone, seed standard deviation
> exceeding its own mean) and the paper withdrew the claim. `tests/test_paper_claims.py`
> fails if it is reinstated.

**YaRN does not shrink the largest angle, and does not come from a bigger base.**
`max|D_k|` is *identical* for plain RoPE and YaRN at every distance measured
(difference `0.0`, 54 distances) — YaRN deliberately leaves the fastest channel
untouched, so `D = 1 rad per token` there is unchanged. What YaRN does change is
the bulk of the spectrum, and with it the fraction of channels that can be
linearized:

| `delta` | scheme | `max|D_k|` | `median|D_k|` | fraction linearizable |
|---:|---|---:|---:|---:|
| 512 | RoPE base 10k | 512 | 5.97 | `0.0625` |
| 512 | YaRN | 512 | 2.67 | `0.3438` |
| 512 | position interpolation | 16 | 0.19 | `0.4375` |
| 4096 | RoPE base 10k | 4096 | 47.79 | `0.0000` |
| 4096 | YaRN | 4096 | 21.34 | `0.2188` |

So the linearization `cos D -> 1 - D^2/2`, `sin D -> D` degrades to nothing for
plain RoPE by `delta = 4096` and retains a fifth of the channels under YaRN. The
honest limit of the story: the amplitude-weighted error is still enormous for
every scheme (ratio RoPE:YaRN of `0.997` at `delta = 4096`), because the handful
of fast channels dominate it. Linearization of the *whole* head is not rescued by
any of these methods; what YaRN buys is a larger linearizable majority
(`figures/fig03`, `figures/fig04`, `figures/fig05`).

**Partial RoPE yields an exactly position-free sub-score.** At `p = 0.25` the
unrotated 75% of channels contribute a sub-score whose spread across distance is
exactly `0.0`, while the rotated quarter varies by `11.34`. This is a clean
separation of a bilinear, position-independent term from the position-carrying
one (`figures/fig07`). Note that partial rotation is *not* orthogonal, so the norm
is preserved exactly as for full RoPE, provided the rotated block is paired within
itself (the GPT-NeoX layout, as in `pythia-160m`). Pairing across the whole head
instead leaves every touched pair half rotated and half not; that variant is not
orthogonal and is not what any published model does.

**YaRN's magnitude term is a temperature.** `mscale = 0.1 * ln(factor) + 1`
(`1.3466` at `scale = 32`) multiplies all logits, pulling attention entropy from
`H/ln T = 0.923` to `0.865` (`figures/fig08`).

## Correcting the earlier drafts

Two claims in the legacy documents do not survive measurement, and the difference
matters for the mechanism:

- **"RoPE breaks bilinearity (3.7x instead of 2x)."** For a fixed position pair,
  RoPE attention is exactly bilinear; doubling the query doubles the score, to
  `7.4e-16` relative error. The legacy demo reached 3.7x by changing the scoring
  function between its two cases, not by exhibiting nonlinearity in one. The
  non-additivity in attention comes from the softmax downstream of the score,
  which is where a real obstruction still lives.
- **"YaRN works by raising the base from 10000 to 500000."** That is plain
  position interpolation, a different method. YaRN keeps `base = 10000` and ramps
  per frequency — short-wavelength channels keep their inverse frequency,
  long-wavelength channels are divided by `scale`, blended linearly in between —
  then rescales by `mscale`. In the reference implementation at `scale = 32`,
  9 of 32 pairs are bit-for-bit unchanged and 21 of 32 differ from uniform
  interpolation. Both `yarn` and `position_interpolation` are computed in
  `experiments.method_spectrum` so the two can be compared directly.

## Figures

| figure | what it shows |
|---|---|
| `fig01_frequency_ladder` | the frequency ladder; YaRN's ramp against plain RoPE and interpolation |
| `fig02_angle_spectrum` | per-channel `\|D_k\|` for RoPE and YaRN at several distances |
| `fig03_max_vs_median_angle` | max angle unchanged by YaRN, median angle reduced |
| `fig04_linearizable_fraction` | fraction of channels admitting the linearization, vs distance |
| `fig05_linearization_error` | amplitude-weighted and per-pair linearization error |
| `fig06_pair_exact_vs_linear` | exact vs linearized per-pair contribution, and where they part |
| `fig07_partial_rope` | the exactly position-free sub-score under partial RoPE |
| `fig08_mscale_entropy` | attention entropy against scale, with and without `mscale` |
| `fig09_position_conditional_attribution` | per-feature contribution against distance |
| `fig10_seed_variance` | the central claim across a 12-seed sweep, with the spread shown |

Each figure ships the plotted data as a CSV beside it, and `figures/README.md`
records which function produced it and its headline number.

## Provenance

`rope.py` is an independent implementation. The inverse-frequency ladder and the
split-half `rotate_half` convention follow `transformers`
(`models/llama/modeling_llama.py`); the YaRN ramp (`find_correction_dim`,
`find_correction_range`, `linear_ramp_mask`) and `get_mscale` are transcribed from
the authors' own reference implementation
(`scaled_rope/LlamaYaRNScaledRotaryEmbedding.py` in `jquesnelle/yarn`).

This repository contains **no third-party source code** — there is no vendored
tree, and `NOTICE.md` records the greps that establish it. See `NOTICE.md`,
`projects/ATTRIBUTION.md` and `CITATION.cff`.

RoPE: Su, Jianlin; Lu, Yu; Pan, Shengfeng; Murtadha, Ahmed; Wen, Bo; Liu, Yunfeng.
"RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021.
<https://arxiv.org/abs/2104.09864>

YaRN: Peng, Bowen; Quesnelle, Jeffrey; Fan, Honglu; Shippole, Enrico.
"YaRN: Efficient Context Window Extension of Large Language Models", 2023.
<https://arxiv.org/abs/2309.00071>

## Scope and honesty

The core measurements here are on the *structure* of the score: exact identities
and their numerical error, on synthetic projections. That is deliberate - those
claims are about the position encoding, not about any particular trained network.

`projects/rope_attribution/real_model.py` repeats the key measurements on real
activations from three trained checkpoints (`pythia-160m` partial rotary,
`llama-160m` and `SmolLM-135M` full rotary), writing `results/real_model.json`. It
needs CPU torch, which is not a test dependency, so it is opt-in:

```bash
pip install -r real_model_requirements.txt
python -m projects.rope_attribution.real_model     # ~3 min, CPU only
```

Three findings there matter for reading the synthetic results:

- Real rotary channels are overwhelmingly position-carrying: only 0.6-0.7% of
  pairs are position-blind at a 1% threshold, and remarkably consistently across
  three unrelated architectures.
- The exact identities hold to the **float32** floor (~1e-7), not 1e-14, because
  real activations are float32. The theorems survive; the floor moved.
- The leading feature's **share** of a score spans `0.011` to `0.9999` depending
  on the head — a factor of 67-93x across the three checkpoints. The repository
  previously had no error bars anywhere; a single-vector measurement would have
  missed this by two orders of magnitude.

`projects/rope_attribution/usefulness.py` then asks what the correction is worth,
on the same checkpoints. It compares the exact per-distance attribution against
the **best possible** position-free scalar — the least-squares constant over the
distance range, not a strawman — and finds that the optimum reports the wrong
sign on `26.8%` of per-pair, per-distance cells, with a median relative error of
`1.13` against a peak of `1`. That is the cost of insisting a RoPE feature's
contribution is a number, and it is the paper's answer to "is this object useful
or merely correct".

```bash
PYTHONPATH=projects python -m rope_attribution.usefulness   # ~4 min, CPU only
```

### What this repository does not claim

- **The algebra is not new.** The per-pair closed form appears in Chachamovits,
  *Phase Structure in Rotary Attention* (arXiv:2607.25507, §3-§4), in
  Liang et al., *RoPE-Aware Bit Allocation* (arXiv:2606.24033, §1), and the
  per-frequency spectral structure of attention has been measured on pretrained
  checkpoints by Li (arXiv:2607.06621). What is claimed here is the attribution
  semantics and its consequence, not the identity. The frozen
  `frontier-01-*` documents assert the opposite — that nobody has done this — and
  that claim is false; see `LEGACY.md`.
- **The sign reversal appears to be unreported**, but that is a search result,
  not a proof of absence.
- Still absent and not measured anywhere here: trained sparse autoencoders,
  retrieval or passkey benchmarks, and any accuracy number. Claims of the form
  "the model retrieves the needle with accuracy 0.7" should not be attributed to
  this repository.

## Legacy material

`projects/frontier-01-*.py` and `projects/frontier-01-*.md` are an earlier,
frozen body of work kept for provenance. Known caveats:

- several scripts **print hardcoded illustrative values** rather than computing
  them — for example `frontier-01-bag-of-words-test.py` assigns retrieval
  accuracies as literals and computes "interaction" as `D**2 / 2` by definition;
- `frontier-01-graphs-ULTIMATE-V11.py` plots literal lists;
- the `projects/fig_*.png` files from that era were plots of those literals and
  have been removed for that reason; they are recoverable from git history at
  `310bf2c`;
- the documents disagree with each other and, in the two cases listed above, with
  the measurements here. Where they disagree, the measurements win.

Do not cite the legacy documents. Cite `projects/rope_attribution/` and
`results/measurements.json`.

## License

Apache-2.0. See `LICENSE`, `NOTICE.md`, `CITATION.cff`.
