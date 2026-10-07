# `figures/` - generated figure set

Everything in this directory is written by
`projects/rope_attribution/figures.py`. No number in these PNGs, in the
CSV sidecars, or in the table below is typed into the source: each one is
returned at run time by `rope_attribution.experiments` or
`rope_attribution.rope`. The literals that do appear in `figures.py` are
configuration (head dim, RoPE base, extension scale, seeds), the distance
grid definition, plotting parameters, and two dimensionless thresholds that
are definitions rather than results: the
`0.1 rad` linearization cut inside
`experiments.linearization_error` and the `1.0` (= 100% relative error)
reference line in fig06.

## Regenerate

```powershell
$env:PYTHONPATH = "projects"
python -m projects.rope_attribution.figures
```

Each figure has a `<stem>.csv` sidecar holding exactly the array that was
plotted, so no claim has to be read off a pixel. Log axes clip at
`1e-18` for display only; the CSVs carry the raw values.

## Figures

| figure | produced by | headline number (computed) |
| --- | --- | --- |
| `fig01_frequency_ladder.png` | rope.inv_freq, rope.yarn_parameters (called through figures.py, head_dim=64, base=10000, scale=32) | YaRN leaves 9 of 32 rotary pairs bit-for-bit unchanged and slows the slowest pair by a measured factor of 32.00. |
| `fig02_angle_spectrum.png` | experiments.pair_terms(...).angle over both frequency ladders | at delta=4493: max \|D_k\| is 4493 rad for RoPE and 4493 rad for YaRN (difference 0.000e+00 rad), while the median angle falls to a 0.4464x fraction. |
| `fig03_max_vs_median_angle.png` | experiments.method_spectrum (which calls experiments.linearization_error), seed=2, 54 distances | max \|D_k\| for rope_base_10k and yarn agree to 0.000e+00 rad across all 54 distances (ratio 1.0000000000 at delta=8192), while the median angle differs by 0.4464x. |
| `fig04_linearizable_fraction.png` | experiments.method_spectrum -> frac_channels_linearizable, seed=2 | at delta=8192: rope_base_10k retains 0.0000 of its channels and yarn retains 0.1250 (the other two retain 0.1250 and 0.1250). |
| `fig05_linearization_error.png` | experiments.method_spectrum -> amplitude_weighted_err and per_pair_err_rms | at delta=8192 the amplitude-weighted error is 1.24572e+06 for rope_base_10k and 1.24179e+06 for yarn, a ratio of only 0.996840. |
| `fig06_pair_exact_vs_linear.png` | experiments.pair_terms(...).contributions() vs the Taylor form | the fastest rotary pair (k=0) has a normalized error as large as the term itself from delta=3 in plain RoPE and delta=3 in YaRN - the same, because YaRN does not touch that channel. |
| `fig07_partial_rope.png` | recomputed here from rope.partial_rope_cos_sin + rope.apply_rope, cross-checked against experiments.partial_rope_split(seed=3) | the clean (unrotated) sub-score varies by 0.000e+00 across all 54 distances, while the rotated sub-score varies by 24.2017. |
| `fig08_mscale_entropy.png` | experiments.mscale_entropy over 18 values of scale, seed=4 | at scale=32, mscale=1.346574 pulls H/lnT from 0.9233 down to 0.8651, a drop of 0.0582. |
| `fig09_position_conditional_attribution.png` | recomputed from experiments.score_relative on a synthetic SAE-like decomposition identical to experiments.feature_attribution(seed=1) | 8 of 8 features reverse sign somewhere in the range [1, 8192], with the score still additive to 4.974e-14. The magnitudes at the two ends of the range agree to within 10^-0.08, so what changes across distance is which way the contribution points, not how large it is. The max/min ratio this figure was once described by (920.04x) is a sampling artefact and is deliberately not reported; see rope_attribution.statistics. |

## Shared configuration

- `head_dim = 64`, RoPE `base = 10000`, legacy `base = 500000`
- extension `scale = 32`, original context = 2048, `T = 256` keys
- distance grid: 54 log-spaced integers from 1 to 8192 positions
- scale grid for fig08: 18 values from 1 to 128
- seeds: q_hat/k_hat = 20250929, method spectrum = 2, attribution = 1, partial RoPE = 3, mscale = 4
- output: 200 dpi PNG on a white ground

## Reading order

fig04 is the headline result, fig09 is the central claim. fig03 and fig05
are the two guards on fig04: they show *why* the fraction collapses (the
maximum angle never changes) and *what that costs* (the amplitude-weighted
error barely moves). fig06 shows the local mechanism, fig07 and fig08 the
two auxiliary structure claims, and fig01 the frequency bookkeeping.
