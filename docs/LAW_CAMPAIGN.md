# The window-law campaign: pre-registration

**Status boundary.** This document fixes the hypotheses, the checkpoint set, the
texts, the windows and the decision rules *before* the campaign is run on the
new checkpoints. The three baseline rows already exist in
`results/collapse.json` and `results/usefulness.json`; the campaign re-measures
them under one provenance block, so a disagreement is a finding about the
runner, not a licence to pick the older number. No number below is a new
result, and nothing here enters `paper/main.tex` until the corresponding
artifact is committed.

## 1. The question

The paper's law is about one rotary pair: over a window of width `L`, a
position-free constant cannot track `c(d) = rho cos(theta d - psi)` better than
relative error `1` once `x = theta L >= 2 pi`, and below the threshold the
attainable error is the exact band `[(1 - cos(x/2))/2, sin(min(x, pi)/2)]`. The
campaign asks four things at once:

- **H1 (law).** On every checkpoint and every rotary pair, the measured
  channel-level minimax error is `1` for `x >= 2 pi` and stays inside the band
  for `x < 2 pi`.
- **H2 (rate).** The equal-weight per-pair sign-flip rate on the 13-point log
  grid `(1, 2, 4, ..., 4096)` reproduces the committed baseline, and the
  uniform-grid rate stays higher by the amount the grid change predicts. The
  rate is a functional of the grid; the law is not.
- **H3 (QK-norm).** A per-head positive norm applied after projection and
  before RoPE folds into `rho` and leaves `psi` and the law unchanged. The
  campaign's test is that a QK-norm checkpoint satisfies H1 and H5 with the
  capture taken at the tensor the model actually rotates; the folding itself is
  algebra, and the adapter's order is pinned separately in
  `tests/test_real_model.py`.
- **H4 (scale).** Moving from 135M--160M to 410M, 2B, 4B and 8B changes which
  fraction of pairs is past the threshold and the rotary fraction of the score;
  it does not change the channel-level law at a given `x`.
- **H5 (softmax).** The log-odds identity, the odds factorisation, the
  closed-form contrast and the centred gradient are exact to the dtype floor;
  the contrast is blind at separation `2 pi / theta`, so the softmax introduces
  no second length scale.

## 2. Checkpoints

`configs/models.json` is the pre-registration. Every revision is a 40-hex Hub
commit (or the sentinel `local`), and the runner refuses a loaded config whose
geometry disagrees with its `expected` block.

| model | params | role | adapter | status at freeze | why |
| --- | ---: | --- | --- | --- | --- |
| `EleutherAI/pythia-160m` | 160M | baseline | flat | baseline | partial RoPE (25%), the dilution reference |
| `JackFram/llama-160m` | 160M | baseline | flat | baseline | full RoPE, no GQA |
| `HuggingFaceTB/SmolLM-135M` | 135M | baseline | flat | baseline | full RoPE with 3 KV groups |
| `EleutherAI/pythia-410m-deduped` | 410M | scale-cheap | flat | planned | cheapest scale point with a public SAE |
| `Qwen/Qwen3-8B-Base` | 8.0B | QK-norm scale | flat | planned | ungated, 32 query / 8 KV heads, QK-norm, theta 1e6 |
| `google/gemma-2-2b` | 2.0B | Gemma Scope | flat | blocked-gated | license not accepted; softcap 50, no QK-norm |
| `google/gemma-3-4b-pt` | 4.0B | QK-norm scale | gemma3 | blocked-adapter | nested text tower, dual RoPE, sliding 1024; adapter missing |
| `meta-llama/Llama-3.1-8B` | 8.0B | scale | flat | blocked-gated | license not accepted; no QK-norm, theta 500k |
| `local:train_303m` | 303M | own model | own | blocked-adapter | no loader yet; filled after the TPU run |

`planned` entries are not run by a default invocation. They are promoted to
`baseline` only after their first successful campaign artifact, so a default
run cannot pull a gated 8B checkpoint or an adapter that does not exist. A
`blocked-*` entry is skipped unless its id is passed explicitly; passing it is
a request to test the blocker, not a promise that it runs.

The 8B rows need either TPU extraction or sharded bf16. Local float32 loading
is not an option: 8B float32 is about 32 GB against roughly 14 GB free. The
campaign therefore runs the 135M--410M rows locally and records the 8B rows as
`blocked` until the extraction path exists.

## 3. Texts, tokens and seeds

- Prompts: `rope_attribution.real_model.DEFAULT_PROMPTS`, the same tuple the
  committed baselines used. The artifact stores `sha256` of every text, not a
  paraphrase of it.
- Seed: `20260906` for every measurement. The seed controls tokenisation and
  sampling, not a stochastic estimator; the per-pair measurement is
  deterministic given the prompt.
- Tokens: `max_tokens = 512` for collapse and usefulness. The softmax identity
  check uses `min(max_tokens, 128)` and the first prompt, because exactness is
  per head and does not need a second text; the artifact records both limits.
- Distances: the 13-point log grid `(1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
  1024, 2048, 4096)`. This is the grid behind the canonical `27.0%`; it is
  fixed, not chosen after seeing a rate.
- Windows: `(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)`. The smallest
  window is 8 because a window must contain at least one distance and the
  collapse measurement needs several; the canonical per-pair rate uses the
  distance grid above, not this list.

## 4. Metrics, exactly

For one pair `k` and one window `L`:

- `channel_rel_minimax`: the minimax relative error of the best constant,
  `max_d |c(d) - midrange| / rho`, computed in closed form from the phase the
  window sweeps. The law predicts `1` at `theta L >= 2 pi`.
- `channel_rel_mean`: the relative error of the window mean, the surrogate a
  practitioner would actually use. It is reported beside the minimax value so
  the difference between "best constant" and "mean constant" is visible.
- `score_rel`: the same error after summing the pairs and the unrotated tail,
  normalised by the score. This is the diluted quantity, not the law.
- `past_threshold_fraction`: the fraction of measured pairs with
  `theta L >= 2 pi`.
- `sign_flip_fraction_pooled`: over all pairs and distances in the grid, the
  fraction where the sign of the constant surrogate differs from the sign of
  the exact contribution. Equal weight per pair, then pooled; the head-pooled
  and per-model variants are recorded separately.
- `rel_err_median`: the median relative error of the mean surrogate on the
  grid. This is the `1.13` baseline, a different functional from the sign-flip
  rate and from the minimax law.
- Softmax rows: `log_odds_residual_max`, `odds_residual_max`,
  `contrast_residual_max`, `blind_at_period_residual_max`,
  `centered_gradient_residual_max`, plus descriptive pair shares. The first
  four are algebraic identities and must sit at the float64 floor; the centred
  gradient is a finite-difference check with `eps = 1e-6`, so its floor is
  `O(eps)` rather than `1e-15`.

## 5. Decision rules

Fixed before the run:

1. **H1 passes** if, for every measured pair with `x >= 2 pi`, the channel
   minimax error is `1` within `1e-6`, and every pair with `x < 2 pi` lies
   inside the band within `1e-6`. A single violation is a failure of the law on
   that checkpoint and is reported with the layer, head, pair and window.
2. **H2 replicates** if the pooled sign-flip rate is `27.0 +/- 1.0` points and
   the per-model rates preserve the baseline ordering (Pythia above the two
   full-RoPE models). The uniform-grid rate is not re-measured here; the
   committed `results/usefulness_grid_independence.json` covers that functional,
   and a disagreement between the two artifacts is a finding about the grid,
   not evidence for a new law.
3. **H3 passes** if the QK-norm checkpoint satisfies H1 and H5. That is the
   campaign-level evidence; the folding itself is algebra, and the capture
   order (after projection, before RoPE) is pinned by
   `tests/test_real_model.py`. A failure of H1 or H5 is an adapter finding, not
   evidence against the folding.
4. **H4 passes** if H1 passes on every scale checkpoint and the measured
   channel-level error, plotted against `x`, lies on the same band at every
   scale. The informative scale-dependent quantities are
   `past_threshold_fraction` and the rotary fraction; a scale-dependent
   channel law is a finding that would narrow the paper's claim.
5. **H5 passes** if the log-odds, odds, contrast and blindness residuals are at
   most `1e-9`, and the centred gradient is at most `1e-4`. These ceilings are
   deliberately above the measured float64 floors, so a dtype or hardware
   change does not turn a correct implementation red.

A failed hypothesis is not repaired by changing the grid, the seed, the prompt
or the tolerance. It is recorded, and the paper's claim is narrowed.

## 6. Artifacts and provenance

One command produces the campaign artifact:

```
python -m rope_attribution.law_campaign --out results/law_campaign.json
```

Each model row contains the registry checks, the resolved Hub revision, the
git commit and dirty flag, Python/torch/transformers/numpy versions, the host,
and the SHA-256 of every prompt. A row with `status: error` is data: the
campaign continues, and the process exits nonzero so a partial file cannot be
mistaken for a complete one. The detailed per-head softmax rows live in
`results/softmax_decomposition.json`; the campaign embeds the full softmax
payload in each model row, so the paper can cite one provenance block.

Re-running after a code change is expected to reproduce the numbers; the
artifact records the commit, so a difference is attributable. A dirty working
tree is recorded as `dirty: true` and is not a release artifact.

## 7. Failure modes the campaign must distinguish

- **Gated, not broken.** A 401 is a license step, recorded as
  `blocked-gated`, not an architecture failure.
- **OOM, not a null result.** An 8B checkpoint that does not fit is a hardware
  limit; it must not be reported as "the law does not hold at 8B".
- **Adapter, not a law violation.** A nested text tower or a dual RoPE base is
  a code gap. The flat adapter refuses it by design; a wrong measurement is
  worse than no measurement.
- **Revision drift.** A Hub commit that no longer resolves to the pinned
  snapshot is refused before measurement. `revision_requested` and
  `revision_resolved` are both recorded.
- **float32 saturation.** Pythia logits saturate float32; the exact softmax
  reconstruction already shows a `9.5e-3` residual there against `2.7e-6` on
  the other two. This is a dtype effect, and the identity ceilings above are
  set to survive it without hiding an algebra error.
- **Partial RoPE and GQA.** Pythia rotates 25% of each head; SmolLM replicates
  each KV head three times. The runner measures one replica per group and
  reports the rotary fraction, so neither is silently counted as a full-RoPE
  head.

## 8. What the campaign does not claim

It does not train a model, evaluate a downstream task, or measure an attention
output against a task metric. It does not establish that a particular
attribution method improves a particular benchmark. It characterizes the
score-level object and the exact softmax identities; the downstream protocol
is separate and remains a pre-registration until it produces its own artifact.

## 9. Change control

The `campaign` block in `configs/models.json` (seed, windows, distance grid,
max tokens, prompt source) is frozen at the values above. Changing any field is
a new campaign version, not an edit, and requires a new artifact path. Adding a
checkpoint is allowed without a version bump only if its `expected` geometry is
complete and its status is `planned` or `blocked-*`; promoting it to `baseline`
is a decision recorded with the artifact that measured it.
