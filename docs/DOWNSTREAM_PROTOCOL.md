# Downstream protocol: attribution-guided KV eviction at 1B--8B

**Status boundary.** This is a pre-registered design, not a result. No
checkpoint in this document has been run, no number here is measured, and
nothing here is quoted in `paper/main.tex` or `results/`. The paper's
limitation --- it characterizes the attribution object and evaluates no task ---
remains true until this protocol produces a committed artifact. A local pilot at
70M--160M was deliberately not promoted to a result: at that scale the
intervention is underpowered and the exact and position-free rankings overlap
almost completely, so a null there would say nothing about the hypothesis.

## 1. The question

The paper's claim is that a position-free per-feature scalar is a
mis-specification of the RoPE score, not an approximation to it. A downstream
experiment has to show that a *practitioner's decision* changes when the exact
per-distance object is used, and that the changed decision produces a better
behavioural outcome. The decision tested here is **which keys stay in a bounded
KV cache**.

This is deliberately not a test of "RoPE matters": deleting the rotation raises
NLL by several nats and is already a known negative control. The contrast is
between two rankings computed from the *same* query and key vectors:

| arm | key score for query $t$, key $j$, distance $\delta=t-j$ |
| --- | --- |
| **exact** | $\lvert A_{ij}(\delta)\rvert$, the per-distance attribution |
| **mean** | $\lvert \bar A_{ij}\rvert$, the best position-free constant over the window |
| **identity** | $\lvert A_{ij}(0)\rvert$, the $\delta=0$ value |
| **attention** | accumulated post-softmax attention (H2O-style) |
| **recency** | $\delta$ only |
| **random** | uniform, matched on the number of retained keys |

The scientific contrast is **exact versus mean**. The other four arms are
controls: attention and recency are the deployed baselines, identity isolates
"use the wrong distance", and random fixes the budget. If exact and mean rank
keys identically, the attribution has no downstream consequence at that scale,
and the paper should say so rather than imply one.

### 1.1 What the ranking functional is

For one query head $h$ at position $t$ and one cached key $j$,

$$
A_{ij}(\delta)=\sum_{k=1}^{d/2}\rho_{k}\cos(\theta_k\delta-\psi_k)
\;+\;(\text{non-rotary channels}),
$$

with $\rho_k$, $\psi_k$ read from the same $q_i,k_j$ the model uses. The
non-rotary channels are position-independent and therefore enter every arm
identically; for partial-RoPE models they dilute the contrast by the rotary
fraction $p$. The protocol records $p$ per model and reports the contrast
separately for full-RoPE and partial-RoPE families, because the paper's dilution
law predicts a smaller effect in the latter.

The primary ranking uses $\lvert A_{ij}\rvert$ per head, summed over the query
heads that share a KV head (grouped-query attention is handled by summing the
per-query-head attributions for each shared key). A secondary ranking weights
each pair by $\lVert v_jW_O^{(h)}\rVert$; it is reported only if the primary
ranking is inconclusive, and the switch is recorded as exploratory.

## 2. Models

Seven instruction-tuned checkpoints, all with public weights and a single
documented RoPE configuration:

| model | params | RoPE | why it is in the grid |
| --- | ---: | --- | --- |
| `meta-llama/Llama-3.2-1B-Instruct` | 1.2B | full, base 500k | smallest Llama-family anchor |
| `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | full, base 500k | same family, different width/depth |
| `meta-llama/Llama-3.1-8B-Instruct` | 8.0B | full, base 500k | upper end of the range |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | full, base 1M | second architecture family |
| `Qwen/Qwen2.5-7B-Instruct` | 7.6B | full, base 1M | second family at scale |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7.2B | full, base 1M | third family, sliding-window attention |
| `google/gemma-2-2b-it` | 2.6B | partial (50%) | the dilution-law test |

Gemma 4 is excluded for the structural reason in `docs/GEMMA4.md`: its
per-layer attention families cannot be described by one frequency ladder. The
grid spans four architecture families, two rotary fractions, and 1.2B--8B.

**Adapter requirement for Gemma 2.** `google/gemma-2-2b-it` applies
`q_norm`/`k_norm` after the projections. A hook on `q_proj`/`k_proj` sees the
unnormalised vector, while RoPE rotates the normalised one, so the eviction
adapter must read the post-normalisation tensors (or apply the same
normalisation before rotating). The repository's attribution adapter
`real_model.collect_head_vectors` refuses Gemma 2 for exactly this reason. The
eviction core in `eviction.py` takes the post-RoPE query and key tensors and is
therefore agnostic; the deferred Hugging Face adapter is where the post-norm
hook has to live. If that hook is not implemented before the run, the
partial-RoPE arm is dropped and the dilution test is reported as not run --- it
is never replaced by a silently wrong tensor.

## 3. Tasks and prompts

Three task families, each with a deterministic answer and a measurable
full-cache baseline:

1. **RULER-style needle-in-a-haystack** (single needle, 4k/8k/16k contexts,
   answer span one to three tokens). This is the task KV eviction is actually
   deployed on.
2. **Multi-needle retrieval** (four needles, same contexts). A method that keeps
   only the most recent needle fails; a method that keeps the right keys passes.
3. **Induction/copy** on repeated natural passages, with the query at the final
   token and the target the first token of the repeated body. This is the one
   task where the causal key is known by construction, so it separates "the
   ranking found the key" from "the task was easy".

Prompts are sampled once, before any arm is run, from a fixed seed and a fixed
public corpus; the prompt file and its SHA-256 are committed with the artifact.
No prompt is filtered after seeing an arm's result.

## 4. Intervention

At each decoding step, for each KV head, rank the cached keys by the arm's
score, retain the top $B$ keys plus the current token and the first four tokens,
and evict the rest. $B\in\{0.25,0.50\}$ of the configured context. The model is
not retrained; only the cache is masked. A full-cache arm is run on every cell
as the ceiling.

Two implementation rules make the result auditable:

- **Identity test.** With $B$ equal to the full cache, the masked forward pass
  must reproduce the unmasked logits to within $10^{-4}$ in float32. This is the
  mutation detector for the masking code: a bug that drops or reorders keys
  fails it before any task number is produced.
- **No cache sharing across arms.** Each arm starts from the same prompt and
  rebuilds the cache; an arm cannot inherit another arm's eviction decisions.

## 5. Endpoints and analysis

**Primary endpoint:** mean answer-span negative log-likelihood, paired by
prompt, exact minus mean, at 50% retention and 8k context, averaged over the
seven models with each model weighted equally. A continuous endpoint is used
because it is more sensitive than exact match at this sample size; accuracy is
the secondary, interpretable endpoint.

**Success:** the paired mean difference is at least **0.10 nats** in favour of
exact, its 95% bootstrap interval (10,000 resamples, clustered by prompt)
excludes zero, and exact is **non-inferior** to the attention arm within
**0.05 nats**. The last clause prevents a win over a strawman: exact must also
be at least as good as the deployed baseline.

**Secondary endpoints:** exact-match accuracy at 25% and 50% retention; the
same contrast at 4k and 16k; the per-model breakdown, with full-RoPE and
partial-RoPE reported separately; and the rank correlation between exact and
mean key order (a mechanistic check on *why* the arms differ or do not).

**Power.** With 300 prompts per cell and a paired difference SD of 0.5 nats,
the standard error is about 0.029 nats, so a 0.10-nat effect is detected with
power above 0.95 at $\alpha=0.05$. For the accuracy endpoint, 300 prompts with
15% discordance detect a 5-point difference at roughly 0.8 power; the protocol
therefore does not claim power for a 3-point accuracy effect. The item-level
analysis (several questions per prompt) is reported with a cluster bootstrap,
never as independent items.

**Multiplicity.** One primary endpoint, one retention level, one context length.
The other cells are secondary and are reported with intervals, not as
significance tests. No alpha adjustment is claimed for exploratory cells.

## 6. Falsification and stopping

- **Kill the downstream claim** if the exact-minus-mean interval contains zero
  at 50% retention and 8k, or if the point estimate is below 0.05 nats. Report
  the null with the same artifact schema; do not search another task for a
  positive result.
- **Kill the ranking functional** if exact is worse than the attention arm by
  more than 0.05 nats. That means $\lvert A_{ij}\rvert$ is the wrong currency
  for eviction, not that the attribution is wrong; the paper's score-level
  claims are unaffected, and the follow-up must say so.
- **Stop early for harm** if any arm drops below the random arm by more than
  0.2 nats on the full-cache ceiling; a ranking that is worse than random is a
  bug or a broken assumption, and the run is not interpretable until it is
  explained.
- **No peeking.** The confirmatory run is one pass over the pre-registered
  grid. Exploratory runs use a 50-prompt development split and are labelled as
  such in the artifact.

## 7. Artifact and code

The run writes `results/downstream_eviction.json` with:

```
schema, status, protocol_version, git_commit, date, seed, prompt_sha256,
models: [{id, params, rope_base, partial_rotary_factor, layers, kv_heads}],
cells: [{model, task, context, retention, arm, n_prompts,
         nll_mean, nll_sd, accuracy, accuracy_ci, rank_correlation_with_exact}],
ceiling: {model, task, context, nll_mean, accuracy},
note
```

The ranking and masking core lives in
`projects/rope_attribution/eviction.py` and is tested by
`tests/test_eviction.py`. It implements the six arms, the grouped-query
reduction, the forced prefix/current keys, and masked attention; the exact
arm is checked against an independent `rope.apply_rope` construction, and the
all-kept mask is checked against ordinary softmax. `--dry-run` writes the
schema with empty `models`, `cells` and `ceiling` and a note saying that no
checkpoint was run. No test asserts a task outcome, because the outcome is the
experiment.

**What is deferred.** The Hugging Face cache adapter is not shipped. Evicting a
real model's cache requires architecture-specific surgery on the attention
forward: Llama/Qwen/Mistral expose a `DynamicCache` with post-RoPE keys but no
query, Gemma 2 adds post-projection QK-normalisation, and Gemma 4 has two
per-layer rotary families. The adapter must be written and identity-tested per
family before the confirmatory run. Until then the artifact is a dry run, and
the paper's limitation --- no task is evaluated --- remains true.

## 8. Compute

Inference only, one 80GB GPU in bf16, no training and no fine-tuning. The grid
is 7 models $\times$ 3 tasks $\times$ 3 contexts $\times$ 2 retentions
$\times$ 6 arms $\times$ 300 prompts $\approx 227{,}000$ sequences, dominated by
the 8k and 16k cells. At an assumed 0.5--1.5 s per prefill-plus-decode on one
H100, the run is roughly 30--95 GPU-hours; the estimate is arithmetic, not a
benchmark. Model download and prompt construction are separate and are not
counted.

## 9. Relation to the paper

Until this protocol produces a committed artifact, the paper keeps its explicit
limitation: the score-level object is characterized, and no task is evaluated.
If the protocol succeeds, the follow-up paper adds a task-level consequence; if
it fails, the follow-up reports the null and the score-level thesis stands
unchanged. Neither outcome is allowed to change a number in the current paper.
