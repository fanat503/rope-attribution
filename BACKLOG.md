# Research backlog

Living document. The operational protocol asks for a self-updating backlog, and
the point of keeping it in the repository rather than in my head is that a
reader can see what was tried, what was abandoned, and why — which is the part a
task list that gets deleted throws away.

Every item is either **done** (with the commit that closed it), **open** (ranked,
with the next concrete action), or **rejected** (with the reason). Rejected items
are the most valuable ones: they are the decisions that would otherwise be made
again silently.

---

## Method, distilled from the literature

Fan et al., arXiv:2609.20804, *An Empirical Study of Harness Design for Coding
Agents*. The transferable part is not their numbers but their method: fix the
execution loop, vary one component, pair every comparison and test it rather
than eyeballing it, and explain each effect at the trajectory level rather than
asserting it. `LESSONS.md` carries the full read and the nine mistakes of mine it
caught.

The four items below are what that paper's findings became here.

---

## Open, ranked

### 1. The headline number has no interval the paper says so

**Done in part.** `uncertainty.py` now reports a percentile bootstrap over heads
for the pooled sign-flip rate: `0.2670 [0.2665, 0.2674]`, SD `0.0129`, over 2790
heads. **Still open:** the paper's §`sec:worth` still prints `26.8%` with no
interval and no statement of which averaging it used. The interval must appear in
the paper, with the weighting named.

### 2. The three checkpoints do not agree, and the paper implies they do

**Found, not yet written.** The paired bootstrap over heads gives:

| comparison | difference | 95% interval |
|---|---|---|
| pythia-160m vs llama-160m | `+0.0134` | `[+0.0119, +0.0148]` |
| pythia-160m vs SmolLM-135M | `+0.0122` | `[+0.0108, ++0.0136]` |
| llama-160m vs SmolLM-135M | `-0.0012` | `[-0.0018, -0.0005]` |

Every interval excludes zero. The effect is ~1.2–1.3 percentage points — small,
real, and currently invisible in the paper's "this is the ordinary case"
phrasing. It is a model-dependence the paper should either report or explicitly
set aside as below its own precision.

### 3. "26.8%" is a checkpoint mean; the head mean is "26.70%"

**Found, not yet written.** The paper's headline averages the three per-checkpoint
means equally. The mean over all 2790 heads is lower, because SmolLM contributes
1350 heads against 720 each. The two differ by more than the pooled interval's
half-width, so this is not a rounding-level distinction. One sentence fixes it.

### 4. The 9-page limit, still the largest structural item

Main text is 13 pages against a 9-page limit. Recorded honestly in
`paper/BUILD.md`: the remaining content is R1, R2, R3 and the per-pair
decomposition, so fitting requires dropping a contribution. The reviewer's
alternative — reframe as a short position note with results in the appendix — is
the author's call and is not one I should make.

---

## Done

| item | closed by |
|---|---|
| Withdraw the `920x` artefact, replace with grid-independent statistics | `feat(statistics)` |
| Prove the propositions the results rest on; correct the two that were false | `proof(paper)` |
| Measure the cost of the position-free scalar (26.8% wrong sign) | `feat(figures)` |
| fig10, a 12-seed sweep, so the reversal is not one lucky draw | `feat(figures)` |
| Correct the novelty claim — the algebra is published (arXiv:2607.25507) | `feat(paper)` |
| Add the five nearest prior works, verified against the arXiv API | `feat(paper)` |
| One author, and a test that keeps it that way across every ref | `metadata` |
| Makefile's `-m` path was a filesystem path; guard it | `fix(Makefile)` |
| The provenance audit's counts were wrong and cited non-existent scripts | `metadata` |
| `fig09` still printed the withdrawn `max/min` ratio as a headline | `fix(figures)` |
| `grid_independence` orphan artefact, now generated, self-describing, tested | `fix(usefulness)` |
| Uncertainty for the headline, and the checkpoint-disagreement finding | `feat(uncertainty)` |
| Post-edit diagnostics, run after each edit rather than at the batch's end | `feat(scripts)` |

---

## Rejected, with the reason

### Hydra / structured argparse for configuration

**Rejected.** Four modules, none of which takes a user-facing parameter beyond a
seed and a model id. The paper's configuration is the *point* of being frozen —
`SEED_QK`, `SEED_ATTRIB`, `GRID_DENSITIES` are module constants precisely so they
cannot be tuned after seeing results. A config system would make them tunable.

### A `plotting_engine.py` with publication themes

**Rejected as a separate module.** `figures.py` already owns a style function,
and a second place that sets `rcParams` is a second thing to keep consistent. The
real gap the directive names is seed-variance bands, and that belongs to the
figure that needs it (`fig10`), not to a shared engine.

### GPU determinism locking, convergence sweeps, learning-rate sweeps

**Rejected as inapplicable.** There is no training, no optimisation and no model
being fitted anywhere in this repository. The honest available version —
byte-identical regeneration of the committed artefacts across three runs — is
already done and reported in the README.

### `THEORY_AND_PROOFS.md` as a separate document

**Rejected.** The proofs belong where the propositions are, in `paper/main.tex`,
where a reader of the paper sees them; a separate file would be a second place
for them to drift from the claim registry. The rigorous parts were instead added
to the paper itself, and `tests/test_paper_theory.py` pins them numerically.

### Deleting `projects/frontier-01-*`

**Rejected.** 85 files of dead, sometimes fabricated exploration. Deleting them
would destroy the `file:line` citations in `LEGACY.md` that document *which*
claims were wrong. `LEGACY.md` is the honest handling: an inventory that says what
is broken and what the maintained replacement is.
