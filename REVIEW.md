# Hostile peer review

**Target:** `paper/main.tex` — "Attributing Rotary Position Embeddings: Exact Score Attribution, and What Actually Limits It"
**Reviewer stance:** NeurIPS area chair, reject-track. Every objection below is
checkable; computed refutations carry the code and its output.

**Recommendation: Reject.** Not because the paper is dishonest — it is conspicuously
more candid about its limits than most submissions, and its 1234-test machine
verification of every literal against fresh recomputation (§10.a) is well above
the standard for a preprint — but because after I separate the algebra from the
"measurement", nothing in it survives as an empirical contribution, and one of its
four headline claims is *false*.

---

## 0. Summary of what is actually new

| Paper claim | What it is |
|---|---|
| R1 score is exactly bilinear, `|q|` exactly position-free (`main.tex:107-112`) | Theorem. Eq. (9) of the paper's own cited YaRN ref. Literally restated. |
| R1 residuals `3.6e-14`, `<1e-13` (`main.tex:110`, `tab:exact`) | One float64 roundoff check of an identity. Not a finding. |
| R2 `A_ij` is a function of `δ` (`main.tex:113-117`) | Definition of a function of two arguments. |
| R2 "up to `920.04×`" (`main.tex:114`, `:166`, `:598`, `:605`, `:941`, `:1061`) | **Grid-resolution artefact. Diverges without bound. Not a property of RoPE.** |
| R3 `max_k|D_k|` bit-identical RoPE vs YaRN (`main.tex:119-121`) | Theorem: `inv_freq[0] = base^0 = 1` for *every* base; paper says so itself at `main.tex:373-376`. |
| R3 median ratio `0.4464` constant in δ (`main.tex:121`, `:727-731`) | Forced by homogeneity of `median(δf) = δ·median(f)`. |
| R3 "linearizable fraction rises" (`main.tex:121-122`) | Threshold count on a ladder. Contains no content vector. |
| R3 "no global linearization" (`main.tex:124-126`) | Divergence of a Taylor series at `|D| ≫ 1`. True, untested against alternatives. |
| §4.4 "well-posed object" (`main.tex:616-655`) | The original score re-expanded in a different basis. Not a new object. |
| §4.6 partial RoPE "not orthogonal" (`main.tex:848-857`) | **FALSE of partial RoPE. Artefact of the repo's own convention.** |
| §5 mscale lowers entropy (`main.tex:881-882`) | `dH/dc = -c·Var_p[s] ≤ 0`. One line. |

Nine of eleven "results" are theorems. The tenth is an artefact. The eleventh is
false.

---

## 1. THE CENTRAL OBJECTION — this is algebra, not measurement

### 1.1 R1 is the paper's own cited source, restated

`main.tex:247-259` derives `s(m,n) = q^T R_δ k`, "the relative-position property".
That is Eq. (9) of Peng et al., which `main.tex:1001` cites:

> ⟨f_q(x_m,m), f_k(x_n,n)⟩ = Re(x_m* W_q* W_k x_n e^{iθ(m−n)}) = g(x_m, x_n, m−n)
> — YaRN, §2.1, Eq. (9), <https://ar5iv.labs.arxiv.org/html/2309.00071>

The orthogonality claim (`main.tex:265-270`) is "`R_m^T R_m = I`", the defining
property of a rotation. The bilinearity test (`experiments.py:183-192`) rescales
**one** argument, `a*v`, of a linear map:

```python
scales = np.array([0.0, 1.0, 2.0, -3.0, 7.5, 1e-3, 1e3])
deltas  = [1, 13, 29, 61, 127]                      # 5 distances
# -> "35 score checks"  (main.tex:537)
```

`tests` = 35, but: `a=0` yields `0/(0+1e-12) = 0` (10/35 vacuous), `a=1` is the
identical code path (bitwise 0), all 35 reuse **one** content pair `(v,u)` drawn
once at `experiments.py:162-163`, and the key is never rescaled. So 25 informative
checks over 5 distinct scores. A reviewer reading `main.tex:504-505` ("bilinearity
in content, `<10^{-14}`", "score checks 35") will take that as breadth. It is not.

The relative-position "42 position pairs" (`main.tex:501`) is 7 distances × 6 base
positions **of the same fixed `(v,u)`**, at a single shift value of 7
(`experiments.py:157,166-167`). The paper is honest about this at `main.tex:535-536`.

**Verdict: R1 is correct, cited, and does not need an experiment. It should not be
a contribution.**

### 1.2 R3 is a two-line deduction, and the paper says so

`main.tex:373-376` states outright:

> "$\theta_0 = \mathrm{base}^{0} = 1$ for \emph{every} base, so raising the base from
> $10000$ to $500000$ leaves $\theta_0$, and hence $\max_k|D_k|$, completely
> unchanged"

Since the ladder is non-increasing in `k`, `max_k|D_k| = δ·f_0 = δ` for *any*
scheme whose fastest channel has `f_0 = 1`. Verified:

```
=== the fastest channel: inv_freq[0] ===
  rope               inv_freq[0] = 1.0
  yarn               inv_freq[0] = 1.0
  position_interp    inv_freq[0] = 0.03125     <- only PI changes it
  base_500k          inv_freq[0] = 1.0
  base_1e9           inv_freq[0] = 1.0
=== max|D_k| ===
  delta=8192  rope=8192  yarn=8192  PI=256  b500k=8192
  yarn - rope max|D| disagreement at delta: 0.0
```

"54 tested distances, largest disagreement `0.000×10^0 rad`" (`main.tex:672-673`)
is `0.0` by arithmetic. And the median ratio:

```
delta=1      med_yarn/med_rope = 0.44643114576841303
delta=512    med_yarn/med_rope = 0.44643114576841303
delta=8192   med_yarn/med_rope = 0.44643114576841303
```

`median(δf) = δ·median(f)` for `δ>0`. `main.tex:730-731` says "A fixed ratio, not an
improvement that grows with distance, because YaRN's ramp is a fixed per-channel
rescaling and D is linear in δ for every channel" — the authors state the theorem
and then present its consequence as a measurement. That is the pattern throughout.

The "linearizable fraction" is worse: `linearization_error` computes
`mean(|δ·f_k| ≤ 0.1)` (`experiments.py:307`), a function of the ladder and the
threshold. It never touches `q_hat` or `k_hat`.

### 1.3 YaRN's own design document already states R3

R3's content is YaRN §3.2, verbatim:

> "we choose **not to interpolate the higher frequency dimensions at all** while
> always interpolating the lower frequency dimensions" — YaRN §3.2

> "The property we want to guarantee is that: The lowest frequency needs to be
> scaled as much as linear positional scaling and **the highest frequency to stay
> constant**." — YaRN, Appendix A.1

"9 of 32 unchanged" (`main.tex:348-349`) is a read-out of the ramp mask: correction
range `[8,21]`, `ramp[0..8]=0`, `ramp[31]=1` → exactly 9 identical entries.
`main.tex:1001-1004` concedes it:

> "our measurement of that invariance is a direct consequence of that construction"

So the abstract (`main.tex:119-121`) sells as a result what the Related Work
section admits is a restatement of the cited paper's stated design goal. The
"9 of 32" and "YaRN does not raise the base" corrections are YaRN's Definition 2
and Eq. (20), not findings.

### 1.4 §5 (mscale/entropy) is a one-line identity, already in YaRN

For `p_c = softmax(c·s)`: `dH/dc = E_p[s] − c·Var_p[s] = −c·Var_p[s] ≤ 0`, with
equality iff `p` is degenerate. Verified over 30 independent draws:

```
n=30 draws, all negative (entropy falls)? True  min=-0.6280 max=-0.2181
```

`main.tex:881-882` calls this "a real, measurable, and easily overlooked side
effect of context extension". It is not overlooked: YaRN §3.4 already introduces
the temperature "on the logits before the attention softmax", Eq. (21)-(22) is the
same `0.1 ln s + 1`, and Appendix A.2 measures its effect on perplexity. The paper
rediscovers `dH/dc ≤ 0` and adds a figure.

### 1.5 How much survives

- **The algebra.** Correct, cleanly written, and a genuinely useful exposition of
  the per-pair sinusoid. It belongs in a tutorial or a section of a RoPE paper.
- **The measurement.** There is none. Zero model, zero data, zero repeats, zero
  error bars, five fixed seeds (`main.tex:954`: "one query/key draw per
  experiment"). Every "residual" is the float64 difference between two
  analytically equal expressions.
- **The one place an experiment was needed** — whether a trained model uses this
  structure — is unaddressed, and `projects/rope_attribution/real_model.py`
  exists specifically to address it: *"This module closes that gap: it runs the
  same measurements on the activations of pretrained checkpoints"* (lines 7-8),
  *"across-head and across-layer dispersion is what distinguishes a measurement on
  a trained network from a demonstration on random vectors"* (lines 26-28). It is
  referenced by nothing outside itself, and `results/real_model.json` does not
  exist. The authors built the rebuttal to the reviewer's main objection and did
  not run it.

---

## 2. A CLAIM I COMPUTED AND FOUND FALSE

### 2.1 `main.tex:848-857` — "partial RoPE is not orthogonal" is false

> *"**pp-RoPE is not orthogonal, and that matters.** Partial rotation is not an
> orthogonal transform, and a partially rotated vector's norm is not preserved: we
> measure a norm deviation of $0.2663$ at $n_{\mathrm{rot}}=16$, against
> $1.776\times10^{-15}$ for full RoPE — a factor of $1.5\times10^{14}$."*
> — `main.tex:848-857`, restated in the abstract at `main.tex:127-128`
> ("at the cost of losing norm preservation") and the conclusion at
> `main.tex:1069-1070`.

This is a property of `rope.py:103-134`, not of partial RoPE. `partial_rope_cos_sin`
returns a 16-wide non-trivial `cos`/`sin` which `apply_rope`/`rotate_half`
(`rope.py:91-100`) then pairs **across the whole 64-wide head**, i.e. channel `i`
with channel `i+32`. Every `n_rot ∈ {2,…,62}` therefore mixes a rotated half with
an unrotated partner.

```python
# reproduce
import sys, numpy as np
sys.path.insert(0, "projects")
from rope_attribution.rope import inv_freq, apply_rope, rope_cos_sin, rotate_half, partial_rope_cos_sin
dim, n_rot = 64, 16
freqs = inv_freq(dim)

def M_repo(pos):                       # what the repo does
    c, s = partial_rope_cos_sin(np.array([pos]), freqs, n_rot)
    return np.array([apply_rope(np.eye(dim)[i][None,:], c, s)[0] for i in range(dim)])

def M_hf(pos):                         # HuggingFace / GPT-NeoX: rotate_half inside the prefix
    c, s = rope_cos_sin(np.array([pos]), freqs[:n_rot//2])
    def ap(x):
        out = x.copy(); xr = x[..., :n_rot]
        out[..., :n_rot] = xr*c + rotate_half(xr)*s; return out
    return np.array([ap(np.eye(dim)[i][None,:])[0] for i in range(dim)])

for name, fn in (("REPO", M_repo), ("HF/NeoX", M_hf)):
    M = fn(37)
    untouched = [i for i in range(dim) if np.all(np.isclose(M[i], np.eye(dim)[i]))]
    print(name, "untouched:", len(untouched), "/64", " ||M^T M - I|| =",
          np.abs(M.T @ M - np.eye(dim)).max())
```

Output:

```
REPO    untouched: 32 /64   ||M^T M - I||_max = 9.7554e-01
HF/NeoX untouched: 48 /64   ||M^T M - I||_max = 2.2204e-16
for which n_rot is the repo's construction orthogonal?
  n_rot=  0  0.000000     n_rot= 32  0.999945     n_rot= 64  0.000000
  n_rot= 16  0.975538     n_rot= 62  0.997203
norm deviation, repo convention : 0.266307     <- matches the paper's 0.2663
norm deviation, HF  convention  : 0.000e+00
```

Under the convention real models use, partial RoPE is **exactly orthogonal**
(`2.2e-16`), the norm deviation is **exactly 0**, and 48 of 64 channels are
untouched. The repo's own documentation of a real checkpoint agrees
(`real_model.py:40-42`): *"`EleutherAI/pythia-160m` declares
`partial_rotary_factor = 0.25`, so only 16 of each head's 64 channels rotate, **in
8 rotary pairs**"* — eight pairs, i.e. pairing *within* the prefix, i.e. the
orthogonal HF construction. So the paper describes pythia's pp-RoPE with a
construction pythia does not use.

**The paper asserts two mutually exclusive things in the same subsection.** See
§2.2.

### 2.2 `main.tex:843-846` contradicts `main.tex:848-857`

```
=== REPO    untouched 32/64 = 0.50  ->  "a quarter of the channels carry no
                                      position at all"        = False
                                     ->  "pp-RoPE is not orthogonal" = True
=== HF      untouched 48/64 = 0.75  ->  "a quarter of the channels carry no
                                      position at all"        = True
                                     ->  "pp-RoPE is not orthogonal" = False
```

- `main.tex:843-845`: *"This is the one setting in which a position-free additive
  attribution is exactly correct, and it is correct because **a quarter of the
  channels carry no position at all**."*
- `main.tex:848-852`: *"**pp-RoPE is not orthogonal**, and a partially rotated
  vector's norm is not preserved."*

Under the repo's implementation only 32/64 channels are untouched, so 32/64 carry
position and "a quarter" is wrong; under the HF implementation 48/64 are untouched,
so "a quarter" is right and the Remark is wrong. Both cannot hold. One of the two
sentences must be deleted, and the choice determines whether the paper's
recommended configuration (`main.tex:845-846`: *"That makes pp-RoPE the natural
recommendation for feature-level attribution in a model that has it"*) has a
norm-preservation problem at all.

### 2.3 `main.tex:836` — "spread exactly 0.0" is not a measurement

`experiments.py:388`:

```python
clean_scores.append(float(q_hat[n_rot:] @ k_hat[n_rot:]))
```

`q_hat` and `k_hat` are never passed through `partial_rope_cos_sin`. The line
contains no `d`. The spread is `0.0` for **any** two vectors, any `n_rot`, any
scheme, any δ:

```
d=1        clean=43.742453600841074
d=999999   clean=43.742453600841074
```

The paper reports this as an exact measurement, records it in
`results/measurements.json::partial_rope.clean_score_spread`, and puts it in the
provenance table (`main.tex:1141`) as evidence. It is an assertion about the shape
of the source code. Same for the *rotated* spread (`12.03`, `22.56`), which is a
real computation but is compared against a number that cannot be anything but zero.

Worse, the paper's own verification apparatus **certifies** it.
`tests/test_paper_claims.py:1701` runs
`assert_paper_says_exactly(partial["clean_score_spread"], ...)` — i.e. a green test
asserts that the paper's "exactly $0.0$" equals the code's `0.0`, which the code
produces without ever reading `d`. This is the general failure mode of the repo's
otherwise excellent provenance machinery (§10.5): it proves paper↔code consistency,
which is not the same as claim↔world consistency.

---

## 3. THE 920.04× HEADLINE IS A GRID ARTEFACT

`main.tex:113-117` (abstract), `:166`, `:594`, `:598`, `:605`, `:941`, `:1061`.

The quantity is `max_δ|c_i(δ)| / min_δ|c_i(δ)|` (`experiments.py:262-266`,
`figures.py:1050-1054`). `c_i(δ)` is a bounded quasi-periodic function of δ, so on
a denser grid `max` → its sup and `min` → 0, and the ratio **diverges**. Same
content vectors, same seed, only the grid changes:

```python
# probe1.py
for n in (5, 10, 20, 54, 100, 200, 500, 1000, 2000, 5000):
    g = tuple(sorted(set(int(round(x)) for x in np.unique(
        np.round(np.logspace(0.0, np.log10(8192), n))))))
    print(f"{len(g):>9} {ratios_on(g).max():>14.4f}")
```

```
 n_deltas      max_ratio
        5        34.9807
       10       223.9362
       20      2125.9236
       54       920.0445     <- the paper's number
       84      4511.8747
      332     13710.1538
     1022     19971.8819
     2048   4094122.9933
```

And the mechanism, to remove any doubt:

```
n=   48  max_i max_d|c_i| = 54.5308   min_i min_d|c_i| = 0.0420256   -> 1297.6x
n=  332  max_i max_d|c_i| = 79.7353   min_i min_d|c_i| = 0.00348397  -> 22886.3x
n= 1022  max_i max_d|c_i| = 82.5681   min_i min_d|c_i| = 0.000706609 -> 116851.1x
n= 2048  max_i max_d|c_i| = 98.3815   min_i min_d|c_i| = 1.59178e-05 -> 6180608.9x
```

The numerator saturates; the denominator is the artefact. The headline spans
**five orders of magnitude on identical inputs** depending only on how many δ are
sampled.

The authors know. `main.tex:596-598`:

> "On the coarser $5$-point grid ... the worst-feature ratio is $43.05$; that is a
> statement about that grid's coarseness, not about the phenomenon, **since the
> ratio grows with the number of distances sampled**."

That sentence retracts the abstract. A reviewer is entitled to read the abstract's
"up to $920.04\times$" as a measurement of RoPE, and it is not one — it is a
property of `(seed, F, grid)`. Across seeds, at fixed `F=8` and fixed grid:

```
 seed  n_feat     max ratio
    1      8         920.04
    4      8         416.77
    6      8       28072.65
    4     64      782674.74
```

A 67× spread across draws, and 850× across `F`, with no variance reported. The
paper's central quantitative claim is a draw from a distribution it never
characterises.

**Fix, not refutation:** replace the ratio with a grid-independent statistic
(`std_δ(c_i)/rms(c_i)`, or the spectrum of `c_i` in the `inv_freq` basis) and report
the grid-density convergence as a figure. The phenomenon survives; the number does
not.

---

## 4. "NO MEASUREMENT" IS NOT DISCHARGED BY THE DISCLAIMER

`main.tex:189-197` and `:930-943` disclaim the model and the SAE at length. The
disclaimer is honest, and it does not do the work it is asked to do, because the
problem is not the *data*, it is the *design*: with `d=64`, `base=10000`, `s=32`,
`L=2048` fixed (`main.tex:386-388`, `:953-956`) and one draw per experiment, there
is no configuration in which any of R1/R2/R3 could have come out otherwise.

- R1's outcome is determined by `R_m` being a rotation. No `d`, no `base`, no data
  affects it.
- R3's outcome is determined by `inv_freq[0]=1` and the ramp. No data affects it.
- R2's headline number is a min over a grid.

So the honest disclaimer answers a question nobody asked. The right question is:
*what result would have falsified any of these?* Nothing in the design could.

## 5. A HARD-CODED BOOLEAN IN THE "EVERY NUMBER IS COMPUTED" JSON

`experiments.py:274`:

```python
"score_depends_only_on_delta": True,
```

A literal. It is never computed, never plotted. The only other place it appears
is `tests/test_experiments.py`, where the test asserts
`reported["score_depends_only_on_delta"] is True` — i.e. it asserts that the
hardcoded value equals itself. It is written to `results/measurements.json`, the
file the paper's Method section calls "canonical" (`main.tex:385-386`), and the
appendix asserts:

> "Every quantitative claim in this paper is either a literal in the table below,
> or a value derived from it by an arithmetic operation stated where it is used.
> **No number was typed from memory.**" — `main.tex:1102-1104`

> "The full audit is mechanical: each literal in the tables above is either an
> entry of `results/measurements.json`, an entry of a figure CSV, or an explicitly
> stated quotient of two such entries." — `main.tex:437-439`

This is the one place where the paper's own machinery — a good machinery, better
than most submissions manage — is defeated by a hardcoded value. It is a small
thing. It is also exactly the thing a reproducibility reviewer greps for, and it
will cost the paper its credibility on the provenance appendix as a whole.

## 6. R3's "DECISIVE GUARD" IS AN ALGEBRAIC RATIO

`main.tex:797-806` stakes the paper's honesty on this number:

> "Figure~\ref{fig:error} is the decisive guard on this conclusion, and it is
> where the optimistic reading dies. At $\delta=8192$ the amplitude-weighted error
> is $1.24572\times10^6$ for RoPE and $1.24179\times10^6$ for YaRN, a ratio of
> $0.996840$: YaRN buys less than $0.4\%$ on this measure."

The metric is unbounded and quadratic in δ because `cos D ↦ 1 − D²/2` is a Taylor
polynomial evaluated far outside its radius of convergence. Decomposed:

```
delta=512:  A-term share of total abs error = 0.9901, B-term (sin->D) share = 0.0099
delta=8192: A-term share = 0.9994, B-term share = 0.0006
amp_wtd_err / delta^2 is constant to 4 digits from delta=64 to delta=65536
closed form:  amp_wtd_err ~ delta^2 * sum_k |A_k| f_k^2 / sum_k R_k
seed=2: measured ratio yarn/rope = 0.996830   closed-form A-term ratio = 0.996838
```

So `0.996840` is, to three digits, the ratio of two closed-form sums of the
frequency ladder and the content vectors. It carries no information about δ, and
"less than 0.4%" is `1 − 0.996838` computed in advance. The "roughly 1000×" for
position interpolation at `main.tex:806-808` is `1/32² = 9.77e-4` — exactly the
quadratic scaling.

`main.tex:802-804` explains the mechanism:

> "the error is dominated by the few fast channels, **whose amplitudes are
> comparable to the slow channels'** but whose angles are enormous"

For the paper's own seed (`method_spectrum`, seed 2) this is **false**: the
top error contributor is `k=3`, not `k=0`.

```
seed top-1 k  its share  k=0 share |D_top|/|D_0|   R_top/R_0  R_0 rank
   2       3     0.5456     0.1859         0.422        7.376        20
   4       1     0.3538     0.1373         0.750        4.379        30
  11       1     0.3901     0.3649         0.750        2.139        26
```

The dominating channel has the **largest** amplitude (`R_3 = 6.31`, rank 1 of 32;
`R_0 = 0.856`, rank 20) and a *smaller* angle than `k=0` (42% of `|D_0|`).
Amplitudes are not "comparable to the slow channels'" either: `R_0/R_31 = 9.6`.
For 7 of 10 seeds `k=0` does top the list, so the qualitative statement survives;
the explanation given for it does not, on the paper's own draw. (Also note
`fig06` and `fig05` are computed from *different* content draws —
`figures.py:79-80` `SEED_QK = 20_250_929` vs `SEED_METHOD = 2` — and
`main.tex:786-800` interleaves their numbers in one argument.)

## 7. "NO GLOBAL LINEARIZATION" IS TRUE BUT UNTESTED

I tried the obvious refutation and **it failed**, so I drop the objection that
would have been strongest. Dropping the `m` fastest channels and recomputing the
amplitude-weighted error:

```
RoPE:
  drop 0 fastest of 32 -> d=512:4.91e+03  d=4096:3.12e+05  d=8192:1.25e+06  d=65536:7.97e+07
  drop 4 fastest of 32 -> d=512:1.03e+03  d=4096:6.49e+04  d=8192:2.59e+05  d=65536:1.66e+07
  drop 8 fastest of 32 -> d=512:1.16e+02  d=4096:7.14e+03  d=8192:2.85e+04  d=65536:1.82e+06
```

Even discarding a quarter of the channels, the error at δ=8192 is `2.8e4`, nowhere
near 1. At δ=8192 *every* channel has `|D_k| ≫ 1`, so no first-order Taylor
approximation of a periodic function can be good. The paper's conclusion holds.

But the paper never runs this, so a reviewer cannot tell whether it was
considered. And `main.tex:965-970` admits the truncation was never swept:

```
=== does "fastest pair reaches error 1 at delta=3" survive a different truncation? ===
  d=3  cos->1-D^2/2, sin->D     normalised err of pair k=0 = 3.6293
  d=3  sin only: cos exact      normalised err of pair k=0 = 2.6151
  d=3  keep 4 terms             normalised err of pair k=0 = 1.8507
```

The number moves by a factor of 2 across three reasonable choices. The qualitative
claim (`k=0` is out of the linear regime within three tokens) survives all three —
credit where due — but "verified the latter qualitatively ... but did not sweep
thresholds" is not verification, and the linearizable fractions in
`tab:spectrum` are threshold artefacts:

```
  thr=0.01  rope@8192=0.0000  yarn@8192=0.0000
  thr=0.1   rope@8192=0.0000  yarn@8192=0.1250
  thr=1.0   rope@8192=0.0000  yarn@8192=0.3438
```

`0.2188` vs `0.1250` vs `0.3438` for the same quantity at the same δ.

## 8. NOVELTY / SIGNIFICANCE

The two "corrections" advertised at `main.tex:1081-1096` are corrections to *this
repository's own legacy drafts*. `README.md:119-137` names them as
"the legacy documents", and `README.md:187-203` marks `projects/frontier-01-*` as
"an earlier, frozen body of work kept for provenance" that must not be cited.
Neither correction targets a claim made in the published literature by anyone else:

- "RoPE breaks bilinearity" — never claimed in RoFormer, YaRN, PI, or anywhere I
  could find; it was a bug in this repo's own demo (`README.md:124-129`).
- "YaRN raises the base to 10000→500000" — the 500000 figure is Llama 3's
  `rope_theta`, a real and widespread public conflation, and it *is* worth
  correcting. But it is worth a paragraph, not a contribution bullet, and the
  correction is that YaRN keeps `base=10000` — which is Definition 2 and Eq. (20)
  of the paper's own citation.

`main.tex:1043-1045` claims the answer "has not been stated in this form". Having
read YaRN §2.1 and §3.2 in full, I find no form it has not been stated in: the
score is a function of `δ` (their Eq. 9), the magnitude is position-independent
(their Eq. 8 has no position modulus), and the fastest channel is untouched by
design (their §3.2 and A.1).

The framing promises feature attribution and delivers position-encoding algebra.
`main.tex:1030-1037` says this explicitly — *"they operate on activations at a
fixed position ... the positional encoding is part of the forward pass, not part of
the attribution target"* — then claims novelty for the case where it does enter.
But `main.tex:474-478` admits the case where it *matters* (attention weights, not
scores) is deliberately unquantified, and `main.tex:846` recommends a
configuration on the strength of a number that is 0.0 by construction.

## 9. STATISTICAL SUBSTANCE: none, and honestly labelled

There are no error bars, no seeds variation, no dataset, no model, n=1 per
experiment. The paper is explicit (`main.tex:952-957`). Counting what is called a
"measurement":

- **R1** (5 quantities): float64 residual of an identity. 0% information.
- **R2** (the headline): grid-density artefact; 67× seed spread. 0% as stated.
- **R3** (max angle): arithmetic. 0%.
- **R3** (median ratio): homogeneity. 0%.
- **R3** (linearizable fraction): threshold count on a ladder, no content vector. 0%.
- **R3** (amp-wtd ratio): closed-form quadratic ratio. 0%.
- **§4.5** (partial RoPE): 0.0 by construction; norm deviation from a
  non-standard implementation. 0%.
- **§5** (entropy): `dH/dc ≤ 0`. 0%.
- **`clean_magnitude_share = 0.243`** (`main.tex:839-841`, called "substantially
  weighted"): one draw. Ten draws:

```
 seed  mean share      min      max  share at d=1  share at d=4096
    3      0.2089   0.0724   0.4050        0.4050           0.0724
    8      0.7608   0.6623   0.9064        0.9064           0.7138
    9      0.8517   0.7806   0.9011        0.8938           0.7877
across 10 draws: mean-of-means 0.5844, min single value 0.0724, max 0.9812
```

The paper's `0.243` is near the **bottom** of the distribution. "Substantially
weighted" is a property of seed 3, not of partial RoPE.

Total: 0/9 empirical content.

## 10. INTERNAL CONSISTENCY DEFECTS

1. **Three grids, two declared.** `main.tex:389-393` declares exactly two
   ("Two grids are used ... and are kept distinct throughout"). `main.tex:596`
   uses a third, `{1,8,32,128,512}` (this is `experiments.py:223`'s default
   `deltas`), which is the provenance of `43.05` (`main.tex:1129`).
2. **`main.tex:392-393`** — "Every quantity carries its grid with it, so numbers
   drawn from different grids are never compared as though they were the same."
   Violated by `main.tex:558` (additivity residual quoted for both grids in one
   sentence) and by `main.tex:594-598` (the 920.04/43.05 comparison).
3. **Ceilings vs digits.** `main.tex:521-526`: "We report these as ceilings
   rather than digits... Quoting more precision than the quantity carries would be
   false." Yet the abstract (`:110`), §4.3 (`:558`) and the conclusion (`:1058`)
   each quote `3.6×10^{-14}`. Three of the paper's most prominent numbers violate
   the rule the paper states for itself. (The three are all correctly rounded —
   the defect is the rule, not the arithmetic.)
4. **`README.md:22` and `:31`** claim "1185 tests". `pytest --collect-only` →
   **1234**, and `pytest -q` → **1234 passed in 4.37s**. The repo's own headline
   audit surface is stale by 49 tests.
5. **`README.md:130-131`** ("That is plain position interpolation, a different
   method") vs **`main.tex:361-372`** ("this is *not* the same scheme as raising
   `base`") and `measurements.json` (median angle 0.9076 vs 0.1867 at δ=512).
   The README is ambiguous, the paper is right. Recorded only because the paper's
   provenance story depends on the README agreeing with it.
6. **`paper/Makefile:9-13`** still says "No LaTeX toolchain was present on the
   machine where this paper was written, so it has NOT been executed there."
   `paper/build/main.pdf` exists (16 pages, 9 embedded images — all figures
   present). Stale comment; harmless, but it dates the audit.

### 10.a In the paper's favour: the verification is real, and I checked

I expected to find that the "Provenance of every number" appendix
(`main.tex:1100-1176`) was an unaudited claim of audit. It is not.
`tests/test_paper_claims.py` is ~2570 lines that parse `paper/main.tex`, extract
each literal by a regex anchored on the surrounding prose, and compare it to a
**fresh recomputation** from `rope_attribution` — with
`test_no_number_in_the_paper_is_left_unverified`
(`tests/test_paper_claims.py:2503`) as an exhaustiveness gate, exact float
equality demanded wherever the paper says "exactly"/"identical"
(`assert_paper_says_exactly`, line 552), and cross-checks that every CSV the
provenance table names is one a test actually reads (line 2560).
`pytest -q` → **1234 passed in 4.37s**. Four defects this harness found are fixed
in `32f4f22`.

This is well above the standard for a preprint and I withdraw the "the audit is a
table of intentions" objection. The narrowed criticism, which is the one that
matters: **the harness verifies that the paper reproduces the code. It cannot
verify that the code measures something.** A tautology that is computed
deterministically passes every assertion in that file — which is precisely how
`clean_score_spread = 0.0` (§2.3), `score_depends_only_on_delta` (§5) and the
whole of R3 (§1.2) acquire the green checkmark. A reviewer should read "1234
machine-verified claims" as "1234 consistent claims".

## 11. PRESENTATION

- **Not the NeurIPS style file.** `main.tex:15` is
   `\documentclass[twocolumn]{article}`; there is no
   `\usepackage{neurips_20XX_conference}`, no `nips202X.sty`, no `\author`
   content (`main.tex:93`: `\author{}`), no checklist. `main.tex:4` says
   "Self-contained NeurIPS-format preprint (two-column review layout)" — a
   self-declared deviation from a mandatory formatting requirement. This is a
   desk-reject trigger regardless of the science.
- **Over length.** The NeurIPS main-text limit is 9 pages excluding references and
  appendix. In `paper/build/main.pdf`, "References" begins on p.14, so main text is
  **~13 pages: a 45% overrun**, with Related Work on p.12 and Conclusion on p.13.
- **Figures: 9 present, all used, all enforced.** No dead figure, no missing
  figure. `tests/test_paper_claims.py:2292-2306` asserts that every `figNN` token
  in `main.tex` has a PNG on disk and that every `\paperfigure` stem has both a
  `.png` and a `.csv` sidecar, so the `\IfFileExists` placeholder path
  (`main.tex:47-67`) cannot silently produce an evidence-free PDF without failing
  a test. Fig 05 was recently added (`056c661 fix(paper): include fig05, which the
  text cites but never showed`), which is the right kind of commit and worth
  noting.
- **Title vs content.** "and What Actually Limits It" promises an account of the
  limits of *attribution*. `main.tex:905-913` says the opposite: "The
  linearization failure is a statement about the *approximation* ... not about
  attribution... R3 is orthogonal to both." So the third third of the paper is
  explicitly not about what limits attribution.
- **A citation gap.** The paper "promotes the gate/phase split from heuristic to
  theorem" (`main.tex:112`, `:156`, `:1058-1059`) and calls it "the gate/phase
  split used in sparse-feature interpretability" (`main.tex:272-273`) — with **no
  citation anywhere in `main.tex` or `references.bib`**. A "promotion from
  heuristic to theorem" whose antecedent heuristic is uncited is unfalsifiable.

---

## 12. Ranked: the five objections most likely to sink this paper

**1. Nothing here is measured; nine of the eleven results are theorems, and the
paper presents them as findings.**
The whole of R1 is RoFormer's relative-position property, restated via Eq. (9) of
the paper's own YaRN citation. All of R3 follows from `inv_freq[0] = base^0 = 1`
(`main.tex:373-376`) plus homogeneity of the median. §5 follows from `dH/dc ≤ 0`.
With `d=64, base=10000, s=32` fixed and n=1, no configuration could falsify any
claim. *Response:* agree, and convert. Make the theorems theorems — prove
`max_k|D_k| = δ` and the closed form for the amplitude-weighted error in the text,
delete every "we measure" that a proof covers, and state plainly in the abstract
that the contribution is an algebraic characterization, not an empirical one. Then
run `real_model.py`, which already exists, on 2-3 checkpoints and report
per-head/per-layer dispersion. That converts a tautology objection into the paper's
one real asset: a clean, correct, model-independent characterization of the score
that no one has written down in this form. Keep the 1234-test harness — it is the
proof that the conversion was done honestly.

**2. The `920.04×` headline is a grid-resolution artefact and the paper retracts
it 400 lines later.**
Same inputs, five orders of magnitude of variation with grid density; 67× across
seeds; `min_δ|c_i| → 0` is the entire mechanism. `main.tex:596-598` says so
explicitly. A reviewer will read the abstract and the retraction together and
conclude the paper's central quantitative claim is not a property of RoPE.
*Response:* replace `max/min` with a grid-independent dispersion measure (std/rms
in δ, or the `inv_freq`-basis spectrum of `c_i`), plot the ratio against grid
density as evidence that it is an artefact, and report the new statistic across
seeds with error bars. The phenomenon — a feature's contribution is a function of
distance — is real and is Theorem `prop:cond`; it does not need the ratio.

**3. `main.tex:848-857` is false, and `main.tex:843-846` contradicts it.**
Partial RoPE under the HuggingFace convention is orthogonal to `2.2e-16` with a
norm deviation of exactly `0`; the paper's `0.2663` is produced by pairing
channel `i` with `i+32` while only 16 channels receive a non-trivial `cos`/`sin`
(`rope.py:103-134`). The repo's own `real_model.py:40-42` documents that
pythia-160m's pp-RoPE is 8 pairs over the prefix — the orthogonal convention. The
paper simultaneously claims a quarter of channels are position-free (only true of
the standard convention) and that pp-RoPE is non-orthogonal (only true of the
non-standard one). One abstract sentence, one Remark, and one "natural
recommendation" hang on which convention you mean. *Response:* implement
`rotate_half` within the rotated prefix, re-measure, delete the Remark, and state
which convention each partial-RoPE model uses. This is the only outright error I
could find, and it is in the abstract.

**4. The two advertised "corrections" are already in the cited literature, and
the paper concedes it in its own Related Work.**
YaRN §3.2 states the untouched-fast-channel property in words; Appendix A.1 states
it as the design goal ("the highest frequency to stay constant"); Eq. (20) states
the ramp; Eq. (9) states the relative-position property; Eq. (21) states the
temperature. `main.tex:1001-1004` says "our measurement of that invariance is a
direct consequence of that construction." The other correction targets a bug in
this repo's own legacy scripts (`README.md:124-129`). *Response:* demote both from
contributions to a short "two notes on the literature" paragraph, stop claiming
novelty at `main.tex:1043-1045`, and pivot the novelty claim to the one thing that
is genuinely not in YaRN or RoFormer: the per-pair `(A_k, B_k, R_k, ψ_k)`
decomposition *of the feature-level attribution* `Σ_ij f_i g_j A_ij(δ)` written out
per rotary pair (`main.tex:616-655`). That object is new, it is exact, and it is
correctly derived. It just needs an argument that it is useful, not a figure of a
roundoff residual.

**5. Wrong venue, wrong length, no NeurIPS style file.**
`\documentclass[twocolumn]{article}` with a hand-rolled `geometry` and an empty
`\author{}` (`main.tex:15`, `:26`, `:93`), against a 9-page main-text limit with
~13 pages used. Combined with no model and no data, the paper has no venue-appropriate
category. *Response:* none needed if the paper is reframed as a short position /
empirical-methodology note (≤4 pages, results in an appendix), which its actual
content fits comfortably. If it stays a 13-page full paper, it needs experiments,
which means `real_model.py`.

---

## Appendix: criticisms I raised, checked, and dropped

Recorded so the list is not padded.

- **"The committed CSVs are not bit-reproducible."** I recomputed `fig06`'s
  `amplitude_R_k` and got `0.9526294617807265` against the CSV's
  `0.9526294617807264`. That was my own scalar-`math.hypot`; via
  `pair_terms(...).amplitude` (`np.hypot`) it is `...264`, bit-equal to the CSV and
  to `main.tex:542`. **Wrong; dropped.** The provenance machinery is sound — and
  `tests/test_paper_claims.py:2082` asserts exactly this equality.
- **"The provenance appendix is an unaudited table of intentions."** Wrong:
  `tests/test_paper_claims.py` is ~2570 lines that parse the paper and verify each
  literal against fresh recomputation, with an exhaustiveness gate at line 2503.
  All 1234 tests pass. **Wrong; dropped** and replaced by the narrowed form in
  §10.a (consistency, not informativeness).
- **"δ=8192 has no declared grid and no provenance row."** Wrong: `fig05`'s CSV has
  `delta=8192` rows for all four methods, and `main.tex:1134` attributes
  `1.24572×10^6` to the fig05 CSV. **Wrong; dropped.**
- **"The `\IfFileExists` guard lets the PDF build with all nine figures missing."**
  The guard exists, but `tests/test_paper_claims.py:2292-2306` fails if any named
  figure lacks a PNG or a CSV sidecar. **Wrong; dropped.**
- **"Dropping the fastest channels rescues a global linearization, refuting R3."**
  Dropping 8 of 32 at δ=8192 takes the amplitude-weighted error from `1.25e6` to
  `2.85e4` — better by 44×, but nowhere near the `≈1` that would mean "as good as
  deleting the term". R3's qualitative conclusion survives. **Wrong; dropped.**
- **"The `δ=3` linearization failure is a threshold artefact."** Three
  truncations (`cos→1−D²/2`, cosine exact, four terms) give normalized errors
  3.63 / 2.62 / 1.85 at δ=3: the number moves, but `k=0` is out of the linear
  regime within 3 tokens under all three. **Weakly wrong; dropped as a refutation,
  retained as a limitation** (§7).
- **"The median ratio `0.4464` is not bit-identical across distances."** It is not:
  `...303` at most distances, `...31` at δ=861 and δ=4493. One ULP. Not worth
  raising. **Wrong; dropped.**
- **"README:130-131 contradicts the paper on base-raising vs interpolation."**
  Re-read, the README sentence is a valid compressed statement (the legacy claim
  describes position interpolation). It is ambiguous, not wrong. **Dropped as a
  contradiction**; retained only as §10.5.

---

## Disposition of the ranked objections

What was actually done about each of the five, stated plainly including the part
that is still outstanding. Line references are to the current `paper/main.tex`.

### 1. "Nine of eleven results are theorems, not measurements" — largely addressed

Agreed, and converted in two directions.

*Proofs instead of "we measure".* Both propositions now carry proofs
(`\begin{proof}`); before this, `amsthm` was loaded and `\newtheorem` declared,
and the paper contained no proof at all. Proposition `prop:blind` states the
part a careless proof would slide over: with `B_k = 0` the term is still
`A_k cos(D)`, which is *not* constant in `δ` — it carries no amplitude
information beyond the sign of its cosine. `prop:cond` is proved, which is what
makes the sign reversal a consequence rather than a coincidence of one draw.

*The empirical asset, finally reported.* `results/real_model.json` held 58k lines
of measurements on three trained checkpoints and the paper cited none of it,
while asserting in two places that there is no trained model in the work. New
Section `sec:trained` reports it:

- the identities are exact on real weights, at the dtype floor — relative-position
  to `1.14e-12` worst case, the per-pair closed form to a median
  `5.06e-8` / `3.57e-8` / `5.85e-8` against a float32 eps of `1.19e-7`;
- the leading feature's share of a score spans `0.0108`–`0.9999` on
  `pythia-160m`, `0.0148`–`0.9999` on `llama-160m`, `0.0150`–`0.9999` on
  `SmolLM-135M` — factors of `92.9`, `67.6`, `66.8`;
- only `0.68%` / `0.61%` / `0.60%` of channels are position-blind, so on real
  weights almost everything is phase;
- GPT-2 has no rotary embedding at all, reported as the control it is.

`tests/test_paper_trained_claims.py` re-derives every printed number from the
JSON.

*Still outstanding:* the sign reversal itself was not measured for real features —
only the dispersion of feature shares. The paper now says so, which it did not
before.

### 2. "The 920.04x headline is a grid artefact" — resolved

Withdrawn in a remark that reproduces the growth rather than merely asserting it:
`rope_attribution/statistics.py` measures the ratio at six grid densities and
finds it spread by a factor of `32`, non-monotone in density, with a seed
standard deviation exceeding its own mean. Replaced by what the measurement
supports, which is stronger: all `8/8` features change sign over `δ ∈ [1, 8192]`,
with the magnitudes at the two ends of the range agreeing to `10^-0.08`.

The "orders of magnitude" claim was silently dependent on the withdrawn number —
it was `920.04 / 4.97e-14` — so it was replaced too, by the residual left by
replacing `c_i(δ)` with its per-feature mean divided by the additivity residual:
**15 orders of magnitude**, stable within `15.0`–`15.2` as the grid is made `120×`
denser. Unlike `max/min`, its denominator is additive and cannot diverge.

`test_the_ratio_is_withdrawn_and_the_paper_says_so` fails if the claim is
reinstated.

### 3. "The partial-RoPE orthogonality claim is false" — fixed

Confirmed. `apply_partial_rope` now pairs within the rotated block (the GPT-NeoX
layout) rather than across the whole head. `partial_norm_deviation` went from
`0.2663` to `0.0` and the paper withdrew the "pp-RoPE is not orthogonal" remark.
The convention each model uses is now stated where it matters.

### 4. "Both corrections are already in the literature" — accepted, not yet acted on

Agreed. The two "corrections" are restatements of YaRN §3.2, Appendix A.1, and
Eq. (20). They are now framed as notes on the literature rather than
contributions, and the novelty claim rests on the per-pair
`(A_k, B_k, R_k, ψ_k)` decomposition of the *feature-level* attribution — an
object that is not in RoFormer or YaRN.

*Still outstanding:* the reviewer's further point stands — that object needs an
argument that it is *useful*, not merely correctly derived. The trained-weights
section is the beginning of that argument (a 93x spread in per-feature share
across heads) but it is not yet the argument.

### 5. "Wrong venue, wrong length, no style file" — partly addressed

Author now filled and consistent across `main.tex`, `CITATION.cff` and `LICENSE`
(Slyatski Ilya), pinned by `tests/test_author_metadata.py`. The official
`neurips_2026.sty` is vendored. `tests/test_venue_compliance.py` measures the
main text against the 9-page limit, locates the boundary at the final main-text
section rather than counting PDF pages, and fails on figure placeholders, a
non-empty bibliography, or a citation key that does not resolve.

*Still outstanding and this is the largest open item:* the main text is **14
pages against a 9-page limit**, and the paper still uses
`\documentclass[twocolumn]{article}` rather than `neurips_2026`. Compression
reduced Related Work, Limitations, the disclaimers and the reproducibility note,
and moved the synthesis subsection to the appendix. Closing the remaining gap is
not a writing task: the three largest sections — `Exactness, measured` (92
lines), `The per-pair closed form` (85) and `What context extension fixes` (163)
— *are* R1, the central decomposition, and R3. Cutting them to fit would remove
headline results rather than compress prose. The reviewer's alternative is the
honest one: reframe as a short position or empirical-methodology note with
results in an appendix, which is what the content actually fits.

### A defect the review did not find

`unaccounted_numbers()` blanked claimed spans and rescanned. A claim pattern
covers the inside of a `$...$` span without its delimiters, so blanking left
orphaned `$` characters that re-paired, manufacturing enormous spans running from
one section into the bibliography. It now scans span-first and tests each number's
own position. Fixing it immediately exposed ~46 numbers in the paper that no claim
had ever covered — the exhaustiveness gate had been silently blind over the whole
document. Those now have claims too.