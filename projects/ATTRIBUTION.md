# Attribution

Per-module provenance for this repository. Companion to `../NOTICE.md`
(repository-level statement) and `../CITATION.cff` (machine-readable citation).

**Summary of the audit result:** the repository contains no third-party source
code, with the one exception named in `../NOTICE.md` (the conference style sheet
under `paper/`). Every Python file in the tree is original to this project. The
RoPE and YaRN methods are cited as academic literature; their formulas were read
from the papers and the canonical public reference implementations and
re-expressed in this project's own code.

---

## Provenance table

| Component | Role | Provenance | Basis for the claim |
| --- | --- | --- | --- |
| `projects/rope_attribution/rope.py` | Core library: plain RoPE, YaRN, and partial-RoPE (pp-RoPE) cos/sin construction and application, in NumPy `float64`. | **Independent implementation, original to this project.** Formulas and algorithm steps consulted from four sources; no upstream code copied. | Module docstring (`rope.py:1-31`) names each source inline per-algorithm: RoPE from Su et al. arXiv:2104.09864; YaRN from Peng et al. arXiv:2309.00071; the **YaRN ramp and `mscale`** (`find_correction_dim`, `find_correction_range`, `linear_ramp_mask`, `get_mscale`) explicitly marked as transcribed from the authors' own reference implementation `jquesnelle/yarn`, `scaled_rope/LlamaYaRNScaledRotaryEmbedding.py`; the **split-half rotation convention** (`rotate_half`, `inv_freq`, `rope_cos_sin`) from `transformers` `models/llama/modeling_llama.py` / `modeling_rope_utils.py`. Docstrings say "transcribed verbatim from upstream" for the ramp/mscale — that word refers to the *formulas*, and the accompanying text says so: the module header states that formulas/algorithms were consulted and re-expressed. The YaRN ramp/mscale is a published algorithm; `transformers` and `jquesnelle/yarn` are named as **reading references consulted while transcribing formulas**, not as licensors and not as sources of copied text. |
| `projects/rope_attribution/__init__.py` | Package re-export surface for the symbols above. | **Original to this project.** 3 lines of imports plus a docstring. | File is 19 lines and contains no algorithm; every name it exports is defined in `rope.py`. |
| `projects/rope_attribution/experiments.py` | Measured experiments for the RoPE / YaRN / pp-RoPE attribution claims; writes `results/measurements.json`. | **Original to this project.** The only non-stdlib import is NumPy. | Every reported value is computed by the package at run time; nothing is assigned as a literal result. The committed artifact is re-derived by `tests/test_experiments.py`. |
| `projects/rope_attribution/figures.py` | Figure generation for the paper; writes the PNGs and CSV sidecars under `figures/`. | **Original to this project.** NumPy and matplotlib only. | Plots only `results/` artifacts and package functions; `tests/test_figures.py` covers the generators, including a guard against a silently dropped figure. |
| `projects/rope_attribution/statistics.py` | Grid-independent statistics behind the paper's aggregate numbers. | **Original to this project.** NumPy only. | Recomputed from `results/measurements.json`; `tests/test_statistics.py`. |
| `projects/rope_attribution/real_model.py` | The structural measurements of `experiments.py`, repeated on a real model; writes `results/real_model.json`. | **Original to this project.** NumPy, `torch` and `transformers` as install-time dependencies. | No model weights are tracked in this repository; the module loads a checkpoint through `transformers` at run time, and the committed artifact is re-derived by `tests/test_real_model.py` and `tests/test_paper_trained_claims.py`. |
| `projects/rope_attribution/usefulness.py` | The pairwise-cost statistic: what it costs to use a position-free per-feature attribution under RoPE. | **Original to this project.** NumPy only. | Computed from the package's own functions; `tests/test_usefulness.py`. |
| `projects/rope_attribution/uncertainty.py` | Uncertainty for the aggregate statistics the paper quotes. | **Original to this project.** NumPy only. | Computed from the package's own functions; `tests/test_uncertainty.py`. |
| `projects/rope_attribution/sensitivity.py` | Sensitivity of the position-free cost statistic to configuration; writes `results/sensitivity.json`. | **Original to this project.** NumPy only. | `tests/test_sensitivity.py` re-runs the sweep and diffs it against the committed artifact. |
| `projects/rope_attribution/agreement.py` | What a position-free scalar gets wrong that a reader would act on. | **Original to this project.** NumPy only. | Computed from the package's own functions; `tests/test_agreement.py`. |
| `projects/rope_attribution/train.py` | Training harness for the model family the paper's claims are about; logs every metric per evaluation and refuses CPU runs above the smoke scale. | **Original to this project.** `torch`, `transformers` and `datasets` as install-time dependencies. | Parameter counts are re-derived from the configs in `tests/test_train.py`, which also pins the CPU guard. |
| `tests/`, `figures/`, `results/`, `paper/`, `scripts/`, `Makefile`, `pyproject.toml`, `.github/` | Test suite, figure outputs and CSV sidecars, measured artifacts, paper sources and build, tooling and CI. | **Original to this project**, except `paper/neurips_2026.sty` (see `../NOTICE.md`). | Written for this project; the artifacts are produced by the package's own generators and re-derived by the test suite. |
| `projects/frontier-01-graphs-ULTIMATE-V11.py`, `-graphs-TOPLAB-IDEAL-V10.py`, `-graphs-BEAUTIFUL-FINAL.py`, `-all-graphs.py`, `-all-graphs-ideal.py`, `-make-figures.py` | Legacy figure generation. | **Original to this project.** | Written for this project; import only `math`, `numpy`, `matplotlib`, `pathlib`. Curves are either computed from elementary closed forms or drawn from synthetic random draws (`-make-figures.py:32-36`) — see the hardcoded-constant caveat below before citing any figure value. |
| Other legacy `projects/frontier-01-*.py` (e.g. `-bilinearity-break-demo.py`, `-bilinearity-break-ideal.py`, `-bilinearity-break-torch-ideal.py`, `-bilinearity-break-ULTIMATE-V11.py`, `-bilinearity-break-KAGGLE-COPY*.py`, `-bag-of-words-test.py`, `-eval-numpy-ideal.py`, `-high-level-demo.py`, `-usual-attention-code*.py`, `-kaggle-*.py`, `-scaffold.py`, `-pre-post-checklist.py`, `-real-hook-demo.py`, `-cli-ideal.py`, `-eval-high-level*.py`, `-hla-extra-tests-IDEAL.py`, `-gemma4-pp-rope.py`, `-kaggle-2xT4.py`, `-bilinearity-standalone-DEMO-FOR-USER.py`) | Early exploratory and demo scripts for the attribution project. | **Original to this project**, but **early-stage**: they are working demonstrations, not a maintained library. **Several print hardcoded illustrative constants instead of computing measurements** — see the caveat section below. | No third-party code, no vendored tree, no upstream imports; provenance audit found 0 hits for `copied from` / `ported from` / `taken from` and 0 hits for `rotary_emb` / `apply_rotary` / `attention_scaling` / `ntk` anywhere in the repository. Behavioural caveat established by reading the files, e.g. `projects/frontier-01-eval-numpy-ideal.py:16-17` assigns `err_linear = 3.55e-15` and `err_score_direct = 1.2e-3` as literals and then formats them. |
| `projects/frontier-01-*.md` (plans, proofs, paper drafts, reviewer guidelines) | Project planning, proofs, paper drafts, post-mortems. | **Original to this project**; they *cite* external work, and those citations are recorded in `../NOTICE.md`. | They contain no executable code. The RoPE-related strings in them are prose and formula sketches (e.g. `frontier-01-best-plan.md:42`, `frontier-01-PAPER-DRAFT-V13-9PAGES-ORAL.md:85`); none is copied code. Measured pattern counts for the whole tree are kept in `../NOTICE.md`. |
| `projects/requirements.txt`, `projects/Dockerfile` | Environment pinning and container image. | **Original to this project.** | Both name third-party *packages* (`torch`, `transformers`, `matplotlib`, …) as install-time dependencies, which is ordinary dependency declaration, not redistribution of code. Nothing from those packages is vendored into the tree. |
| `README.md`, `LICENSE` | Repository documentation and license. | **Original to this project.** | `LICENSE` is the unmodified Apache-2.0 text with the appendix `Copyright [yyyy] [name of copyright owner]` line left unfilled — i.e. no copyright holder is named anywhere in the repository. |
| `NOTICE.md`, `CITATION.cff`, `projects/ATTRIBUTION.md` | Provenance and citation metadata. | **Original to this project.** | This file and its two siblings. |

---

## Known caveat: hardcoded illustrative constants in the legacy scripts

**A number of the legacy `projects/frontier-01-*.py` scripts print hardcoded
illustrative constants rather than computing measurements.** A reader will
otherwise assume that every number printed by these scripts is a computed
result. It is not, in the following cases.

The clearest example, `projects/frontier-01-eval-numpy-ideal.py`, assigns the
result values as literals and then formats them:

```python
# projects/frontier-01-eval-numpy-ideal.py:16-19
err_linear = 3.55e-15
err_score_direct = 1.2e-3
print(f"1. Conservation linear err={err_linear:.2e} <1e-10 PASS, ...")
```

The same holds for `R2_high = 0.62`, `R2_low = 0.08`,
`phi_err_low = 111.7` (`-eval-numpy-ideal.py:55-58`), for the gate/phase
numbers, and for the Bag-of-Words metrics further down the same file. The
figure scripts annotate axes with the same class of claimed values
(`frontier-01-make-figures.py:21-23`, `-graphs-ULTIMATE-V11.py:39`), while the
curves they plot are either elementary closed forms or synthetic random draws.

Practical consequences:

- Numbers printed by these scripts are **illustrative targets / expected
  values**, not measurements, and must not be quoted as results.
- The PASS/FAIL strings they print describe the hardcoded literals, not any
  executed check.
- The same constants are restated in the `projects/frontier-01-*.md` documents,
  so they propagate into the drafts.
- The maintained package `projects/rope_attribution/` and its tests do not have
  this problem: every value they report is computed, and the core identities
  are checked against closed forms in `tests/test_rope.py`. Prefer them for any
  number that will be cited; the paper's numbers are the `results/` artifacts
  produced by this package.

The legacy scripts and documents are retained only as the audit record
described in `../LEGACY.md`.

---

## Reproducing the "no third-party code" claim

Run from the repository root. Requires no dependencies. The repository's
top-level entries are `.github/`, `figures/`, `paper/`, `projects/`, `results/`,
`scripts/`, `tests/`, `.gitignore`, `CITATION.cff`, `LEGACY.md`, `LICENSE`,
`Makefile`, `NOTICE.md`, `README.md` and `pyproject.toml`; there is no `src/`
and no vendored tree beyond the style sheet named in `../NOTICE.md`.

The commands below deliberately exclude `NOTICE.md`, `CITATION.cff` and
`projects/ATTRIBUTION.md`. Those three files have to name the audit patterns in
order to report on them, so including them would make the counts
self-referential and the numbers below non-reproducible.

### PowerShell

```powershell
$pats = @("inv_freq","rotary","rotary_emb","apply_rotary","attention_scaling",
          "mscale","ntk","copied from","ported from","taken from")
$docs = @(".\NOTICE.md", ".\CITATION.cff", ".\projects\ATTRIBUTION.md") |
        ForEach-Object { (Resolve-Path $_).Path }
$files = Get-ChildItem -Recurse -File -Path . -Include *.py,*.md,*.json,*.txt |
         Where-Object { $_.FullName -notmatch '\\\.git\\|__pycache__' -and
                         $docs -notcontains $_.FullName }
foreach ($p in $pats) {
  $hits = Select-String -Path $files.FullName -Pattern $p -SimpleMatch -CaseSensitive:$false
  "{0} -> {1} hit(s)" -f $p, @($hits).Count
  $hits | ForEach-Object { "    {0}:{1}: {2}" -f $_.Path, $_.LineNumber, $_.Line.Trim() }
}
```

### Equivalent POSIX form

```sh
for p in inv_freq rotary rotary_emb apply_rotary attention_scaling mscale ntk \
         "copied from" "ported from" "taken from"; do
  echo "== $p"
  grep -rni --include='*.py' --include='*.md' --include='*.json' --include='*.txt' \
       --exclude-dir=.git --exclude-dir=__pycache__ \
       --exclude=NOTICE.md --exclude=CITATION.cff --exclude=ATTRIBUTION.md \
       "$p" . | head -50
done
```

The counts this method returns over the current tree are recorded in
`../NOTICE.md`, which is where they are maintained — this file keeps only the
method, so the two documents cannot drift apart. Two properties of the output
are worth stating so it is not misread: `inv_freq` and `mscale` legitimately
appear in this project's own code and tests, and the zero rows (`rotary_emb`,
`apply_rotary`, `attention_scaling`, `ntk`, `copied from`, `ported from`,
`taken from`) are the part that matters. A non-zero row for any of them would
be a provenance failure, not a counting difference.
