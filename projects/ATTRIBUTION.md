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
| `projects/rope_attribution/__init__.py` | Package re-export surface for the symbols above. | **Original to this project.** Imports plus a docstring. | Contains no algorithm; every name it exports is defined in `rope.py`. |
| `projects/rope_attribution/experiments.py` | Measured experiments for the RoPE / YaRN / pp-RoPE attribution claims; writes `results/measurements.json`. | **Original to this project.** The only non-stdlib import is NumPy. | Every reported value is computed by the package at run time; nothing is assigned as a literal result. The committed artifact is re-derived by `tests/test_experiments.py`. |
| `projects/rope_attribution/figures.py` | Figure generation for the paper; writes the PNGs and CSV sidecars under `figures/`. | **Original to this project.** NumPy and matplotlib only. | Plots only `results/` artifacts and package functions; `tests/test_figures.py` covers the generators, including a guard against a silently dropped figure. |
| `projects/rope_attribution/statistics.py` | Grid-independent statistics behind the paper's aggregate numbers. | **Original to this project.** NumPy only. | `run_all()` is recomputed and compared value by value against `results/statistics.json` in `tests/test_statistics.py`; the test writes to a temporary path and asserts that the committed artifact is unchanged. |
| `projects/rope_attribution/real_model.py` | The structural measurements of `experiments.py`, repeated on real checkpoints; writes `results/real_model.json`. | **Original to this project.** NumPy, `torch` and `transformers` as install-time dependencies. | No model weights are tracked in this repository; the module loads a checkpoint through `transformers` at run time. `tests/test_real_model.py` covers the pure logic and the committed artifact's schema; the live checkpoint sweep is opt-in because it needs CPU torch and downloads weights. |
| `projects/rope_attribution/usefulness.py` | The pairwise-cost statistic: what it costs to use a position-free per-feature attribution under RoPE. | **Original to this project.** NumPy only. | Computed from the package's own functions; the live sweep is re-run against the committed artifact by `tests/test_usefulness.py` when torch and the checkpoints are available. |
| `projects/rope_attribution/uncertainty.py` | Uncertainty for the aggregate statistics the paper quotes. | **Original to this project.** NumPy only. | Computed from the package's own functions; `tests/test_uncertainty.py`. |
| `projects/rope_attribution/sensitivity.py` | Sensitivity of the position-free cost statistic to configuration; writes `results/sensitivity.json`. | **Original to this project.** NumPy only. | `tests/test_sensitivity.py` re-runs the sweep and diffs it against the committed artifact. |
| `projects/rope_attribution/agreement.py` | What a position-free scalar gets wrong that a reader would act on. | **Original to this project.** NumPy only. | Computed from the package's own functions; `tests/test_agreement.py`. |
| `projects/rope_attribution/train.py` | Training harness for the model family the paper's claims are about; logs every metric per evaluation and refuses CPU runs above the smoke scale. | **Original to this project.** `torch`, `transformers` and `datasets` as install-time dependencies. | Parameter counts are re-derived from the configs in `tests/test_train.py`, which also pins the CPU guard. |
| `projects/rope_attribution/report_run.py` | Turns a training run's JSONL log into a self-contained report. | **Original to this project.** Standard library only. | Log parsing and HTML rendering are exercised by `tests/test_report_run.py`. |
| `projects/rope_attribution/tpu_configs.py`, `tpu_launch.py` | TPU v5e-8 presets and single-host launcher. | **Original to this project.** The launcher imports `torch_xla` only at run time. | Arithmetic and dry-run behaviour are pinned by `tests/test_tpu_configs.py`; no real XLA run is claimed. |
| `projects/rope_attribution/downstream.py` | Optional NLL kernel-replacement probe; not used by the paper. | **Original to this project.** `torch`, `transformers` as install-time dependencies. | `tests/test_downstream.py` covers its pure logic; its scratch outputs are not committed and are not paper evidence. |
| `tests/`, `figures/`, `results/`, `paper/`, `scripts/`, `docs/`, `Makefile`, `pyproject.toml`, `.github/`, `THEORY_AND_PROOFS.md` | Test suite, figure outputs and CSV sidecars, measured artifacts, paper sources and build, documentation, tooling and CI. | **Original to this project**, except `paper/neurips_2026.sty` (see `../NOTICE.md`). | Written for this project; the artifacts are produced by the package's own generators and re-derived by the test suite. |
| `README.md`, `LICENSE` | Repository documentation and license. | **Original to this project.** | `LICENSE` is the Apache-2.0 text with the appendix boilerplate filled as `Copyright 2026 Ilya Slyatski`; `tests/test_author_metadata.py` refuses the unfilled `[yyyy] [name of copyright owner]` form. |
| `NOTICE.md`, `CITATION.cff`, `projects/ATTRIBUTION.md` | Provenance and citation metadata. | **Original to this project.** | This file and its two siblings. |

The early exploratory archive that used to live under `projects/frontier-01-*`
was removed from the working tree in October 2026. It is preserved only in git
history at tag `pre-legacy-cleanup-2026-10-20`; it printed illustrative constants
rather than measurements and is not part of the maintained provenance record.

---

## Reproducing the "no third-party code" claim

Run from the repository root. Requires no dependencies. The tracked top-level
entries are `.github/`, `docs/`, `figures/`, `paper/`, `projects/`, `results/`,
`scripts/`, `tests/`, `.gitignore`, `CITATION.cff`, `LICENSE`, `Makefile`,
`NOTICE.md`, `README.md`, `THEORY_AND_PROOFS.md`, `pyproject.toml`, and the
three requirement files; there is no `src/` and no vendored tree beyond the
style sheet named in `../NOTICE.md`.

The commands below deliberately exclude `NOTICE.md`, `CITATION.cff` and
`projects/ATTRIBUTION.md`. Those three files have to name the audit patterns in
order to report on them, so including them would make the counts self-referential
and the numbers below non-reproducible.

### PowerShell

```powershell
$pats = @("inv_freq","rotary","rotary_emb","apply_rotary","attention_scaling",
          "mscale","ntk","copied from","ported from","taken from")
$docs = @(".\NOTICE.md", ".\CITATION.cff", ".\projects\ATTRIBUTION.md") |
        ForEach-Object { (Resolve-Path $_).Path }
$files = Get-ChildItem -Recurse -File -Path . -Include *.py,*.md,*.json,*.txt |
         Where-Object { $_.FullName -notmatch '\\\.git\\|__pycache__|\\\.tmp\\|\\build\\' -and
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
       --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.tmp \
       --exclude-dir=build \
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
