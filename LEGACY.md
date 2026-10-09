# `LEGACY.md` — inventory of `projects/frontier-01-*`

This document exists for one purpose: to stop a reader from citing a number that
was typed in by hand.

Everything under `projects/frontier-01-*` is a frozen body of work from before
the maintained core existed. The maintained core is
`projects/rope_attribution/` plus `tests/`, `figures/`, `results/`. It computes
every number it reports. The legacy tree mostly does not.

Read this file before quoting anything from `projects/frontier-01-*`. When the
legacy documents and `results/measurements.json` disagree, the measurements win.

### A factual claim in the legacy tree that is now known to be false

Many legacy documents assert, as a headline result, a "search proof" that nobody
has previously solved exact feature attribution for RoPE. That claim is false
and should not be repeated.

Chachamovits, *Phase Structure in Rotary Attention* (arXiv:2607.25507, §3--§4)
writes a rotary head as a sum of per-pair terms
$\rho\cos(\alpha - \beta + (m-n)\theta)$ with $\rho = |q_j||k_j|$ --- the same
closed form these documents present as new --- and remarks that the term goes
negative near $\pi$, which is the sign behaviour they claim to have discovered.
The same identity also appears in Liang et al., *RoPE-Aware Bit Allocation for
KV-Cache Quantization* (arXiv:2606.24033, §1).

Both were verified against the arXiv API before being written here. What the
maintained paper now claims is narrower and defensible: not the algebra, which is
standard, but the attribution semantics and the consequence that no
position-free per-feature scalar exists. The legacy tree's broader novelty claim
should be treated as unsupported.

---

## 0. Counts

Enumerated with `git ls-files`:

| set | command | count |
|---|---|---:|
| legacy markdown | `git ls-files 'projects/*.md'` | **54** |
| all Python under `projects/` | `git ls-files 'projects/*.py'` | 35 |
| of which the maintained package | `projects/rope_attribution/{__init__,experiments,figures,rope}.py` | 4 |
| **legacy Python** | the other 31 | **31** |
| **legacy total** | 54 + 31 | **85 files** |

There is also non-`.md`/`.py` legacy material in the same directory
(`settings.json`, `settings-ideal.json`, `Dockerfile`, `requirements.txt`,
`frontier-01-figures.html`, `frontier-01-figures-ideal.html`). It is covered in
the tables where it matters — see the `settings.json` entry, which is the single
most citable artifact of the whole legacy tree.

Of the 54 markdown files, **49 contain at least one fabricated result number**.
The five that do not are: `PUSH-INSTRUCTIONS-FASTEST.md`,
`frontier-01-REVIEWER-GUIDELINES-TOP3-FULL-V10.md`, `frontier-01-best-plan.md`,
`frontier-01-nonlinear-qk-attribution.md`, `frontier-01-reviewer-guidelines-full.md`
(the last two and the `full` guidelines file are clean; the 2025-2026 variant is
not — see the table).

---

## 1. Categories

| category | meaning |
|---|---|
| `illustrative` | prints or plots hardcoded values; nothing is measured |
| `partly-computed` | runs real arithmetic on toy data, but the *reported results* are literals, or the arithmetic does not test the claim attached to it |
| `genuinely-computed` | actually measures the thing it reports |
| `planning-document` | design/plan/note prose; no code, but restates the fabricated numbers as results |
| `reference-summary` | verbatim fetched third-party text (conference reviewer guidelines) |
| `runbook` | operational instructions (Kaggle / TPU / git) |
| `provenance` | records where code came from — accurate, kept current |
| `scaffold` | function bodies that `raise NotImplementedError`; nothing runs |

The `canonical` column names the file a near-duplicate should be resolved
against. No two legacy files are byte-identical; every "duplicate" below is a
near-duplicate (measured with `git diff --no-index --numstat`).

### 1.1 Python (31 files)

| file | category | evidence | canonical |
|---|---|---|---|
| `frontier-01-all-graphs-ideal.py` | `illustrative` | `entropy_ratio = [0.94, 0.23, 0.17]` / `retrieval = [0.2, 0.7, 0.75]` / `interaction = [0.8, 0.089, 0.005]` (`:281-283`); `err = [111.7, 5.0]`, `R2 = [0.08, 0.62]` (`:136-137`); `errs = [3.55e-15, 1.2e-3]` (`:240`) | — |
| `frontier-01-all-graphs.py` | `illustrative` | `err = [111.7, 5.0]` / `R2 = [0.08, 0.62]` (`:106-107`); `errs = [3.55e-15, 1.2e-3]` (`:178`) | `all-graphs-ideal.py` |
| `frontier-01-bag-of-words-test.py` | `illustrative` | accuracy assigned inside an f-string: `"… ~1 bag-of-words, acc 0.2 FAIL"` (`:32`), `"… real learning, acc 0.7 PASS"` (`:35`); `"pp-RoPE … ratio 0.17 … acc 0.75 PASS"` (`:37`); `inter = D**2 / 2` (`:60`) is the definition of the metric being reported | — |
| `frontier-01-eval-numpy-ideal.py` | `illustrative` | `err_linear = 3.55e-15` / `err_score_direct = 1.2e-3` (`:16-17`); `R2_high = 0.62` / `R2_low = 0.08` / `phi_err_low = 111.7` (`:55-58`); `retrieval_full = 0.2` / `retrieval_cond = 0.7` (`:68-69`); `"oral_guaranteed_even_if_phase_R2_0.62": True` (`:177`). The only computed quantities in the file are two SHA-256 digests of a JSON string and of the literal `b"FineWeb-Edu-10B"` (`:107-108`) | — |
| `frontier-01-gemma4-pp-rope.py` | `illustrative` | `theta_local = 10000 ** (-2 * 0 / 16)` and `theta_global = 1000000 ** (-2 * 0 / 16)` (`:54-55`) are both exactly `1.0` at `i = 0`, so `D_local == D_global == 10.0`; the script then prints `"global D 100x smaller"` (`:59`). Verified by running it: `theta_local= 1.0 theta_global= 1.0 D_local= 10.0 D_global= 10.0` | — |
| `frontier-01-graphs-BEAUTIFUL-FINAL.py` | `illustrative` | `err = [111.7, 5.0]` / `R2 = [0.08, 0.62]` (`:246-247`); `errs = [3.55e-15, 1.2e-3]` (`:425`); `entropy_ratio` / `retrieval` / `interaction` literals (`:477-479`) | `graphs-ULTIMATE-V11.py` |
| `frontier-01-graphs-TOPLAB-IDEAL-V10.py` | `illustrative` | same literals at `:234-235`, `:379`, `:426-428` | `graphs-ULTIMATE-V11.py` |
| `frontier-01-graphs-ULTIMATE-V11.py` | `illustrative` | `err = [111.7, 5.0]` / `R2 = [0.08, 0.62]` (`:318-319`); `inter_RoPE = D_vals**2 * 0.5` (`:367`); `errs = [3.55e-15, 1.2e-3]` (`:483`); `entropy_ratio = [0.94, 0.23, 0.17]`, `retrieval = [0.2, 0.7, 0.75]`, `interaction = [0.8, 0.089, 0.005]` (`:541-543`). Also claims `"error bars 3 seeds"` in six figure titles/annotations (`:334`, `:352`, `:503`, `:590`, `:628`, `:651`) while setting `np.random.seed(42)` exactly once (`:251`) and never looping over seeds; the bars are hand-entered (`yerr=inter_RoPE * 0.2` at `:374`, `yerr=[20, 2]` at `:328`, `yerr=0.05` at `:556`) | — |
| `frontier-01-kaggle-2xT4.py` | `illustrative` | the eight "falsifications" exist only as comments (`:63-71`), including `# 1 linear 3.55e-15 … 7 R2 high 0.62 vs low 0.08 phi err 5° vs 111°`; the script itself prints `"Kaggle 2xT4 Ideal Ready"` (`:83`) having measured nothing | — |
| `frontier-01-kaggle-notebook-ideal-v2.py` | `illustrative` | cells are Python *string literals* that are never executed; cell 2 hardcodes `print(f"Контент: … 1.08->4.00 ratio 3.7x FAIL не билинейно")` (`:34`) and `cos(90+90)=-1 !=0+0` (`:36`) | `KAGGLE-NOTEBOOK-V16-FINAL.py` |
| `frontier-01-kaggle-notebook-ideal.py` | `illustrative` | same construction; cell 2 hardcodes `"Fixed pos 1.08->2.16 … bilinear PASS"` (`:22`) and `"Content … 1.08->4.0 3.7x NOT linear FAIL"` (`:25`) | `KAGGLE-NOTEBOOK-V16-FINAL.py` |
| `frontier-01-make-figures.py` | `illustrative` | `err = [111.7, 5.0]` / `R2 = [0.08, 0.62]` (`:52-53`); the gate/phase scatter is `np.random.randn(30)` (`:33-36`) labelled `"gate specialists \|q\|"` (`:38`) | — |
| `frontier-01-cli-ideal.py` | `illustrative` | the only number it computes is a file size, and the "PASS" is `status = "PASS" if sz > 150_000 else "FAIL too small"` (`:61`) on a PNG produced from typed-in lists; it then prints `"ALL DONE IDEAL"` (`:78`) | — |
| `frontier-01-bilinearity-break-ideal.py` | `partly-computed` | real arithmetic, wrong question. Case 1 uses `score = x_q * x_k * math.cos(delta_fixed)` (`:19`, `:24`); case 2 uses `score = x_q * x_k * math.cos(x_q - x_k)` (`:36`, `:42`). The scoring function is swapped between the two cases, so the 3.70x is an artifact of the swap, not of RoPE. Its "pp-RoPE validation" is `torch.randn` with `q_rot = q[:n_rot]` (`:174`) — a slice, no rotation applied — followed by an unconditional `print("Torch PASS")` (`:182`) inside a `try` whose only failure mode is `ImportError` (`:164`) | `bilinearity-break-demo.py` (smallest form) |
| `frontier-01-bilinearity-break-demo.py` | `partly-computed` | same swap in its minimal form: `math.cos(delta_fixed)` at `:19`/`:22` vs `math.cos(x_q - x_k)` at `:32`/`:37` | — (canonical of this family) |
| `frontier-01-bilinearity-break-torch-ideal.py` | `partly-computed` | same swap (`:34`,`:40` vs `:69`,`:74`); prints `"fp64 err 3.55e-15 <1e-10 PASS vs score direct err 1.2e-3 FAIL"` (`:251`) without computing either | `bilinearity-break-demo.py` |
| `frontier-01-bilinearity-break-ULTIMATE-V11.py` | `partly-computed` | same swap (`:43`,`:48` vs `:69`,`:75`); prints `"Conservation q \|q-sum f_i q_i\|=0.00e+00 <1e-10 PASS"` (`:121`) as a literal | `bilinearity-break-demo.py` |
| `frontier-01-bilinearity-break-KAGGLE-COPY.py` | `partly-computed` | same swap (`:18-19` vs `:31-32`) | `bilinearity-break-KAGGLE-COPY-V10-TOPLAB.py` |
| `frontier-01-bilinearity-break-KAGGLE-COPY-V10-TOPLAB.py` | `partly-computed` | same swap (`:21` vs `:41`) | — (canonical of this family) |
| `frontier-01-bilinearity-standalone-DEMO-FOR-USER.py` | `partly-computed` | same swap (`:45-46` vs `:65-68`); prints `"FAIL: 3.70× ≠ 2.00× — не линейно, билинейность сломана"` (`:75`) | `bilinearity-break-demo.py` |
| `frontier-01-eval-high-level-IDEAL.py` | `partly-computed` | `err_linear` is really computed (`:36`); `err_score_direct = 1.2e-3` (`:37`), `score_real_remove = 2.7` / `score_rand_remove = 5.15` (`:48-49`), `align_l6 = 2.1` / `align_l0 = 0.1` (`:73-74`), `overlap = 5` (`:80`), `R2_high = 0.62` / `R2_low = 0.08` / `phi_err_low = 111.7` (`:86-89`), `retrieval_full = 0.2` / `retrieval_cond = 0.7` (`:99-100`) are all literals | `eval-high-level.py` (v1) |
| `frontier-01-eval-high-level.py` | `partly-computed` | same file, same literals (`:37`, `:48-49`, `:72-77`, `:82-85`, `:95-96`) | — (canonical of this family) |
| `frontier-01-high-level-demo-IDEAL.py` | `partly-computed` | conservation really computed (`:42`), high-L0 phi really computed (`:108-123`); but `inter = D**2 / 2` (`:135`) is the definition, and accuracy 0.2 / 0.7 / 0.75 is typed into print strings (`:151-157`) | `high-level-demo.py` |
| `frontier-01-high-level-demo.py` | `partly-computed` | same file; `"Sim margin: kill real feature 5.2->2.7, kill random same norm 5.2->5.15"` is a print literal (`:105-107`) | — (canonical of this family) |
| `frontier-01-usual-attention-code-IDEAL.py` | `partly-computed` | `test_conservation` really computes and asserts (`:72-98`); but `err_score = 1.2e-3` (`:94`), and the whole pipeline is a comment — `big_pipeline(...)` at `:225-228` is a docstring of the eight claims followed by `pass`; `collect_for_seed` (`:169-186`) is a docstring with no body | `usual-attention-code.py` |
| `frontier-01-usual-attention-code.py` | `partly-computed` | same; `phase_gate_interaction_per_token` is `raise NotImplementedError` (`:80`), `big_pipeline` is `pass` (`:130-133`) | — (canonical of this family) |
| `frontier-01-KAGGLE-NOTEBOOK-V16-FINAL.py` | `partly-computed` | the demo at the top does run (`:24-63`); but it prints `"Seed 42 config_hash 9bd59cac dataset_hash 848bb0b0 conservation 3.55e-15 PASS"` (`:18`), `"R2 0.08 … R2 0.62 PASS"` (`:58`) and `"BoW entropy 0.94 vs 0.23 vs 0.17 retrieval 0.2 vs 0.7 vs 0.75"` (`:72`) as literals, and the rest of the file is cell-as-string | — (canonical of this family) |
| `frontier-01-hla-extra-tests-IDEAL.py` | `illustrative` | `base_last = 3.2 + 0.15 * math.log(T / 128 + 1)` / `hla_last = 3.18 + 0.05 * math.log(...)` (`:188-189`) — losses generated by a closed-form formula and printed as "simulating base vs OLD_MODEL loss" (`:183`); it also imports `src.model` from `./reviews/laplace-attention` (`:29-34`), a tree that does not exist in this repository | — |
| `frontier-01-pre-post-checklist.py` | `scaffold` | every function body is `raise NotImplementedError` (`:25`, `:46`, `:79`, `:99`, `:128`); `run_full_pipeline_before_post` returns an empty dict (`:151-192`) | `scaffold.py` |
| `frontier-01-scaffold.py` | `scaffold` | `rotate_pairwise`, `compute_base_score`, `compute_hla_score`, `decompose_diff` all `raise NotImplementedError` (`:16`, `:24`, `:35`, `:48`); the `test_*` functions below them call those, so running it raises | — (canonical of this family) |
| `frontier-01-real-hook-demo.py` | `genuinely-computed` | loads a real model — `HookedTransformer.from_pretrained("gpt2-small", device="cpu")` (`:10`) — reads `cache["blocks.0.ln1.hook_normalized"]` (`:23`) and computes real attention scores `scores = (q @ k_all.T) / (d_head**0.5)` (`:48`). It reports no fabricated metric. Caveat: the payload is a 9-token prompt, so it demonstrates plumbing, not a result | — |

Python summary: **14 `illustrative`, 14 `partly-computed`, 1 `genuinely-computed`, 2 `scaffold`.**
Files whose output is not a measurement: **30 of 31.**

### 1.2 Non-Python legacy artifacts worth naming

| file | category | evidence |
|---|---|---|
| `settings.json`, `settings-ideal.json` | `illustrative` | both carry `"R2_high": 0.62` (`:37`) and `"oral_guaranteed_even_if_phase_R2_0.62": true` (`:71`); both also carry `"base_global": 1000000` (`:8`) while the same tree's prose says "base 500k". Written by `eval-numpy-ideal.py:193-197` and `eval-high-level-IDEAL.py:244-247`. These are the most citable objects in the legacy tree, because they are machine-readable and look like a results record |
| `Dockerfile`, `requirements.txt` | `runbook` | environment pinning; no results |

### 1.3 Markdown (54 files)

#### Claim documents — restate the fabricated numbers as measured results

| file | canonical / note |
|---|---|
| `frontier-01-PAPER-DRAFT-BEST-POSSIBLE.md` | **the most dangerous file in the repo.** Line 13 is a complete abstract carrying every fabricated number. Also `:29` "NOT bilinear 3.7x FAIL", `:33` |
| `frontier-01-PAPER-DRAFT-V13-9PAGES-ORAL.md` | `:14` contribution list with 3.7x / 0.089 / 3.55e-15 / 111.7° / R² 0.62 / retrieval 0.2→0.7; `:24` the 3.70x counterexample |
| `frontier-01-paper-draft-IDEAL.md` | `:17` "Content x_q*x_k*cos(x_q-x_k) 1.08->4 3.7x not linear FAIL"; `:63` |
| `frontier-01-paper-draft.md` | 29-line stub of the above |
| `frontier-01-lesswrong-post-IDEAL.md` | `:5` full abstract; `:20`, `:26`, `:136` |
| `frontier-01-lesswrong-post.md` | 12-line stub of the above |
| `frontier-01-proofs-ideal.md` | **the document the whole tree cites as its proof.** See §3 for the specific false claims at `:45` and `:76` |
| `frontier-01-final-verification-REAL.md` | titled "REAL"; `:77`, `:99`, `:112`, `:122`, `:154` assert the literals were verified |
| `frontier-01-reviewer-guidelines-FULL-2025-2026.md` | `:1-165` is genuine fetched conference text. `:168-200` is not: it maps the project onto those criteria *using the fabricated numbers as evidence* — e.g. `:171` "Our proofs: conservation … err 3.55e-15 <1e-10 PASS" |
| `frontier-01-ANTHROPIC-STRUCTURE-TOPLAB-V11.md` | `:12` "YaRN — base 10k->500k"; `:22`, `:34`, `:98`, `:100`, `:106`, `:120` |
| `frontier-01-audit-final-IDEAL.md` | `:35`, `:51`, `:166`, `:174` |
| `frontier-01-bag-of-words-method.md` | **the method design is worth keeping (§4).** The table at `:105-107` is fabricated |
| `frontier-01-falsifications.md` | the eight-falsification *design* is worth keeping (§4). Every `Честно:` line (`:6`, `:12`, `:18`, `:24`, `:31`, `:37`, `:44`, `:50`) asserts a typed-in value |
| `frontier-01-method-full-IDEAL.md` | `:7`, `:9`, `:73`, `:77`, `:163`, `:165` |
| `frontier-01-method-full.md` | 78-line earlier draft of the above |
| `frontier-01-oral-format-IDEAL.md`, `frontier-01-oral-format.md`, `frontier-01-ORAL-FORMAT-V10-TOPLAB.md` | talk scripts; `:20`, `:11`, `:35`, `:44` carry 3.7x and 500k |
| `frontier-01-efficiency-max-IDEAL.md` | `:26` |
| `frontier-01-final-report.md` | `:10` "base 10k->500k … interaction 0.089 small vs 0.8 large" |
| `frontier-01-gemma4-4b-best.md` | `:26`, `:30` |
| `frontier-01-usual-attention-best.md` | 28-line project blurb |
| `frontier-01-RISK-MITIGATION-ORAL-PLAN-B-V25.md` | 39 lines; the "risk mitigation" is `oral_guaranteed_even_if_phase_R2_0.62` |
| `frontier-01-audit/…`, `frontier-01-reviewer-audit.md` | self-assessment against the guidelines, scored using the fabricated numbers |
| `README-HIGHEST.md`, `README-HIGHEST-IDEAL.md`, `README-V13-TOPLAB-FINAL.md`, `README-V29-ULTIMATE-FINAL.md` | project READMEs; `README-V29-ULTIMATE-FINAL.md:68` states the bilinearity claim; `README-HIGHEST-IDEAL.md:51` restates every constant in one line |

#### Near-duplicate "FINAL" bundles (restate all of the above)

| file | note |
|---|---|
| `frontier-01-FINAL-V13-ABSOLUTE-IDEAL-ULTIMATE-BEST-POSSIBLE.md` | 230 lines |
| `frontier-01-FINAL-V25-…-BEST-POSSIBLE.md` | 327 lines |
| `frontier-01-FINAL-V26-…-BEST-POSSIBLE.md` | differs from V25 by 109 added / 109 deleted lines, almost all of it the version number and the audit timestamp |
| `frontier-01-FINAL-V27-…-BEST-POSSIBLE.md` | same, 109/109 against V26. Canonical: **V27** |
| `frontier-01-FINAL-ANSWER-IDEAL-v4.md` | canonical for itself |
| `frontier-01-FINAL-HIGHEST-IDEAL-v3.md` | 288 lines |
| `frontier-01-FINAL-IDEAL-PACKAGE.md` | 155 lines |
| `frontier-01-FINAL-HIGHEST-PACKAGE.md` | 42 lines |
| `frontier-01-README-USAGE-IDEAL.md`, `frontier-01-USAGE-ONE-CLICK-V23.md`, `frontier-01-final-organization-USAGE-IDEAL.md` | usage docs; `:7`, `:46`, `:60` of the first carry 3.7x |

Note: every one of these V25/V26/V27 files opens with a single ~2 KB audit-banner line (`:3`) that is one long unbroken sentence enumerating "0 bugs", "0 absolute paths", "0 OLD_MODEL refs", "PNG all >150K True PASS". That banner is itself a fabricated measurement: the PNGs it certifies were removed on this branch, and the file sizes it quotes (`273K 289K 162K …`) were never produced by any code in the repository.

#### `reference-summary` — genuinely useful, keep

| file | note |
|---|---|
| `frontier-01-reviewer-guidelines-full.md` | NeurIPS/ICML/ICLR 2025 reviewer instructions, verbatim, with source URLs. No fabricated results |
| `frontier-01-REVIEWER-GUIDELINES-TOP3-FULL-V10.md` | ICLR 2026 reviewer guide, verbatim. No fabricated results |
| `frontier-01-nonlinear-qk-attribution.md` | the earliest framing of the project; honest about its own status (`:24` "the second claim is forbidden until the first is calibrated", `:69` "this is a promising gap, not proven novelty"). No fabricated results |
| `frontier-01-best-plan.md` | an early experimental design (Pythia 410M, cross-layer contrast, discovery/confirmatory split at `:21`). No fabricated results |

#### `runbook` — useful, keep with the numbers stripped

| file | note |
|---|---|
| `frontier-01-TPU-runbook-IDEAL.md` | per-query chunking pattern (`:11-23`) and the half-save-without-logits discipline (`:41-45`, `:105`) are correct and worth reusing. But `:27-37` and `:69-77` present all eight falsifications as "already PASS" |
| `frontier-01-TPU-runbook.md` | earlier Qwen3 draft of the above |
| `frontier-01-kaggle-howto-IDEAL.md` | `:33`, `:155` carry 3.7x |
| `frontier-01-kaggle-howto.md` | 87-line earlier draft |
| `frontier-01-KAGGLE-IDEAL-HOWTO-V10.md` | `:22`, `:108` |
| `COMMIT-SAFE-INSTRUCTION.md`, `COMMIT-SAFE-CODESPACE-PATCH.md` | git/push procedure. Usable, but their "expected output" blocks assert the fabricated results — e.g. `COMMIT-SAFE-INSTRUCTION.md:47` "Ожидаешь: conservation 3.55e-15 PASS, 3.7× vs 2× FAIL" and `:50` "Ожидаешь: RoPE ratio 1.00 FAIL, YaRN ratio 0.23 PASS" |
| `PUSH-INSTRUCTIONS-FASTEST.md` | git/push only; clean |

#### `provenance` — accurate, keep current

| file | note |
|---|---|
| `projects/ATTRIBUTION.md` | the provenance table and the "Known caveat: hardcoded illustrative constants" section (`:35-70`) are correct and are the pre-existing statement of this file's subject. **But its status note is stale**: `:20` and `:157-163` say `experiments.py` "did not exist in the working tree at the time this file was written" and that no `verify_rope*.py` figure script exists. Both landed on this branch. Update when acting on §5 |

Markdown summary: **38 claim documents, 9 near-duplicate FINAL/README bundles, 4
reference summaries, 8 runbooks, 1 provenance, and 4 usage/stub files whose content
is inherited from the claim documents.**

### 1.4 Totals

| category | Python | Markdown | total |
|---|---:|---:|---:|
| `illustrative` | 14 | — | 14 |
| `partly-computed` | 14 | — | 14 |
| `genuinely-computed` | 1 | — | 1 |
| `scaffold` | 2 | — | 2 |
| `planning-document` / claim documents | — | 38 | 38 |
| near-duplicate bundles | — | 9 | 9 |
| `reference-summary` | — | 4 | 4 |
| `runbook` | — | 8 | 8 |
| `provenance` | — | 1 | 1 |
| usage/stub | — | 4 | 4 |
| **total** | **31** | **54** | **85** |

**Files whose output is not a measurement: 30 Python + 53 markdown (all but
`PUSH-INSTRUCTIONS-FASTEST.md` and the three clean reference summaries) = 83 of
85.** Exactly one legacy file, `frontier-01-real-hook-demo.py`, computes and
reports something it actually measured.

---

## 2. What the maintained core measured, for contrast

From `results/measurements.json` (produced by
`python -m projects.rope_attribution.experiments`):

| quantity | measured | where |
|---|---:|---|
| bilinearity in content, 35 score checks | `7.4e-16` | `experiments.py:208` |
| relative-position property, 42 pairs | `3.4e-14` | `experiments.py:204` |
| feature additivity | `3.6e-14` | `experiments.py:272` |
| per-pair closed form vs brute force | `3.6e-15` | `experiments.py:209` |
| per-feature contribution spread across distance | `43.0x` on the 5-point grid — **withdrawn**: a `max/min` over a sampled grid, it grows without bound as the grid gets denser (`920x` on the 54-point grid, drifting 30.9x across six densities). The paper now reports sign reversal instead. See `projects/rope_attribution/statistics.py`. | `experiments.py:273` |
| `mscale` at `scale = 32` | `1.3466` | `experiments.py:452` |
| attention entropy `H/lnT`, unscaled → mscaled | `0.9233 → 0.8651` | `experiments.py:456-457` |

The legacy tree's equivalents (`3.55e-15`, `0.62`, `0.089`, `0.2/0.7/0.75`,
`111.7°`) have no measured counterpart anywhere in this repository.

---

## 3. False claims, with citations

The two the maintained work corrects are confirmed below. The rest are found.

### 3.1 "RoPE breaks bilinearity (3.7x instead of 2x)" — **false**

**Where the claim is made:** `projects/frontier-01-proofs-ideal.md:43-45` —
"#### Контрпример 1: удвоение не удваивает / x_q=1,x_k=2, score = x_q x_k
cos(x_q - x_k) =1*2*cos(-1)=1.08 / x_q=2 => 2*2*cos(0)=4, отношение 3.7x !=2x.
Не линейно."

Repeated in: `frontier-01-PAPER-DRAFT-BEST-POSSIBLE.md:13` (abstract),
`frontier-01-PAPER-DRAFT-V13-9PAGES-ORAL.md:24`,
`frontier-01-lesswrong-post-IDEAL.md:26`,
`frontier-01-graphs-ULTIMATE-V11.py:47` (legend text
`"Content cos(x_q-x_k) - not bilinear 3.7x FAIL"`), and the demo scripts.

**Why it is false:** the 2x case and the 3.7x case do not use the same scoring
function. `projects/frontier-01-bilinearity-break-ideal.py:19` computes
`score = x_q * x_k * math.cos(delta_fixed)`; `:36` computes
`score = x_q * x_k * math.cos(x_q - x_k)`. Under a fixed position the phase is a
constant and the score is bilinear in `(x_q, x_k)`. Under
`cos(x_q - x_k)` the *content* has been substituted for the *position*, which is
a different model of the world, not a property of RoPE. The 3.70x is an
artifact of the swap.

**What the measurement says:** `projects/rope_attribution/experiments.py:188-192`
tests bilinearity directly — it computes `score_relative(a * v, u, freqs, d)`
against `a * score_relative(v, u, freqs, d)` over 5 distances and 7 scales
(`scales = np.array([0.0, 1.0, 2.0, -3.0, 7.5, 1e-3, 1e3])`, `:183`) and reports
`bilinearity_max_rel_err` (`:208`). `results/measurements.json` gives
**`7.40285616369109e-16`** over 35 score checks. That is machine precision:
doubling the query doubles the score exactly, at a fixed position pair.

**What survives:** the non-additivity of `cos(a+b)`
(`frontier-01-proofs-ideal.md:47-51`) is a real fact and is restated correctly in
`experiments.py:21-24` — but the locus is wrong. The non-additivity in
*attention* comes from the softmax downstream of the score, not from RoPE.

### 3.2 "YaRN raises the base from 10000 to 500000" — **false**

**Where the claim is made:** `projects/frontier-01-proofs-ideal.md:76` — "YaRN:
theta_i = base^{-2i/d}, base 10k->500k." Repeated in
`frontier-01-bilinearity-break-ideal.py:127`, `-bilinearity-break-torch-ideal.py:215`,
`frontier-01-graphs-ULTIMATE-V11.py:388` (legend `"YaRN base 500k interaction
small PASS real"`), `frontier-01-final-report.md:10`,
`frontier-01-falsifications.md:48`, `frontier-01-TPU-runbook-IDEAL.md:110`.

**Why it is false:** `projects/rope_attribution/rope.py:24-30` states it directly
— "YaRN does *not* change `base` from 10000 to 500000. Changing the base, or
uniformly dividing positions by a factor, is plain Position Interpolation.
YaRN keeps the base and instead ramps per frequency." The implementation at
`rope.py:201-210` keeps `pos_freqs = base ** (arange(0, dim, 2) / dim)` and blends
`inv_freq_interpolation` into `inv_freq_extrapolation` with a per-frequency ramp
`linear_ramp_mask` (`:170-175`). The base is a parameter it never rewrites.

**Why it matters mechanically:** the legacy chain
`base 10k→500k → theta 50x smaller → D small → interaction 0.089 vs 0.8` is a
single-link argument resting on the wrong link. `experiments.py:336-341` computes
all four schemes side by side, and the measured difference is not the one claimed:
`max|D_k|` is *identical* for `rope_base_10k` and `yarn` at every distance
measured (difference `0.0` over 54 distances, `figures/README.md:32`) because
YaRN deliberately leaves the fastest channel alone. What YaRN actually changes is
the *median* angle and the linearizable fraction
(`README.md:91-97`), and even the amplitude-weighted error barely moves — ratio
`0.9968` at `delta = 8192` (`figures/README.md:34`).

**Also note the internal inconsistency:** `projects/settings.json:8` says
`"base_global": 1000000`, not 500000, while the same file's sibling prose says
"base 500k". Two different numbers for the same fictional mechanism inside one
tree.

### 3.3 Additional false claims found in this audit

**3.3.1 — "retrieval accuracy 0.2 / 0.7 / 0.75" is not measured anywhere.**
`projects/frontier-01-bag-of-words-test.py:32`, `:35`, `:37` place the accuracies
inside f-strings that also print a *computed* entropy, so the accuracy reads as
a measurement. It is not; the script never loads a model. The same three numbers
are plotted as bars in `frontier-01-graphs-ULTIMATE-V11.py:542`
(`retrieval = [0.2, 0.7, 0.75]`) and the 8 `fig_*.png` files removed at
`abf4605` were plots of them. `README.md:183-185` already states these are not
measured. The one legacy script that does load a real model
(`frontier-01-real-hook-demo.py`) evaluates no retrieval task at all.

**3.3.2 — "interaction = D²/2" is a definition presented as a result.**
`projects/frontier-01-bag-of-words-test.py:60` computes `inter = D**2 / 2` and
`:61` derives the verdict from it: `status = "PASS real learning" if inter < 0.1
else "FAIL bag-of-words"`. The threshold test is applied to the formula that
defines the quantity, so the PASS/FAIL is tautological. The same construction
appears at `frontier-01-high-level-demo-IDEAL.py:135-136`,
`frontier-01-all-graphs-ideal.py:167-169`, and
`frontier-01-graphs-ULTIMATE-V11.py:367-369`. The maintained core measures the
real thing instead — the deviation of `cos D, sin D` from their Taylor forms,
per rotary pair, against the exact closed form
(`experiments.py:294-312`).

**3.3.3 — "error bars 3 seeds" is false in six places.**
`frontier-01-graphs-ULTIMATE-V11.py:334`, `:352`, `:503`, `:590`, `:628`, `:651`
all print "error bars 3 seeds". The file calls `np.random.seed(42)` exactly once
(`:251`), never loops over seeds, and the bars are hand-entered constants
(`yerr=inter_RoPE * 0.2` at `:374`, `yerr=[20, 2]` at `:328`, `yerr=0.05` at
`:556`). The `3 seeds` claim is restated in the runbooks, e.g.
`frontier-01-TPU-runbook-IDEAL.md:106`.

**3.3.4 — the conservation error is quoted with two different fabricated values.**
`frontier-01-proofs-ideal.md:102` says `fp64 tiny err 1.78e-15`;
`frontier-01-proofs-ideal.md:162` says `err = 3.55e-15`; and
`frontier-01-kaggle-2xT4.py:55` says `1.78e-15`. All three describe the same
claim, and none of them is what the code computes: the scripts that do compute
conservation print a value from `torch.randn` at
`frontier-01-eval-high-level-IDEAL.py:36` and it does not equal either literal.
The maintained measurement is `3.6e-14` (`results/measurements.json`,
`additivity_max_abs_err`) — a different quantity, and honestly reported.

**3.3.5 — "score direct err 1.2e-3" is a literal in every file that states it.**
`eval-numpy-ideal.py:17`, `eval-high-level.py:37`, `eval-high-level-IDEAL.py:37`,
`usual-attention-code-IDEAL.py:94`, `graphs-ULTIMATE-V11.py:483`,
`proofs-ideal.md:164`, `falsifications.md:6`. There is no code anywhere in the
tree that sums per-feature score contributions and compares them to the total, so
the error whose magnitude is quoted as `1.2e-3` was never produced. This one is
load-bearing: the "score is not additive" argument in every paper draft rests on
it.

**3.3.6 — the "pp-RoPE validation" applies no rotation and passes unconditionally.**
`frontier-01-bilinearity-break-ideal.py:174` is `q_rot = q[:n_rot]` — a slice of
the un-rotated vector. No rotation is applied. `:182` then prints
`"Torch PASS"` unconditionally; the enclosing `try` (`:163`) catches only
`ImportError` (`:183-184`). There is no assertion in the block.

**3.3.7 — the "8 falsifications all PASS" claim describes code that does not exist.**
`frontier-01-kaggle-2xT4.py:63-71` lists all eight as comments.
`frontier-01-usual-attention-code-IDEAL.py:225-228` is where a pipeline that
would run them should be, and it is a comment block followed by `pass`; the
non-IDEAL version at `:130-133` is the same. The honest fraction: of the eight,
only #1 (conservation) and #7's underlying high-L0 toy
(`high-level-demo.py:112-128`) are executed anywhere, and #7's real SAE
version is not.

**3.3.8 — `frontier-01-gemma4-pp-rope.py` contradicts itself in its own output.**
`:54-55` set `theta = base ** (-2 * 0 / 16)`, which is `base ** 0 == 1.0` for
*any* base, so `D_local` and `D_global` are both `10.0`. `:59` then prints
`"global D 100x smaller, linearization 1+iD works better, interaction small"`.
Verified by execution: `theta_local= 1.0 theta_global= 1.0 D_local= 10.0
D_global= 10.0`. The index `i = 0` is the *fastest* channel, where the base has
no effect at all.

**3.3.9 — `hla-extra-tests-IDEAL.py` cannot run here and fabricates its losses.**
`:29-34` imports `src.model` from `./reviews/laplace-attention`; that path does
not exist in this repository (`Test-Path .\reviews` → `False`), and there is no
`src/`. When the import fails the script falls back to `None` (`:79`) and skips
the real work — but `mode_long_context` still runs and prints losses generated
by `:188-189`, `base_last = 3.2 + 0.15*math.log(T/128+1)` and
`hla_last = 3.18 + 0.05*math.log(T/128+1)`, under the heading `"Simulating base
vs OLD_MODEL loss"` (`:183`).

**3.3.10 — every "Gemma 4 4B" number in the tree is about a model that was never run.**
The specs (`head_dim 512`, `kv_reduction 0.375`, `kv_sharing 18/42`,
`pp_rope_p 0.25`, `base_global 1M`) are asserted in
`frontier-01-proofs-ideal.md:7`, `:84-88` and
`frontier-01-eval-numpy-ideal.py:113-122`. Every script that would load it
substitutes a proxy: `frontier-01-kaggle-2xT4.py:18-20` uses
`"gemma-2-2b"  # or "gemma-3-4b" … or "google/gemma-4-4b" via nnsight`, and
`frontier-01-kaggle-notebook-ideal.py:35` does the same. So every claim
attributed to Gemma 4 4B is unmeasured. (The cited source
"Gemma 4 Technical Report 2607.02770" could not be checked from inside this
repository; that is a separate question from the measurements, which are absent
regardless.)

**3.3.11 — the V25/V26/V27 audit banner certifies artifacts that do not exist.**
`frontier-01-FINAL-V27-…-BEST-POSSIBLE.md:3` asserts, as a single verified
claim, `PNG V11 ultimate 273K 289K 165K 285K 168K 185K 290K 213K all >150K True
PASS … — 0 БАГОВ … — V27 FINAL NUMERIC VERIFIED`. Those PNGs were removed on this
branch (recoverable at `310bf2c`) precisely because they were plots of typed-in
literals. The banner also asserts "numeric verification all proofs PASS 2x vs
3.7x" — §3.1 above. Lines `:46`, `:365`, `:406` repeat the same unverifiable
inventory.

**3.3.12 — `settings.json` bakes a fabricated R² into a key name.**
`projects/settings.json:71` and `projects/settings-ideal.json:71` both contain
`"oral_guaranteed_even_if_phase_R2_0.62": true`, written by
`frontier-01-eval-numpy-ideal.py:177`. The field name asserts a guarantee that is
conditional on a number (`R2_high = 0.62`, `:55`) that nothing measured. It is
the most dangerous single line in the legacy tree, because it is
machine-readable and survives any grep for prose.

---

## 4. What is still worth keeping

Being fair: the legacy tree is not worthless. Four things in it are good, and
the maintained core does not cover all of them.

**4.1 — The rotation-cleaning idea (keep, and it is already superseded).**
`frontier-01-proofs-ideal.md:170-180` asks the right question — *what does it
mean to remove the RoPE rotation to get a pure content score?* — and gives the
right answer: removing `R(m)` removes `pos*theta` but leaves `phi_q`, which is
itself content-dependent, so cleaning the rotation is not the same as removing
position. The maintained core does this properly rather than by argument: it
works in the relative frame, where orthogonality gives
`q_relative = R_m^T (R_m q_hat) = q_hat` and `k_relative(n) = R_{n-m} k_hat`
(`experiments.py:11-17`), and it then *measures* the consequence. The legacy
framing survives only as an intuition; the derivation does not.

**4.2 — The gate/phase vocabulary (keep; the maintained core adopts it).**
`score = |q||k| cos(phi_q - phi_k + pos_diff*theta)`, with gate = magnitude =
"what", phase = angle = "where", is a genuinely good decomposition and it is the
right one. `experiments.py:33-45` restates it exactly, in the stronger form
`A_k` (aligned, position-free) / `B_k` (crossed, carries position) per rotary
pair, with the theorem that `|R_delta k_hat| = |k_hat|` so the gate is exactly
position-independent (`:26-29`). The legacy *vocabulary* is worth keeping; the
legacy *numbers* attached to it (`gate_only 1.84`, `phase_only 3.15`,
`interaction -0.089`) are not.

**4.3 — The reviewer-guidelines summaries (keep verbatim; the maintained core
does not cover this).** `frontier-01-reviewer-guidelines-full.md` and
`frontier-01-REVIEWER-GUIDELINES-TOP3-FULL-V10.md` are genuine fetched conference
text with source URLs and dates. `frontier-01-reviewer-guidelines-FULL-2025-2026.md:1-165`
is the best of the three. The maintained core has no submission material at all,
so there is nothing to deduplicate against. **Caveat:** do not keep
`reviewer-guidelines-FULL-2025-2026.md:168-200` — that section re-uses the
fabricated numbers as evidence of guideline compliance.

**4.4 — The Kaggle / TPU runbooks (keep the engineering, drop the numbers).**
The per-query chunking pattern at `frontier-01-TPU-runbook-IDEAL.md:11-23` (loop
over query positions so scores are `[B,T]` and never `[B,T,T]`) and the
save-`x`-half-but-never-logits discipline at `:41-45` and `:105` are correct,
practical, and not covered anywhere in the maintained core — which never loads a
model. Strip `:27-37` and `:69-77`, which present the eight falsifications as
already-passing.

**4.5 — The Bag-of-Words test *definitions* (keep the design, drop the table).**
`frontier-01-bag-of-words-method.md:66-90` gives standard, correct definitions:
attention entropy `H = -sum p log p` with `H/log T` as a bag-of-words ratio
(`:68-73`), order sensitivity by shuffling (`:75-80`), and needle retrieval as
`1 if w_p = max_i w_i` (`:82-84`). The maintained core implements the entropy
half on synthetic scores (`experiments.py:442-445`); the retrieval and
order-sensitivity halves are implemented nowhere. Only the numbers at `:105-107`
are fabricated. Same for `frontier-01-falsifications.md`, whose eight
interventions are a reasonable falsification design even though every `Честно:`
line is typed in.

**4.6 — The honest early framing (keep).** `frontier-01-nonlinear-qk-attribution.md`
is the only legacy document that states its own epistemic status accurately —
`:24` "the second claim is forbidden until the first is calibrated", `:69` "this
is a promising gap, not proven novelty". It reads as the project it was, before
the numbers were invented. It is worth more as a record of that than the
V25/V26/V27 bundles are.

**4.7 — One piece of algebra worth recovering.**
`frontier-01-scaffold.py:38-48` docstrings state the *correct* additive partition
of a gated-rotated score:
`phase_only = (<Rq,Rk> - <q,k>)/sqrt(d)`, `gate_only = (m-1)*<q,k>/sqrt(d)`,
`interaction = (m-1)*phase / sqrt(d)`, with the conservation test that the four
sum to `s_HLA - s_base`. That is right, and it contradicts the legacy "interaction"
story — here interaction is a defined algebraic term of order `(m-1)*phase`, not
`D²/2`. The maintained core's per-pair closed form (`experiments.py:37-40`) is the
same idea done exactly. The file itself is a stub and cannot run.

---

## 5. Recommendation

**Move the tree to `projects/legacy/`, keep every file, and label the directory.
Do not delete it. Do not leave it where it is.**

**Why not leave it in place.** The failure mode here is not that the legacy
files are wrong — it is that they are *findable and quotable*. `git grep '0.089'`
returns 40-odd hits across paper drafts, oral scripts and runbooks. `retrieval
0.2` returns hits inside a figure-plotting script. A reviewer or a future reader
who lands on `frontier-01-PAPER-DRAFT-BEST-POSSIBLE.md:13` gets a complete,
fluent, confident abstract containing eight invented numbers, with no marker on
the page saying any of it is invented. `README.md:187-203` does warn about this,
but a warning in the root README does not travel with a file that gets copied,
quoted, or cited on its own. The warning has to live *at the point of use*.

**Why not delete.** Three reasons. (a) Four items in §4 are genuinely good and
are not covered by the maintained core — the reviewer-guidelines text, the TPU
runbook, the BoW test definitions, the honest early framing — and deleting them
destroys work that took real effort. (b) The correction is only credible if the
original is visible. A repository that says "we used to claim this, here it is,
and here is the measurement that contradicts it" reads as a project that caught
itself; a repository where the evidence has been quietly removed reads as a
project with something to hide, and invites the suspicion that the same thing was
done to the numbers that remain. (c) `abf4605` already removed the eight `fig_*.png`
and the self-referential patch for exactly this reason; that removal was correct,
but it was done on the *artifacts* and not on the *documents*, so the numbers
themselves are still there. The half-finished state is the worst of both.

**Why a subdirectory rather than a header in each file.** 85 files, 49 of them
markdown. Editing 49 files to add a banner is 49 chances to fat-finger a citation
and a large diff whose content is not about the research. Moving the directory is
one commit, is greppable, and puts the explanation in exactly one place that a
reader hits *before* the content rather than after it.

**The concrete shape.** A single commit that:

1. `git mv projects/frontier-01-* projects/legacy/` (this also moves
   `settings.json`, `settings-ideal.json`, `Dockerfile`, `requirements.txt` and
   the two figure HTML files, which is the right grouping — they are the
   artifacts those scripts emit).
2. Leaves `projects/ATTRIBUTION.md` and `projects/rope_attribution/` where they
   are. `ATTRIBUTION.md` is current provenance, not legacy material, and it
   already carries the hardcoded-constant caveat at `:35-70`. It does need two
   edits: its status note (`:20`, `:157-163`) says `experiments.py` and
   `verify_rope*.py` do not exist, and they now do.
3. Adds `projects/legacy/README.md` — a pointer to this file plus the four
   §4 items, so a reader who opens the directory knows which four files to read
   and which 81 to ignore.
4. Puts a one-line banner in the root `README.md` at the existing "Legacy
   material" section, pointing at `projects/legacy/README.md` and at this file.

**Two narrower calls, if the full move is too disruptive:**

- **Delete `projects/settings.json` and `settings-ideal.json` outright.** They
  are outputs of `eval-numpy-ideal.py`, they are regenerable by re-running that
  script, and they are the most citable objects in the tree — machine-readable,
  results-shaped, with `oral_guaranteed_even_if_phase_R2_0.62` baked into a key
  name at `:71`. If only one thing is removed, remove these two.
- **Delete or truncate the six "FINAL" bundles** (`FINAL-V13`, `V25`, `V26`,
  `V27`, `FINAL-ANSWER-IDEAL-v4`, `FINAL-HIGHEST-IDEAL-v3`). They are
  109-line-apart restatements of each other whose only unique content is a
  version number and an audit banner that certifies files which no longer exist
  (§3.3.11). They add no recoverable information and they are the documents a
  reader is most likely to mistake for a result.

**Do not** relabel the file *contents*. Leaving the fabricated numbers in place
inside a clearly-marked `legacy/` directory is the correct outcome: it preserves
the record of what was claimed, which is what makes the correction in
`README.md:119-137` checkable. Rewriting 49 files to remove numbers would
destroy the evidence that the correction was necessary.

---

*Generated by audit of `projects/frontier-01-*` on branch state `a2f9792`.
Enumerated with `git ls-files 'projects/*.md'` (54) and `git ls-files
'projects/*.py'` (35, of which 4 are `projects/rope_attribution/`). Every
`illustrative` and `partly-computed` classification above cites a `file:line`
that was read directly.*
