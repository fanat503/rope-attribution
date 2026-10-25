# NOTICE

This repository is distributed under the Apache License, Version 2.0. See `LICENSE`.

This file records the provenance of the repository: what is original work, what is
an academic citation, and what third-party source is included. The answer to the
last question is **none, with one named exception** — stated explicitly, because a
bare assertion would not be auditable. The exception is the paper's conference
style sheet, named in "Third-party code" below.

## Academic citations (methods, not code)

The methods studied and re-implemented here are published algorithms, cited as
academic literature:

- **RoPE** — Su, Jianlin; Lu, Yu; Pan, Shengfeng; Murtadha, Ahmed; Wen, Bo;
  Liu, Yunfeng. *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
  arXiv:2104.09864 (2021). <https://arxiv.org/abs/2104.09864>
- **YaRN** — Peng, Bowen; Quesnelle, Jeffrey; Fan, Honglu; Shippole, Enrico.
  *YaRN: Efficient Context Window Extension of Large Language Models*.
  arXiv:2309.00071 (2023). <https://arxiv.org/abs/2309.00071>

Citing a method is not the same as shipping its implementation. The formulas and
algorithms behind RoPE and YaRN were read from the papers and re-expressed in
this repository's own code. No source code from those works, or from any
implementation of them, is present in this repository.

## Third-party code: none in the measured code

There is no third-party source code in the Python modules, tests, tooling or
measurement artifacts this repository's claims rest on.

One file is third-party by construction, and it is named here rather than left
to be found: `paper/neurips_2026.sty`, the NeurIPS conference style sheet,
vendored verbatim so that `paper/` builds without network access. Its own header
records that it is a partial rewrite of `nips15submit_e.sty`. It carries no
copyright line and no license header of its own, nothing under `projects/`
imports or derives from it, and it is the only such file in the tree.

Beyond it: no vendored directory, no `src/` tree, no bundled third-party
library, and no file carrying a third-party copyright or SPDX header. Exactly
one license file exists in the tree — `LICENSE`, Apache-2.0, with the appendix
copyright line filled as `Copyright 2026 Ilya Slyatski` — and a case-insensitive
search for `copyright`, `SPDX-License`, `MIT License` and `BSD` across every
`.py`, `.json` and `.txt` file returns six `copyright` hits, all in
`tests/test_authorship.py` and `tests/test_author_metadata.py`, where they
assert the project's own Apache line; `SPDX-License`, `MIT License` and `BSD`
return zero. None of the six is a third-party claim.

That statement is auditable rather than asserted. It is backed by a provenance
audit first run on 2026-09-29 and re-measured since, reproducible with the exact
commands in `projects/ATTRIBUTION.md`. Observed result, over all `*.py`, `*.md`,
`*.json` and `*.txt` files, excluding `.git/`, `__pycache__/`, and the three
provenance documents themselves (`NOTICE.md`, `CITATION.cff`,
`projects/ATTRIBUTION.md` — which necessarily name these patterns in order to
report on them), matched case-insensitively:

| Pattern | Hits | Reading |
| --- | ---: | --- |
| `inv_freq` | 240 | This project's own code, tests and measured artifacts. |
| `rotary` | 256 | This project's own code, tests and measured artifacts. |
| `mscale` | 290 | This project's own code, tests and measured artifacts. |
| `rotary_emb` | 4 | Two are the upstream config keyword `rotary_emb_base` in synthetic test models; two are this project's own fallback lookup for that keyword in `downstream.py`. Not an implementation. |
| `apply_rotary` | 3 | A test monkeypatches `transformers`' `apply_rotary_pos_emb` to a no-op to compare against a no-RoPE model; no upstream code is shipped. |
| `attention_scaling` | 0 | — |
| `ntk` | 0 | — |
| `copied from` | 1 | A comment in `tpu_configs.py` saying the learning rates are *not* copied from the CPU ladder. |
| `ported from` | 0 | — |
| `taken from` | 0 | — |

The table is a re-measurement on 2026-10-23, after the dataset-preparation
module and the Gemma 4 / partial-rotation architecture guards were added.
Earlier figures attributed 13 `inv_freq` and 3 `mscale` hits to `verify_rope*.py`
scripts that do not exist in this repository. Patterns are matched
case-insensitively over `.py`, `.md`, `.json` and `.txt` files, with `.git/`,
`paper/build/`, cache directories and the untracked `.tmp/` scratch tree
excluded, and with this file, `CITATION.cff` and `projects/ATTRIBUTION.md`
excluded too --- they enumerate the patterns, so counting them would make the
audit measure itself.

Every non-zero hit above is either this project's own code, a test of that code,
an emitted measurement artifact, or an upstream API name used as a test seam.
None is a fragment of an upstream implementation. The four identifier rows
`attention_scaling`, `ntk`, `ported from` and `taken from` remain at zero. The
four `rotary_emb` hits are one public Hugging Face configuration keyword: two
occurrences in synthetic test models that set it, and two in this project's own
fallback read of that keyword. The three `apply_rotary` hits are the name of a
public Hugging Face function used as a monkeypatch seam. Neither is a copied
body or a vendored file.

Scope of that audit, stated so the claim is not read as broader than it is: it
covers the text sources `.py`, `.md`, `.json`, `.txt`. The LaTeX sources under
`paper/` and the figure CSV sidecars under `figures/` sit outside that scope and
were checked separately on 2026-10-17: `rotary_emb`, `apply_rotary`,
`attention_scaling`, `ntk`, `copied from` and `ported from` return 0 hits across
them, the single hit for `taken from` is the phrase "activations taken from three
trained checkpoints" in `paper/main.tex` (a description of where activations come
from, not a code-provenance marker), and no file among them carries a
`copyright` or `SPDX` marker.

## The new `rope.py` is an independent implementation

`projects/rope_attribution/rope.py` is original to this project. It is ordinary
NumPy code that re-implements published algorithms, written against the papers
and against the canonical public reference implementations, which were consulted
as **reading** references while transcribing formulas:

- Su et al., arXiv:2104.09864 — the RoPE definition and the inverse-frequency
  schedule.
- Peng et al., arXiv:2309.00071 — the YaRN method.
- `jquesnelle/yarn`, `scaled_rope/LlamaYaRNScaledRotaryEmbedding.py` — the YaRN
  ramp expressions (`find_correction_dim`, `find_correction_range`,
  `linear_ramp_mask`) and the magnitude scaling (`get_mscale`).
- Hugging Face `transformers` (`models/llama/modeling_llama.py`,
  `modeling_rope_utils.py`) — the split-half rotation convention
  (`rotate_half`, `compute_default_rope_parameters`).

What was consulted is the mathematics: formulas, algorithm steps, and the two
standard layout conventions that define the method. What was not done is copying
code. No source file, function body, or line of upstream code was pasted,
vendored, or mechanically transformed into this repository. The implementation is
written independently, in NumPy, and its behaviour is checked by
`tests/test_rope.py`, whose assertions compare the implementation against
closed-form identities rather than against upstream output. An earlier version
of this paragraph credited the check to `verify_rope.py` through
`verify_rope4.py` at the repository root; those scripts do not exist here, and
the tests are what actually does the checking.

Where `rope.py` shares a convention with upstream — the split-half pairing of
dimension `i` with `i + dim/2`, the ramp bounds derived from `beta_fast` and
`beta_slow` — the shared element is a published algorithm or a standard
arithmetic layout, not copied expression. The `transformers` and `jquesnelle/yarn`
projects are named here as sources consulted, not as licensors of any part of
this repository.

No third-party license is reproduced or claimed: no third-party source is
distributed beyond the conference style sheet named above, and that file carries
no license text of its own to reproduce. The Apache-2.0 `LICENSE` at the
repository root covers only this project's own original work.

## Attribution

Per-module provenance is recorded in `projects/ATTRIBUTION.md`. The early
exploratory scripts and drafts that used to live under `projects/frontier-01-*`
were removed from the tree in October 2026; they printed illustrative constants
rather than measurements and are not part of the maintained record. Machine-
readable citation metadata is in `CITATION.cff`.
