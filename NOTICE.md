# NOTICE

This repository is distributed under the Apache License, Version 2.0. See `LICENSE`.

This file records the provenance of the repository: what is original work, what is
an academic citation, and what third-party source code is included. The answer to
the last question is **none** — stated explicitly, because that is the fact the
provenance audit established, and because a bare assertion would not be auditable.

## Academic citations (methods, not code)

The methods studied and re-implemented here are published algorithms, cited as
academic literature:

- **RoPE** — Su, Jianlin; Lu, Yu; Pan, Shengfeng; Murtadha, Ahmed; Wen, Bo;
  Liu, Yunfeng. *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
  arXiv:2104.09864 (2021). <https://arxiv.org/abs/2104.09864>
- **YaRN** — Peng, Bowen; Quesnelle, Jeffrey; Fan, Honglu; Shippole, Enrico.
  *YaRN: Efficient Context Window Extension of Large Language Models*.
  arXiv:2309.00071 (2023). <https://arxiv.org/abs/2309.00071>
- *Gemma 4 Technical Report*, arXiv:2607.02770 (2026) — cited across the
  `projects/frontier-01-*.md` planning documents as a model-architecture
  reference (pp-RoPE layout, head dimension, local/global ratio).
- *LeRoPE: Learnable RoPE Frequencies Improve Language Modeling*,
  arXiv:2607.10134 (2026) — cited in the same planning documents as prior work on
  RoPE frequency schedules.

Citing a method is not the same as shipping its implementation. The formulas and
algorithms behind RoPE and YaRN were read from the papers and re-expressed in
this repository's own code. No source code from those works, or from any
implementation of them, is present in this repository.

## Third-party code: none

There is no third-party source code in this repository.

There is no vendored directory, no `src/` tree, no bundled third-party library,
and no file carrying a third-party copyright or SPDX header. Exactly one license
file exists in the tree — `LICENSE`, Apache-2.0, with the appendix copyright line
left unfilled — and a case-insensitive search for `copyright`, `SPDX-License`,
`MIT License` and `BSD` across every `.py`, `.json` and `.txt` file returns
0 hits.

That statement is auditable rather than asserted. It is backed by a provenance
audit run on 2026-09-29, reproducible with the exact commands in
`projects/ATTRIBUTION.md`. Observed result, over all `*.py`, `*.md`, `*.json`
and `*.txt` files, excluding `.git/`, `__pycache__/`, and the three provenance
documents themselves (`NOTICE.md`, `CITATION.cff`, `projects/ATTRIBUTION.md` —
which necessarily name these patterns in order to report on them), matched
case-insensitively:

| Pattern | Hits | Where they are |
| --- | --- | --- |
| `inv_freq` | 223 | This project's own code and its own test/measurement artefacts: `tests/test_rope.py` (77), `projects/rope_attribution/real_model.py` (29), `results/real_model.json` (24), `tests/test_real_model.py` (19), `projects/rope_attribution/rope.py` (12). |
| `rotary` | 176 | `results/real_model.json` (55), `projects/rope_attribution/real_model.py` (36), `tests/test_real_model.py` (13), `projects/rope_attribution/rope.py` (10). |
| `mscale` | 344 | `tests/test_paper_claims.py` (99), `tests/test_rope.py` (71), `tests/test_experiments.py` (43), `projects/rope_attribution/figures.py` (41), `results/measurements.json` (24). |
| `rotary_emb` | 0 | — |
| `apply_rotary` | 0 | — |
| `attention_scaling` | 0 | — |
| `ntk` | 0 | — |
| `copied from` | 0 | — |
| `ported from` | 0 | — |
| `taken from` | 0 | — |

The counts above are a re-measurement, not the ones this table originally carried. Three things made the earlier figures wrong: they attributed 13 `inv_freq` and 3 `mscale` hits to `verify_rope*.py` scripts that do not exist in this repository, and the repository has since gained `real_model.py`, `usefulness.py`, `statistics.py` and their tests, which legitimately add hits of their own. Patterns are matched case-insensitively over `.py`, `.md`, `.json` and `.txt` files, with `.git/`, `paper/build/` and the untracked `.tmp/` scratch tree excluded, and with this file and `projects/ATTRIBUTION.md` excluded too - they enumerate the patterns, so counting them would make every zero row non-zero and the audit would measure itself.

Every non-zero hit above is either this project's own newly written code, this project's own test of that code, or this project's own emitted measurement artefact. None is a fragment of an upstream implementation, and the zero rows remain zero, which is the part that matters.

Scope of that audit, stated so the claim is not read as broader than it is: it
covers the text sources `.py`, `.md`, `.json`, `.txt`. The only other non-binary
text files in the tree are `projects/frontier-01-figures.html` and
`projects/frontier-01-figures-ideal.html`, which are figure preview pages with no
scripts (0 hits for `<script`, `import ` and `require(`).

This audit previously claimed to cover `rope-attribution-anthropic.patch`, a git
diff of this repository's own history. That file does not exist here and never did,
so the claim was unverifiable and has been removed rather than repeated.

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
`tests/test_rope.py`, 77 of whose assertions reference `inv_freq` and which compare
the implementation against closed-form identities rather than against upstream
output. An earlier version of this paragraph credited the check to `verify_rope.py`
through `verify_rope4.py` at the repository root; those scripts do not exist here,
and the tests are what actually does the checking.

Where `rope.py` shares a convention with upstream — the split-half pairing of
dimension `i` with `i + dim/2`, the ramp bounds derived from `beta_fast` and
`beta_slow` — the shared element is a published algorithm or a standard
arithmetic layout, not copied expression. The `transformers` and `jquesnelle/yarn`
projects are named here as sources consulted, not as licensors of any part of
this repository.

No third-party license is reproduced or claimed, because no third-party source
is distributed. The Apache-2.0 `LICENSE` at the repository root covers only this
project's own original work.

## Attribution

Per-module provenance is recorded in `projects/ATTRIBUTION.md`, which also
documents a substantive caveat: several legacy `projects/frontier-01-*.py`
scripts print hardcoded illustrative constants rather than computed
measurements. Machine-readable citation metadata is in `CITATION.cff`.
