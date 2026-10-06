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
| `inv_freq` | 24 | `projects/rope_attribution/rope.py` (9), `projects/rope_attribution/__init__.py` (1), `verify_rope*.py` (13) — this project's own new module and its checks — plus one prose formula sketch in `projects/frontier-01-best-plan.md:42` |
| `rotary` | 9 | `projects/rope_attribution/rope.py` (8), `__init__.py` (1) — this project's own new module — plus one citation line in `projects/frontier-01-PAPER-DRAFT-V13-9PAGES-ORAL.md:85` |
| `mscale` | 15 | `projects/rope_attribution/rope.py` (12), `__init__.py` (1), `verify_rope.py` (2), `verify_rope4.py` (1) — this project's own new module and its checks |
| `rotary_emb` | 0 | — |
| `apply_rotary` | 0 | — |
| `attention_scaling` | 0 | — |
| `ntk` | 0 | — |
| `copied from` | 0 | — |
| `ported from` | 0 | — |
| `taken from` | 0 | — |

Every non-zero hit above is either this project's own newly written code or
prose in this project's own documents. None is a fragment of an upstream
implementation.

Scope of that audit, stated so the claim is not read as broader than it is: it
covers the text sources `.py`, `.md`, `.json`, `.txt`. The only other non-binary
text files in the tree are `projects/frontier-01-figures.html` and
`projects/frontier-01-figures-ideal.html`, which are figure preview pages with no
scripts (0 hits for `<script`, `import ` and `require(`), and
`rope-attribution-anthropic.patch`, which is a git diff of *this repository's own*
history — the Python appearing in it is this project's own earlier code being
removed, not third-party code.

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
written independently, in NumPy, and its behaviour is checked by the
`verify_rope.py` … `verify_rope4.py` scripts at the repository root, which
compare it against closed-form identities rather than against upstream output.

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
