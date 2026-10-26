# Building the paper

The paper is `main.tex`. It needs the vendored `neurips_2026.sty` in this
directory and the standard `plainnat` bibliography style, but no other local
style file. No figure is required to be present (see *Missing figures* below).

## With tectonic (what CI uses)

```sh
mkdir -p build
tectonic -X compile --outdir build main.tex
```

tectonic does not create the output directory, so `mkdir -p build` is required
and is the single most common reason a fresh checkout fails to build.

The pinned version used by `.github/workflows/paper.yml` is 0.15.0. It is
installed on the runner, not vendored into the repository.

## With pdflatex

```sh
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`bibtex` is required for the bibliography to resolve on the first pass; the two
following `pdflatex` runs pick up the citation numbers and the cross-references.

## Style files

`neurips_2026.sty` is vendored in this directory and is loaded as
`\usepackage[preprint,position]{neurips_2026}` on top of
`\documentclass{article}`, exactly as the official template does. The style file
is single-column, so no layout option is passed to the class and the page
geometry is the one the style itself sets.

It is the only third-party file in the repository, vendored verbatim and
unmodified so that `paper/` builds without network access. Its own header
records that it is a partial rewrite of `nips15submit_e.sty`; it carries no
copyright line or license header of its own. `../NOTICE.md` is the
repository-level statement that names this exception, and nothing outside
`paper/` imports or derives from it. The `preprint` option is deliberate: it
produces a non-anonymous build for checking the formatting. The submission's
anonymity requirement is an open decision, not something this file asserts.

The official NeurIPS 2026 package ships `neurips_2026.sty`, `neurips_2026.tex`
and `checklist.tex`; there is **no** `neurips_2026.cls`, so
`\documentclass{neurips_2026}` is not a form the conference supports. The
vendored `neurips_2026.sty` is byte-identical to the official one (SHA-256
`C3FC2894E83D2517CA18B66741D6C595986D97957DC08EC08BB2125A7EC4555A`), and
`main.tex` uses `\documentclass{article}` plus `\usepackage{neurips_2026}`,
which is the supported form.

The style file loads `natbib` and sets the geometry itself, so `main.tex` must
not load either. It also explicitly forbids `fullpage`.

## Missing figures

`\paperfigure` takes `(png basename, caption, label)`. If `../figures/NAME.png`
is absent it emits a framed placeholder naming the file and pointing at the CSV
sidecar in `../figures/`, which carries the plotted array. The document
therefore compiles with an empty `../figures/` directory — but the resulting PDF
is not submittable, and `tests/test_venue_compliance.py` fails on any surviving
placeholder.

No claim in the paper is read off a pixel: every figure ships the array it
plotted as a `figures/NAME.csv` sidecar, and the numbers in the prose come from
those files or from the structured artifacts under `results/` (the provenance
appendix names the artifact for each row).

## What the tests check about the build

- `tests/test_venue_compliance.py::test_tectonic_can_compile_the_paper` compiles
  the committed source into a fresh temporary directory, which catches a paper
  that only builds because of stale intermediates in `build/`.
- `tests/test_venue_compliance.py::test_every_numbered_figure_reaches_the_document`
  fails if any figure fell back to its placeholder.
- `tests/test_venue_compliance.py::test_the_figures_are_all_embedded` requires at
  least one embedded image per `\paperfigure` call, counting the calls in
  `main.tex` rather than hard-coding a number.
- `tests/test_venue_compliance.py::test_the_main_text_fits_in_nine_pages` locates
  the appendix's first section in the built PDF and requires it to start no later
  than page 10. It also rejects a `Figure N:` or `Table N:` caption above that
  heading, because a float queued behind the Conclusion can otherwise occupy the
  first uncounted page while the heading still lands on page 10. The appendix is
  forced onto a fresh page, so this is a real measure of where the main text ends.
  A PDF older than the source is skipped locally rather than used as evidence;
  the paper workflow sets `ROPE_PAPER_STRICT=1`, which makes a missing or stale
  PDF a failure, so CI cannot pass this gate by skipping it.

## Page limit accounting

NeurIPS caps the main text at nine pages. References and the appendix are not
counted. The appendix is preceded by `\clearpage`, so its first section starts
on a fresh page. `tests/test_venue_compliance.py` locates that section in the
extracted page text and requires it to start no later than page 10: page 10 is
the first uncounted page, so this is equivalent to a main text of at most nine
pages. The same test rejects any float caption before that heading, because a
figure queued at the end of the main text can otherwise sit on page 10 while the
heading itself still starts there. Both conditions are measured from the built
PDF; neither is inferred from the source.

## Regenerating the numbers

The paper's numbers live in `results/*.json` and the CSV sidecars under
`figures/`; nothing is typed from memory. The principal generators are:

```sh
PYTHONPATH=../projects python -m rope_attribution.experiments   # measurements.json
PYTHONPATH=../projects python -m rope_attribution.statistics   # statistics.json
PYTHONPATH=../projects python -m rope_attribution.sensitivity  # sensitivity.json
PYTHONPATH=../projects python -m rope_attribution.figures      # figures/*.png and *.csv
```

The trained-checkpoint artifacts need CPU torch, `transformers` and the three
public checkpoints (the first run downloads them):

```sh
PYTHONPATH=../projects python -m rope_attribution.real_model    # real_model.json
PYTHONPATH=../projects python -m rope_attribution.usefulness    # usefulness*.json
PYTHONPATH=../projects python -m rope_attribution.uncertainty   # uncertainty values
PYTHONPATH=../projects python -m rope_attribution.seed_variance # usefulness_seeds.json
PYTHONPATH=../projects python -m rope_attribution.agreement     # agreement.json
PYTHONPATH=../projects python -m rope_attribution.attention_cost # attention_cost.json
```

The trained-dictionary artifact additionally needs the public
`ctigges/pythia-70m-deduped__res-sm_processed` SAE
(`3-res-sm`, `d_sae=32768`), which is not vendored. The artifact records the
exact dictionary it used; pass the downloaded file explicitly:

```sh
PYTHONPATH=../projects python -m rope_attribution.qk_attribution \
    --sae /path/to/sae_ctigges_l3.safetensors                   # qk_attribution.json
```

All generators are seeded. The structural artifacts are re-run by the suite and
compared value by value (floats to a relative `1e-12`, which tolerates last-bit
drift between platforms). The checkpoint artifacts need network access and CPU
torch; where the suite has them, it re-runs the headline measurement and
compares the fields the paper quotes, not byte-identical files. The figure CSV
sidecars are re-derived against the paper's claims in
`tests/test_paper_claims.py`.