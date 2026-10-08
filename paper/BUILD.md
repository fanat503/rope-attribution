# Building the paper

The paper is `main.tex`. It is self-contained: it needs no `.sty` or `.bst`
file beyond the ones in this directory, and no figure is required to be present
(see *Missing figures* below).

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
`\documentclass[twocolumn]{article}`.

The matching `neurips_2026.cls` is **not** vendored. It could not be fetched from
the NeurIPS repositories during this work (`neurips.cc/Conferences/2026/...`
returns 404 for the class), so the `\documentclass{neurips_2026}` form is not
available. NeurIPS ships the two files together; before submitting, download both
from the conference style-files page and switch line 15 of `main.tex` to
`\documentclass[preprint,position]{neurips_2026}`, dropping the `\usepackage`.

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
those files or from `results/measurements.json`.

## What the tests check about the build

- `tests/test_venue_compliance.py::test_tectonic_can_compile_the_paper` compiles
  the committed source into a fresh temporary directory, which catches a paper
  that only builds because of stale intermediates in `build/`.
- `tests/test_venue_compliance.py::test_every_numbered_figure_reaches_the_document`
  fails if any figure fell back to its placeholder.
- `tests/test_venue_compliance.py::test_the_figures_are_all_embedded` requires at
  least nine embedded images.
- `tests/test_venue_compliance.py::test_the_main_text_fits_in_nine_pages` measures
  the main text against NeurIPS' nine-page limit. It is currently `xfail`: the
  main text runs to 14 pages.

## Page limit accounting

NeurIPS caps the main text at nine pages. References and the appendix are not
counted. The boundary is located by finding the final main-text section in the
extracted page text — `LAST_MAIN_TEXT_SECTION` in
`tests/test_venue_compliance.py` — rather than by counting pages of the PDF,
which would silently include the appendix.

## Regenerating the numbers

The paper's numbers come from two sources, both regenerated from the code:

```sh
PYTHONPATH=../projects python -m rope_attribution.experiments   # results/measurements.json
PYTHONPATH=../projects python -m rope_attribution.statistics   # results/statistics.json
PYTHONPATH=../projects python -m rope_attribution.figures      # figures/*.png and *.csv
```

All three are seeded and reproduce byte-identically.
`tests/test_measurements_json_is_current` and its statistics counterpart re-run
them and require the committed files to match, so the paper cannot drift from
the data.