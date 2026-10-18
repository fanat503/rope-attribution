# One-command reproduction for the rope-attribution repository.
#
# `make all` runs the test suite, regenerates the committed structural
# artefacts and the figure set, and recompiles the paper. Every step is
# reproducible: the test suite recomputes the structural JSON artefacts
# and compares them with the committed copies value by value (floats to
# a relative 1e-12).
#
# The checkpoint measurements (real_model.py, usefulness.py, agreement.py) are
# deliberately not part of `all`: they need CPU torch, which is not a test
# dependency, and they take about four minutes each. Run `make checkpoints`
# when you want them.

# `-m` takes a DOTTED module path, not a filesystem path. This Makefile shipped
# `projects/rope_attribution.experiments` with slashes and CI rejected it with a
# ModuleNotFoundError that read like a missing file rather than a typo.
#
# `projects/` has no __init__.py and does not need one: Python resolves it as an
# implicit namespace package, so the dotted form runs straight from the repository
# root with no PYTHONPATH at all. That is the form every document here uses.
PYTHON  ?= python
PKG      = projects.rope_attribution

.PHONY: all test results figures paper checkpoints postedit clean help

## all: test + results + figures + paper
all: test results figures paper

## test: run the full suite
test:
	$(PYTHON) -m pytest -q

## results: regenerate measurements.json, statistics.json, sensitivity.json
results:
	$(PYTHON) -m $(PKG).experiments
	$(PYTHON) -m $(PKG).statistics
	$(PYTHON) -m $(PKG).sensitivity

## figures: regenerate figures/*.png and the CSV sidecars, plus figures/README.md
figures:
	$(PYTHON) -m $(PKG).figures

## paper: compile paper/main.tex (needs tectonic on PATH, or .tmp/tectonic/)
paper:
	cd paper && (command -v tectonic >/dev/null && tectonic -X compile --outdir build main.tex \
		|| ../.tmp/tectonic/tectonic.exe -X compile --outdir build main.tex)

## postedit: syntax + ruff + the affected module's own tests, on what changed.
# The check Fan et al. (arXiv:2609.20804) found worth running after every
# edit rather than at the end of a batch. Use it between edits, not before a
# commit - by commit time it should already have passed.
postedit:
	$(PYTHON) scripts/postedit.py

## checkpoints: re-measure on trained checkpoints (needs real_model_requirements.txt)
checkpoints:
	$(PYTHON) -m $(PKG).real_model
	$(PYTHON) -m $(PKG).usefulness
	$(PYTHON) -m $(PKG).agreement

## clean: remove generated build output (never touches committed artefacts)
clean:
	rm -rf paper/build .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

## help: list targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'