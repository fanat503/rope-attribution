# One-command reproduction for the rope-attribution repository.
#
# `make all` runs the test suite, regenerates both committed artefacts and the
# figure set, and recompiles the paper. Every step is byte-reproducible: the two
# JSON artefacts are verified identical across three independent runs.
#
# The checkpoint measurements (real_model.py, usefulness.py) are deliberately not
# part of `all`: they need CPU torch, which is not a test dependency, and they
# take about four minutes each. Run `make checkpoints` when you want them.

PYTHON  ?= python
PKG      = projects/rope_attribution
PYTHONPATH = projects

.PHONY: all test results figures paper checkpoints clean help

## all: test + results + figures + paper
all: test results figures paper

## test: run the full suite
test:
	$(PYTHON) -m pytest -q

## results: regenerate results/measurements.json and results/statistics.json
results:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(PKG).experiments
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(PKG).statistics

## figures: regenerate figures/*.png and the CSV sidecars, plus figures/README.md
figures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(PKG).figures

## paper: compile paper/main.tex (needs tectonic on PATH, or .tmp/tectonic/)
paper:
	cd paper && (command -v tectonic >/dev/null && tectonic -X compile --outdir build main.tex \
		|| ../.tmp/tectonic/tectonic.exe -X compile --outdir build main.tex)

## checkpoints: re-measure on trained checkpoints (needs real_model_requirements.txt)
checkpoints:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(PKG).real_model
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(PKG).usefulness

## clean: remove generated build output (never touches committed artefacts)
clean:
	rm -rf paper/build .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

## help: list targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'