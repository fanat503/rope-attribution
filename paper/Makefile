# Build the paper. Run from this directory (paper/).
#
# Figures are resolved by relative path ../figures/ and guarded with
# \IfFileExists inside main.tex, so this also succeeds with figures/ absent.
#
# NOTE: the maintained build path is in BUILD.md and uses tectonic; the CI
# workflow .github/workflows/paper.yml compiles the paper with it. This
# convenience Makefile drives pdflatex + bibtex when that toolchain is
# installed. It is not exercised by CI, so prefer the BUILD.md command if the
# two ever disagree.

PDF := main.pdf
AUX := main.aux main.bbl main.blg main.log main.out

.PHONY: all clean check-toolchain

all: $(PDF)

$(PDF): main.tex references.bib
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

check-toolchain:
	@for t in pdflatex xelatex lualatex tectonic bibtex; do \
	  if command -v $$t >/dev/null 2>&1; then \
	    echo "found: $$t -> $$(command -v $$t)"; \
	  else \
	    echo "MISSING: $$t"; \
	  fi; \
	done

clean:
	rm -f $(PDF) $(AUX)
