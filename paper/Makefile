# Build the paper. Run from this directory (paper/).
#
# Figures are resolved by relative path ../figures/ and guarded with
# \IfFileExists inside main.tex, so this also succeeds with figures/ absent.
#
# NOTE: this target requires a LaTeX installation (pdflatex + bibtex).
# No LaTeX toolchain was present on the machine where this paper was
# written, so it has NOT been executed there. The source was instead
# validated statically (balanced environments/braces, label/ref
# consistency, citation keys vs references.bib, and \includegraphics
# path resolution).

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
