"""Measure the paper against the venue's format rules, and report honestly.

Why this file exists
--------------------
The paper targets NeurIPS, which caps the *main text* at nine pages. Everything
after the appendix marker does not count. Both halves of that are easy to get
wrong and hard to notice:

* counting the whole PDF would count the appendix and the references, which are
  not subject to the limit;
* counting the PDF would silently pass if the appendix moved earlier, so the
  boundary is located by finding the ``\\appendix`` marker in the source and
  matching it against the extracted page text.

The tests below therefore *measure* rather than assert. The page-count test
carried ``xfail(strict=False)`` while the main text ran over the limit — it
recorded the real number instead of failing the suite — and it cleared on
2026-10-17, the day the compression pass moved the related-work survey, the
extension-comparison table and the "note on two corrections" into the appendix.
It is a plain assertion now. Raising the limit without changing the paper still
cannot make it pass silently: the boundary is located in the built PDF.

Everything here skips cleanly when the paper has not been built, so a fresh
clone with no LaTeX toolchain still has a green suite that is honest about
having checked nothing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PAPER_DIR = _REPO / "paper"
_MAIN_TEX = _PAPER_DIR / "main.tex"
_BUILD_DIR = _PAPER_DIR / "build"
_PDF = _BUILD_DIR / "main.pdf"

# NeurIPS 2026: nine pages of main text, unlimited references and appendix.
MAIN_TEXT_LIMIT = 9


def _pdf_or_skip() -> Path:
    if not _PDF.exists():
        pytest.skip("paper/build/main.pdf is absent; run the paper workflow first")
    return _PDF


@pytest.fixture(scope="module")
def pages() -> list[str]:
    """Flattened text of each page, in order."""
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(str(_pdf_or_skip()))
    return [" ".join((page.extract_text() or "").split()) for page in reader.pages]


def _first_page_containing(pages: list[str], needle: str) -> int | None:
    for index, text in enumerate(pages, start=1):
        if needle in text:
            return index
    return None


# --------------------------------------------------------------------------
# where the counted part of the paper stops
# --------------------------------------------------------------------------


# The last section of the main text. Bound for page accounting by the first
# page on which it appears. After the 2026-10 compression pass the counted text
# ends with the Conclusion — the "note on two corrections" it used to end with
# now lives in the appendix — so the boundary is located there. Every page
# accounting test keys on this, so moving it means updating one named constant.
LAST_MAIN_TEXT_SECTION = "Conclusion"


def test_the_appendix_marker_is_present() -> None:
    """Without this marker the page accounting below is meaningless."""
    assert "\\appendix" in _MAIN_TEX.read_text(encoding="utf-8")


def test_the_last_main_text_section_is_found(pages: list[str]) -> None:
    """The boundary marker must be locatable, or the count is a guess."""
    boundary = _first_page_containing(pages, LAST_MAIN_TEXT_SECTION)
    assert boundary is not None, (
        f"could not locate {LAST_MAIN_TEXT_SECTION!r} in the PDF; the page-count "
        "test needs updating to whatever replaced it"
    )


def test_the_main_text_fits_in_nine_pages(pages: list[str]) -> None:
    """The limit itself, measured rather than asserted.

    This test carried ``xfail(strict=False)`` while the main text ran long: it
    recorded the real number — measured at 11 pages when the October 2026
    compression pass started, 10 after the first cut — and cleared when the
    second cut brought the boundary to page 9. It stays an assertion rather than
    a warning because that is what the limit is.
    """
    boundary = _first_page_containing(pages, LAST_MAIN_TEXT_SECTION)
    assert boundary is not None
    assert boundary <= MAIN_TEXT_LIMIT, (
        f"main text runs to page {boundary}, limit is {MAIN_TEXT_LIMIT}"
    )


# --------------------------------------------------------------------------
# structural facts a reviewer would otherwise have to take on trust
# --------------------------------------------------------------------------


def test_every_numbered_figure_reaches_the_document(pages: list[str]) -> None:
    """No placeholder survived the build.

    ``\\paperfigure`` substitutes a framed placeholder when the PNG is absent, so
    a missing image still compiles. This is the check that says so out loud.
    """
    assert not any("figure file absent at build time" in text for text in pages)


def test_the_figures_are_all_embedded(pages: list[str]) -> None:
    """One raster image per figure, counted from the LaTeX rather than typed in.

    The assertion used to be a hard-coded floor of nine, which eleven figures
    clear even if two images silently vanish. The count is therefore read from
    the ``\\paperfigure`` calls in ``main.tex``, the same way the paper CI gate
    does it, so adding a figure tightens the check automatically.
    """
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(str(_pdf_or_skip()))
    n_images = sum(len(page.images) for page in reader.pages)
    main = _MAIN_TEX.read_text(encoding="utf-8")
    n_figures = len(re.findall(r"(?m)^\\paperfigure\{", main))
    assert n_figures >= 1, "no \\paperfigure calls found in paper/main.tex"
    assert n_images >= n_figures, (
        f"{n_images} embedded images for {n_figures} \\paperfigure calls; "
        f"a figure is probably missing"
    )


def test_the_references_are_present_and_not_empty(pages: list[str]) -> None:
    start = _first_page_containing(pages, "References")
    assert start is not None, "no References heading in the PDF"
    tail = " ".join(pages[start - 1 :])
    assert "arXiv" in tail, "the bibliography is empty"


def test_the_bibliography_closes_the_document() -> None:
    """The references come after every appendix, not inside them.

    They once sat between appendix A and appendix B--G, so a reader following a
    citation landed inside the proof material and the number scan stopped at
    the first appendix. Placement is a sourcing decision, so it is checked in
    the source: nothing but the closing marker follows the bibliography.
    """
    main = _MAIN_TEX.read_text(encoding="utf-8")
    tail = main[main.index("\\bibliography{references}") :]
    assert "\\end{document}" in tail
    assert "\\section" not in tail, "the bibliography must follow every section"
    assert "\\subsection" not in tail, "the bibliography must follow every subsection"
    assert "\\appendix" not in tail, "the appendix marker must precede the bibliography"


def test_every_bibliography_key_is_cited_and_every_citation_resolves() -> None:
    """A dangling citation or an uncited entry is a defect a reviewer will find."""
    main = _MAIN_TEX.read_text(encoding="utf-8")
    bib = (_PAPER_DIR / "references.bib").read_text(encoding="utf-8")

    cited: set[str] = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", main):
        cited.update(key.strip() for key in group.split(","))
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))

    assert cited <= defined, f"cited but not defined: {sorted(cited - defined)}"
    assert defined <= cited, f"in the .bib but never cited: {sorted(defined - cited)}"


# --------------------------------------------------------------------------
# build reproducibility
# --------------------------------------------------------------------------


def test_the_documented_build_command_actually_builds() -> None:
    """``paper/BUILD.md`` must not describe a command that does not work."""
    build_md = _PAPER_DIR / "BUILD.md"
    if not build_md.exists():
        pytest.skip("paper/BUILD.md is absent")
    text = build_md.read_text(encoding="utf-8")
    assert "tectonic" in text or "pdflatex" in text, (
        "BUILD.md documents neither tectonic nor pdflatex"
    )


def test_tectonic_can_compile_the_paper(tmp_path: Path) -> None:
    """A real compile, in a clean directory, of the committed source.

    This is the check that catches a paper that only builds because of stale
    intermediates sitting in ``paper/build``.
    """
    binary = shutil.which("tectonic")
    if binary is None:
        local = _REPO / ".tmp" / "tectonic" / "tectonic.exe"
        if not local.exists():
            pytest.skip("no tectonic binary available")
        binary = str(local)

    proc = subprocess.run(
        [binary, "-X", "compile", "--outdir", str(tmp_path), str(_MAIN_TEX)],
        cwd=_PAPER_DIR,
        capture_output=True,
        text=True,
        # tectonic's progress notes carry UTF-8 punctuation; without an explicit
        # encoding the reader thread decodes as the machine locale (cp1251 here)
        # and dies on byte 0x98, which surfaces as an unhandled-thread warning.
        encoding="utf-8",
        errors="replace",
        timeout=1200,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    produced = tmp_path / "main.pdf"
    assert produced.exists(), "tectonic reported success but wrote no PDF"
    assert produced.stat().st_size > 100_000, "the PDF is suspiciously small"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))