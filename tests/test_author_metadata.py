"""The author's name must agree everywhere it appears, and actually be present.

Why this file exists
--------------------
The author name was, until recently, deliberately absent from all three places it
belongs: ``paper/main.tex``, ``CITATION.cff`` and the ``LICENSE`` appendix. That
was the right call while it was unknown -- a wrong attribution in a legal
document is worse than a missing one that fails validation loudly -- but it left
nothing to stop the three from drifting apart once it was filled in.

These tests are therefore not about the name itself. They are about the
*consistency* of the three, and they fail if any one of them goes back to being a
placeholder. The name is read from the CFF and compared against the other two, so
correcting one file is never enough.

``paper/main.tex`` is checked for an author command with a non-empty argument
rather than for a literal string, so that LaTeX markup inside the argument does
not have to be reproduced here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CFF = _REPO / "CITATION.cff"
_LICENSE = _REPO / "LICENSE"
_MAIN_TEX = _REPO / "paper" / "main.tex"

EXPECTED_GIVEN = "Ilya"
EXPECTED_FAMILY = "Slyatski"


@pytest.fixture(scope="module")
def cff_text() -> str:
    return _CFF.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cff_authors(cff_text: str) -> list[tuple[str, str]]:
    """Every ``(family-names, given-names)`` pair under a top-level ``authors:``."""
    match = re.search(r"^authors:\s*\n((?:[ \t]+.*\n|\s*\n)*)", cff_text, re.MULTILINE)
    assert match, "CITATION.cff has no top-level `authors:` block"
    body = match.group(1)
    pairs = []
    for chunk in re.split(r"^\s*-\s+", body, flags=re.MULTILINE)[1:]:
        family = re.search(r'family-names:\s*"([^"]+)"', chunk)
        given = re.search(r'given-names:\s*"([^"]+)"', chunk)
        if family and given:
            pairs.append((family.group(1), given.group(1)))
    return pairs


def test_the_author_is_not_a_placeholder_in_the_cff(cff_authors: list[tuple[str, str]]) -> None:
    assert cff_authors, "CITATION.cff lists no author; `authors` is a required field"
    for family, given in cff_authors:
        assert "<" not in family and ">" not in family, (family, given)
        assert "<" not in given and ">" not in given, (family, given)


def test_the_cff_names_exactly_one_author(cff_authors: list[tuple[str, str]]) -> None:
    assert len(cff_authors) == 1, f"expected a single author, found {cff_authors}"
    assert cff_authors[0] == (EXPECTED_FAMILY, EXPECTED_GIVEN), cff_authors[0]


def test_the_license_names_the_same_holder() -> None:
    text = _LICENSE.read_text(encoding="utf-8")
    line = re.search(r"^\s*Copyright (.+)$", text, re.MULTILINE)
    assert line, "the LICENSE appendix carries no Copyright line"
    holder = line.group(1).strip()
    assert "[yyyy]" not in holder and "[name of copyright owner]" not in holder, (
        f"the LICENSE copyright line is still the unfilled boilerplate: {holder!r}"
    )
    assert EXPECTED_GIVEN in holder and EXPECTED_FAMILY in holder, holder
    assert holder.split()[0].isdigit(), f"the LICENSE line needs a year: {holder!r}"


def test_the_paper_names_the_same_author() -> None:
    tex = _MAIN_TEX.read_text(encoding="utf-8")
    match = re.search(r"\\author\{(.+?)\}", tex, re.S)
    assert match, "paper/main.tex has no \\author{...}"
    author = " ".join(match.group(1).split())
    assert author, "paper/main.tex has an empty \\author{}"
    assert EXPECTED_GIVEN in author and EXPECTED_FAMILY in author, author


def test_the_three_sources_do_not_disagree(cff_authors: list[tuple[str, str]]) -> None:
    """The single test that would catch one file being corrected on its own."""
    family, given = cff_authors[0]
    tex_author = " ".join(re.search(r"\\author\{(.+?)\}", _MAIN_TEX.read_text(encoding="utf-8"), re.S).group(1).split())
    license_holder = re.search(
        r"^\s*Copyright (.+)$", _LICENSE.read_text(encoding="utf-8"), re.MULTILINE
    ).group(1)

    for label, text in (("paper/main.tex", tex_author), ("LICENSE", license_holder)):
        assert given in text and family in text, (
            f"{label} says {text!r}, but CITATION.cff says {given} {family}"
        )


def test_the_preferred_citation_credits_the_same_author(cff_text: str) -> None:
    """The article entry must not be left crediting nobody, or a different person."""
    block = re.search(
        r"^preferred-citation:\s*\n((?:[ \t]+.*\n|\s*\n)*)", cff_text, re.MULTILINE
    )
    assert block, "CITATION.cff has no preferred-citation block"
    body = block.group(1)
    assert f'family-names: "{EXPECTED_FAMILY}"' in body, (
        "preferred-citation does not credit the author"
    )
    assert f'given-names: "{EXPECTED_GIVEN}"' in body