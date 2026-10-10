"""Tests for repository authorship metadata.

These tests exist because of one specific failure mode that happened 26 times
before anyone noticed: the paper has a single author, and an automated helper
attached a co-author trailer to every commit. That is not cosmetic. A contributor
line is a claim about who is responsible for the result, and it was wrong in the
git history of the very repository asserting it.

The rule enforced here is narrower than "no automated assistance", which would be
unverifiable. It is: the repository's metadata files name exactly one author, and
the git record attributes the work to nobody else. What is checked is the record,
not the process.

Commits that predate the paper's reconstruction are deliberately not rewritten.
They are the owner's own earlier history, and `PRE_EXISTING_AUTHORS` pins what is
there so a *new* name appearing is caught rather than absorbed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

EXPECTED_AUTHOR = "Slyatski Ilya"

# AI assistants that were attached as co-authors and had to be stripped. Matched
# case-insensitively, since a trailer may be written "Co-Authored-By" or
# "co-authored-by".
FORBIDDEN_ASSISTANTS = ("Claude", "Anthropic", "OpenAI", "Gemini", "Copilot")

# Author names in commits predating this work. Recorded, not forbidden: the owner
# left them alone deliberately, and rewriting them was not authorised.
PRE_EXISTING_AUTHORS = frozenset(
    {"Slyatski Ilya", "fanat503", "Vitebsk Research", "Arena Student"}
)

# The earliest commit that predates the paper's reconstruction. Bounds the range
# whose authorship is asserted rather than the whole history.
FIRST_PAPER_COMMIT = "27698cb"


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=_REPO, capture_output=True, text=True, encoding="utf-8"
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr.strip()}"
    return proc.stdout


# ---------------------------------------------------------------------------
# the metadata files
# ---------------------------------------------------------------------------


def test_the_paper_names_exactly_one_author() -> None:
    """``\\author`` is a formal attribution, so it is pinned exactly."""
    tex = (_REPO / "paper" / "main.tex").read_text(encoding="utf-8")
    assert "\\author{Ilya Slyatski}" in tex
    # No second \\author, no \\author with an "and", no \\thanks credit.
    assert tex.count("\\author{") == 1, tex.count("\\author{")
    for other in FORBIDDEN_ASSISTANTS:
        assert other.lower() not in tex.lower(), f"paper/main.tex names {other!r}"


def test_citation_metadata_names_exactly_one_author() -> None:
    cff = (_REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r'family-names:\s*"?Slyatski"?', cff), (
        "CITATION.cff must name the author"
    )
    assert re.search(r'given-names:\s*"?Ilya"?', cff), "CITATION.cff must name the given name"
    # Two author blocks belong to this repository: its own, and its
    # preferred-citation. Confined to the part of the file before the
    # `references` list, which cites other papers and legitimately names many
    # authors - counting the whole file reports twelve and rejects a correct
    # file, which is what the first version of this check did.
    head = cff.split("references:")[0]
    assert head.count("family-names:") == 2, (
        "CITATION.cff should name the author twice: the repository itself and"
        " its preferred-citation"
    )
    assert head.count("Slyatski") == 2, (
        "both author blocks before the references must name Slyatski"
    )
    for other in FORBIDDEN_ASSISTANTS:
        assert other.lower() not in cff.lower(), f"CITATION.cff names {other!r}"


def test_the_license_names_the_author_and_nobody_else() -> None:
    text = (_REPO / "LICENSE").read_text(encoding="utf-8", errors="ignore")
    assert "Copyright 2026 Ilya Slyatski" in text
    for other in FORBIDDEN_ASSISTANTS:
        assert other.lower() not in text.lower(), f"LICENSE names {other!r}"


def test_pyproject_names_the_author_and_nobody_else() -> None:
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'authors = [{ name = "Ilya Slyatski" }]' in text
    assert text.count('name = "Ilya Slyatski"') == 1
    for other in FORBIDDEN_ASSISTANTS:
        assert other.lower() not in text.lower(), f"pyproject.toml names {other!r}"


# ---------------------------------------------------------------------------
# the git record
# ---------------------------------------------------------------------------


def test_no_ref_carries_a_co_author_trailer() -> None:
    """The exact defect, checked as a trailer rather than as a name.

    A co-author trailer is the specific thing that was added without
    authorisation, so asserting the trailers are empty is a stronger and more
    local statement than asserting the names are absent.
    """
    out = _git(
        "log", "--all", "--format=%H%x01%(trailers:key=Co-Authored-By,valueonly,separator=;)"
    )
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) > 20, f"only {len(lines)} commits scanned; the check is vacuous"
    for line in lines:
        sha, _sep, trailers = line.partition("\x01")
        assert trailers.strip() == "", (
            f"commit {sha[:10]} carries a co-author trailer: {trailers!r}"
        )


def test_no_ref_attributes_the_work_to_an_ai_assistant() -> None:
    """Every assistant that ever appeared here must be gone from every ref.

    ``--all`` rather than ``HEAD``, because the contamination reached three
    branches before it was caught, and one branch is easy to forget.
    """
    for fmt in ("%an%x01%ae", "%cn%x01%ce", "%(trailers:key=Co-Authored-By,valueonly)"):
        out = _git("log", "--all", "--format=" + fmt)
        for line in out.splitlines():
            for assistant in FORBIDDEN_ASSISTANTS:
                assert assistant.lower() not in line.lower(), (
                    f"an author or co-author field still names {assistant!r}: {line!r}"
                )


def test_the_commits_written_for_the_paper_name_only_the_author() -> None:
    """Of the commits made for the paper, every one names one author."""
    out = _git(
        "log", "--format=%h%x01%an%x01%(trailers:key=Co-Authored-By,valueonly)",
        f"{FIRST_PAPER_COMMIT}^..HEAD",
    )
    rows = [r.split("\x01") for r in out.splitlines() if r.strip()]
    assert len(rows) >= 26, f"expected at least 26 rewritten commits, found {len(rows)}"
    for sha, author, trailers in rows:
        assert author == EXPECTED_AUTHOR, f"{sha} is attributed to {author!r}"
        assert trailers.strip() == "", f"{sha} carries a co-author trailer: {trailers!r}"


def test_the_pre_existing_author_set_is_what_is_recorded() -> None:
    """Pin the owner's earlier history instead of pretending it was cleaned."""
    out = _git("log", "--all", "--format=%an")
    authors = {line.strip() for line in out.splitlines() if line.strip()}
    assert authors == set(PRE_EXISTING_AUTHORS), (
        f"the author set changed; it was {sorted(PRE_EXISTING_AUTHORS)}, "
        f"it is now {sorted(authors)}"
    )


def test_the_local_identity_produces_the_right_attribution() -> None:
    """The next ``git commit`` must not re-introduce a different name.

    Without this the history is correct only until someone commits again, which
    is how the handle reappeared after the name was already fixed in the paper.
    """
    name = _git("config", "user.name").strip()
    assert name == EXPECTED_AUTHOR, f"user.name is {name!r}; expected {EXPECTED_AUTHOR!r}"