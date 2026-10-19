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

import json
import re
import subprocess
from pathlib import Path

import pytest

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

# Subject of the earliest commit written for the paper. Located by subject rather
# than by hash, because the history was rewritten once to strip the co-author
# trailers and a hard-coded hash would then point at a commit that no longer
# exists - which is exactly how this test failed.
FIRST_PAPER_SUBJECT = "feat(statistics): withdraw the 920x claim"


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=_REPO, capture_output=True, text=True, encoding="utf-8"
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr.strip()}"
    return proc.stdout


def _git_ok(*args: str) -> tuple[bool, str]:
    """Run git, returning (succeeded, output) instead of asserting.

    A shallow clone - which is what actions/checkout produces by default - has
    neither the full history nor a configured identity. That is a fact about the
    checkout, not a defect in the repository, so the history-based checks have to
    be able to say so rather than fail.
    """
    proc = subprocess.run(
        ["git", *args], cwd=_REPO, capture_output=True, text=True, encoding="utf-8"
    )
    return proc.returncode == 0, proc.stdout


def _first_paper_commit() -> str | None:
    """The hash of the earliest commit written for the paper, or None.

    Located by grepping for its subject. Returns None when the history is not
    present, which is what a shallow clone looks like.
    """
    ok, out = _git_ok(
        "log", "--all", "--format=%H%x01%s", "--grep=" + FIRST_PAPER_SUBJECT, "--fixed-strings"
    )
    if not ok or not out.strip():
        return None
    rows = [r for r in out.splitlines() if r.strip()]
    return rows[-1].split("\x01")[0] if rows else None


def _history_is_shallow() -> bool:
    ok, out = _git_ok("rev-list", "--count", "HEAD")
    return (not ok) or int(out.strip() or "0") < 20


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
    if _history_is_shallow():
        pytest.skip(
            f"shallow clone: {len(lines)} commit(s) visible; this check needs the "
            f"full history. CI fetches it, so a skip here means the checkout is "
            f"incomplete, not that the guard passed."
        )
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
    base = _first_paper_commit()
    if base is None:
        pytest.skip(
            "the earliest paper commit is not in this history; the range this "
            "check owns cannot be located"
        )
    out = _git(
        "log", "--format=%h%x01%an%x01%(trailers:key=Co-Authored-By,valueonly)",
        f"{base}^..HEAD",
    )
    rows = [r.split("\x01") for r in out.splitlines() if r.strip()]
    assert len(rows) >= 26, f"expected at least 26 rewritten commits, found {len(rows)}"
    for sha, author, trailers in rows:
        assert author == EXPECTED_AUTHOR, f"{sha} is attributed to {author!r}"
        assert trailers.strip() == "", f"{sha} carries a co-author trailer: {trailers!r}"


def test_the_pre_existing_author_set_is_what_is_recorded() -> None:
    """Pin the owner's earlier history instead of pretending it was cleaned."""
    if _history_is_shallow():
        pytest.skip("shallow clone: the pre-existing commits are not present")
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
    ok, out = _git_ok("config", "user.name")
    if not ok or not out.strip():
        pytest.skip("no git identity configured in this checkout")
    name = out.strip()
    assert name == EXPECTED_AUTHOR, f"user.name is {name!r}; expected {EXPECTED_AUTHOR!r}"


# ---------------------------------------------------------------------------
# local filesystem paths inside tracked files
# ---------------------------------------------------------------------------


def test_published_source_fields_are_repo_relative() -> None:
    """A published artifact cites its source in repository terms, not home terms.

    ``results/real_model.json`` shipped ``published_source`` with the
    machine-local absolute path of ``results/measurements.json`` - the owner's
    home directory, Cyrillic username included, in all three records and
    therefore in every commit published so far. It survived the authorship
    audits above because it is stored as JSON ``\\u`` escapes, decrypting only
    at the reader. Real location: ``real_model.py:1316`` wrote ``str(path)``.
    """
    artifact = json.loads(
        (_REPO / "results" / "real_model.json").read_text(encoding="utf-8")
    )

    sources: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "published_source" and isinstance(value, str):
                    sources.append(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(artifact)
    assert len(sources) == 3, f"expected 3 published_source records, found {len(sources)}"
    for source in sources:
        assert source == "results/measurements.json", (
            f"published_source is not a repository-relative path: {source!r}"
        )


def test_no_tracked_file_embeds_a_local_home_path() -> None:
    """No tracked file may embed a local home directory.

    The needle is assembled at runtime so that this file does not itself
    contain the byte sequence it hunts: ``"Users"`` and the separator live on
    opposite sides of an operator, and the guard cannot become its own false
    positive (which is exactly what its first revision was).
    """
    tracked = [line for line in _git("ls-files").splitlines() if line.strip()]
    assert tracked, "git ls-files returned nothing; is this a checkout?"
    needle = b"Users" + b"\\"
    for rel in tracked:
        blob = (_REPO / rel).read_bytes()
        assert needle not in blob, f"{rel} embeds a local home path"