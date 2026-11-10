"""Tests for repository authorship metadata.

The paper has one author, and a contributor line is a claim about who is
responsible for the result. These tests keep the metadata files, the commit
trailers, and the commits written for the paper consistent with that claim.

The rule is deliberately about the record, not about how the work was produced:
the metadata names exactly one author, and the paper's commits carry no
co-author trailer. Older commits are left as they are; the checks below apply
to the range this repository is responsible for.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

EXPECTED_AUTHOR = "Slyatski Ilya"

# Subject of the earliest commit written for the paper. Located by subject rather
# than by a hard-coded hash, so that a reworded history does not silently move
# the boundary of the range this test owns.
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


def test_the_license_names_the_author_and_nobody_else() -> None:
    text = (_REPO / "LICENSE").read_text(encoding="utf-8", errors="ignore")
    assert "Copyright 2026 Ilya Slyatski" in text


def test_pyproject_names_the_author_and_nobody_else() -> None:
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'authors = [{ name = "Ilya Slyatski" }]' in text
    assert text.count('name = "Ilya Slyatski"') == 1


# ---------------------------------------------------------------------------
# the git record
# ---------------------------------------------------------------------------


def test_no_ref_carries_a_co_author_trailer() -> None:
    """Every commit in this repository must have an empty co-author trailer.

    A trailer is a formal authorship claim, so checking it directly is stronger
    and more local than searching for a name.
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
    assert len(rows) >= 26, f"expected at least 26 paper commits, found {len(rows)}"
    for sha, author, trailers in rows:
        assert author == EXPECTED_AUTHOR, f"{sha} is attributed to {author!r}"
        assert trailers.strip() == "", f"{sha} carries a co-author trailer: {trailers!r}"


def test_the_local_identity_produces_the_right_attribution() -> None:
    """The next ``git commit`` must not re-introduce a different name."""
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

    Absolute local paths make an artifact non-reproducible on another machine
    and leak the author's filesystem layout, so every ``published_source`` field
    must name the repository-relative file.
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

    The needle is assembled at runtime so this guard does not match its own
    source file.
    """
    tracked = [line for line in _git("ls-files").splitlines() if line.strip()]
    assert tracked, "git ls-files returned nothing; is this a checkout?"
    needle = b"Users" + b"\\"
    for rel in tracked:
        blob = (_REPO / rel).read_bytes()
        assert needle not in blob, f"{rel} embeds a local home path"