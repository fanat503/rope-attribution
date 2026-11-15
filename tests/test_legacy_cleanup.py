"""Tombstones for the pre-paper archive removed on 2026-10-17.

Why these tests exist
---------------------
An audit of the pre-paper tree recommended removing three classes of file, which
was done on 2026-10-17: machine-readable, results-shaped JSON whose contents were
typed-in illustrative constants; near-duplicate "final" status documents that a
reader could mistake for a result; and process documents that operated against a
directory which does not exist. The rest of that archive was removed from the
working tree in October 2026 and remains in git history at tag
`pre-legacy-cleanup-2026-10-20`.

The check runs against `git ls-files`, not the filesystem, and the difference
matters. A local regeneration of a removed file is not the hazard: a *tracked*
file is, because it is what a reader of the repository would find and could cite.

The patterns are deliberately broad. They name the *classes* the audit removed,
not the individual filenames, so a renamed copy of the same hazard is caught too.

Mutation verdict: track any file matching one of the patterns below (for
example `git add -f projects/settings.json`) and the corresponding test fails
naming that path; `git rm --cached` it again and the test passes. No maintained
code path produces such a file, so the same verdict holds for the rest of each
class.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# One pattern per class of removed file. The patterns are the contract: they are
# intentionally broader than the removed names so that a renamed copy is caught.
_FORBIDDEN_PATTERNS = (
    "projects/*settings*.json",
    "projects/*FINAL*.md",
    "projects/*COMMIT-SAFE*",
    "projects/*PUSH-INSTRUCTION*",
)


def _tracked(pattern: str) -> list[str]:
    if not (_REPO / ".git").exists():
        pytest.skip("not a git checkout; the tombstone needs git ls-files")
    proc = subprocess.run(
        ["git", "ls-files", "--", pattern],
        cwd=_REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, (
        f"git ls-files failed ({proc.returncode}); the tombstone cannot be "
        f"evaluated: {proc.stderr.strip()}"
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def test_the_tombstone_patterns_are_not_silently_shrunk() -> None:
    """Guard: the pattern list is the contract; shrinking it would mute checks."""
    assert len(_FORBIDDEN_PATTERNS) == 4, (
        f"expected 4 tombstoned classes, found {len(_FORBIDDEN_PATTERNS)}; if a "
        f"class was intentionally restored, remove its pattern and record why in "
        f"the commit message"
    )


@pytest.mark.parametrize("pattern", _FORBIDDEN_PATTERNS)
def test_no_tracked_file_recreates_a_removed_class(pattern: str) -> None:
    tracked = _tracked(pattern)
    assert not tracked, (
        f"{tracked!r} matches {pattern!r} and is tracked again, but that class "
        f"was removed on 2026-10-17 as part of the legacy cleanup (history at tag "
        f"pre-legacy-cleanup-2026-10-20). Re-adding it re-creates the hazard the "
        f"cleanup removed: machine-readable fake results, or a process document "
        f"that asserts them."
    )
