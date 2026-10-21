"""Tombstones for the legacy-file cleanup executed on 2026-10-17.

Why these tests exist
---------------------
An audit of the pre-paper tree under `projects/frontier-01-*` recommended
removing two classes of file, which was done on 2026-10-17:
`settings.json` / `settings-ideal.json` — machine-readable, results-shaped
objects whose contents were typed-in illustrative constants — and the six
near-duplicate "FINAL" bundles the audit called the documents a reader is most
likely to mistake for a result. Three process documents that operated against a
`projects/bundle/` directory that does not exist were removed with them. The
rest of that archive was removed from the working tree in October 2026; it
remains in git history at tag `pre-legacy-cleanup-2026-10-20`.

The check runs against `git ls-files`, not the filesystem, and the difference
matters. A local regeneration of the two JSON files is not the hazard: a
*tracked* file is, because it is what a reader of the repository would find and
could cite.

Mutation verdict: regenerate `projects/settings.json`, run `git add -f
projects/settings.json`, and `test_removed_legacy_file_stays_untracked` fails
naming that path; `git rm --cached` it again and the test passes. No maintained
code path produces any of the other ten files, so for them the same verdict
holds by construction.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

_REMOVED_RESULT_ARTIFACTS = (
    "projects/settings.json",
    "projects/settings-ideal.json",
)

_REMOVED_FINAL_BUNDLES = (
    "projects/frontier-01-FINAL-V13-ABSOLUTE-IDEAL-ULTIMATE-BEST-POSSIBLE.md",
    "projects/frontier-01-FINAL-V25-ABSOLUTE-IDEAL-ULTIMATE-BEST-POSSIBLE.md",
    "projects/frontier-01-FINAL-V26-ABSOLUTE-IDEAL-ULTIMATE-BEST-POSSIBLE.md",
    "projects/frontier-01-FINAL-V27-ABSOLUTE-IDEAL-ULTIMATE-BEST-POSSIBLE.md",
    "projects/frontier-01-FINAL-ANSWER-IDEAL-v4.md",
    "projects/frontier-01-FINAL-HIGHEST-IDEAL-v3.md",
)

_REMOVED_PROCESS_DOCUMENTS = (
    "projects/COMMIT-SAFE-INSTRUCTION.md",
    "projects/COMMIT-SAFE-CODESPACE-PATCH.md",
    "projects/PUSH-INSTRUCTIONS-FASTEST.md",
)

_REMOVED_CASES = (
    [
        pytest.param(path, id=f"result-artifact:{Path(path).name}")
        for path in _REMOVED_RESULT_ARTIFACTS
    ]
    + [
        pytest.param(path, id=f"final-bundle:{Path(path).name}")
        for path in _REMOVED_FINAL_BUNDLES
    ]
    + [
        pytest.param(path, id=f"process-document:{Path(path).name}")
        for path in _REMOVED_PROCESS_DOCUMENTS
    ]
)


def _tracked(path: str) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "--", path],
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


def test_the_tombstone_list_is_not_silently_shrunk() -> None:
    """Guard: the list is the contract; shrinking it would mute the checks."""
    assert len(_REMOVED_CASES) == 11, (
        f"expected 11 tombstoned paths, found {len(_REMOVED_CASES)}; if a path "
        f"was intentionally restored to the repository, remove its case and "
        f"record why in the commit message"
    )


@pytest.mark.parametrize("path", _REMOVED_CASES)
def test_removed_legacy_file_stays_untracked(path: str) -> None:
    tracked = _tracked(path)
    assert not tracked, (
        f"{tracked!r} is tracked again, but it was removed on 2026-10-17 as "
        f"part of the legacy cleanup (history at tag "
        f"pre-legacy-cleanup-2026-10-20). Re-adding it re-creates the hazard "
        f"the cleanup removed: machine-readable fake results, or a procedure "
        f"addressed to an agent that asserts them."
    )
