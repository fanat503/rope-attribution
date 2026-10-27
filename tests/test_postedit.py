"""Tests for the post-edit diagnostics' changed-path discovery.

The parser is the part that can silently skip a file: a rename used to appear
as the literal string ``old -> new`` and a quoted non-ASCII path was returned
with Git's escapes intact, so the changed file was never checked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import postedit  # noqa: E402  (needs the sys.path tweak above)


def test_parse_porcelain_reads_verbatim_paths_and_renames() -> None:
    raw = "R  new.py\0old.py\0 M caf\u00e9.py\0?? added.py\0"
    assert postedit._parse_porcelain(raw) == ["new.py", "caf\u00e9.py", "added.py"]


def test_changed_paths_survives_a_staged_rename(tmp_path, monkeypatch) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "old.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "old.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "mv", "old.py", "new.py"], cwd=tmp_path, check=True)
    monkeypatch.setattr(postedit, "_REPO", tmp_path)
    assert postedit._changed_paths([]) == [tmp_path / "new.py"]
