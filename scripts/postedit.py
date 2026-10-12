#!/usr/bin/env python
"""Post-edit diagnostics: the checks that must run immediately after an edit.

Why this exists
---------------
Fan et al. (arXiv:2609.20804) found that running a fast, read-only lint right
after each edit - and appending the findings to the same tool result - is what
keeps a defect from costing a whole turn to discover. Their version runs ruff,
pyflakes, or a syntax-only fallback on the file just written.

In this repository the counterpart ran at the *end* of a batch, and the cost was
paid: a ``NameError`` sat in ``figures.py`` through two commits, because nothing
checked the module until much later. This script is the fix. It is deliberately
small and has no dependencies beyond ruff, which is already a dev requirement.

What it does
------------
Given paths (or, with no arguments, everything git reports as modified), it:

1. checks each ``.py`` file parses at all, and reports a syntax error with the
   line and the offending source line, which is the failure mode that a lint
   tool cannot even start on;
2. runs ruff over exactly those files, not the whole tree, so the output is
   about what just changed;
3. runs the tests whose names correspond to a changed module, so a behavioural
   break is caught now rather than at ``make all``.

It returns non-zero if any stage fails, so ``make postedit`` stops a commit.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# A changed module maps to the test file that owns it. Only the modules with a
# dedicated test file are listed; everything else is covered by the suite.
_MODULE_TESTS = {
    "rope": "tests/test_rope.py",
    "experiments": "tests/test_experiments.py",
    "statistics": "tests/test_statistics.py",
    "figures": "tests/test_figures.py",
    "real_model": "tests/test_real_model.py",
    "usefulness": "tests/test_usefulness.py",
    "uncertainty": "tests/test_uncertainty.py",
}


def _syntax_check(path: Path) -> list[str]:
    """Report a syntax error with its line and text, or nothing."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{path}: cannot be read ({exc})"]
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        lines = source.splitlines()
        lineno = exc.lineno or 0
        text = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        return [
            f"{path}:{lineno}: {exc.msg}",
            f"    {text.strip()[:120]}",
        ]
    return []


def _changed_paths(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(a) for a in explicit]
    proc = subprocess.run(
        ["git", "status", "--porcelain"], cwd=_REPO, capture_output=True,
        text=True, encoding="utf-8",
    )
    paths = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        candidate = line[3:].strip()
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1]
        paths.append(_REPO / candidate)
    return [p for p in paths if p.exists() and p.suffix == ".py"]


def _tests_for(paths: list[Path]) -> list[str]:
    tests = {"tests/test_authorship.py"}
    for path in paths:
        stem = path.stem
        if stem in _MODULE_TESTS:
            tests.add(_MODULE_TESTS[stem])
    # anything that asserts on the paper or its figures must run with the paper
    for path in paths:
        text = path.as_posix()
        if "rope_attribution" in text:
            tests.add("tests/test_paper_claims.py")
    return sorted(t for t in tests if (_REPO / t).exists())


def main(argv: list[str]) -> int:
    paths = _changed_paths(argv)
    if not paths:
        print("no changed Python files; nothing to check")
        return 0

    print(f"post-edit diagnostics over {len(paths)} changed file(s)")
    failures = 0

    for path in paths:
        problems = _syntax_check(path)
        if problems:
            failures += 1
            for line in problems:
                print(f"  SYNTAX {line}")
        else:
            print(f"  parses  {path.relative_to(_REPO).as_posix()}")

    rel = [p.relative_to(_REPO).as_posix() for p in paths]
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *rel], cwd=_REPO,
        capture_output=True, text=True, encoding="utf-8",
    )
    if ruff.returncode != 0:
        failures += 1
        print("  ruff:")
        for line in ruff.stdout.splitlines():
            print(f"    {line}")
    else:
        print("  ruff    clean")

    tests = _tests_for(paths)
    if tests:
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *tests], cwd=_REPO,
            capture_output=True, text=True, encoding="utf-8",
        )
        tail = [l for l in run.stdout.splitlines() if l.strip()][-1:] or [""]
        if run.returncode != 0:
            failures += 1
            print("  pytest  FAILED")
            print(f"    {tail[0].strip()[:160]}")
            for line in run.stdout.splitlines():
                if line.startswith("FAILED"):
                    print(f"    {line.strip()[:160]}")
        else:
            print(f"  pytest  {tail[0].strip()[:120]} ({len(tests)} file(s))")

    print()
    if failures:
        print(f"FAILED: {failures} stage(s); fix before committing")
        return 1
    print("all stages clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
