"""Read a training run's JSONL records and write a portable report.

Why this exists
---------------
``train.py`` logs one JSON object per line and a closing ``summary.json``. That
is the right format for a run, but not a useful thing to look at: a reader who
wants to know whether a loss curve converged, which learning rate was used, or
what the attribution diagnostics did over training should not have to open the
JSONL by hand.

This module is the viewer. It validates the record stream and writes two files
next to the run by default:

* ``report/curves.csv`` -- the scalar series, one row per evaluation, ready for
  a spreadsheet or a plotting script;
* ``report/report.md`` -- a compact human-readable table and the final
  diagnostics;
* optionally, a self-contained HTML report via ``--html PATH``.

No new dependencies, and no plotting side effects: the CSV is the data, and the
figures under ``figures/`` are generated separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from html import escape
from pathlib import Path
from typing import Any

#: The columns a reader always wants, in reading order. Diagnostics that
#: ``train.py`` adds to an eval record are appended after these.
CORE_COLUMNS: tuple[str, ...] = (
    "step",
    "lr",
    "grad_norm",
    "train_loss_ema",
    "val_loss",
    "tokens_seen",
    "tok_per_s",
)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load and validate a ``metrics.jsonl`` file.

    A malformed *interior* line is an error, not something to skip: a report
    that silently drops records is worse than no report. The final line is the
    exception a crashed run actually leaves behind -- a prefix of a JSON object
    with no closing brace. That tail is dropped so an interrupted run can still
    be inspected; a malformed last line that does not look like a truncated
    object is still refused.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no metrics file at {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    last_nonempty = max(
        (index for index, line in enumerate(lines) if line.strip()), default=-1
    )
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, 1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            looks_truncated = text.startswith("{") and not text.endswith("}")
            if lineno - 1 == last_nonempty and looks_truncated:
                break
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        records.append(record)
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def split_records(
    records: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """Return ``(run_start, evals, summary)`` and refuse an ambiguous stream.

    A run directory is wiped at the start of a run, so more than one
    ``run_start`` means the file has been concatenated by hand. Reporting the
    first run and ignoring the second would be a silent lie about the series.
    """
    starts = [r for r in records if r.get("event") == "run_start"]
    evals = [r for r in records if r.get("event") == "eval"]
    summaries = [r for r in records if r.get("event") == "summary"]
    if len(starts) != 1:
        raise ValueError(f"expected exactly one run_start record, found {len(starts)}")
    if len(summaries) > 1:
        raise ValueError(f"expected at most one summary record, found {len(summaries)}")
    return starts[0], evals, (summaries[0] if summaries else None)


def scalar_columns(evals: Iterable[dict[str, Any]]) -> list[str]:
    """Core columns first, then any additional numeric diagnostics in order."""
    columns = list(CORE_COLUMNS)
    for record in evals:
        for key, value in record.items():
            if key == "event" or key in columns:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            columns.append(key)
    return columns


def write_csv(
    evals: Sequence[dict[str, Any]], path: Path, columns: Sequence[str]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for record in evals:
            writer.writerow({key: record.get(key, "") for key in columns})
    return path


def _value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(
    start: dict[str, Any],
    evals: Sequence[dict[str, Any]],
    summary: dict[str, Any] | None,
    columns: Sequence[str],
) -> str:
    config = start.get("config", {})
    lines = [
        f"# Training run: {start.get('config_name', 'unknown')}",
        "",
        f"- parameters: {_value(start.get('n_params'))}",
        f"- device: {_value(start.get('device'))}",
        f"- seed: {_value(start.get('seed'))}",
        f"- tokens available: {_value(start.get('tokens_total'))}",
        f"- evaluations: {len(evals)}",
    ]
    if summary is not None:
        lines.extend(
            [
                f"- steps completed: {_value(summary.get('steps'))}",
                f"- final train loss EMA: {_value(summary.get('final_train_loss_ema'))}",
            ]
        )
    lines.extend(["", "## Configuration", ""])
    if config:
        lines.append("| key | value |")
        lines.append("| --- | --- |")
        for key in sorted(config):
            lines.append(f"| `{key}` | {_value(config[key])} |")
    else:
        lines.append("No configuration record was written.")

    lines.extend(["", "## Curves", ""])
    if evals:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for record in evals:
            lines.append(
                "| " + " | ".join(_value(record.get(key)) for key in columns) + " |"
            )
    else:
        lines.append("No evaluation records were written.")

    if evals:
        final = evals[-1]
        diagnostics = {
            key: value
            for key, value in final.items()
            if key != "event" and key not in CORE_COLUMNS and not isinstance(value, (dict, list))
        }
        if diagnostics:
            lines.extend(["", "## Final diagnostics", "", "| key | value |", "| --- | --- |"])
            for key in sorted(diagnostics):
                lines.append(f"| `{key}` | {_value(diagnostics[key])} |")
    lines.append("")
    return "\n".join(lines)


def render_html(
    start: dict[str, Any],
    evals: Sequence[dict[str, Any]],
    summary: dict[str, Any] | None,
    columns: Sequence[str],
) -> str:
    """A self-contained HTML view: no scripts, no external assets."""
    title = escape(str(start.get("config_name", "training run")))
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{escape(_value(record.get(column)))}</td>" for column in columns)
        + "</tr>"
        for record in evals
    )
    config = start.get("config", {})
    config_rows = "".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(_value(value))}</td></tr>"
        for key, value in sorted(config.items())
    )
    summary_line = ""
    if summary is not None:
        summary_line = (
            f"<p>steps: {escape(_value(summary.get('steps')))}; "
            f"final train loss EMA: {escape(_value(summary.get('final_train_loss_ema')))}</p>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Training run: {title}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #17202a; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccd1d1; padding: .3rem .5rem; text-align: right; }}
th {{ background: #f4f6f7; }}
td:first-child, th:first-child {{ text-align: left; }}
code {{ background: #f4f6f7; padding: .1rem .25rem; }}
</style>
</head>
<body>
<h1>Training run: {title}</h1>
<p>parameters: {escape(_value(start.get('n_params')))}; device: {escape(_value(start.get('device')))}; seed: {escape(_value(start.get('seed')))}</p>
{summary_line}
<h2>Configuration</h2>
<table>{config_rows}</table>
<h2>Curves</h2>
<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>
</body>
</html>
"""


def write_report(
    run_dir: Path,
    out_dir: Path | None = None,
    html_path: Path | None = None,
) -> dict[str, Any]:
    """Write ``curves.csv`` and ``report.md``, and optionally an HTML view."""
    run_dir = Path(run_dir)
    records = load_records(run_dir / "metrics.jsonl")
    start, evals, summary = split_records(records)
    columns = scalar_columns(evals)
    destination = Path(out_dir) if out_dir is not None else run_dir / "report"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(evals, destination / "curves.csv", columns)
    markdown_path = destination / "report.md"
    markdown_path.write_text(
        render_markdown(start, evals, summary, columns), encoding="utf-8"
    )
    html_result = None
    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(
            render_html(start, evals, summary, columns), encoding="utf-8"
        )
        html_result = str(html_path)
    return {
        "run_dir": str(run_dir),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "html": html_result,
        "n_evals": len(evals),
        "columns": columns,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="directory containing metrics.jsonl")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: RUN_DIR/report)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="also write a self-contained HTML report to this path",
    )
    args = parser.parse_args(argv)
    result = write_report(args.run_dir, args.out_dir, args.html)
    print(f"wrote {result['csv']} ({result['n_evals']} evaluations)")
    print(f"wrote {result['markdown']}")
    if result["html"]:
        print(f"wrote {result['html']}")


if __name__ == "__main__":
    main()
