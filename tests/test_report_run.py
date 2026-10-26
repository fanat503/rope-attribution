"""Tests for the training-run report viewer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from rope_attribution import report_run as R


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _write_run(path: Path, *, summary: bool = True) -> Path:
    run = path / "run"
    run.mkdir()
    records = [
        {
            "event": "run_start",
            "config_name": "smoke",
            "config": {"dim": 64, "n_layers": 2},
            "n_params": 12345,
            "device": "cpu",
            "seed": 7,
            "tokens_total": 4096,
        },
        {
            "event": "eval",
            "step": 4,
            "lr": 1e-3,
            "grad_norm": 1.5,
            "train_loss_ema": 3.25,
            "val_loss": 3.5,
            "tokens_seen": 1024,
            "tok_per_s": 900.0,
            "sign_flip_fraction": 0.21,
        },
        {
            "event": "eval",
            "step": 8,
            "lr": 5e-4,
            "grad_norm": 1.1,
            "train_loss_ema": 2.75,
            "val_loss": 3.0,
            "tokens_seen": 2048,
            "tok_per_s": 950.0,
            "sign_flip_fraction": 0.19,
        },
    ]
    if summary:
        records.append(
            {
                "event": "summary",
                "config_name": "smoke",
                "n_params": 12345,
                "steps": 8,
                "final_train_loss_ema": 2.75,
                "device": "cpu",
            }
        )
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    if summary:
        (run / "summary.json").write_text(json.dumps(records[-1]), encoding="utf-8")
    return run


def test_load_records_refuses_a_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"event": "run_start"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"metrics\.jsonl:2"):
        R.load_records(path)


def test_load_records_tolerates_only_a_truncated_final_object(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"event": "run_start"}\n'
        '{"event": "eval", "step": 4}\n'
        '{"event": "eval", "step": 8, "val_loss": 3.',
        encoding="utf-8",
    )
    records = R.load_records(path)
    assert [record.get("step") for record in records] == [None, 4]


def test_load_records_refuses_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        R.load_records(path)


def test_missing_metrics_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="metrics.jsonl"):
        R.load_records(tmp_path / "metrics.jsonl")


def test_split_records_uses_the_first_start_and_the_last_summary() -> None:
    """Legacy and resumed files carry several starts/summaries; that is the run.

    A file written by an older trainer has one ``run_start`` per resume, and a
    resumed file has a ``summary`` from every segment that ended. The report
    describes the run as a whole, so the first start (the original
    configuration) and the last summary (the final state) are the honest
    choice; reporting the first summary would stop the story at step 4.
    """
    records = [
        {"event": "run_start", "config_name": "a"},
        {"event": "run_start", "config_name": "b"},
        {"event": "eval", "step": 4},
        {"event": "eval", "step": 8},
        {"event": "summary", "steps": 4},
        {"event": "summary", "steps": 8},
    ]
    start, evals, summary = R.split_records(records)
    assert start["config_name"] == "a"
    assert [record["step"] for record in evals] == [4, 8]
    assert summary is not None and summary["steps"] == 8


def test_split_records_refuses_a_file_without_a_start() -> None:
    with pytest.raises(ValueError, match="run_start"):
        R.split_records([{"event": "eval", "step": 1}])


def test_a_resumed_metrics_file_reports_the_whole_run(tmp_path: Path) -> None:
    """The records a resumed run actually writes must be reportable."""
    run = tmp_path / "run"
    run.mkdir()
    base = {"config_name": "smoke", "config": {"dim": 64}}
    _write_jsonl(
        run / "metrics.jsonl",
        [
            {"event": "run_start", **base, "device": "cpu", "seed": 7, "n_params": 10},
            {"event": "eval", "step": 4, "val_loss": 3.5, "train_loss_ema": 3.25},
            {"event": "summary", "steps": 4},
            {"event": "resume", "resumed_from_step": 4, "global_batch_size": 4},
            {"event": "eval", "step": 8, "val_loss": 3.0, "train_loss_ema": 2.75},
            {"event": "summary", "steps": 8},
        ],
    )
    result = R.write_report(run)
    assert result["n_evals"] == 2
    report = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert "steps completed: 8" in report
    assert "steps completed: 4" not in report
    rows = list(csv.DictReader((run / "report" / "curves.csv").open(encoding="utf-8")))
    assert [row["step"] for row in rows] == ["4", "8"]


def test_step_and_checkpoint_records_stay_out_of_the_eval_table(tmp_path: Path) -> None:
    """Telemetry answers a different question and belongs in its own section.

    Mutation verdict: filtering by ``event == "eval"`` is what keeps the step
    rows out of ``curves.csv``; a report that treats every record as an eval
    would list steps 2, 4, 6, 8 in the curves table and print ``—`` for
    ``val_loss``.
    """
    run = tmp_path / "run"
    run.mkdir()
    _write_jsonl(
        run / "metrics.jsonl",
        [
            {"event": "run_start", "config_name": "smoke", "config": {},
             "n_params": 10, "device": "cpu", "seed": 7},
            {"event": "step", "step": 4, "lr": 1e-3, "train_loss": 3.3,
             "grad_norm": 1.5, "tokens_seen": 1024,
             "interval_tok_per_s": 900.0, "interval_s": 1.1, "elapsed_s": 1.1},
            {"event": "checkpoint", "step": 4, "path": "checkpoint_resume.pt",
             "tokens_seen": 1024},
            {"event": "eval", "step": 8, "val_loss": 3.0, "train_loss_ema": 2.75},
            {"event": "step", "step": 8, "lr": 5e-4, "train_loss": 2.9,
             "grad_norm": 1.1, "tokens_seen": 2048,
             "interval_tok_per_s": 950.0, "interval_s": 1.0, "elapsed_s": 2.1},
            {"event": "summary", "steps": 8},
        ],
    )
    result = R.write_report(run, html_path=run / "report.html")
    assert result["n_evals"] == 1
    curves = list(
        csv.DictReader((run / "report" / "curves.csv").open(encoding="utf-8"))
    )
    assert [row["step"] for row in curves] == ["8"]

    assert result["steps_csv"] is not None
    with Path(result["steps_csv"]).open(encoding="utf-8") as handle:
        step_rows = list(csv.DictReader(handle))
    assert [row["step"] for row in step_rows] == ["4", "8"]
    assert step_rows[1]["interval_tok_per_s"] == "950.0"
    assert step_rows[1]["elapsed_s"] == "2.1"

    report = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert "## Step telemetry" in report
    assert "`checkpoint_resume.pt`" in report
    html = (run / "report.html").read_text(encoding="utf-8")
    assert "Step telemetry" in html
    assert "950" in html


def test_the_report_contains_every_eval_and_the_diagnostics(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    result = R.write_report(run)
    assert result["n_evals"] == 2
    assert "sign_flip_fraction" in result["columns"]

    rows = list(csv.DictReader((run / "report" / "curves.csv").open(encoding="utf-8")))
    assert [row["step"] for row in rows] == ["4", "8"]
    assert rows[0]["sign_flip_fraction"] == "0.21"
    assert rows[1]["val_loss"] == "3.0"

    report = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert "# Training run: smoke" in report
    assert "| `dim` | 64 |" in report
    assert "final train loss EMA: 2.75" in report
    assert "| `sign_flip_fraction` | 0.19 |" in report


def test_an_unfinished_run_can_still_be_reported(tmp_path: Path) -> None:
    run = _write_run(tmp_path, summary=False)
    result = R.write_report(run)
    assert result["n_evals"] == 2
    report = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert "steps completed" not in report


def test_the_cli_writes_to_an_explicit_directory(tmp_path: Path, capsys) -> None:
    run = _write_run(tmp_path)
    out = tmp_path / "elsewhere"
    R.main([str(run), "--out-dir", str(out)])
    assert (out / "curves.csv").exists()
    assert (out / "report.md").exists()
    assert "wrote" in capsys.readouterr().out


def test_the_cli_writes_a_self_contained_html_view(tmp_path: Path, capsys) -> None:
    run = _write_run(tmp_path)
    html = tmp_path / "report.html"
    R.main([str(run), "--html", str(html)])
    text = html.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "Training run: smoke" in text
    assert "sign_flip_fraction" in text
    assert "wrote" in capsys.readouterr().out
