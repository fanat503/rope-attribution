"""Tests for the training-run report viewer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from rope_attribution import report_run as R


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


def test_split_records_refuses_two_runs_in_one_file() -> None:
    records = [
        {"event": "run_start", "config_name": "a"},
        {"event": "run_start", "config_name": "b"},
    ]
    with pytest.raises(ValueError, match="exactly one run_start"):
        R.split_records(records)


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
