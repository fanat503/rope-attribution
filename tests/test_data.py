"""Tests for ``projects/rope_attribution/data.py``.

The token stream is the one artefact every training number sits on. These
tests pin four properties a reviewer would check by hand: the ids survive the
round trip, a changed byte is detected, a rerun is idempotent, and a corpus
that does not fit the declared dtype is refused rather than truncated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "projects") not in sys.path:
    sys.path.insert(0, str(_REPO / "projects"))

from rope_attribution import data as D  # noqa: E402
from rope_attribution import train as T  # noqa: E402


def _encode(text: str) -> list[int]:
    return [ord(char) for char in text]


def _build(tmp_path: Path, *, shard_tokens: int = 4) -> tuple[Path, dict]:
    out = tmp_path / "tokens"
    manifest = D.prepare(
        _encode,
        iter(["abc", "", "de"]),
        out,
        name="toy",
        source="unit-test",
        tokenizer="ord",
        shard_tokens=shard_tokens,
    )
    return out, manifest


def test_the_round_trip_is_the_concatenation_of_the_documents(tmp_path: Path) -> None:
    out, manifest = _build(tmp_path)
    assert manifest["n_tokens"] == 5
    assert manifest["n_documents"] == 3
    assert manifest["shards"], "no shard was written"
    assert sum(s["n_tokens"] for s in manifest["shards"]) == 5
    tokens = D.load_tokens(out)
    assert isinstance(tokens, np.memmap)
    assert tokens.dtype == np.uint16
    assert [int(token) for token in tokens] == [97, 98, 99, 100, 101]
    assert (out / "manifest.json").exists()
    assert not list(out.glob("*.tmp")), "an atomic write left a temporary file"


def test_more_than_one_shard_is_written_when_asked(tmp_path: Path) -> None:
    out, manifest = _build(tmp_path, shard_tokens=2)
    assert len(manifest["shards"]) >= 2, manifest["shards"]
    assert all(s["sha256"] for s in manifest["shards"])
    assert [int(t) for t in D.load_tokens(out)] == [97, 98, 99, 100, 101]


def test_a_changed_byte_is_detected(tmp_path: Path) -> None:
    """Mutation verdict: dropping the sha256 comparison accepts the edit."""
    out, manifest = _build(tmp_path)
    D.verify_manifest(out)  # the pristine directory must pass
    shard = out / manifest["shards"][0]["file"]
    payload = bytearray(shard.read_bytes())
    payload[0] ^= 0xFF
    shard.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="sha256"):
        D.verify_manifest(out)


def test_the_merged_stream_is_verified_too(tmp_path: Path) -> None:
    out, manifest = _build(tmp_path)
    merged = out / manifest["tokens"]["file"]
    payload = bytearray(merged.read_bytes())
    payload[-1] ^= 0xFF
    merged.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="sha256"):
        D.verify_manifest(out)


def test_a_rerun_reuses_the_artefact_without_calling_the_encoder(tmp_path: Path) -> None:
    """Mutation verdict: ignoring the manifest calls the raising encoder."""

    def refuse(_text: str) -> list[int]:
        raise AssertionError("the encoder was called on a reusable artefact")

    out, first = _build(tmp_path)
    second = D.prepare(
        refuse,
        iter([]),
        out,
        name="toy",
        source="unit-test",
        tokenizer="ord",
        shard_tokens=4,
    )
    assert second["reused"] is True
    assert second["tokens"]["sha256"] == first["tokens"]["sha256"]
    assert second["n_tokens"] == first["n_tokens"]


def test_a_different_identity_forces_a_rebuild(tmp_path: Path) -> None:
    calls: list[str] = []

    def counting(text: str) -> list[int]:
        calls.append(text)
        return _encode(text)

    out, _ = _build(tmp_path)
    rebuilt = D.prepare(
        counting,
        iter(["xy"]),
        out,
        name="toy",
        source="another-source",
        tokenizer="ord",
        shard_tokens=4,
    )
    assert rebuilt["reused"] is False
    assert calls == ["xy"], "a changed source must be tokenized again"
    assert [int(t) for t in D.load_tokens(out)] == [120, 121]


def test_a_token_that_does_not_fit_uint16_is_refused(tmp_path: Path) -> None:
    """Mutation verdict: truncating the id would write 4464 and pass silently."""
    out = tmp_path / "tokens"
    with pytest.raises(ValueError, match="uint32"):
        D.prepare(
            lambda _text: [70_000],
            iter(["x"]),
            out,
            name="toy",
            source="unit-test",
            tokenizer="ord",
        )
    assert not (out / "tokens.u16").exists()
    assert not (out / "manifest.json").exists()


def test_uint32_is_available_for_a_larger_vocabulary(tmp_path: Path) -> None:
    out = tmp_path / "tokens"
    manifest = D.prepare(
        lambda _text: [70_000, 1],
        iter(["x"]),
        out,
        name="toy",
        source="unit-test",
        tokenizer="ord",
        dtype="uint32",
    )
    assert manifest["tokens"]["file"] == "tokens.u32"
    assert [int(t) for t in D.load_tokens(out)] == [70_000, 1]


def test_training_windows_keep_the_memmap_dtype() -> None:
    """A uint16 stream must not be widened before the batch is cut.

    Mutation verdict: restoring ``np.asarray(tokens, dtype=np.int64)`` in
    ``_split_windows`` materializes the corpus and fails the dtype assertion;
    removing the per-batch cast in ``_id_batch`` would hand the embedding a
    uint16 tensor.
    """
    stream = np.arange(200, dtype=np.uint16)
    train_in, train_tg, held_in, _held_tg = T._split_windows(stream, 8)
    assert train_in.dtype == np.uint16
    assert train_tg.dtype == np.uint16
    assert held_in.dtype == np.uint16
    if T.TORCH_AVAILABLE:
        import torch

        batch = T._id_batch(train_in[:2])
        assert batch.dtype == torch.int64


def test_the_check_command_reports_the_counts(tmp_path: Path, capsys) -> None:
    out, manifest = _build(tmp_path)
    D.main(["check", "--out", str(out)])
    captured = capsys.readouterr().out
    assert f"{manifest['n_tokens']:,} tokens" in captured
    assert "sha256 matches" in captured


def test_the_manifest_is_strict_json(tmp_path: Path) -> None:
    out, _ = _build(tmp_path)
    text = (out / "manifest.json").read_text(encoding="utf-8")

    def reject(token: str):
        raise ValueError(token)

    manifest = json.loads(text, parse_constant=reject)
    assert manifest["format"] == D.DATA_FORMAT
    assert manifest["dtype"] == "uint16"
