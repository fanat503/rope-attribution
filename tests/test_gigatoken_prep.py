"""Tests for the gigatoken lane.

The lane exists to make Kaggle tokenization fast without changing the
artifact training reads.  These tests pin exactly that contract with an
injected encoder: order is preserved, empty documents vanish without
reordering, the dtype check refuses a silent wrap, and the manifest is the
one ``data.import_tokens`` writes.

Mutation verdicts
-----------------
* ``test_flatten_documents_concatenates_in_order`` fails if rows are sorted,
  deduplicated or emptied: the expected ids would no longer match.
* ``test_prepare_rejects_ids_above_uint16`` fails if the range check is
  removed: the id wraps to a different token and the stream silently changes.
* ``test_prepare_writes_the_same_manifest_as_import_tokens`` fails if this
  module grows its own writer: the format field or the checksum would differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "projects") not in sys.path:
    sys.path.insert(0, str(_REPO / "projects"))

from rope_attribution import data as D  # noqa: E402
from rope_attribution import gigatoken_prep as G  # noqa: E402


def _fake_encoder(rows):
    def encode(paths, *, tokenizer, separator):
        assert tokenizer == G.DEFAULT_TOKENIZER
        assert separator == G.DEFAULT_SEPARATOR
        assert [str(p) for p in paths] == ["a.txt"]
        return rows

    return encode


def test_flatten_documents_concatenates_in_order() -> None:
    rows = [np.array([1, 2]), np.array([], dtype=np.int64), np.array([3])]
    flat = G.flatten_token_documents(rows)
    assert flat.tolist() == [1, 2, 3]
    assert flat.dtype == np.uint32


def test_flatten_documents_accepts_a_flat_array() -> None:
    flat = G.flatten_token_documents(np.array([7, 8, 9], dtype=np.int64))
    assert flat.tolist() == [7, 8, 9]
    assert flat.dtype == np.uint32


def test_flatten_documents_empty_input_is_empty() -> None:
    assert G.flatten_token_documents([]).size == 0


def test_encode_text_files_uses_the_injected_backend() -> None:
    class Encoder:
        def encode_files(self, source):
            assert source == "source"
            return [[4, 5], [6]]

    def factory(name):
        assert name == "openai-community/gpt2"
        return Encoder()

    ids = G.encode_text_files(
        ["a.txt"],
        tokenizer_factory=factory,
        source_factory=lambda paths, separator: "source",
    )
    assert ids.tolist() == [4, 5, 6]


def test_missing_gigatoken_names_the_install_command(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "gigatoken", None)
    with pytest.raises(RuntimeError, match="pip install gigatoken"):
        G.encode_text_files(["a.txt"])


def test_prepare_rejects_ids_above_uint16(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not fit uint16"):
        G.prepare(
            ["a.txt"],
            tmp_path / "out",
            name="toy",
            encoder=_fake_encoder([[1, 70_000]]),
        )


def test_prepare_accepts_uint32_when_asked(tmp_path: Path) -> None:
    manifest = G.prepare(
        ["a.txt"],
        tmp_path / "out",
        name="toy",
        dtype="uint32",
        encoder=_fake_encoder([[1, 70_000]]),
    )
    assert manifest["n_tokens"] == 2
    assert manifest["dtype"] == "uint32"


def test_prepare_writes_the_same_manifest_as_import_tokens(tmp_path: Path) -> None:
    out = tmp_path / "out"
    manifest = G.prepare(
        ["a.txt"],
        out,
        name="toy",
        encoder=_fake_encoder([[11, 12, 13]]),
    )
    assert manifest["format"] == D.DATA_FORMAT
    assert manifest["kind"] == "imported"
    assert manifest["tokenizer"] == G.DEFAULT_TOKENIZER
    assert manifest["tokens"]["n_tokens"] == 3
    assert (out / "tokens.u16").exists()
    assert (out / "manifest.json").exists()
    assert D.load_tokens(out).tolist() == [11, 12, 13]


def test_prepare_round_trips_through_check(tmp_path: Path) -> None:
    out = tmp_path / "out"
    G.prepare(["a.txt"], out, name="toy", encoder=_fake_encoder([[1, 2]]))
    verified = D.verify_manifest(out)
    assert verified["n_tokens"] == 2


def test_prepare_truncates_after_encoding(tmp_path: Path) -> None:
    out = tmp_path / "out"
    manifest = G.prepare(
        ["a.txt"],
        out,
        name="toy",
        max_tokens=2,
        encoder=_fake_encoder([[1, 2, 3, 4]]),
    )
    assert manifest["n_tokens"] == 2
    assert D.load_tokens(out).tolist() == [1, 2]


def test_prepare_rejects_a_nonpositive_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        G.prepare(
            ["a.txt"],
            tmp_path / "out",
            name="toy",
            max_tokens=0,
            encoder=_fake_encoder([[1]]),
        )
