"""Tests for ``projects/rope_attribution/data.py``.

The token stream is the one artefact every training number sits on. These
tests pin the properties a reviewer would check by hand: ids survive the round
trip, a changed byte is detected, a rerun is idempotent, documents are
separated by EOS, an interrupted build resumes without duplicates, a
pre-tokenized import is validated, and a corpus that does not fit the declared
dtype is refused rather than truncated. No test touches the network.
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
        eos_id=None,
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
        eos_id=None,
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


def test_documents_are_joined_by_eos_and_not_terminated(tmp_path: Path) -> None:
    """The model must see a boundary between documents, and no trailing one.

    Mutation verdict: dropping the separator write leaves five tokens and the
    exact-list assertion fails; appending after the last document adds a sixth
    separator and ``tokens[-1] != ord("f")`` fails.
    """
    out = tmp_path / "tokens"
    manifest = D.prepare(
        _encode,
        iter(["abc", "de", "f"]),
        out,
        name="toy",
        source="unit-test",
        tokenizer="ord",
        shard_tokens=100,
    )
    tokens = [int(token) for token in D.load_tokens(out)]
    assert tokens == [97, 98, 99, D.DEFAULT_EOS_ID, 100, 101, D.DEFAULT_EOS_ID, 102]
    assert manifest["eos_id"] == D.DEFAULT_EOS_ID
    assert manifest["n_tokens"] == 8


def test_eos_id_is_configurable_and_can_be_disabled(tmp_path: Path) -> None:
    """Mutation verdict: ignoring the argument always writes 50256, so the
    custom-id assertion fails; ignoring ``None`` writes a separator where the
    disabled run expects adjacency.
    """
    custom = D.prepare(
        _encode,
        iter(["ab", "cd"]),
        tmp_path / "custom",
        name="toy",
        source="unit-test",
        tokenizer="ord",
        eos_id=7,
    )
    assert [int(t) for t in D.load_tokens(tmp_path / "custom")] == [97, 98, 7, 99, 100]
    assert custom["eos_id"] == 7

    disabled = D.prepare(
        _encode,
        iter(["ab", "cd"]),
        tmp_path / "disabled",
        name="toy",
        source="unit-test",
        tokenizer="ord",
        eos_id=None,
    )
    assert [int(t) for t in D.load_tokens(tmp_path / "disabled")] == [97, 98, 99, 100]
    assert disabled["eos_id"] is None


def test_source_resolution_is_pure_and_needs_no_network(tmp_path: Path) -> None:
    """Mutation verdict: hard-coding the wikitext dataset makes the openwebtext
    and c4 assertions fail; dropping the required-argument checks turns the
    expected ValueError into a later, less specific failure.
    """
    assert D.resolve_source("wikitext-103") == D.resolve_source("wikitext")
    wikitext = D.resolve_source("wikitext")
    assert wikitext.dataset == "wikitext"
    assert wikitext.subset == "wikitext-103-raw-v1"
    assert wikitext.split == "train"
    assert wikitext.text_field == "text"
    assert wikitext.label == "wikitext:wikitext-103-raw-v1:train"

    openwebtext = D.resolve_source("openwebtext")
    assert openwebtext.dataset == "Skylion007/openwebtext"

    c4 = D.resolve_source("c4", subset="en.noblocklist")
    assert c4.dataset == "allenai/c4"
    assert c4.subset == "en.noblocklist"

    custom = D.resolve_source(
        "dataset",
        dataset="user/corpus",
        subset="clean",
        split="validation",
        text_field="content",
    )
    assert (custom.dataset, custom.subset, custom.split) == (
        "user/corpus",
        "clean",
        "validation",
    )
    assert custom.text_field == "content"

    local = D.resolve_source("text", text=tmp_path / "corpus.txt")
    assert local.kind == "text"
    assert local.text == tmp_path / "corpus.txt"
    assert local.label == f"text:{tmp_path / 'corpus.txt'}"

    with pytest.raises(ValueError, match="--dataset"):
        D.resolve_source("dataset")
    with pytest.raises(ValueError, match="--text"):
        D.resolve_source("text")
    with pytest.raises(ValueError, match="source"):
        D.resolve_source("not-a-corpus")


class _FakeStream:
    """The part of a Hugging Face IterableDataset the reader actually uses."""

    def __init__(self, examples: list[dict], skipped: list[int] | None = None) -> None:
        self.examples = examples
        self.skipped = skipped if skipped is not None else []

    def skip(self, count: int):
        self.skipped.append(count)
        return _FakeStream(self.examples[count:], self.skipped)

    def __iter__(self):
        return iter(self.examples)


def test_huggingface_source_skips_with_the_dataset_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation verdict: replacing ``.skip(n)`` with re-iteration yields the
    first document instead of the third, and ``skipped`` stays empty.
    """
    source = D.HuggingFaceDocuments(D.resolve_source("openwebtext"))
    stream = _FakeStream([{"text": "a"}, {"text": "b"}, {"text": "c"}])
    monkeypatch.setattr(source, "_open_dataset", lambda: stream)
    assert list(source.open(skip_docs=2)) == ["c"]
    assert stream.skipped == [2]

    custom = D.HuggingFaceDocuments(
        D.resolve_source("dataset", dataset="user/corpus", text_field="content")
    )
    monkeypatch.setattr(custom, "_open_dataset", lambda: _FakeStream([{"text": "x"}]))
    with pytest.raises(ValueError, match="text-field"):
        list(custom.open())


class _InterruptingDocuments:
    """A source that yields ``docs`` and then fails, like a dropped network."""

    def __init__(self, docs: list[str], fail_at: int) -> None:
        self.docs = docs
        self.fail_at = fail_at

    def __iter__(self):
        for index, document in enumerate(self.docs):
            if index == self.fail_at:
                raise RuntimeError("interrupted")
            yield document


def test_resume_keeps_complete_shards_and_skips_done_documents(
    tmp_path: Path,
) -> None:
    """A resumed run must concatenate to exactly the uninterrupted stream.

    Mutation verdict: ignoring progress.json re-tokenizes from document zero,
    so the resumed stream has duplicate tokens and the equality assertion
    fails; deleting shards on resume changes the first shard hash.
    """
    docs = ["abc", "de", "fg", "hi", "jk"]
    common = {
        "name": "toy",
        "source": "unit-test",
        "tokenizer": "ord",
        "shard_tokens": 4,
    }
    full_out = tmp_path / "full"
    full = D.prepare(_encode, iter(docs), full_out, **common)

    partial_out = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="interrupted"):
        D.prepare(_encode, _InterruptingDocuments(docs, 2), partial_out, **common)

    progress = json.loads((partial_out / "progress.json").read_text(encoding="utf-8"))
    assert progress["docs_done"] == 2
    assert progress["tokens_done"] == 6
    assert progress["eos_id"] == D.DEFAULT_EOS_ID
    assert len(progress["shards"]) == 1
    kept = [record["sha256"] for record in progress["shards"]]
    encoded: list[str] = []

    def counting(text: str) -> list[int]:
        encoded.append(text)
        return _encode(text)

    resumed = D.prepare(
        counting,
        iter(docs),
        partial_out,
        resume=True,
        **common,
    )
    assert resumed["reused"] is False
    assert resumed["n_documents"] == len(docs)
    assert encoded == docs[2:], "resume must not re-tokenize finished documents"
    assert [int(t) for t in D.load_tokens(partial_out)] == [
        int(t) for t in D.load_tokens(full_out)
    ]
    assert resumed["tokens"]["sha256"] == full["tokens"]["sha256"]
    assert [record["sha256"] for record in resumed["shards"][: len(kept)]] == kept
    assert not (partial_out / "progress.json").exists(), "progress must be cleared"


def test_a_handwritten_progress_file_with_only_the_documented_fields_resumes(
    tmp_path: Path,
) -> None:
    """The documented progress contract is the five fields, not our extras.

    Mutation verdict: requiring the module's ``format``/``kind``/``dtype``
    keys treats this progress file as foreign, deletes the shard, and encodes
    all five documents again.
    """
    docs = ["abc", "de", "fg", "hi", "jk"]
    common = {
        "name": "toy",
        "source": "unit-test",
        "tokenizer": "ord",
        "shard_tokens": 4,
    }
    full_out = tmp_path / "full"
    full = D.prepare(_encode, iter(docs), full_out, **common)

    partial_out = tmp_path / "partial"
    first = D.prepare(_encode, iter(docs[:2]), partial_out, **common)
    (partial_out / "manifest.json").unlink()
    (partial_out / "tokens.u16").unlink()
    (partial_out / "progress.json").write_text(
        json.dumps(
            {
                "docs_done": 2,
                "tokens_done": first["n_tokens"],
                "source": common["source"],
                "tokenizer": common["tokenizer"],
                "eos_id": D.DEFAULT_EOS_ID,
            }
        ),
        encoding="utf-8",
    )

    encoded: list[str] = []

    def counting(text: str) -> list[int]:
        encoded.append(text)
        return _encode(text)

    resumed = D.prepare(counting, iter(docs), partial_out, resume=True, **common)
    assert encoded == docs[2:]
    assert resumed["n_documents"] == len(docs)
    assert [int(t) for t in D.load_tokens(partial_out)] == [
        int(t) for t in D.load_tokens(full_out)
    ]
    assert resumed["tokens"]["sha256"] == full["tokens"]["sha256"]


class _InterruptingTextFile(D.TextFileDocuments):
    def open(self, **kwargs):
        for index, document in enumerate(super().open(**kwargs)):
            if index == 2:
                raise RuntimeError("interrupted")
            yield document


def test_text_source_resumes_at_the_recorded_byte_offset(tmp_path: Path) -> None:
    """A local file must resume by seeking, not by re-reading from byte zero.

    Mutation verdict: dropping ``byte_offset`` from the source call makes the
    resumed source start at zero and ``last_start`` differ; re-tokenizing every
    document fails the ``encoded`` list equality.
    """
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abc\n\nde\n\nfg\n\nhi\n")
    common = {
        "name": "toy",
        "source": f"text:{corpus}",
        "tokenizer": "ord",
        "shard_tokens": 5,
    }
    full_out = tmp_path / "full"
    D.prepare(_encode, D.TextFileDocuments(corpus), full_out, **common)

    partial_out = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="interrupted"):
        D.prepare(
            _encode,
            _InterruptingTextFile(corpus),
            partial_out,
            **common,
        )
    progress = json.loads((partial_out / "progress.json").read_text(encoding="utf-8"))
    assert progress["docs_done"] == 2
    assert progress["byte_offset"] == len(b"abc\n\nde\n\n")

    encoded: list[str] = []

    def counting(text: str) -> list[int]:
        encoded.append(text)
        return _encode(text)

    source = D.TextFileDocuments(corpus)
    D.prepare(counting, source, partial_out, resume=True, **common)
    assert encoded == ["fg\n", "hi\n"]
    assert source.last_start == progress["byte_offset"]
    assert [int(t) for t in D.load_tokens(partial_out)] == [
        int(t) for t in D.load_tokens(full_out)
    ]


def test_a_crlf_text_file_keeps_the_old_newline_normalisation(tmp_path: Path) -> None:
    """Mutation verdict: dropping the normalisation yields ``"abc\\r\\n"``,
    which the byte-exact assertion rejects; the original text-mode reader
    produced ``"abc\\n"``.
    """
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abc\r\n\r\nde\r\n")
    assert list(D.TextFileDocuments(corpus).open()) == ["abc\n", "de\n"]


def test_a_pretokenized_npy_file_is_imported_without_an_encoder(tmp_path: Path) -> None:
    """Mutation verdict: routing the import through ``prepare`` calls the
    encoder (there is none) or writes a manifest without the imported hash.
    """
    ids = np.array([0, 1, D.DEFAULT_EOS_ID, 65_535], dtype=np.uint16)
    source = tmp_path / "ids.npy"
    np.save(source, ids)

    out = tmp_path / "tokens"
    manifest = D.import_tokens(
        source,
        out,
        name="toy",
        source=f"tokens-npy:{source}",
        tokenizer="gpt2",
        dtype="uint16",
        npy=True,
    )
    assert manifest["n_documents"] is None
    assert manifest["shards"] == []
    assert manifest["n_tokens"] == 4
    assert [int(t) for t in D.load_tokens(out)] == [0, 1, D.DEFAULT_EOS_ID, 65_535]
    assert D.verify_manifest(out)["verified"] is True
    assert D.load_tokens(out, verify=True).dtype == np.uint16


def test_a_uint32_pretokenized_bin_file_is_imported(tmp_path: Path) -> None:
    """Mutation verdict: always choosing tokens.u16 truncates 70,000 or writes
    a file whose manifest disagrees with its dtype.
    """
    source = tmp_path / "ids.u32"
    source.write_bytes(np.array([70_000, 1], dtype="<u4").tobytes())
    out = tmp_path / "tokens"
    manifest = D.import_tokens(
        source,
        out,
        name="toy",
        source=f"tokens-bin:{source}",
        tokenizer="gpt2",
        dtype="uint32",
    )
    assert manifest["tokens"]["file"] == "tokens.u32"
    assert [int(t) for t in D.load_tokens(out)] == [70_000, 1]
    assert D.verify_manifest(out)["verified"] is True


def test_a_pretokenized_bin_file_is_imported_and_size_checked(tmp_path: Path) -> None:
    """Mutation verdict: skipping the divisibility check silently drops the
    trailing byte; accepting 70,000 silently truncates it to 4,464.
    """
    source = tmp_path / "ids.u16"
    ids = np.array([3, 4, 5], dtype="<u2")
    source.write_bytes(ids.tobytes())
    out = tmp_path / "tokens"
    manifest = D.import_tokens(
        source,
        out,
        name="toy",
        source=f"tokens-bin:{source}",
        tokenizer="gpt2",
        dtype="uint16",
    )
    assert manifest["n_tokens"] == 3
    assert [int(t) for t in D.load_tokens(out)] == [3, 4, 5]

    truncated = tmp_path / "bad.u16"
    truncated.write_bytes(ids.tobytes() + b"\x00")
    with pytest.raises(ValueError, match="multiple"):
        D.import_tokens(
            truncated,
            tmp_path / "bad-out",
            name="toy",
            source="tokens-bin:bad",
            tokenizer="gpt2",
            dtype="uint16",
        )

    too_large = tmp_path / "large.npy"
    np.save(too_large, np.array([70_000], dtype=np.uint32))
    with pytest.raises(ValueError, match="uint32"):
        D.import_tokens(
            too_large,
            tmp_path / "large-out",
            name="toy",
            source="tokens-npy:large",
            tokenizer="gpt2",
            dtype="uint16",
            npy=True,
        )


def test_the_cli_imports_npy_and_prints_a_compact_summary(tmp_path: Path, capsys) -> None:
    """Mutation verdict: dropping the summary fields loses the exact strings;
    loading the GPT-2 tokenizer for an import would need the network.
    """
    source = tmp_path / "ids.npy"
    np.save(source, np.array([7, 8, 9], dtype=np.uint16))
    out = tmp_path / "tokens"
    D.main(
        [
            "prepare",
            "--out",
            str(out),
            "--name",
            "toy",
            "--tokens-npy",
            str(source),
        ]
    )
    captured = capsys.readouterr().out
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "3 tokens" in captured
    assert "0 shard(s)" in captured
    assert "bytes" in captured
    assert manifest["tokens"]["sha256"] in captured
    assert str(out / "tokens.u16") in captured


def test_the_cli_prepares_a_text_file_with_and_without_eos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Mutation verdict: not wiring ``--no-eos`` leaves 50256 in the second
    stream; dropping the summary fields loses the asserted strings. The
    encoder is stubbed so the CLI test never downloads GPT-2.
    """
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abc\n\nde\n")
    monkeypatch.setattr(D, "_gpt2_encoder", lambda: (_encode, 50257))

    separated = tmp_path / "separated"
    D.main(
        [
            "prepare",
            "--out",
            str(separated),
            "--source",
            "text",
            "--text",
            str(corpus),
        ]
    )
    assert [int(t) for t in D.load_tokens(separated)] == [
        97,
        98,
        99,
        10,
        D.DEFAULT_EOS_ID,
        100,
        101,
        10,
    ]
    captured = capsys.readouterr().out
    assert "2 docs, 8 tokens, 1 shard(s)" in captured
    assert "sha256" in captured
    assert str(separated / "tokens.u16") in captured

    adjacent = tmp_path / "adjacent"
    D.main(
        [
            "prepare",
            "--out",
            str(adjacent),
            "--source",
            "text",
            "--text",
            str(corpus),
            "--no-eos",
        ]
    )
    assert [int(t) for t in D.load_tokens(adjacent)] == [97, 98, 99, 10, 100, 101, 10]


def test_reuse_verifies_a_same_size_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation verdict: returning early on matching sizes reuses the damaged
    stream, so ``reused`` is True and the encoder is never called.
    """
    out, first = _build(tmp_path)
    merged = out / first["tokens"]["file"]
    damaged = np.fromfile(merged, dtype=np.uint16)
    damaged[0] = 0
    damaged.tofile(merged)
    assert merged.stat().st_size == first["tokens"]["n_tokens"] * 2

    with pytest.raises(ValueError, match="sha256"):
        D.load_tokens(out, verify=True)
    assert int(D.load_tokens(out)[0]) == 0, "the hot path must not hash by default"

    calls: list[str] = []

    def counting(text: str) -> list[int]:
        calls.append(text)
        return _encode(text)

    rebuilt = D.prepare(
        counting,
        iter(["abc", "", "de"]),
        out,
        name="toy",
        source="unit-test",
        tokenizer="ord",
        shard_tokens=4,
        eos_id=None,
    )
    assert rebuilt["reused"] is False
    assert calls == ["abc", "", "de"]
    assert [int(t) for t in D.load_tokens(out)] == [97, 98, 99, 100, 101]


def test_trust_size_is_an_explicit_opt_out_of_the_reuse_hash(tmp_path: Path) -> None:
    """Mutation verdict: ignoring ``trust_size`` re-tokenizes and calls the
    refusing encoder, so this test raises instead of reporting reuse.
    """
    out, first = _build(tmp_path)
    merged = out / first["tokens"]["file"]
    damaged = np.fromfile(merged, dtype=np.uint16)
    damaged[0] = 0
    damaged.tofile(merged)

    def refuse(_text: str) -> list[int]:
        raise AssertionError("trust_size must not hash or tokenize")

    reused = D.prepare(
        refuse,
        iter([]),
        out,
        name="toy",
        source="unit-test",
        tokenizer="ord",
        eos_id=None,
        trust_size=True,
    )
    assert reused["reused"] is True


def test_max_tokens_stops_cleanly_and_records_truncation(tmp_path: Path) -> None:
    """Mutation verdict: ignoring the budget writes both documents; omitting
    the flag leaves ``truncated`` false on a deliberately shortened stream.
    """
    out = tmp_path / "tokens"
    manifest = D.prepare(
        _encode,
        iter(["abcdef", "ghij"]),
        out,
        name="toy",
        source="unit-test",
        tokenizer="ord",
        shard_tokens=4,
        max_tokens=6,
        eos_id=None,
    )
    assert manifest["truncated"] is True
    assert manifest["n_tokens"] == 6
    assert manifest["n_documents"] == 1
    assert [int(t) for t in D.load_tokens(out)] == [97, 98, 99, 100, 101, 102]
    assert not (out / "progress.json").exists()

    generous = D.prepare(
        _encode,
        iter(["abc"]),
        tmp_path / "generous",
        name="toy",
        source="unit-test",
        tokenizer="ord",
        max_tokens=100,
        eos_id=None,
    )
    assert generous["truncated"] is False


def test_changing_eos_invalidates_a_reused_artefact(tmp_path: Path) -> None:
    """Mutation verdict: leaving eos_id out of the identity reuses the old
    bytes and the refusing encoder is never called.
    """
    out, _ = _build(tmp_path)
    rebuilt = D.prepare(
        _encode,
        iter(["abc", "", "de"]),
        out,
        name="toy",
        source="unit-test",
        tokenizer="ord",
        shard_tokens=4,
        eos_id=7,
    )
    assert rebuilt["reused"] is False
    assert [int(t) for t in D.load_tokens(out)] == [
        97,
        98,
        99,
        7,
        7,
        100,
        101,
    ]
