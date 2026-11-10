"""Turn a streamed corpus into the token stream a training run consumes.

Pipeline
--------
``prepare`` streams documents from the source, tokenizes them in one process
and writes document-bounded shards of about ``--shard-tokens`` each,
atomically; a second pass merges those shards into the single file that
training memory-maps. Documents are joined by the tokenizer's EOS id
(GPT-2's 50256 by default) so the model sees document boundaries; pass
``eos_id=None`` / ``--no-eos`` to concatenate raw text.

Sources
-------
``wikitext`` (default), ``openwebtext`` and ``c4`` are opened with
``datasets.load_dataset(..., streaming=True)``, as is any other Hugging Face
dataset through ``--source dataset --dataset NAME``. ``--source text`` reads a
local UTF-8 file with one document per blank-line-separated block. The source
resolution in :func:`resolve_source` is pure, so its tests need no network.

Layout
------
::

    data/<name>/
        shard-00000.u16      tokenized shards, written once, atomically
        shard-00001.u16
        tokens.u16           their concatenation; what training mmaps
        manifest.json        counts, dtype, EOS id and SHA-256 of every file
        progress.json        resume point, while a build is incomplete

``uint16`` holds the GPT-2 vocabulary (50,257 ids) with room to spare and
halves the bytes a ``uint16`` memmap moves against ``int32``. A vocabulary
that does not fit is refused rather than silently truncated; pass
``dtype="uint32"`` for a larger tokenizer.

Resume and verification
-----------------------
``--resume`` keeps the shards recorded in ``progress.json``, skips the
documents they already contain (Hugging Face ``.skip(n)``; a byte offset for a
local file), and continues the shard numbering. ``progress.json`` records
``docs_done``, ``tokens_done``, the source and tokenizer identity, ``eos_id``,
the kept shard records and the local-file byte offset; it is rewritten
atomically after every shard. The interrupted and the uninterrupted run
therefore concatenate to the same bytes. ``prepare`` verifies the merged
file's SHA-256 before reusing an artefact; ``--trust-size`` opts out of that
read, and ``check`` re-hashes every shard and the merged stream.

``--tokens-bin`` and ``--tokens-npy`` import an existing uint16/uint32 array,
validate its dtype and range, write the same checksummed artefact, and skip
tokenization entirely.

What is not claimed
-------------------
Tokenization is single-process and single-threaded: one core reads, encodes and
writes. There is no parallelism, no approximate token count and no corpus
redistribution. The test suite needs neither ``datasets`` nor a network.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DATA_FORMAT",
    "DEFAULT_EOS_ID",
    "HuggingFaceDocuments",
    "SourceSpec",
    "TOKEN_DTYPES",
    "TextFileDocuments",
    "import_tokens",
    "load_tokens",
    "main",
    "prepare",
    "resolve_source",
    "verify_manifest",
    "wikitext_documents",
]

#: Bumped when the on-disk contract changes; a reader refuses anything else.
#: v2 added document separators, ``eos_id`` and the resumable progress file.
DATA_FORMAT = "rope_attribution/token-data/v2"

TOKEN_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}

#: 50M tokens is 100 MB as uint16: large enough that a shard is a useful unit
#: of retry, small enough that a rejected document does not cost a whole run.
DEFAULT_SHARD_TOKENS = 50_000_000

#: GPT-2's end-of-text id. It is written between documents, never after the
#: last one, so the stream ends on content rather than on a separator.
DEFAULT_EOS_ID = 50256

MANIFEST_NAME = "manifest.json"
PROGRESS_NAME = "progress.json"

#: Tokens per read when importing a pre-tokenized array; keeps a 100M-token
#: file from being copied into RAM in one piece.
_IMPORT_CHUNK = 1 << 24


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _itemsize(dtype: str) -> int:
    return int(TOKEN_DTYPES[dtype]().itemsize)


def _shard_suffix(dtype: str) -> str:
    return "u16" if dtype == "uint16" else "u32"


def _shard_name(index: int, dtype: str) -> str:
    return f"shard-{index:05d}.{_shard_suffix(dtype)}"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, writer: Callable[[object], None]) -> None:
    """Write through a temporary sibling, then replace."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            writer(handle)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _token_array(chunks: Sequence[np.ndarray], dtype: str) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=TOKEN_DTYPES[dtype])
    tokens = np.concatenate(chunks)
    limit = int(np.iinfo(TOKEN_DTYPES[dtype]).max)
    if tokens.size and int(tokens.max()) > limit:
        raise ValueError(
            f"token id {int(tokens.max())} does not fit {dtype}; use dtype='uint32'"
        )
    if tokens.size and int(tokens.min()) < 0:
        raise ValueError("token ids must be non-negative")
    return tokens.astype(TOKEN_DTYPES[dtype], copy=False)


def _validated_eos(eos_id: int | None, dtype: str) -> int | None:
    if eos_id is None:
        return None
    value = int(eos_id)
    if value < 0:
        raise ValueError(f"eos id must be non-negative, not {value}")
    limit = int(np.iinfo(TOKEN_DTYPES[dtype]).max)
    if value > limit:
        raise ValueError(f"eos id {value} does not fit {dtype}; use dtype='uint32'")
    return value


@dataclass(frozen=True)
class SourceSpec:
    """A resolved corpus location: no I/O, no ``datasets`` import."""

    kind: str
    dataset: str | None
    subset: str | None
    split: str
    text_field: str
    text: Path | None

    @property
    def label(self) -> str:
        if self.kind == "text":
            return f"text:{self.text}"
        if self.subset:
            return f"{self.dataset}:{self.subset}:{self.split}"
        return f"{self.dataset}:{self.split}"


def resolve_source(
    source: str,
    *,
    subset: str | None = None,
    dataset: str | None = None,
    text: Path | str | None = None,
    split: str = "train",
    text_field: str = "text",
) -> SourceSpec:
    """Turn the CLI source flags into a :class:`SourceSpec`.

    Pure by design: tests can check the mapping without importing
    ``datasets`` or touching the network. ``wikitext-103`` is accepted as an
    alias for the old CLI spelling.
    """
    key = str(source).strip().lower()
    if key == "text":
        if text is None:
            raise ValueError("--source text needs --text PATH")
        return SourceSpec("text", None, None, split, text_field, Path(text))
    if dataset is not None:
        return SourceSpec("hf", dataset, subset, split, text_field, None)
    if key in {"wikitext", "wikitext-103"}:
        return SourceSpec(
            "hf",
            "wikitext",
            subset or "wikitext-103-raw-v1",
            split,
            text_field,
            None,
        )
    if key == "openwebtext":
        return SourceSpec(
            "hf", "Skylion007/openwebtext", subset, split, text_field, None
        )
    if key == "c4":
        return SourceSpec("hf", "allenai/c4", subset or "en", split, text_field, None)
    if key == "dataset":
        raise ValueError("--source dataset needs --dataset NAME")
    raise ValueError(
        f"unknown source {source!r}; choose wikitext, openwebtext, c4, dataset or text"
    )


class HuggingFaceDocuments:
    """A streaming Hugging Face corpus, resumable with ``.skip(n)``.

    The import is lazy so importing this module (and running its tests) does
    not require ``datasets``. A missing cache or network is an error with a
    command, not a silent empty corpus.
    """

    def __init__(self, spec: SourceSpec) -> None:
        if spec.kind != "hf":
            raise ValueError(f"{spec.label} is not a Hugging Face source")
        self.spec = spec
        self.byte_offset: int | None = None
        self._dataset = None

    def _open_dataset(self):
        if self._dataset is None:
            try:
                from datasets import load_dataset
            except Exception as exc:  # pragma: no cover - optional dependency
                raise SystemExit(
                    "streaming a corpus needs `datasets`; install "
                    "train_requirements.txt or pass --source text PATH"
                ) from exc
            try:
                self._dataset = load_dataset(
                    self.spec.dataset,
                    self.spec.subset,
                    split=self.spec.split,
                    streaming=True,
                )
            except Exception as exc:  # pragma: no cover - network/cache path
                raise SystemExit(f"could not open {self.spec.label} ({exc})") from exc
        return self._dataset

    def open(
        self, *, skip_docs: int = 0, byte_offset: int | None = None
    ) -> Iterator[str]:
        if byte_offset is not None:
            raise ValueError(f"{self.spec.label} is not byte-addressable")
        dataset = self._open_dataset()
        if skip_docs:
            skipper = getattr(dataset, "skip", None)
            if callable(skipper):
                dataset = skipper(int(skip_docs))
            else:  # pragma: no cover - current datasets always offers .skip
                dataset = itertools.islice(dataset, int(skip_docs), None)
        field = self.spec.text_field
        for example in dataset:
            if field not in example:
                raise ValueError(
                    f"{self.spec.label}: examples have no {field!r} field; "
                    "pass --text-field"
                )
            yield example[field]


def _decode_document(block: Sequence[bytes]) -> str:
    """UTF-8 text with universal newlines, matching the old text-mode reader."""
    return b"".join(block).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


class TextFileDocuments:
    """Blank-line-separated UTF-8 blocks with a resumable byte offset.

    ``byte_offset`` always points at the first unread byte after the most
    recently yielded document, so an interrupted run can seek before decoding
    a single earlier block. ``last_start`` records where the current iterator
    began, which is what the resume test inspects.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.byte_offset = 0
        self.last_start = 0

    def open(
        self, *, skip_docs: int = 0, byte_offset: int | None = None
    ) -> Iterator[str]:
        start = self.byte_offset if byte_offset is None else int(byte_offset)
        self.last_start = start
        return self._iterate(start, max(0, int(skip_docs)))

    def _iterate(self, start: int, skip_docs: int) -> Iterator[str]:
        with self.path.open("rb") as handle:
            handle.seek(start)
            block: list[bytes] = []
            remaining = skip_docs
            while True:
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    block.append(line)
                elif block:
                    document = _decode_document(block)
                    block = []
                    self.byte_offset = handle.tell()
                    if remaining:
                        remaining -= 1
                    else:
                        yield document
            if block:
                document = _decode_document(block)
                self.byte_offset = handle.tell()
                if remaining:
                    remaining -= 1
                else:
                    yield document


def _read_text_documents(path: Path) -> Iterator[str]:
    """One document per blank-line-separated block, newlines preserved."""
    yield from TextFileDocuments(path).open()


def _document_iterator(
    documents: Iterable[str], *, skip_docs: int, byte_offset: int | None
) -> Iterator[str]:
    """Open a source at the resume point, preferring a seek over a skip."""
    opener = getattr(documents, "open", None)
    if callable(opener):
        skipped = 0 if byte_offset is not None else int(skip_docs)
        return iter(opener(skip_docs=skipped, byte_offset=byte_offset))
    if skip_docs:
        skipper = getattr(documents, "skip", None)
        if callable(skipper):
            documents = skipper(int(skip_docs))
        else:
            documents = itertools.islice(documents, int(skip_docs), None)
    return iter(documents)


def _write_shards(
    encode: Callable[[str], Sequence[int]],
    documents: Iterator[str],
    out_dir: Path,
    *,
    dtype: str,
    shard_tokens: int,
    max_tokens: int | None,
    eos_id: int | None,
    source: str,
    tokenizer: str,
    prior_shards: Sequence[dict] = (),
    start_docs: int = 0,
    start_tokens: int = 0,
    progress_path: Path | None = None,
    byte_position: Callable[[], int | None] | None = None,
) -> tuple[list[dict], int, int, bool]:
    """Tokenize documents into shards. Returns records, tokens, docs, truncated."""
    if shard_tokens < 1:
        raise ValueError(f"shard_tokens must be positive, not {shard_tokens!r}")
    if max_tokens is not None and int(max_tokens) < 0:
        raise ValueError(f"max_tokens must be non-negative, not {max_tokens!r}")
    limit = int(np.iinfo(TOKEN_DTYPES[dtype]).max)
    typed = TOKEN_DTYPES[dtype]
    records = list(prior_shards)
    pending: list[np.ndarray] = []
    pending_tokens = 0
    pending_docs = 0
    n_documents = int(start_docs)
    total_tokens = int(start_tokens)
    flushed_docs = int(start_docs)
    flushed_tokens = int(start_tokens)
    truncated = False

    def write_progress(*, at_budget: bool = False) -> None:
        if progress_path is None:
            return
        position = byte_position() if byte_position is not None else None
        _atomic_write_json(
            progress_path,
            {
                "format": DATA_FORMAT,
                "kind": "tokenized",
                "source": source,
                "tokenizer": tokenizer,
                "eos_id": eos_id,
                "dtype": dtype,
                "docs_done": int(flushed_docs),
                "tokens_done": int(flushed_tokens),
                "shards": records,
                "byte_offset": None if position is None else int(position),
                "truncated": bool(at_budget),
                "updated_utc": _now(),
            },
        )

    def flush() -> None:
        nonlocal pending, pending_tokens, pending_docs, flushed_docs, flushed_tokens
        if not pending_tokens:
            return
        tokens = _token_array(pending, dtype)
        name = _shard_name(len(records), dtype)
        path = Path(out_dir) / name
        _atomic_write(path, lambda handle: tokens.tofile(handle))
        records.append(
            {
                "file": name,
                "n_tokens": int(tokens.size),
                "sha256": _sha256_file(path),
                "n_documents": int(pending_docs),
            }
        )
        flushed_docs += pending_docs
        flushed_tokens += int(tokens.size)
        pending = []
        pending_tokens = 0
        pending_docs = 0
        write_progress()

    for document in documents:
        ids = encode(document)
        array = np.asarray(ids, dtype=np.int64).reshape(-1)
        if eos_id is not None and n_documents > 0:
            array = np.concatenate((np.asarray([eos_id], dtype=np.int64), array))
        if array.size == 0:
            # An empty document contributes no tokens, but it is still a
            # document in the manifest and still owns a boundary.
            n_documents += 1
            pending_docs += 1
            continue
        if int(array.min()) < 0:
            raise ValueError("token ids must be non-negative")
        if int(array.max()) > limit:
            raise ValueError(
                f"token id {int(array.max())} does not fit {dtype}; "
                "use dtype='uint32'"
            )
        # Store each document in its final width now: a 50M-token shard then
        # costs 100 MB pending instead of 400 MB as int64.
        pending.append(array.astype(typed, copy=False))
        pending_tokens += int(array.size)
        total_tokens += int(array.size)
        pending_docs += 1
        n_documents += 1
        if max_tokens is not None and total_tokens >= int(max_tokens):
            flush()
            truncated = True
            write_progress(at_budget=True)
            break
        if pending_tokens >= shard_tokens:
            flush()
    else:
        flush()
    return records, total_tokens, n_documents, truncated


def _merge_shards(
    shards: Sequence[dict], out_dir: Path, *, name: str, dtype: str
) -> dict:
    """Concatenate the shards into the single file training mmaps."""
    path = Path(out_dir) / name
    total = 0

    def write(handle) -> None:
        nonlocal total
        for shard in shards:
            with (Path(out_dir) / shard["file"]).open("rb") as source:
                while True:
                    block = source.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
            total += int(shard["n_tokens"])

    _atomic_write(path, write)
    return {
        "file": name,
        "n_tokens": int(total),
        "sha256": _sha256_file(path),
        "dtype": dtype,
    }


def _identity(manifest: dict) -> tuple:
    return (
        manifest.get("format"),
        manifest.get("kind"),
        manifest.get("source"),
        manifest.get("tokenizer"),
        manifest.get("dtype"),
        manifest.get("eos_id"),
    )


def _clear_artifact(out_dir: Path) -> None:
    for name in (
        "tokens.u16",
        "tokens.u32",
        MANIFEST_NAME,
        PROGRESS_NAME,
    ):
        (out_dir / name).unlink(missing_ok=True)
        (out_dir / f"{name}.tmp").unlink(missing_ok=True)
    for pattern in ("shard-*.u16", "shard-*.u32", "shard-*.tmp"):
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)


def _progress_matches(payload: dict, identity: dict) -> bool:
    """Match a progress file, tolerating a hand-written one.

    ``progress.json`` only needs the fields the CLI documents
    (``docs_done``, ``tokens_done``, ``source``, ``tokenizer``, ``eos_id``);
    ``format``, ``kind`` and ``dtype`` are written by this module but a
    hand-written progress file may omit them.
    """
    if payload.get("format") not in (None, identity["format"]):
        return False
    if payload.get("kind") not in (None, identity["kind"]):
        return False
    if payload.get("dtype") not in (None, identity["dtype"]):
        return False
    return all(
        payload.get(key) == identity[key] for key in ("source", "tokenizer", "eos_id")
    )


def _read_progress(out_dir: Path, identity: dict) -> dict | None:
    path = out_dir / PROGRESS_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: unreadable progress file ({exc})") from exc
    if not _progress_matches(payload, identity):
        return None
    docs_done = payload.get("docs_done")
    tokens_done = payload.get("tokens_done")
    if (
        not isinstance(docs_done, int)
        or not isinstance(tokens_done, int)
        or docs_done < 0
        or tokens_done < 0
    ):
        raise ValueError(f"{path}: docs_done/tokens_done must be non-negative integers")
    return payload


def _resume_shards(
    out_dir: Path, progress: dict, *, dtype: str, trust_size: bool
) -> tuple[list[dict], int, int, int | None]:
    """Validate and return the shards a resume may keep, in merge order."""
    docs_done = int(progress["docs_done"])
    tokens_done = int(progress["tokens_done"])
    itemsize = _itemsize(dtype)
    listed = progress.get("shards")
    if listed is None:
        records = []
        for index, path in enumerate(sorted(out_dir.glob(f"shard-*.{_shard_suffix(dtype)}"))):
            if path.name != _shard_name(index, dtype):
                raise ValueError(f"{path}: shard numbering has a gap")
            size = path.stat().st_size
            if size % itemsize:
                raise ValueError(f"{path}: size {size} is not a whole number of tokens")
            records.append(
                {
                    "file": path.name,
                    "n_tokens": size // itemsize,
                    "sha256": _sha256_file(path),
                }
            )
    else:
        records = list(listed)
        for index, record in enumerate(records):
            if record.get("file") != _shard_name(index, dtype):
                raise ValueError(
                    f"progress lists {record.get('file')!r} at position {index}; "
                    f"expected {_shard_name(index, dtype)}"
                )
    total = 0
    docs_in_records = 0
    docs_counted = False
    for record in records:
        path = out_dir / str(record["file"])
        expected = int(record["n_tokens"])
        if not path.exists():
            raise ValueError(f"{path} is listed in progress but missing")
        size = path.stat().st_size
        if size != expected * itemsize:
            raise ValueError(
                f"{path}: size {size} disagrees with {expected} tokens in progress"
            )
        expected_sha = record.get("sha256")
        if expected_sha is None:
            record["sha256"] = _sha256_file(path)
        elif not trust_size and _sha256_file(path) != expected_sha:
            raise ValueError(f"{path}: sha256 does not match progress")
        total += expected
        if "n_documents" in record:
            docs_counted = True
            docs_in_records += int(record["n_documents"])
    if total != tokens_done:
        raise ValueError(
            f"progress says {tokens_done} tokens but its shards hold {total}"
        )
    if docs_counted and docs_in_records != docs_done:
        raise ValueError(
            f"progress says {docs_done} documents but its shards hold "
            f"{docs_in_records}"
        )
    byte_offset = progress.get("byte_offset")
    return records, docs_done, tokens_done, (
        None if byte_offset is None else int(byte_offset)
    )


def prepare(
    encode: Callable[[str], Sequence[int]],
    documents: Iterable[str],
    out_dir: Path,
    *,
    name: str,
    source: str,
    tokenizer: str,
    dtype: str = "uint16",
    eos_id: int | None = DEFAULT_EOS_ID,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    max_tokens: int | None = None,
    reuse: bool = True,
    trust_size: bool = False,
    resume: bool = False,
) -> dict:
    """Tokenize ``documents`` into ``out_dir`` and return the manifest.

    ``encode`` is the tokenizer as a plain callable, which keeps this module
    free of a hard ``transformers`` dependency and lets the tests use an
    encoder whose output is countable by hand. ``reuse`` accepts an existing
    manifest whose identity fields and file sizes match, but hashes the merged
    file first unless ``trust_size`` is set. ``resume`` continues from
    ``progress.json`` instead of starting over.
    """
    if dtype not in TOKEN_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(TOKEN_DTYPES)}, not {dtype!r}")
    if shard_tokens < 1:
        raise ValueError(f"shard_tokens must be positive, not {shard_tokens!r}")
    if max_tokens is not None and int(max_tokens) < 0:
        raise ValueError(f"max_tokens must be non-negative, not {max_tokens!r}")
    eos_id = _validated_eos(eos_id, dtype)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_name = "tokens.u16" if dtype == "uint16" else "tokens.u32"
    manifest_path = out_dir / MANIFEST_NAME
    identity = {
        "format": DATA_FORMAT,
        "kind": "tokenized",
        "source": source,
        "tokenizer": tokenizer,
        "dtype": dtype,
        "eos_id": eos_id,
    }
    if reuse and manifest_path.exists() and (out_dir / final_name).exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _identity(existing) == _identity(identity):
            tokens = existing.get("tokens", {})
            path = out_dir / final_name
            expected = int(tokens.get("n_tokens", -1))
            if (
                tokens.get("file") == final_name
                and expected >= 0
                and path.stat().st_size == expected * _itemsize(dtype)
                and (
                    trust_size
                    or (tokens.get("sha256") and _sha256_file(path) == tokens["sha256"])
                )
            ):
                return {**existing, "reused": True}

    resumed = _read_progress(out_dir, identity) if resume else None
    if resumed is not None:
        prior, start_docs, start_tokens, byte_offset = _resume_shards(
            out_dir, resumed, dtype=dtype, trust_size=trust_size
        )
        for stale in (out_dir / final_name, manifest_path):
            stale.unlink(missing_ok=True)
    else:
        _clear_artifact(out_dir)
        prior, start_docs, start_tokens, byte_offset = [], 0, 0, None

    iterator = _document_iterator(
        documents, skip_docs=start_docs, byte_offset=byte_offset
    )
    byte_position = None
    if hasattr(documents, "byte_offset"):
        byte_position = lambda: getattr(documents, "byte_offset", None)  # noqa: E731

    if (
        resumed is not None
        and resumed.get("truncated")
        and max_tokens is not None
        and start_tokens >= int(max_tokens)
    ):
        records, n_tokens, n_documents, truncated = (
            prior,
            start_tokens,
            start_docs,
            True,
        )
    else:
        records, n_tokens, n_documents, truncated = _write_shards(
            encode,
            iterator,
            out_dir,
            dtype=dtype,
            shard_tokens=shard_tokens,
            max_tokens=max_tokens,
            eos_id=eos_id,
            source=source,
            tokenizer=tokenizer,
            prior_shards=prior,
            start_docs=start_docs,
            start_tokens=start_tokens,
            progress_path=out_dir / PROGRESS_NAME,
            byte_position=byte_position,
        )
    merged = _merge_shards(records, out_dir, name=final_name, dtype=dtype)
    if merged["n_tokens"] != n_tokens:
        raise RuntimeError(
            f"merged {merged['n_tokens']} tokens but shards hold {n_tokens}"
        )
    for shard in records:
        size = (out_dir / shard["file"]).stat().st_size
        if size != int(shard["n_tokens"]) * _itemsize(dtype):
            raise RuntimeError(f"{shard['file']}: size {size} disagrees with count")
    manifest = {
        "format": DATA_FORMAT,
        "kind": "tokenized",
        "name": name,
        "source": source,
        "tokenizer": tokenizer,
        "dtype": dtype,
        "eos_id": eos_id,
        "n_tokens": int(n_tokens),
        "n_documents": int(n_documents),
        "shards": records,
        "tokens": merged,
        "truncated": bool(truncated),
        "created_utc": _now(),
    }
    _atomic_write_json(manifest_path, manifest)
    (out_dir / PROGRESS_NAME).unlink(missing_ok=True)
    return {**manifest, "reused": False}


def _token_chunks(
    path: Path, *, dtype: str, npy: bool
) -> tuple[Iterator[np.ndarray], int]:
    """Open a token array and return (chunks, total), validating the header."""
    path = Path(path)
    itemsize = _itemsize(dtype)
    if npy:
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"{path}: not a readable .npy file ({exc})") from exc
        if getattr(array, "ndim", None) != 1:
            raise ValueError(f"{path}: expected a 1-D array, got shape {array.shape}")
        if array.dtype.kind != "u" or array.dtype.itemsize not in (2, 4):
            raise ValueError(
                f"{path}: dtype {array.dtype} is not uint16 or uint32; "
                "convert the tokens first"
            )
        total = int(array.size)
    else:
        size = path.stat().st_size
        if size % itemsize:
            raise ValueError(
                f"{path}: {size} bytes is not a multiple of {itemsize}; "
                f"expected raw little-endian {dtype}"
            )
        if size == 0:
            return iter(()), 0
        array = np.memmap(path, dtype=np.dtype(f"<u{itemsize}"), mode="r")
        total = int(array.size)

    def chunks() -> Iterator[np.ndarray]:
        for start in range(0, total, _IMPORT_CHUNK):
            yield np.asarray(array[start : start + _IMPORT_CHUNK])

    return chunks(), total


def import_tokens(
    path: Path,
    out_dir: Path,
    *,
    name: str,
    source: str,
    tokenizer: str,
    dtype: str = "uint16",
    eos_id: int | None = DEFAULT_EOS_ID,
    npy: bool = False,
) -> dict:
    """Import a pre-tokenized array, write the checksummed artefact, skip encoding.

    ``npy=False`` reads raw little-endian ``uint16``/``uint32``; ``npy=True``
    reads a one-dimensional ``.npy`` file. Values are range-checked against
    ``dtype`` and never silently truncated. Document boundaries are unknown
    here, so the manifest records ``n_documents: null``.
    """
    if dtype not in TOKEN_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(TOKEN_DTYPES)}, not {dtype!r}")
    eos_id = _validated_eos(eos_id, dtype)
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_artifact(out_dir)
    final_name = "tokens.u16" if dtype == "uint16" else "tokens.u32"
    final_path = out_dir / final_name
    chunks, total = _token_chunks(path, dtype=dtype, npy=npy)
    itemsize = _itemsize(dtype)
    limit = int(np.iinfo(TOKEN_DTYPES[dtype]).max)

    def write(handle) -> None:
        for chunk in chunks:
            values = np.asarray(chunk)
            if values.size:
                high = int(values.max())
                if high > limit:
                    raise ValueError(
                        f"token id {high} does not fit {dtype}; use dtype='uint32'"
                    )
                if int(values.min()) < 0:
                    raise ValueError("token ids must be non-negative")
            little = np.asarray(values, dtype=np.dtype(f"<u{itemsize}"))
            handle.write(little.tobytes(order="C"))

    _atomic_write(final_path, write)
    size = final_path.stat().st_size
    if size != total * itemsize:
        raise RuntimeError(
            f"{final_path}: wrote {size} bytes for {total} {dtype} tokens"
        )
    manifest = {
        "format": DATA_FORMAT,
        "kind": "imported",
        "name": name,
        "source": source,
        "tokenizer": tokenizer,
        "dtype": dtype,
        "eos_id": eos_id,
        "n_tokens": int(total),
        "n_documents": None,
        "shards": [],
        "tokens": {
            "file": final_name,
            "n_tokens": int(total),
            "sha256": _sha256_file(final_path),
            "dtype": dtype,
        },
        "truncated": False,
        "imported_from": str(path),
        "created_utc": _now(),
    }
    _atomic_write_json(out_dir / MANIFEST_NAME, manifest)
    (out_dir / PROGRESS_NAME).unlink(missing_ok=True)
    return {**manifest, "reused": False}


def load_tokens(
    data_dir: Path, *, mmap_mode: str = "r", verify: bool = False
) -> np.memmap:
    """The training stream, memory-mapped. Read-only by default.

    This is what ``train(..., tokens=...)`` wants: ``_split_windows`` keeps
    the dtype and only the batch is ever copied. ``verify=True`` hashes the
    merged file once and raises ``ValueError`` on a mismatch; it is off by
    default because the hot path must not re-read a multi-gigabyte stream.
    """
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format") != DATA_FORMAT:
        raise ValueError(
            f"{data_dir}: {manifest.get('format')!r} is not {DATA_FORMAT}"
        )
    dtype = str(manifest["dtype"])
    if dtype not in TOKEN_DTYPES:
        raise ValueError(f"{data_dir}: unknown dtype {dtype!r}")
    path = data_dir / manifest["tokens"]["file"]
    expected = int(manifest["tokens"]["n_tokens"])
    actual = path.stat().st_size // _itemsize(dtype)
    if actual != expected:
        raise ValueError(f"{path}: {actual} tokens on disk, manifest says {expected}")
    if verify and _sha256_file(path) != manifest["tokens"]["sha256"]:
        raise ValueError(f"{path}: sha256 does not match the manifest")
    return np.memmap(path, dtype=TOKEN_DTYPES[dtype], mode=mmap_mode, shape=(expected,))


def verify_manifest(data_dir: Path) -> dict:
    """Re-hash every shard and the merged stream; raise on any mismatch."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != DATA_FORMAT:
        raise ValueError(f"{data_dir}: not a {DATA_FORMAT} directory")
    dtype = str(manifest["dtype"])
    for record in manifest["shards"]:
        path = data_dir / record["file"]
        if not path.exists():
            raise ValueError(f"{path} is listed in the manifest but missing")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"{path}: sha256 does not match the manifest")
        size = path.stat().st_size
        if size != int(record["n_tokens"]) * _itemsize(dtype):
            raise ValueError(f"{path}: size disagrees with n_tokens")
    tokens = manifest["tokens"]
    path = data_dir / tokens["file"]
    if _sha256_file(path) != tokens["sha256"]:
        raise ValueError(f"{path}: sha256 does not match the manifest")
    return {**manifest, "verified": True}


def wikitext_documents(config: str = "wikitext-103-raw-v1") -> Iterator[str]:
    """The default corpus, streamed from the Hugging Face cache.

    Kept as a module-level helper for callers that pass their own iterable to
    :func:`prepare`. Imported lazily so the module and its tests need neither
    ``datasets`` nor a network.
    """
    spec = resolve_source("wikitext", subset=config)
    yield from HuggingFaceDocuments(spec).open()


def _gpt2_encoder():
    try:
        from transformers import GPT2TokenizerFast
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "the gpt2 tokenizer needs `transformers`; install train_requirements.txt"
        ) from exc
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return encode, tokenizer.vocab_size


def _summary(action: str, out: Path, manifest: dict) -> str:
    tokens = manifest["tokens"]
    path = Path(out) / tokens["file"]
    documents = manifest.get("n_documents")
    docs = "unknown" if documents is None else f"{int(documents):,}"
    return (
        f"{action} {out}: {docs} docs, {manifest['n_tokens']:,} tokens, "
        f"{len(manifest['shards'])} shard(s), {path.stat().st_size:,} bytes, "
        f"sha256 {tokens['sha256']}, path {path}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("prepare", help="tokenize a corpus into a checksummed stream")
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--name", default=None)
    build.add_argument(
        "--source",
        default="wikitext",
        help="wikitext (default), openwebtext, c4, dataset or text",
    )
    build.add_argument("--text", type=Path, default=None, help="with --source text")
    build.add_argument(
        "--dataset", default=None, help="arbitrary Hugging Face dataset name"
    )
    build.add_argument("--subset", default=None, help="dataset config / c4 subset")
    build.add_argument("--split", default="train")
    build.add_argument("--text-field", default="text")
    build.add_argument("--tokenizer", default="gpt2")
    build.add_argument("--dtype", choices=sorted(TOKEN_DTYPES), default="uint16")
    build.add_argument("--eos-id", type=int, default=DEFAULT_EOS_ID)
    build.add_argument("--no-eos", action="store_true", help="do not separate documents")
    build.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    build.add_argument("--max-tokens", type=int, default=None)
    build.add_argument("--resume", action="store_true", help="continue progress.json")
    build.add_argument(
        "--trust-size",
        action="store_true",
        help="reuse by size without verifying the merged sha256",
    )
    build.add_argument("--no-reuse", action="store_true")
    build.add_argument("--tokens-bin", type=Path, default=None)
    build.add_argument("--tokens-npy", type=Path, default=None)

    check = sub.add_parser("check", help="re-hash an existing stream")
    check.add_argument("--out", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "check":
        manifest = verify_manifest(args.out)
        print(
            f"OK: {manifest['n_tokens']:,} tokens, "
            f"{len(manifest['shards'])} shard(s), every sha256 matches"
        )
        return

    if args.tokens_bin and args.tokens_npy:
        raise SystemExit("pass only one of --tokens-bin and --tokens-npy")
    eos_id = None if args.no_eos else args.eos_id
    if args.tokens_bin or args.tokens_npy:
        npy = bool(args.tokens_npy)
        source_path = args.tokens_npy or args.tokens_bin
        kind = "tokens-npy" if npy else "tokens-bin"
        manifest = import_tokens(
            source_path,
            args.out,
            name=args.name or args.out.name or "tokens",
            source=f"{kind}:{source_path}",
            tokenizer=args.tokenizer,
            dtype=args.dtype,
            eos_id=eos_id,
            npy=npy,
        )
    else:
        if args.tokenizer != "gpt2":
            raise SystemExit(
                "only --tokenizer gpt2 is wired up; pass an encode callable in Python"
            )
        try:
            spec = resolve_source(
                args.source,
                subset=args.subset,
                dataset=args.dataset,
                text=args.text,
                split=args.split,
                text_field=args.text_field,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        encode, vocab = _gpt2_encoder()
        documents = (
            TextFileDocuments(spec.text)
            if spec.kind == "text"
            else HuggingFaceDocuments(spec)
        )
        manifest = prepare(
            encode,
            documents,
            args.out,
            name=args.name or args.out.name or "tokens",
            source=spec.label,
            tokenizer=f"gpt2/vocab{vocab}",
            dtype=args.dtype,
            eos_id=eos_id,
            shard_tokens=args.shard_tokens,
            max_tokens=args.max_tokens,
            reuse=not args.no_reuse,
            trust_size=args.trust_size,
            resume=args.resume,
        )
    action = "reused" if manifest.get("reused") else "wrote"
    print(_summary(action, args.out, manifest))


if __name__ == "__main__":
    main()
