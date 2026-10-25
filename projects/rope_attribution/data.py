"""Turn a corpus into the token stream a training run consumes, once.

Why this module exists
----------------------
``train.py`` takes an array of token ids and cuts next-token windows out of it.
The default CLI path calls ``datasets`` and tokenizes WikiText-103 in every
process, which is fine for a smoke run and wrong for a real one: eight TPU
ranks would each tokenize the corpus, each hold it as ``int64``, and none of
them could say afterwards which bytes produced the ids. Tokenization is a
deterministic function of (corpus, tokenizer, code), so it belongs in a
separate, checksummed artefact that training only reads.

Layout
------
::

    data/<name>/
        shard-00000.u16      tokenized shards, written once
        shard-00001.u16
        tokens.u16           their concatenation; what training mmaps
        manifest.json        counts, dtype and SHA-256 of every file

``uint16`` holds the GPT-2 vocabulary (50,257 ids) with room to spare and
halves the bytes a ``uint16`` memmap moves against ``int32``. A vocabulary
that does not fit is refused rather than silently truncated; pass
``dtype="uint32"`` for a larger tokenizer.

The training loop keeps the memory map: ``_split_windows`` preserves the
dtype and the loop casts one batch to ``int64`` at a time, so a 200 GB stream
costs page cache, not 200 GB of RAM.

Idempotence and verification
----------------------------
``prepare`` writes each shard to a temporary sibling and replaces it
atomically. A rerun with the same identity fields and matching file sizes
reuses the existing artefact instead of tokenizing again; ``check`` re-hashes
every byte and is the command to run before trusting a copied directory.

What is not claimed
-------------------
No corpus is redistributed here and no download is required by the test
suite. The WikiText reader uses the Hugging Face cache if ``datasets`` is
installed; without it, ``prepare`` still works from any iterable of strings.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np

__all__ = [
    "DATA_FORMAT",
    "TOKEN_DTYPES",
    "prepare",
    "load_tokens",
    "verify_manifest",
    "wikitext_documents",
    "main",
]

#: Bumped when the on-disk contract changes; a reader refuses anything else.
DATA_FORMAT = "rope_attribution/token-data/v1"

TOKEN_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}

#: 50M tokens is 100 MB as uint16: large enough that a shard is a useful unit
#: of retry, small enough that a rejected document does not cost a whole run.
DEFAULT_SHARD_TOKENS = 50_000_000


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


def _write_shards(
    encode: Callable[[str], Sequence[int]],
    documents: Iterable[str],
    out_dir: Path,
    *,
    dtype: str,
    shard_tokens: int,
    max_tokens: int | None,
) -> tuple[list[dict], int, int]:
    """Tokenize documents into shards. Returns records, token count, doc count."""
    if shard_tokens < 1:
        raise ValueError(f"shard_tokens must be positive, not {shard_tokens!r}")
    records: list[dict] = []
    pending: list[np.ndarray] = []
    pending_tokens = 0
    total_tokens = 0
    n_documents = 0

    def flush() -> None:
        nonlocal pending, pending_tokens
        if not pending_tokens:
            return
        tokens = _token_array(pending, dtype)
        name = f"shard-{len(records):05d}.u16"
        path = Path(out_dir) / name
        _atomic_write(path, lambda handle: tokens.tofile(handle))
        records.append(
            {
                "file": name,
                "n_tokens": int(tokens.size),
                "sha256": _sha256_file(path),
            }
        )
        pending = []
        pending_tokens = 0

    for document in documents:
        ids = encode(document)
        if len(ids) == 0:
            # An empty document contributes no tokens, but it is still a
            # document in the manifest: the count is provenance, not training.
            n_documents += 1
            continue
        array = np.asarray(ids, dtype=np.int64).reshape(-1)
        pending.append(array)
        pending_tokens += int(array.size)
        total_tokens += int(array.size)
        n_documents += 1
        if max_tokens is not None and total_tokens >= int(max_tokens):
            flush()
            break
        if pending_tokens >= shard_tokens:
            flush()
    flush()
    return records, total_tokens, n_documents


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
        manifest.get("source"),
        manifest.get("tokenizer"),
        manifest.get("dtype"),
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
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    max_tokens: int | None = None,
    reuse: bool = True,
) -> dict:
    """Tokenize ``documents`` into ``out_dir`` and return the manifest.

    ``encode`` is the tokenizer as a plain callable, which keeps this module
    free of a hard ``transformers`` dependency and lets the tests use an
    encoder whose output is countable by hand. ``reuse`` trusts an existing
    manifest whose identity fields and file sizes match; set it to ``False``
    to tokenize again, and run :func:`verify_manifest` to check checksums.
    """
    if dtype not in TOKEN_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(TOKEN_DTYPES)}, not {dtype!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_name = "tokens.u16" if dtype == "uint16" else "tokens.u32"
    manifest_path = out_dir / "manifest.json"
    if reuse and manifest_path.exists() and (out_dir / final_name).exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = _identity(
            {
                "format": DATA_FORMAT,
                "source": source,
                "tokenizer": tokenizer,
                "dtype": dtype,
            }
        )
        if _identity(existing) == expected:
            # Size check only: hashing a multi-hundred-gigabyte stream on
            # every launch would dominate the run. ``check`` is the deep mode.
            tokens = existing.get("tokens", {})
            path = out_dir / final_name
            if (
                tokens.get("file") == final_name
                and int(tokens.get("n_tokens", -1)) >= 0
                and path.stat().st_size
                == int(tokens["n_tokens"]) * TOKEN_DTYPES[dtype]().itemsize
            ):
                return {**existing, "reused": True}

    for stale in sorted(out_dir.glob("shard-*.u16")):
        stale.unlink()
    for stale in (out_dir / final_name, manifest_path):
        if stale.exists():
            stale.unlink()

    shards, n_tokens, n_documents = _write_shards(
        encode,
        documents,
        out_dir,
        dtype=dtype,
        shard_tokens=shard_tokens,
        max_tokens=max_tokens,
    )
    merged = _merge_shards(shards, out_dir, name=final_name, dtype=dtype)
    if merged["n_tokens"] != n_tokens:
        raise RuntimeError(
            f"merged {merged['n_tokens']} tokens but shards hold {n_tokens}"
        )
    for shard in shards:
        size = (out_dir / shard["file"]).stat().st_size
        if size != int(shard["n_tokens"]) * TOKEN_DTYPES[dtype]().itemsize:
            raise RuntimeError(f"{shard['file']}: size {size} disagrees with count")
    manifest = {
        "format": DATA_FORMAT,
        "name": name,
        "source": source,
        "tokenizer": tokenizer,
        "dtype": dtype,
        "n_tokens": int(n_tokens),
        "n_documents": int(n_documents),
        "shards": shards,
        "tokens": merged,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    return {**manifest, "reused": False}


def load_tokens(data_dir: Path, *, mmap_mode: str = "r") -> np.memmap:
    """The training stream, memory-mapped. Read-only by default.

    This is what ``train(..., tokens=...)`` wants: ``_split_windows`` keeps
    the dtype and only the batch is ever copied.
    """
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != DATA_FORMAT:
        raise ValueError(
            f"{data_dir}: {manifest.get('format')!r} is not {DATA_FORMAT}"
        )
    dtype = str(manifest["dtype"])
    if dtype not in TOKEN_DTYPES:
        raise ValueError(f"{data_dir}: unknown dtype {dtype!r}")
    path = data_dir / manifest["tokens"]["file"]
    expected = int(manifest["tokens"]["n_tokens"])
    actual = path.stat().st_size // TOKEN_DTYPES[dtype]().itemsize
    if actual != expected:
        raise ValueError(
            f"{path}: {actual} tokens on disk, manifest says {expected}"
        )
    return np.memmap(path, dtype=TOKEN_DTYPES[dtype], mode=mmap_mode, shape=(expected,))


def verify_manifest(data_dir: Path) -> dict:
    """Re-hash every shard and the merged stream; raise on any mismatch."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != DATA_FORMAT:
        raise ValueError(f"{data_dir}: not a {DATA_FORMAT} directory")
    for record in manifest["shards"]:
        path = data_dir / record["file"]
        if not path.exists():
            raise ValueError(f"{path} is listed in the manifest but missing")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"{path}: sha256 does not match the manifest")
        size = path.stat().st_size
        if size != int(record["n_tokens"]) * TOKEN_DTYPES[manifest["dtype"]]().itemsize:
            raise ValueError(f"{path}: size disagrees with n_tokens")
    tokens = manifest["tokens"]
    path = data_dir / tokens["file"]
    if _sha256_file(path) != tokens["sha256"]:
        raise ValueError(f"{path}: sha256 does not match the manifest")
    return {**manifest, "verified": True}


def wikitext_documents(config: str = "wikitext-103-raw-v1") -> Iterator[str]:
    """The default corpus, streamed from the Hugging Face cache.

    Imported lazily so the module and its tests need neither ``datasets`` nor
    a network. A missing cache is an error with a command, not a silent empty
    corpus.
    """
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "the wikitext reader needs `datasets`; install train_requirements.txt "
            "or pass a text file / your own iterable of documents"
        ) from exc
    try:
        split = load_dataset("wikitext", config, split="train", streaming=True)
    except Exception as exc:  # pragma: no cover - network/cache path
        raise SystemExit(f"could not open wikitext ({exc})") from exc
    for example in split:
        yield example["text"]


def _read_text_documents(path: Path) -> Iterator[str]:
    """One document per blank-line-separated block, newlines preserved."""
    block: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                block.append(line)
            elif block:
                yield "".join(block)
                block = []
    if block:
        yield "".join(block)


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("prepare", help="tokenize a corpus into a checksummed stream")
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--name", default=None)
    build.add_argument("--source", choices=("wikitext-103", "text"), default="wikitext-103")
    build.add_argument("--text", type=Path, default=None, help="with --source text")
    build.add_argument("--tokenizer", default="gpt2")
    build.add_argument("--dtype", choices=sorted(TOKEN_DTYPES), default="uint16")
    build.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    build.add_argument("--max-tokens", type=int, default=None)
    build.add_argument("--no-reuse", action="store_true")

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

    if args.tokenizer != "gpt2":
        raise SystemExit("only --tokenizer gpt2 is wired up; pass an encode callable in Python")
    encode, vocab = _gpt2_encoder()
    if args.source == "text":
        if args.text is None:
            raise SystemExit("--source text needs --text PATH")
        documents = _read_text_documents(args.text)
        source = f"text:{args.text}"
    else:
        documents = wikitext_documents()
        source = "wikitext-103-raw-v1"
    manifest = prepare(
        encode,
        documents,
        args.out,
        name=args.name or args.out.name,
        source=source,
        tokenizer=f"gpt2/vocab{vocab}",
        dtype=args.dtype,
        shard_tokens=args.shard_tokens,
        max_tokens=args.max_tokens,
        reuse=not args.no_reuse,
    )
    action = "reused" if manifest.get("reused") else "wrote"
    print(
        f"{action} {args.out}: {manifest['n_tokens']:,} tokens from "
        f"{manifest['n_documents']:,} documents in {len(manifest['shards'])} shard(s)"
    )


if __name__ == "__main__":
    main()
