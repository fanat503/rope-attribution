"""Gigatoken lane: tokenize a text corpus at GB/s, then keep the same artifact.

Why this module exists
----------------------
``data.py`` can tokenize with a Hugging Face tokenizer, which is correct but
slow: one Python process, one document at a time.  On Kaggle the tokenizer is
the bottleneck before a training run, not the model.  `gigatoken` is a Rust
tokenizer that reads files itself and parallelises across cores; its published
throughput on GPT-2 is tens of GB/s on a server and several GB/s on a desktop.

The point of this module is *not* to add a second data format.  It encodes,
flattens the per-document result, and hands the ids to
``data.import_tokens``, so the output is the same immutable ``tokens.u16``
plus ``manifest.json`` with a SHA-256 per file that the rest of the repository
already reads.  Swapping the encoder cannot change what training sees.

Availability
------------
``gigatoken`` is an optional dependency and is imported inside the function
that needs it; the test suite exercises this module with an injected encoder
and skips nothing.  ``pip install gigatoken`` on Kaggle is the only extra step.

Mutation verdicts
-----------------
* ``test_flatten_documents_concatenates_in_order`` fails if the flattening
  sorts, deduplicates or drops empty documents, which would silently change
  the token stream.
* ``test_prepare_rejects_ids_above_uint16`` fails if the range check is
  removed: a uint32 id would wrap to a different token instead of raising.
* ``test_prepare_writes_the_same_manifest_as_import_tokens`` fails if this
  module writes its own manifest instead of delegating, which would let the
  two lanes drift.
"""

from __future__ import annotations

import argparse
import itertools
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from rope_attribution import data as data_module

__all__ = [
    "DEFAULT_SEPARATOR",
    "DEFAULT_TOKENIZER",
    "encode_text_files",
    "flatten_token_documents",
    "prepare",
    "main",
]

#: The GPT-2 tokenizer the training presets use, by Hub id.
DEFAULT_TOKENIZER = "openai-community/gpt2"
#: Document separator used by the OpenWebText-style corpora.
DEFAULT_SEPARATOR = "<|endoftext|>"
#: Largest id a uint16 stream can hold.
_UINT16_MAX = int(np.iinfo(np.uint16).max)


def flatten_token_documents(documents: Iterable) -> np.ndarray:
    """Concatenate per-document token sequences into one ``uint32`` vector.

    ``gigatoken.encode_files`` returns one row per document, and rows have
    different lengths.  Empty documents are legal (two adjacent separators)
    and contribute nothing; they are not an error, because dropping or
    reordering rows would change the stream.
    """
    if isinstance(documents, np.ndarray) and documents.ndim == 1:
        return np.asarray(documents, dtype=np.uint32)

    # Awkward arrays (what gigatoken actually returns) expose ``layout`` and
    # flatten in one pass; a plain list of lists goes through itertools.
    if hasattr(documents, "layout"):
        import awkward as ak

        flat = ak.flatten(documents, axis=None)
        return np.asarray(flat, dtype=np.uint32)

    values = list(
        itertools.chain.from_iterable(np.asarray(row).reshape(-1) for row in documents)
    )
    if not values:
        return np.zeros(0, dtype=np.uint32)
    return np.asarray(values, dtype=np.uint32)


def encode_text_files(
    paths: Sequence[str | Path],
    *,
    tokenizer: str = DEFAULT_TOKENIZER,
    separator: str | bytes | None = DEFAULT_SEPARATOR,
    tokenizer_factory: Callable[[str], object] | None = None,
    source_factory: Callable[..., object] | None = None,
) -> np.ndarray:
    """Encode plain-text files with gigatoken and return one flat id vector.

    ``separator`` is the document boundary inside a file, not a token added
    to the stream; pass ``None`` to treat each whole file as one document.
    ``tokenizer_factory`` and ``source_factory`` exist so the test suite can
    drive this function without the optional Rust extension; production code
    passes neither.
    """
    if tokenizer_factory is None or source_factory is None:
        try:
            import gigatoken as gt
        except ImportError as exc:  # pragma: no cover - exercised without it
            raise RuntimeError(
                "gigatoken is not installed; run `pip install gigatoken` on the "
                "machine that prepares the stream, or use the Hugging Face lane "
                "in `data.py`"
            ) from exc
        if tokenizer_factory is None:
            tokenizer_factory = gt.Tokenizer
        if source_factory is None:

            def source_factory(paths, separator):
                return gt.TextFileSource(
                    [str(p) for p in paths], separator=separator
                )

    encoder = tokenizer_factory(tokenizer)
    source = source_factory(paths, separator=separator)
    return flatten_token_documents(encoder.encode_files(source))


def prepare(
    text_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    name: str,
    tokenizer: str = DEFAULT_TOKENIZER,
    separator: str | bytes | None = DEFAULT_SEPARATOR,
    dtype: str = "uint16",
    eos_id: int | None = data_module.DEFAULT_EOS_ID,
    max_tokens: int | None = None,
    encoder: Callable[..., np.ndarray] | None = None,
) -> dict:
    """Tokenize ``text_paths`` and write the checksummed stream to ``out_dir``.

    The ids are written to a temporary raw file and imported with
    ``data.import_tokens``, so the manifest, SHA-256 records and ``check``
    command are the ones the repository already documents.  ``max_tokens``
    truncates the *encoded* stream, which is what a Kaggle storage budget
    constrains; it does not change the tokenizer or the shard format.
    """
    encode = encode_text_files if encoder is None else encoder
    tokens = np.asarray(
        encode(text_paths, tokenizer=tokenizer, separator=separator),
        dtype=np.uint32,
    ).reshape(-1)
    if tokens.size and int(tokens.max()) > _UINT16_MAX and dtype == "uint16":
        raise ValueError(
            f"token id {int(tokens.max())} does not fit uint16; use --dtype uint32"
        )
    if max_tokens is not None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        tokens = tokens[: int(max_tokens)]

    if dtype not in data_module.TOKEN_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(data_module.TOKEN_DTYPES)}")
    itemsize = int(data_module.TOKEN_DTYPES[dtype]().itemsize)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gigatoken-") as tmp:
        raw = Path(tmp) / "tokens.raw"
        # The raw file must use the same width the importer will read, or a
        # uint16 stream would be read as half as many uint32 values.
        raw.write_bytes(
            np.asarray(tokens, dtype=f"<u{itemsize}").tobytes(order="C")
        )
        return data_module.import_tokens(
            raw,
            out_dir,
            name=name,
            source=f"gigatoken:{','.join(str(p) for p in text_paths)}",
            tokenizer=tokenizer,
            dtype=dtype,
            eos_id=eos_id,
            npy=False,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", nargs="+", required=True, help="UTF-8 text files")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--separator",
        default=DEFAULT_SEPARATOR,
        help="document separator inside a file; pass an empty string for none",
    )
    parser.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16")
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args(argv)
    separator = args.separator if args.separator else None
    manifest = prepare(
        args.text,
        args.out,
        name=args.name,
        tokenizer=args.tokenizer,
        separator=separator,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
    )
    print(
        f"{manifest['name']}: {manifest['n_tokens']} tokens, "
        f"{manifest['tokens']['sha256'][:12]}..., {args.out}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
