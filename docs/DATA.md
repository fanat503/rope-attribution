# The token stream: pre-tokenized, sharded, memory-mapped

`projects/rope_attribution/data.py` prepares the corpus once and hands training
a flat array it can `mmap`. The design follows the "gigatoken" principle rather
than a runtime tokenizer: tokenization is an offline build step, the artefact is
immutable and checksummed, and the training loop only reads windows from it.

This document maps each principle to the code. It does not claim a throughput
number: tokenization is deliberately single-process and single-threaded, and the
honest statement is the design, not a benchmark on one machine.

## The artefact

```
data/<name>/
    shard-00000.u16      tokenized shards, written once, atomically
    shard-00001.u16
    tokens.u16           their concatenation; what training mmaps
    manifest.json        counts, dtype, EOS id and SHA-256 of every file
    progress.json        resume point, while a build is incomplete
```

`tokens.u16` is the only file on the training hot path. `uint16` holds the
GPT-2 vocabulary (50,257 ids) with room to spare and halves the bytes moved
against `int32`; a vocabulary that does not fit is refused, and
`dtype="uint32"` is the explicit alternative.

## Principles and where they live

| Principle | Implementation |
| --- | --- |
| Tokenize once, offline | `prepare` writes the stream; `train.py` never constructs a tokenizer when `--data-dir` is given |
| Immutable shards | each shard is written to a temporary name and renamed atomically (`_atomic_write`); a finished shard is never edited |
| Bounded retry unit | `--shard-tokens` defaults to 50M tokens (100 MB as `uint16`), so a rejected document costs one shard, not the run |
| Document boundaries | the tokenizer's EOS id is written between documents, never after the last one; `--no-eos` is the explicit raw-concatenation mode |
| One flat readable array | `_merge_shards` concatenates the shards into `tokens.u16`; `load_tokens` returns a read-only `np.memmap` |
| Narrow dtype | `TOKEN_DTYPES` admits `uint16` and `uint32`; the range is validated on import and on read |
| Integrity | `manifest.json` records the SHA-256, token count and byte size of every shard and of the merged file; `check` re-hashes all of them |
| Resume | `progress.json` records `docs_done`, `tokens_done`, the source/tokenizer identity, `eos_id`, the kept shards and a local-file byte offset; it is rewritten atomically after every shard |
| No silent reuse | `prepare` re-hashes an existing merged file before reusing it; `--trust-size` is the explicit opt-out, `--no-reuse` forces a rebuild |
| Deterministic order | the source order is fixed; training draws windows with a per-rank `numpy` generator seeded from `data_seed`, so a resumed rank resumes its own stream |
| No network in the loop | Hugging Face streaming happens only in `prepare`; `check`, `load_tokens` and `train.py --data-dir` read local bytes |

The window sampler keeps the memmap's dtype and casts only the batch it hands to
the model (`train._id_batch`), so a multi-gigabyte stream is never materialised
as `int64`.

## Building and checking

```sh
python -m projects.rope_attribution.data prepare \
    --source wikitext-103 --out data/wikitext103
python -m projects.rope_attribution.data check --out data/wikitext103
python -m projects.rope_attribution.train --config train_124m \
    --run-name my_124m --data-dir data/wikitext103
```

`prepare` also accepts `--source openwebtext`, `--source c4`, an arbitrary
Hugging Face dataset (`--source dataset --dataset NAME`), or a local UTF-8 file
(`--source text --text FILE`). An existing token array can be imported without
re-tokenizing:

```sh
python -m projects.rope_attribution.data prepare \
    --source text --out data/imported --tokens-bin tokens.u16
python -m projects.rope_attribution.data prepare \
    --source text --out data/imported --tokens-npy tokens.npy
```

The import path validates dtype and range, then writes the same manifest and
checksums.

## What this is not

* No deduplication or quality filtering beyond what the source already did.
* No curriculum or mixture weights: the sampler draws uniformly from windows.
* No packing across documents beyond the EOS separator.
* No parallel tokenization. One core reads, encodes and writes; the design buys
  a reproducible, resumable artefact, not maximum tokenizer throughput.
* No corpus redistribution: the stream is built locally from the source the
  caller names, and no token file is committed.

The training-side contract is in `docs/TPU_V5E8.md`; the tests for this module
need neither `datasets` nor a network.
