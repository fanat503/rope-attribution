# Preparing a token stream on Kaggle

This is the Kaggle-specific recipe for `projects/rope_attribution/data.py`. It
stops at the immutable token stream; training is a separate job and is not run
on Kaggle by this repository. The design is the one in `docs/DATA.md`: tokenize
once, write immutable shards, checksum every file, and let training memory-map
one flat `uint16` array.

**Status boundary.** The commands below exercise the same code path as the CPU
tests in `tests/test_data.py`; no Kaggle kernel has been run as part of this
repository's evidence. The sizes are arithmetic, not a throughput benchmark.

## Why Kaggle needs a different recipe

Kaggle mounts input datasets read-only at `/kaggle/input/<slug>/` and keeps only
`/kaggle/working/` writable. A preparation job that writes beside its input
fails, and one that trains directly from `/kaggle/input` cannot resume an
interrupted shard. The path below therefore writes the whole artefact under
`/kaggle/working/data/<name>/` and leaves the source untouched.

There are two lanes. Prefer the first when the dataset already ships tokens.

## Lane A: a pre-tokenized dataset, no tokenizer and no network

Use this when the input is a flat `uint16` or `uint32` file, or a NumPy `.npy`
array. The source is imported, range-checked, and written with the same manifest
and SHA-256 records as a freshly tokenized corpus. No tokenizer is constructed.

```sh
python -m projects.rope_attribution.data prepare \
    --source text \
    --out /kaggle/working/data/my-corpus \
    --name my-corpus \
    --tokens-bin /kaggle/input/my-token-dataset/tokens.u16 \
    --dtype uint16

python -m projects.rope_attribution.data check \
    --out /kaggle/working/data/my-corpus
```

Use `--tokens-npy /kaggle/input/.../tokens.npy` instead of `--tokens-bin` for a
NumPy array. The `--source text` value is ignored on this lane; it is present
because `prepare` requires a source label.

This lane is the sterile one: set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` if the kernel has no network, and nothing in the import
path will try to fetch a tokenizer.

## Lane B: raw UTF-8 text, tokenized once

Use this when the dataset ships text. `TextFileDocuments` reads
blank-line-separated UTF-8 blocks, so a file with one document per line and a
blank line between documents keeps document boundaries; a file with no blank
lines is one document.

```sh
python -m projects.rope_attribution.data prepare \
    --source text \
    --text /kaggle/input/my-text-dataset/corpus.txt \
    --out /kaggle/working/data/my-corpus \
    --name my-corpus \
    --tokenizer gpt2 \
    --dtype uint16 \
    --shard-tokens 50000000

python -m projects.rope_attribution.data check \
    --out /kaggle/working/data/my-corpus
```

The GPT-2 tokenizer is fetched once and cached by `transformers`; after that the
job can be rerun with `HF_HUB_OFFLINE=1`. The vocabulary is 50,257 ids, so
`uint16` is sufficient and halves the bytes moved against `int32`.

## Size arithmetic

`uint16` is two bytes per token. A 50M-token shard is about 100 MB, and the
merged `tokens.u16` is about 2 GB per billion tokens. Kaggle's writable space is
finite, so a preparation job should either cap the stream with `--max-tokens`
or write only the shards a training job will consume. The manifest records the
exact byte size; use it rather than this paragraph when planning storage.

## Resume and integrity

* Every shard is written to a temporary name and renamed atomically.
* `progress.json` records the document count, token count, source identity,
  tokenizer identity, EOS id and local byte offset; rerun with `--resume` after
  an interruption.
* `manifest.json` records the SHA-256, token count and byte size of every shard
  and of the merged file. `check` re-hashes all of them.
* `prepare` re-hashes an existing merged file before reusing it. `--trust-size`
  skips that check and `--no-reuse` forces a rebuild; neither is a default.
* The stream is read-only to training: `load_tokens` returns a read-only
  `numpy.memmap`, and the sampler casts only the batch it hands to the model.

## What this recipe does not do

* It does not deduplicate, filter for quality, or reweight a mixture beyond what
  the source already did.
* It does not parallelize tokenization. One process reads, encodes and writes;
  the design buys a reproducible, resumable artefact, not tokenizer throughput.
* It does not train. `train.py` refuses a CPU run of the 100M--300M presets
  unless `--force-cpu` is passed, and this document does not ask you to pass it.
* It does not redistribute a corpus. The stream is built locally from the source
  the caller names, and no token file is committed.
