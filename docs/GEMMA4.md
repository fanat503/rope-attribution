# Gemma 4: config plan, and why it is not a measurement

**Status boundary.** `projects/rope_attribution/gemma4.py` reads a
`config.json` and describes the rotary geometry of each layer family. It reads
no weights, downloads nothing by itself, and produces no measured number. No
Gemma 4 result is part of the paper or of `results/`. The adapter in
`real_model.py` refuses Gemma 4 before fetching a checkpoint, and this document
records why that refusal is structural rather than cautious.

## What the adapter assumes

`real_model.validate_config` reads one `rope_theta`, one
`partial_rotary_factor`, one `head_dim`, and rebuilds a single frequency ladder.
`collect_head_vectors` hooks one `q_proj`/`k_proj` pair per attention module and
assumes the projection output is the vector RoPE rotates. Gemma 4 violates all
of those assumptions at once.

## What the E4B config declares

The fields below are the ones transcribed in `tests/test_gemma4.py` from the
public `config.json` of `google/gemma-4-E4B-it`. They are configuration, not a
measurement.

| Field | Sliding layers | Full-attention layers |
| --- | ---: | ---: |
| Layer count in the 30-layer tower | 25 | 5 |
| `head_dim` | 256 | 512 (`global_head_dim`) |
| `rope_theta` | 10,000 | 1,000,000 |
| `rope_type` | `default` | `proportional` |
| `partial_rotary_factor` | 1.0 | 0.25 |
| Channels that rotate | 256 of 256 | 128 of 512 |
| Frequency denominator | 256 | 512 |
| Zero-frequency pairs in the ladder | 0 | 192 |

The full-attention row is the subtle one. Under `proportional` RoPE the
frequency ladder still spans the full 512-wide head; the unrotated 384 channels
are padded with zero frequencies, so their cosine is one and their sine is zero.
The 64 live frequencies are `theta ** (-2 i / 512)`, not
`theta ** (-2 i / 128)`. An adapter that sets
`rope_dim = int(head_dim * partial)` and rebuilds `inv_freq(rope_dim, theta)`
would therefore compare the checkpoint against a different ladder, and the
residuals would look finite while describing a rotation the model never applies.
This is the reason the config-level plan records both `rotated_channels` and
`ladder_denominator` rather than one `rope_dim`.

Two more fields change the measurement:

* **QK normalisation.** Gemma 4 applies `q_norm` and `k_norm` after the
  projections. A hook on `q_proj`/`k_proj` sees the unnormalised vector; RoPE
  rotates the normalised one. `collect_head_vectors` already refuses
  QK-normalised attention for Gemma 2/3 for this reason.
* **KV sharing.** E4B sets `num_kv_shared_layers = 18`: a shared layer reuses
  another layer's key instead of projecting a fresh one. The 31B variant sets
  `attention_k_eq_v = true`, so its key is the value projection and there is no
  separate key projection to hook.

## Reading a config

Only the config is needed; the weights are not:

```sh
python -c "from huggingface_hub import hf_hub_download; \
print(hf_hub_download('google/gemma-4-E4B-it', 'config.json'))"
```

Gated checkpoints need a Hugging Face token and an accepted licence. Then:

```sh
PYTHONPATH=projects python -m projects.rope_attribution.gemma4 \
    --config /path/to/config.json
PYTHONPATH=projects python -m projects.rope_attribution.gemma4 \
    --config /path/to/config.json --json
```

The text output is a table of families, followed by the adapter blockers and
the changes a future adapter would need. The JSON output is the same plan under
`schema: gemma4-plan/v1`. The planner accepts the top-level `Gemma4Config`
(with a nested `text_config`) or the text tower directly.

## What a future adapter would need

1. **One pass per layer family.** The families differ in head dimension, theta
   and rotated width; pooling them into one residual would average two
   different geometries.
2. **The proportional ladder.** Build the full-head ladder and slice its live
   prefix; do not rebuild a ladder on the rotated width.
3. **Post-normalisation hooks.** Read `q` and `k` after `q_norm`/`k_norm`, or
   apply the same normalisation to the projection output before rotating.
4. **The KV graph.** Follow `num_kv_shared_layers` and `attention_k_eq_v` so a
   shared layer is attributed to the layer that actually produced its key.

None of this is implemented in the measurement adapter today. The planner is
the checked description of the gap, not a workaround that produces numbers.

## What is tested

`tests/test_gemma4.py` covers the two public variants, the proportional ladder,
the six-layer pattern expansion, malformed configs, the JSON/CLI path, and the
planner's refusal text. One test builds a 105k-parameter randomly initialised
Gemma 4 text tower with `transformers` and checks the plan against the tensors
that library constructs: the per-layer `head_dim`, the presence of
`q_norm`/`k_norm`, the two `inv_freq` buffers and the zero-frequency tail. That
test checks the planner, not a released checkpoint.

The planner imports no `transformers`; only the tiny-model test does, and it is
skipped when `torch` or `transformers` is absent.
