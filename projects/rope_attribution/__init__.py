"""Faithful, computed RoPE / YaRN / partial-RoPE machinery.

See `rope.py` for the transcriptions and their original sources.
"""

from rope_attribution.rope import (
    apply_rope,
    find_correction_dim,
    find_correction_range,
    get_mscale,
    inv_freq,
    linear_ramp_mask,
    partial_rope_cos_sin,
    per_dim_angle,
    rope_cos_sin,
    rotary_angles,
    rotate_half,
    yarn_parameters,
)

