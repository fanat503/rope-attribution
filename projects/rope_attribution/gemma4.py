"""Config-level audit of a Gemma 4 text tower.

``real_model.py`` refuses Gemma 4 before it fetches a weight, because the
measurement adapter assumes one rotary geometry for the whole network. This
module reads a ``config.json`` and reports the geometry the checkpoint would
actually run, family by family. It reads no weights and imports no
``transformers``; it is a plan, not a measurement.

Why the plan is not a formality
-------------------------------
Gemma 4 does not merely use two head dimensions. Under ``proportional`` RoPE
the frequency ladder spans the *full* head dimension while only
``partial_rotary_factor`` of the channels rotate. The full-attention layers of
the E4B tower rotate 128 of 512 channels, but their 64 frequencies are
``theta ** (-2 i / 512)``, not ``theta ** (-2 i / 128)``. An adapter that takes
``rope_dim = int(head_dim * partial)`` and rebuilds ``inv_freq`` on that width --
which is exactly what ``real_model.validate_config`` does for a flat config --
gets a different ladder and would report finite residuals for a rotation the
model never applies. ``docs/GEMMA4.md`` records the rest of the refusal.

The output is deliberately explicit about what is missing: ``config_flat`` says
whether the config declares one geometry, and ``adapter_blockers`` says why the
current adapter still cannot measure it. A flat Gemma 4 configuration remains
refused, because Gemma 4 applies ``q_norm``/``k_norm`` after the projections and
the adapter hooks the projections.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rope_attribution.real_model import unsupported_architecture

__all__ = ["SCHEMA", "audit_config", "load_config", "main"]

SCHEMA = "gemma4-plan/v1"

_FAMILIES = ("sliding_attention", "full_attention")

#: RoPE initialisers this planner can reproduce. Anything else is reported, not
#: guessed: a made-up ladder is worse than a named gap.
_MODELLED_ROPE_TYPES = frozenset({"default", "proportional"})


def _positive_int(value: Any, what: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be an integer, got {value!r}") from exc
    if number <= 0:
        raise ValueError(f"{what} must be positive, got {number}")
    return number


def _tower(config: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """The text-tower mapping, and whether the config nested it."""
    nested = config.get("text_config")
    if nested is None:
        return dict(config), False
    if not isinstance(nested, Mapping):
        raise ValueError("text_config must be an object")
    return dict(nested), True


def _layer_types(tower: Mapping[str, Any], n_layers: int) -> list[str]:
    """One layer type per layer, expanding the six-layer default pattern."""
    raw = tower.get("layer_types")
    if raw is None:
        return [
            "sliding_attention" if (index + 1) % 6 else "full_attention"
            for index in range(n_layers)
        ]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("layer_types must be a list")
    pattern = [str(name) for name in raw]
    if not pattern:
        raise ValueError("layer_types must not be empty")
    unknown = sorted(set(pattern) - set(_FAMILIES))
    if unknown:
        raise ValueError(f"unknown layer types: {', '.join(unknown)}")
    if len(pattern) == n_layers:
        return pattern
    if n_layers % len(pattern):
        raise ValueError(
            f"layer_types has {len(pattern)} entries, which does not divide "
            f"num_hidden_layers={n_layers}"
        )
    return pattern * (n_layers // len(pattern))


def _rope_parameters(tower: Mapping[str, Any], family: str) -> dict[str, Any]:
    """The rope block for one family, accepting the flat legacy form."""
    rope = tower.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("rope_parameters must be an object")
    if {"rope_theta", "rope_type", "partial_rotary_factor"} & set(rope):
        return dict(rope)
    block = rope.get(family)
    if not isinstance(block, Mapping):
        raise ValueError(f"rope_parameters has no block for {family!r}")
    return dict(block)


def _geometry(head_dim: int, params: Mapping[str, Any]) -> dict[str, Any]:
    """The ladder and the rotated width, for one family.

    ``proportional`` keeps the denominator at the full head dimension and pads
    the unrotated tail with zero frequencies. ``default`` rotates the whole
    head, so a declared partial factor is recorded but not applied. A declared
    factor that does not land on a whole number of pairs is rejected rather
    than rounded: a half pair is not a rotation.
    """
    theta = float(params.get("rope_theta", 10_000.0))
    partial = float(params.get("partial_rotary_factor", 1.0))
    rope_type = str(params.get("rope_type", "default"))
    if theta <= 0.0:
        raise ValueError(f"rope_theta must be positive, got {theta}")
    if not 0.0 < partial <= 1.0:
        raise ValueError(f"partial_rotary_factor must be in (0, 1], got {partial}")

    exact_pairs = partial * head_dim / 2.0
    if abs(exact_pairs - round(exact_pairs)) > 1e-9:
        raise ValueError(
            f"partial_rotary_factor={partial} of head_dim={head_dim} is "
            f"{exact_pairs:.6f} pairs, not a whole number"
        )
    declared_pairs = int(round(exact_pairs))

    if rope_type == "proportional":
        rotated_pairs = declared_pairs
        denominator = head_dim
    elif rope_type == "default":
        # The default initialiser ignores partial_rotary_factor and rotates
        # every channel; recording the declared factor keeps the config visible.
        rotated_pairs = head_dim // 2
        denominator = head_dim
    else:
        rotated_pairs = declared_pairs
        denominator = head_dim

    if rotated_pairs * 2 > head_dim:
        raise ValueError(
            f"rotated width {2 * rotated_pairs} exceeds head_dim={head_dim}"
        )
    zero_pairs = head_dim // 2 - rotated_pairs
    return {
        "rope_theta": theta,
        "rope_type": rope_type,
        "partial_rotary_factor_declared": partial,
        "rotated_channels": 2 * rotated_pairs,
        "rotated_pairs": rotated_pairs,
        "zero_frequency_pairs": zero_pairs,
        "ladder_denominator": denominator,
        "geometry_modelled": rope_type in _MODELLED_ROPE_TYPES,
    }


def audit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe every layer family of a Gemma 4 text tower.

    Raises ``ValueError`` on a config that cannot be described: a missing head
    dimension, an unknown layer type, a partial factor that is not a whole
    number of pairs, or layer families whose layers disagree on a dimension.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    tower, nested = _tower(config)
    tower_type = str(tower.get("model_type", "") or "")
    top_type = str(config.get("model_type", "") or "")
    if not (tower_type.startswith("gemma4") or top_type.startswith("gemma4")):
        raise ValueError("this planner only reads Gemma 4 text towers")

    n_layers = _positive_int(tower.get("num_hidden_layers"), "num_hidden_layers")
    hidden_size = _positive_int(tower.get("hidden_size"), "hidden_size")
    head_dim = _positive_int(tower.get("head_dim"), "head_dim")
    global_head_dim = tower.get("global_head_dim")
    if global_head_dim is not None:
        global_head_dim = _positive_int(global_head_dim, "global_head_dim")

    layer_types = _layer_types(tower, n_layers)
    per_layer = tower.get("per_layer_config") or {}
    if not isinstance(per_layer, Mapping):
        raise ValueError("per_layer_config must be an object")

    k_eq_v = bool(tower.get("attention_k_eq_v", False))
    shared = int(tower.get("num_kv_shared_layers", 0) or 0)
    if shared < 0:
        raise ValueError("num_kv_shared_layers must be non-negative")

    grouped: dict[str, list[int]] = {}
    for index, family in enumerate(layer_types):
        grouped.setdefault(family, []).append(index)

    families: list[dict[str, Any]] = []
    for family in _FAMILIES:
        layers = grouped.get(family)
        if not layers:
            continue
        head_dims: set[int] = set()
        kv_heads: set[int] = set()
        for index in layers:
            override = per_layer.get(str(index), {})
            if not isinstance(override, Mapping):
                raise ValueError(f"per_layer_config[{index!r}] must be an object")
            layer_head = override.get("head_dim")
            if layer_head is None:
                if family == "full_attention" and global_head_dim is not None:
                    layer_head = global_head_dim
                else:
                    layer_head = head_dim
            head_dims.add(_positive_int(layer_head, f"head_dim at layer {index}"))
            layer_kv = override.get(
                "num_key_value_heads", tower.get("num_key_value_heads")
            )
            if layer_kv is not None:
                kv_heads.add(_positive_int(layer_kv, f"num_key_value_heads at layer {index}"))

        if len(head_dims) != 1:
            raise ValueError(
                f"{family} layers disagree on head_dim: {sorted(head_dims)}"
            )
        family_head_dim = next(iter(head_dims))
        geometry = _geometry(family_head_dim, _rope_parameters(tower, family))
        families.append(
            {
                "name": family,
                "layers": layers,
                "n_layers": len(layers),
                "head_dim": family_head_dim,
                "num_key_value_heads": (
                    next(iter(kv_heads)) if len(kv_heads) == 1 else None
                ),
                "num_key_value_heads_values": sorted(kv_heads) if kv_heads else [],
                **geometry,
            }
        )

    blockers: list[str] = [
        "Gemma 4 applies q_norm/k_norm after q_proj/k_proj; "
        "real_model.collect_head_vectors refuses QK-normalised attention, "
        "because a projection hook would measure a vector the model never rotates"
    ]
    adapter_reason = unsupported_architecture(config)
    if adapter_reason:
        blockers.append(adapter_reason)
    for family in families:
        if not family["geometry_modelled"]:
            blockers.append(
                f"{family['name']} uses rope_type={family['rope_type']!r}, "
                "which this planner does not reproduce"
            )
        if (
            family["rope_type"] == "proportional"
            and family["rotated_channels"] < family["head_dim"]
        ):
            blockers.append(
                f"{family['name']} rotates {family['rotated_channels']} of "
                f"{family['head_dim']} channels on a "
                f"{family['ladder_denominator']}-wide ladder; rebuilding the "
                f"ladder on {family['rotated_channels']} channels would change "
                "every frequency"
            )

    requirements: list[str] = []
    if len(families) > 1:
        requirements.append(
            "measure each layer family in its own pass: the families differ in "
            "head dimension, theta and rotated width"
        )
    for family in families:
        if family["rotated_channels"] != family["head_dim"]:
            requirements.append(
                f"build {family['name']}'s ladder on "
                f"{family['ladder_denominator']} channels and rotate only the "
                f"first {family['rotated_channels']}; int(head_dim * partial) "
                "is the wrong denominator"
            )
    if k_eq_v:
        requirements.append(
            "read the key from the value projection; attention_k_eq_v means "
            "there is no separate key projection"
        )
    if shared:
        requirements.append(
            f"follow the KV-sharing graph for {shared} layer(s); a shared "
            "layer reuses another layer's key"
        )
    requirements.append(
        "hook q and k after q_norm/k_norm; the projections are not the tensors "
        "RoPE rotates"
    )

    config_flat = (
        len(families) == 1
        and families[0]["rotated_channels"] == families[0]["head_dim"]
        and not k_eq_v
        and not shared
    )
    return {
        "schema": SCHEMA,
        "model_type": top_type or tower_type,
        "tower_type": tower_type,
        "text_tower_nested": nested,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": int(tower.get("num_attention_heads", 0) or 0),
        "attention_k_eq_v": k_eq_v,
        "num_kv_shared_layers": shared,
        "config_flat": config_flat,
        "measurement_supported": not blockers,
        "families": families,
        "adapter_blockers": blockers,
        "adapter_requirements": requirements,
        "note": (
            "config-level plan only: no weights were read, and no Gemma 4 "
            "number is a measurement in this repository"
        ),
    }


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a ``config.json`` and return the audited plan."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} does not contain a JSON object")
    return audit_config(payload)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="a local config.json; only the config is read, never the weights",
    )
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    plan = load_config(args.config)
    if args.json:
        print(json.dumps(plan, indent=2))
        return
    print(
        f"{plan['tower_type']}: {plan['n_layers']} layers, "
        f"hidden {plan['hidden_size']}, "
        f"{'nested' if plan['text_tower_nested'] else 'top-level'} text tower"
    )
    print(
        f"{'family':<18} {'layers':>6} {'head':>4} {'rot':>4} {'ladder':>6} "
        f"{'theta':>10} {'type':>12}"
    )
    for family in plan["families"]:
        print(
            f"{family['name']:<18} {family['n_layers']:>6} "
            f"{family['head_dim']:>4} {family['rotated_channels']:>4} "
            f"{family['ladder_denominator']:>6} "
            f"{family['rope_theta']:>10.0f} {family['rope_type']:>12}"
        )
    print(f"config_flat: {plan['config_flat']}")
    print(f"measurement_supported: {plan['measurement_supported']}")
    print("adapter blockers:")
    for blocker in plan["adapter_blockers"]:
        print(f"  - {blocker}")
    print("what an adapter would need:")
    for requirement in plan["adapter_requirements"]:
        print(f"  - {requirement}")


if __name__ == "__main__":
    main()
