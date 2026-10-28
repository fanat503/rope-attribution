"""The Gemma 4 config planner: geometry per family, and the refusal it explains.

These tests read configs, not weights. The one model they build is a 105k
parameter randomly initialised Gemma 4 text tower, used only to check that the
planner's reading of the config matches the tensors ``transformers`` actually
constructs.
"""

from __future__ import annotations

import json

import pytest
from rope_attribution import gemma4 as G
from rope_attribution import real_model as RM

_E4B_ROPE = {
    "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1_000_000.0,
        "rope_type": "proportional",
    },
    "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
}


def _layer_pattern(n_layers: int) -> list[str]:
    pattern = ["sliding_attention"] * 5 + ["full_attention"]
    return (pattern * (n_layers // len(pattern) + 1))[:n_layers]


def _e4b_config(n_layers: int = 30) -> dict:
    return {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "num_hidden_layers": n_layers,
            "hidden_size": 2304,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "global_head_dim": 512,
            "attention_k_eq_v": False,
            "num_kv_shared_layers": 18,
            "layer_types": _layer_pattern(n_layers),
            "rope_parameters": _E4B_ROPE,
        },
    }


def test_e4b_has_two_families_with_different_ladders() -> None:
    """The full-attention ladder spans 512 channels even though 128 rotate.

    Mutation verdict: changing ``ladder_denominator`` to the rotated width, or
    ``rotated_channels`` to the whole head, fails the exact comparisons below.
    """
    plan = G.audit_config(_e4b_config())

    assert plan["text_tower_nested"] is True
    assert plan["tower_type"] == "gemma4_text"
    assert plan["n_layers"] == 30
    assert [family["name"] for family in plan["families"]] == [
        "sliding_attention",
        "full_attention",
    ]

    sliding, full = plan["families"]
    assert sliding["n_layers"] == 25
    assert sliding["layers"][:6] == [0, 1, 2, 3, 4, 6]
    assert sliding["head_dim"] == 256
    assert sliding["rotated_channels"] == 256
    assert sliding["ladder_denominator"] == 256
    assert sliding["zero_frequency_pairs"] == 0
    assert sliding["rope_theta"] == 10_000.0
    assert sliding["rope_type"] == "default"

    assert full["layers"] == [5, 11, 17, 23, 29]
    assert full["head_dim"] == 512
    assert full["rotated_channels"] == 128
    assert full["rotated_pairs"] == 64
    assert full["ladder_denominator"] == 512
    assert full["zero_frequency_pairs"] == 192
    assert full["rope_theta"] == 1_000_000.0
    assert full["rope_type"] == "proportional"


def test_e4b_is_refused_for_four_distinct_reasons() -> None:
    """The plan repeats the adapter's reason instead of inventing a second one.

    Mutation verdict: deleting any one of the nested-tower, per-family-rope,
    head-dimension or KV-sharing checks removes its phrase from the blocker.
    """
    config = _e4b_config()
    plan = G.audit_config(config)
    reason = RM.unsupported_architecture(config)

    assert plan["measurement_supported"] is False
    assert plan["config_flat"] is False
    assert reason is not None
    assert reason in plan["adapter_blockers"]
    for phrase in (
        "text_config",
        "full_attention",
        "sliding_attention",
        "head_dim is 256",
        "global_head_dim is 512",
        "18 layer(s) share key/value state",
    ):
        assert phrase in reason, phrase
    assert any(
        "rotates 128 of 512 channels on a 512-wide ladder" in blocker
        for blocker in plan["adapter_blockers"]
    )
    assert any(
        "q_norm/k_norm" in requirement
        for requirement in plan["adapter_requirements"]
    )


def test_31b_plan_names_the_tied_key_and_value() -> None:
    """The two public variants differ; the plan reports the variant it read."""
    config = _e4b_config()
    config["text_config"]["attention_k_eq_v"] = True
    config["text_config"]["num_kv_shared_layers"] = 0
    plan = G.audit_config(config)

    assert plan["attention_k_eq_v"] is True
    assert plan["num_kv_shared_layers"] == 0
    assert any(
        "key and value projections are tied" in blocker
        for blocker in plan["adapter_blockers"]
    )
    assert any(
        "read the key from the value projection" in requirement
        for requirement in plan["adapter_requirements"]
    )
    assert not any(
        "share key/value state" in blocker for blocker in plan["adapter_blockers"]
    )


def test_a_flat_config_is_flat_but_still_not_measurable() -> None:
    """One geometry removes the geometry blockers, not the QK-normalisation one.

    Mutation verdict: dropping the q_norm blocker makes ``measurement_supported``
    true here, which is the false claim this test exists to prevent.
    """
    config = {
        "model_type": "gemma4_text",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "layer_types": ["full_attention", "full_attention"],
        "attention_k_eq_v": False,
        "num_kv_shared_layers": 0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
    }
    plan = G.audit_config(config)

    assert plan["config_flat"] is True
    assert plan["measurement_supported"] is False
    assert any("q_norm/k_norm" in blocker for blocker in plan["adapter_blockers"])
    assert RM.unsupported_architecture(config) is None


def test_a_short_layer_type_list_is_expanded_and_never_truncated() -> None:
    """A six-entry pattern describes twelve layers by repetition."""
    config = _e4b_config(n_layers=12)
    config["text_config"]["layer_types"] = _layer_pattern(6)
    plan = G.audit_config(config)

    assert plan["n_layers"] == 12
    assert [family["n_layers"] for family in plan["families"]] == [10, 2]
    assert plan["families"][1]["layers"] == [5, 11]


def test_an_unknown_rope_type_is_reported_not_guessed() -> None:
    """A ladder this planner cannot rebuild is a named gap, not a number."""
    config = _e4b_config(n_layers=6)
    config["text_config"]["rope_parameters"]["full_attention"]["rope_type"] = "yarn"
    plan = G.audit_config(config)

    full = plan["families"][1]
    assert full["geometry_modelled"] is False
    assert any("yarn" in blocker for blocker in plan["adapter_blockers"])


def test_a_missing_head_dimension_is_rejected() -> None:
    config = _e4b_config(n_layers=6)
    del config["text_config"]["head_dim"]
    with pytest.raises(ValueError, match="head_dim"):
        G.audit_config(config)


def test_a_non_gemma4_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="only reads Gemma 4"):
        G.audit_config({"model_type": "llama", "num_hidden_layers": 1})


def test_the_cli_reads_a_local_config_and_emits_json(tmp_path, capsys) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_e4b_config(n_layers=6)), encoding="utf-8")

    G.main(["--config", str(path), "--json"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == G.SCHEMA
    assert plan["n_layers"] == 6
    assert len(plan["families"]) == 2

    G.main(["--config", str(path)])
    text = capsys.readouterr().out
    assert "sliding_attention" in text
    assert "full_attention" in text
    assert "measurement_supported: False" in text


def test_the_plan_matches_the_model_transformers_builds() -> None:
    """The config reading is checked against real tensors, not against itself.

    Mutation verdict: setting the full-attention denominator to the rotated
    width, or reading the sliding theta for both families, fails the buffer
    comparison. The model is randomly initialised and tiny; this is a check of
    the planner, not a Gemma 4 measurement.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    cfg = transformers.Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        max_position_embeddings=128,
        layer_types=["sliding_attention", "full_attention"],
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=16,
        sliding_window=32,
    )
    model = transformers.Gemma4TextModel(cfg)
    plan = G.audit_config(cfg.to_dict())
    sliding, full = plan["families"]

    assert sliding["head_dim"] == model.layers[0].self_attn.head_dim
    assert full["head_dim"] == model.layers[1].self_attn.head_dim
    assert hasattr(model.layers[0].self_attn, "q_norm")
    assert hasattr(model.layers[0].self_attn, "k_norm")

    rotary = next(
        module
        for module in model.modules()
        if type(module).__name__ == "Gemma4TextRotaryEmbedding"
    )
    sliding_freq = rotary.sliding_attention_inv_freq
    full_freq = rotary.full_attention_inv_freq
    assert sliding_freq.shape[0] == sliding["head_dim"] // 2
    assert full_freq.shape[0] == full["head_dim"] // 2
    assert int((full_freq == 0).sum()) == full["zero_frequency_pairs"]

    expected_sliding = torch.tensor(
        [10_000.0 ** (-2.0 * index / sliding["head_dim"]) for index in range(8)]
    )
    torch.testing.assert_close(sliding_freq, expected_sliding)
    expected_full = torch.tensor(
        [1_000_000.0 ** (-2.0 * index / full["head_dim"]) for index in range(4)]
    )
    torch.testing.assert_close(full_freq[:4], expected_full)
    assert torch.count_nonzero(full_freq[4:]) == 0
