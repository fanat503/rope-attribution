"""Tests for the campaign runner and its model registry.

The runner is the place where a number acquires its provenance, so the tests
are about refusal as much as about success: an unpinned revision, a duplicated
id, a geometry that disagrees with the registry, or a NaN in the artifact must
stop the campaign rather than produce a plausible file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from rope_attribution import law_campaign as LC


def _entry(**overrides) -> dict:
    entry = {
        "id": "fake/model",
        "revision": "a" * 40,
        "role": "test",
        "family": "llama",
        "params_m": 1,
        "gated": False,
        "adapter": "flat",
        "hardware": "cpu",
        "status": "baseline",
        "expected": {"hidden_size": 32},
    }
    entry.update(overrides)
    return entry


def _registry(*entries) -> dict:
    return {
        "schema": LC.REGISTRY_SCHEMA,
        "campaign": {"seed": 7, "windows": [8, 16], "distance_grid": [1, 2]},
        "models": list(entries) or [_entry()],
    }


def test_the_shipped_registry_is_valid_and_every_revision_is_pinned() -> None:
    """The pre-registration must not contain a moving branch name.

    Mutation verdict: replacing any 40-hex revision with ``main`` makes the
    validation raise, and this test fails before a checkpoint is downloaded.
    """
    registry = LC.load_registry("configs/models.json")
    assert registry["schema"] == LC.REGISTRY_SCHEMA
    ids = [entry["id"] for entry in registry["models"]]
    assert len(ids) == len(set(ids))
    assert "EleutherAI/pythia-160m" in ids
    for entry in registry["models"]:
        assert entry["revision"] == "local" or len(entry["revision"]) == 40


def test_a_missing_field_is_rejected() -> None:
    registry = _registry()
    del registry["models"][0]["expected"]
    with pytest.raises(ValueError, match="missing"):
        LC.validate_registry(registry)


def test_a_floating_revision_is_rejected() -> None:
    with pytest.raises(ValueError, match="40-hex"):
        LC.validate_registry(_registry(_entry(revision="main")))


def test_a_duplicate_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate model id"):
        LC.validate_registry(_registry(_entry(), _entry()))


def test_an_unknown_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="status"):
        LC.validate_registry(_registry(_entry(status="probably-fine")))


def _tiny_llama():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


def test_check_expected_accepts_a_matching_config() -> None:
    model = _tiny_llama()
    facts = LC.config_facts(model)
    entry = _entry(expected={k: v for k, v in facts.items() if v is not None})
    rows = LC.check_expected(model, entry)
    assert all(row["ok"] for row in rows)
    assert {row["field"] for row in rows} >= {"hidden_size", "head_dim", "qk_norm"}


def test_check_expected_names_the_mismatched_field() -> None:
    model = _tiny_llama()
    entry = _entry(expected={"hidden_size": 999, "head_dim": 8})
    with pytest.raises(RuntimeError, match="hidden_size: expected 999"):
        LC.check_expected(model, entry)


def test_config_facts_detect_qk_norm_on_a_tiny_qwen3() -> None:
    """The registry's ``qk_norm`` is read from modules, not from ``model_type``.

    Mutation verdict: inferring QK-norm from the family name would report
    ``False`` for a Llama whose attention was given a ``q_norm``; reading the
    modules reports what will actually run.
    """
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    facts = LC.config_facts(model)
    assert facts["qk_norm"] is True
    assert facts["head_dim"] == 8
    assert facts["num_key_value_heads"] == 2


def test_provenance_hashes_the_exact_texts() -> None:
    entry = _entry()
    prov = LC.provenance(entry, ["alpha", "beta"])
    assert prov["text_sha256"] == [
        LC.sha256_text("alpha"),
        LC.sha256_text("beta"),
    ]
    assert prov["n_texts"] == 2
    assert prov["revision_requested"] == "a" * 40
    commit = prov["git"]["commit"]
    assert commit is None or len(commit) == 40


def test_write_results_refuses_a_non_finite_float() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        LC.write_results({"x": float("nan")}, "unused.json")


def test_write_results_round_trips_numpy_scalars(tmp_path) -> None:
    path = LC.write_results(
        {"a": np.int64(3), "b": np.float64(0.5), "c": np.arange(3)},
        tmp_path / "artifact.json",
    )
    payload = json.loads(path.read_text())
    assert payload == {"a": 3, "b": 0.5, "c": [0, 1, 2]}


def test_run_campaign_records_a_load_error_instead_of_aborting(monkeypatch) -> None:
    """A gated checkpoint is one failed row, not a lost campaign.

    Mutation verdict: removing the per-model ``try`` makes the exception escape
    and this test fails with the loader's error instead of an ``error`` row.
    """

    def refuse(*_args, **_kwargs):
        raise OSError("401 Unauthorized")

    monkeypatch.setattr(LC.RM, "load_model", refuse)
    payload = LC.run_campaign(_registry(_entry()))
    assert payload["n_models"] == 1
    assert payload["n_errors"] == 1
    row = payload["models"][0]
    assert row["status"] == "error"
    assert row["error_type"] == "OSError"
    assert "401" in row["error"]


def test_a_blocked_model_is_skipped_unless_explicitly_requested(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run(entry, **_kwargs):
        calls.append(entry["id"])
        return {"status": "ok", "provenance": {"model_id": entry["id"]}}

    monkeypatch.setattr(LC, "run_model", fake_run)
    registry = _registry(
        _entry(id="ready/model"),
        _entry(id="blocked/model", status="blocked-adapter"),
    )
    LC.run_campaign(registry)
    assert calls == ["ready/model"]
    LC.run_campaign(registry, model_ids=["blocked/model"])
    assert calls == ["ready/model", "blocked/model"]


def test_a_planned_model_needs_an_explicit_request(monkeypatch) -> None:
    """Default means the baseline set, not every entry that might run later.

    Mutation verdict: restoring the old ``status not in (blocked, ...)`` filter
    makes the planned entry run and the first assertion fails; removing the
    explicit branch makes the second fail.
    """
    calls: list[str] = []

    def fake_run(entry, **_kwargs):
        calls.append(entry["id"])
        return {"status": "ok", "provenance": {"model_id": entry["id"]}}

    monkeypatch.setattr(LC, "run_model", fake_run)
    registry = _registry(
        _entry(id="ready/model"),
        _entry(id="planned/model", status="planned"),
    )
    LC.run_campaign(registry)
    assert calls == ["ready/model"]
    LC.run_campaign(registry, model_ids=["planned/model"])
    assert calls == ["ready/model", "planned/model"]


def test_run_model_refuses_a_snapshot_that_moved_off_the_pin(monkeypatch) -> None:
    """A registry pin that resolves to a different commit is not that experiment.

    Mutation verdict: deleting the revision comparison lets the call fall
    through to ``check_expected`` on a fake without a model, so the expected
    ``RuntimeError`` about the pin never appears and the test fails.
    """

    class Moved:
        revision = "b" * 40

    monkeypatch.setattr(LC.RM, "load_model", lambda *_a, **_k: Moved())
    with pytest.raises(RuntimeError, match="registry pins"):
        LC.run_model(
            _entry(),
            seed=7,
            prompts=("x",),
            windows=(8,),
            n_pairs=1,
            distance_grid=(1,),
            max_tokens=8,
        )


def test_the_headline_is_copied_from_the_payloads() -> None:
    collapse = {
        "by_window": [
            {"span": 4096, "channel_rel_mean_median": 1.13,
             "channel_rel_minimax_median": 1.0, "score_rel_median": 0.91,
             "past_threshold_fraction": 0.62}
        ]
    }
    useful = {"sign_flip_fraction_pooled": 0.27, "rel_err_median": 1.13}
    softmax = {
        "log_odds_residual_max": 1e-12,
        "centered_gradient_residual_max": 1e-6,
        "participation_ratio_mean": 1.9,
    }
    headline = LC._headline(collapse, useful, softmax)
    assert headline["channel_rel_mean_median_4096"] == 1.13
    assert headline["sign_flip_fraction_pooled"] == 0.27
    assert headline["softmax_participation_ratio_mean"] == 1.9
    assert LC._headline(collapse, useful, None)["softmax_log_odds_residual_max"] is None
