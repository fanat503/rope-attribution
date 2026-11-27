"""One command that turns a checkpoint into a pre-registered campaign artifact.

Why a runner exists
-------------------
Before this module the repository had four measurement scripts (``collapse``,
``usefulness``, ``attention_cost``, ``downstream``), each with its own CLI, its
own model list and its own JSON shape.  Running a new checkpoint meant four
invocations, four loads and four chances to forget a seed, a revision or a
prompt hash.  A number that cannot be traced to a pinned revision, a pinned
text and a pinned commit is not evidence; it is a rumour with decimal places.

This runner fixes the provenance once and reuses one loaded checkpoint across
the measurements.

The registry
------------
``configs/models.json`` is the pre-registration.  Every entry pins:

* ``id`` and ``revision`` -- the exact Hub commit, not ``main``;
* ``gated`` -- ``false``, ``"manual"`` or ``"auto"``, so a missing token is a
  known access step rather than a mysterious 401;
* ``expected`` -- the architecture facts the adapter is about to rely on
  (``head_dim``, head counts, ``rope_theta``, ``partial_rotary_factor``,
  ``qk_norm``, sliding window).  The runner compares the loaded config against
  them and refuses to measure when they disagree;
* ``adapter`` and ``status`` -- whether the code can measure this family today.

What the artifact contains
--------------------------
For each model: the provenance block, the registry checks, the full collapse
payload, the full usefulness payload, an optional softmax-decomposition payload,
and a small ``headline`` object for tables.  The headline is a convenience, not
a second source of truth: every number in it is copied from a payload that sits
next to it.

Failure is data
---------------
A model that cannot be loaded (gated, OOM, missing adapter) is recorded as an
``error`` entry and the campaign continues.  A campaign that stops at the first
401 hides the four models that would have run.  The process exits nonzero if any
requested model failed, so CI cannot mistake a partial artifact for a complete
one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rope_attribution import position_free_collapse as PC
from rope_attribution import real_model as RM
from rope_attribution import softmax_decomposition as SD
from rope_attribution import usefulness as U

__all__ = [
    "SCHEMA",
    "REGISTRY_SCHEMA",
    "REQUIRED_MODEL_FIELDS",
    "load_registry",
    "validate_registry",
    "git_state",
    "sha256_text",
    "config_facts",
    "check_expected",
    "provenance",
    "run_model",
    "run_campaign",
    "write_results",
    "main",
]

SCHEMA = "rope_attribution/law_campaign/v1"
REGISTRY_SCHEMA = "rope_attribution/model_registry/v1"

REQUIRED_MODEL_FIELDS = {
    "id",
    "revision",
    "role",
    "family",
    "params_m",
    "gated",
    "adapter",
    "hardware",
    "status",
    "expected",
}

# The only status values the runner understands.  A typo in a hand-edited
# registry must fail loudly rather than silently excluding a model.
STATUS_VALUES = {"baseline", "planned", "optional", "blocked-adapter", "blocked-gated"}
GATED_VALUES = {False, "manual", "auto"}


def load_registry(path: str | Path = "configs/models.json") -> dict:
    """Read and validate the pre-registered model list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_registry(data)
    return data


def validate_registry(data: dict) -> None:
    """Structural checks that do not need a model download."""
    if data.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(
            f"registry schema is {data.get('schema')!r}, expected {REGISTRY_SCHEMA!r}"
        )
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("registry needs a non-empty 'models' list")
    seen: set[str] = set()
    for i, entry in enumerate(models):
        missing = REQUIRED_MODEL_FIELDS - set(entry)
        if missing:
            raise ValueError(
                f"model entry {i} ({entry.get('id', '?')}) is missing "
                f"{sorted(missing)}"
            )
        model_id = entry["id"]
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"model entry {i} has an empty id")
        if model_id in seen:
            raise ValueError(f"duplicate model id {model_id!r}")
        seen.add(model_id)
        revision = entry["revision"]
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"{model_id}: revision must be a non-empty string")
        if revision != "local" and not (
            len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
        ):
            raise ValueError(
                f"{model_id}: revision {revision!r} is not a 40-hex commit and "
                "is not the sentinel 'local'"
            )
        if entry["gated"] not in GATED_VALUES:
            raise ValueError(f"{model_id}: gated must be false, 'manual' or 'auto'")
        if entry["status"] not in STATUS_VALUES:
            raise ValueError(
                f"{model_id}: status {entry['status']!r} is not one of "
                f"{sorted(STATUS_VALUES)}"
            )
        if not isinstance(entry["expected"], dict) or not entry["expected"]:
            raise ValueError(f"{model_id}: expected must be a non-empty object")


def git_state() -> dict:
    """Commit and dirty flag, or nulls when git is unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def sha256_text(text: str) -> str:
    """Stable hash of the exact text a checkpoint sees."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tower(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _rope_block(config: Any) -> dict:
    params = getattr(config, "rope_parameters", None)
    return dict(params) if isinstance(params, dict) else {}


def _float_field(config: Any, names: Sequence[str], default: float) -> float:
    block = _rope_block(config)
    for name in names:
        if name in block and not isinstance(block[name], dict):
            return float(block[name])
        value = getattr(config, name, None)
        if value is not None and not isinstance(value, dict):
            return float(value)
    return float(default)


def _int_field(config: Any, name: str) -> int | None:
    value = getattr(config, name, None)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def config_facts(model: Any) -> dict:
    """The architecture facts the registry claims, read from the live model.

    ``qk_norm`` is detected on the attention modules rather than inferred from
    ``model_type``: an implementation can drop the norm, and the registry must
    describe the code that will actually run.
    """
    cfg = getattr(model, "config", None)
    tower = _tower(cfg)
    attns = RM.attention_modules(model)
    head_dim = None
    n_kv = None
    if attns:
        head_dim = int(RM.head_dim_of(attns[0], tower))
        k_proj = getattr(attns[0], "k_proj", None)
        weight = getattr(k_proj, "weight", None)
        if weight is not None and head_dim:
            n_kv = int(weight.shape[0] // head_dim)
    if head_dim is None:
        head_dim = _int_field(tower, "head_dim")
    if head_dim is None:
        hidden = _int_field(tower, "hidden_size") or _int_field(tower, "n_embd")
        heads = _int_field(tower, "num_attention_heads") or _int_field(tower, "n_head")
        if hidden and heads:
            head_dim = hidden // heads
    if n_kv is None:
        n_kv = _int_field(tower, "num_key_value_heads") or _int_field(
            tower, "num_attention_heads"
        )
    qk_norm = any(
        hasattr(attn, "q_norm") or hasattr(attn, "k_norm") for attn in attns
    )
    return {
        "model_type": str(getattr(cfg, "model_type", "") or ""),
        "hidden_size": _int_field(tower, "hidden_size") or _int_field(tower, "n_embd"),
        "num_hidden_layers": _int_field(tower, "num_hidden_layers")
        or _int_field(tower, "n_layer"),
        "num_attention_heads": _int_field(tower, "num_attention_heads")
        or _int_field(tower, "n_head"),
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "rope_theta": _float_field(tower, ("rope_theta", "rotary_emb_base"), 10_000.0),
        "partial_rotary_factor": _float_field(
            tower, ("partial_rotary_factor", "rotary_pct"), 1.0
        ),
        "qk_norm": qk_norm,
        "sliding_window": getattr(tower, "sliding_window", None),
        "attn_logit_softcapping": getattr(tower, "attn_logit_softcapping", None),
        "rope_local_base_freq": getattr(tower, "rope_local_base_freq", None),
        "sliding_window_pattern": getattr(tower, "sliding_window_pattern", None),
    }


def _same(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) is bool(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-12)
    return expected == actual


def check_expected(model: Any, entry: dict) -> list[dict]:
    """Compare the registry's claims with the loaded config.

    Raises when a field disagrees.  A measurement on a checkpoint whose geometry
    is not the one that was pre-registered is not the pre-registered experiment,
    even if the code runs.
    """
    actual = config_facts(model)
    rows = []
    failures = []
    for name, expected in entry["expected"].items():
        got = actual.get(name)
        ok = _same(expected, got)
        rows.append({"field": name, "expected": expected, "actual": got, "ok": ok})
        if not ok:
            failures.append(f"{name}: expected {expected!r}, got {got!r}")
    if failures:
        raise RuntimeError(
            f"{entry['id']} does not match its registry entry: " + "; ".join(failures)
        )
    return rows


def provenance(entry: dict, texts: Sequence[str]) -> dict:
    """Everything needed to re-run and to disbelieve this artifact later."""
    return {
        "runner_schema": SCHEMA,
        "model_id": entry["id"],
        "revision_requested": entry["revision"],
        "registry_role": entry["role"],
        "registry_family": entry["family"],
        "registry_params_m": entry["params_m"],
        "registry_gated": entry["gated"],
        "registry_adapter": entry["adapter"],
        "registry_hardware": entry["hardware"],
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": RM._version("torch"),
        "transformers": RM._version("transformers"),
        "hostname": platform.node(),
        "text_sha256": [sha256_text(t) for t in texts],
        "n_texts": len(texts),
    }


def _headline(collapse: dict, useful: dict, softmax: dict | None) -> dict:
    window = next(
        (w for w in collapse.get("by_window", []) if int(w["span"]) == 4096), None
    )
    return {
        "channel_rel_mean_median_4096": (
            window["channel_rel_mean_median"] if window else None
        ),
        "channel_rel_minimax_median_4096": (
            window["channel_rel_minimax_median"] if window else None
        ),
        "score_rel_median_4096": window["score_rel_median"] if window else None,
        "past_threshold_fraction_4096": (
            window["past_threshold_fraction"] if window else None
        ),
        "sign_flip_fraction_pooled": useful.get("sign_flip_fraction_pooled"),
        "rel_err_median": useful.get("rel_err_median"),
        "softmax_log_odds_residual_max": (
            softmax.get("log_odds_residual_max") if softmax else None
        ),
        "softmax_centered_gradient_residual_max": (
            softmax.get("centered_gradient_residual_max") if softmax else None
        ),
        "softmax_participation_ratio_mean": (
            softmax.get("participation_ratio_mean") if softmax else None
        ),
    }


def run_model(
    entry: dict,
    *,
    seed: int,
    prompts: Sequence[str],
    windows: Sequence[int],
    n_pairs: int,
    distance_grid: Sequence[int],
    max_tokens: int,
    with_softmax: bool = True,
    softmax_max_tokens: int = 128,
) -> dict:
    """Load once, validate the registry, run every requested measurement."""
    revision = None if entry["revision"] == "local" else entry["revision"]
    loaded = RM.load_model(entry["id"], seed=seed, revision=revision)
    if revision is not None and loaded.revision not in (None, revision):
        raise RuntimeError(
            f"{entry['id']}: registry pins {revision}, loaded {loaded.revision}"
        )
    checks = check_expected(loaded.model, entry)
    prov = provenance(entry, prompts)
    prov["revision_resolved"] = loaded.revision
    try:
        collapse = PC.measure_model(
            entry["id"],
            seed=seed,
            prompts=prompts,
            windows=windows,
            n_pairs=n_pairs,
            loaded=loaded,
        )
        useful = U.measure_model(
            entry["id"],
            seed=seed,
            prompts=prompts,
            deltas=np.asarray(distance_grid, dtype=np.int64),
            n_pairs=n_pairs,
            loaded=loaded,
        )
        softmax = (
            SD.measure_model(
                entry["id"],
                seed=seed,
                prompts=prompts[:1],
                max_tokens=min(int(max_tokens), int(softmax_max_tokens)),
                loaded=loaded,
            )
            if with_softmax
            else None
        )
    finally:
        del loaded
    return {
        "schema": SCHEMA,
        "status": "ok",
        "provenance": prov,
        "registry_checks": checks,
        "collapse": collapse,
        "usefulness": useful,
        "softmax_decomposition": softmax,
        "headline": _headline(collapse, useful, softmax),
    }


def _jsonable(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and reject NaN."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"artifact contains a non-finite float: {value!r}")
    return value


def write_results(payload: dict, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return out


def run_campaign(
    registry: dict,
    *,
    model_ids: Sequence[str] | None = None,
    seed: int | None = None,
    prompts: Sequence[str] | None = None,
    windows: Sequence[int] | None = None,
    n_pairs: int | None = None,
    max_tokens: int | None = None,
    with_softmax: bool = True,
    softmax_max_tokens: int = 128,
) -> dict:
    """Run every requested entry, recording failures instead of aborting."""
    campaign = dict(registry.get("campaign", {}))
    seed = int(campaign.get("seed", RM.DEFAULT_SEED) if seed is None else seed)
    prompts = tuple(RM.DEFAULT_PROMPTS if prompts is None else prompts)
    windows = tuple(campaign.get("windows", PC.WINDOWS) if windows is None else windows)
    distance_grid = tuple(campaign.get("distance_grid", U.DISTANCE_GRID))
    n_pairs = int(PC.N_CONTENT_PAIRS if n_pairs is None else n_pairs)
    max_tokens = int(campaign.get("max_tokens", 512) if max_tokens is None else max_tokens)

    wanted = set(model_ids) if model_ids is not None else None
    if wanted is None:
        # Default means "the pre-registered baseline set", not "everything that
        # might one day run".  ``planned`` entries are promoted to ``baseline``
        # after their first successful campaign; until then they need an
        # explicit ``model_ids`` request, so a default invocation cannot pull a
        # gated 8B checkpoint or an adapter that does not exist yet.
        entries = [
            entry for entry in registry["models"] if entry["status"] == "baseline"
        ]
    else:
        entries = [entry for entry in registry["models"] if entry["id"] in wanted]
        missing = wanted - {entry["id"] for entry in entries}
        if missing:
            raise ValueError(f"registry has no entry for {sorted(missing)}")

    rows = []
    for entry in entries:
        try:
            rows.append(
                run_model(
                    entry,
                    seed=seed,
                    prompts=prompts,
                    windows=windows,
                    n_pairs=n_pairs,
                    distance_grid=distance_grid,
                    max_tokens=max_tokens,
                    with_softmax=with_softmax,
                    softmax_max_tokens=softmax_max_tokens,
                )
            )
        except Exception as exc:  # noqa: BLE001 - the artifact records the reason
            rows.append(
                {
                    "schema": SCHEMA,
                    "status": "error",
                    "provenance": provenance(entry, prompts),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return {
        "schema": SCHEMA,
        "campaign": {
            "seed": seed,
            "windows": [int(w) for w in windows],
            "distance_grid": [int(d) for d in distance_grid],
            "n_pairs": n_pairs,
            "max_tokens": max_tokens,
            "softmax_max_tokens": softmax_max_tokens,
            "with_softmax": with_softmax,
        },
        "models": rows,
        "n_models": len(rows),
        "n_errors": sum(1 for row in rows if row["status"] == "error"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/law_campaign.json"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--softmax-max-tokens", type=int, default=128)
    parser.add_argument("--skip-softmax", action="store_true")
    parser.add_argument("--n-pairs", type=int, default=None)
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    payload = run_campaign(
        registry,
        model_ids=args.models,
        seed=args.seed,
        max_tokens=args.max_tokens,
        with_softmax=not args.skip_softmax,
        softmax_max_tokens=args.softmax_max_tokens,
        n_pairs=args.n_pairs,
    )
    write_results(payload, args.out)
    print(
        f"wrote {args.out} ({payload['n_models']} models, "
        f"{payload['n_errors']} errors)"
    )
    return 1 if payload["n_errors"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
