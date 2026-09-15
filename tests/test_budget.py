"""Tests for the shared acceleration budget and manifest contract."""

import json

import pytest

from hicache_pp import (
    CacheBudget,
    CacheBudgetRuntime,
    RunIdentity,
)


def _identity(**overrides):
    values = {
        "model_id": "demo-model",
        "weights_digest": "weights-v1",
        "schedule_digest": {"steps": 20, "sampler": "flow"},
        "cfg_branch": "cond",
        "conditioning_id": "conditioning-v1",
        "stage": "shape",
        "token_layout_digest": {"n": 128},
        "dtype": "float16",
        "device": "cuda:0",
        "object_id": "object-1",
        "batch_id": "batch-1",
    }
    values.update(overrides)
    return RunIdentity.from_mapping(values)


def test_budget_is_immutable_and_aliases_are_normalized():
    budget = CacheBudget.from_mapping({
        "method": "dmd",
        "stages": ["shape", "texture"],
        "max_interval": 3,
        "quality": "conservative",
        "memory_cap_mb": 1024,
        "audit_budget": 2,
    })
    assert budget.backend == "dmd"
    assert budget.max_horizon == 3
    assert budget.allowed_stages == ("shape", "texture")
    with pytest.raises((AttributeError, TypeError)):
        budget.max_horizon = 4


def test_identity_digest_changes_when_schedule_or_branch_changes():
    first = _identity()
    changed = _identity(schedule_digest={"steps": 30, "sampler": "flow"})
    branch = _identity(cfg_branch="uncond")
    assert first.fingerprint != changed.fingerprint
    assert first.fingerprint != branch.fingerprint


def test_runtime_records_full_forecast_and_fallback_reasons():
    budget = CacheBudget(
        backend="hermite",
        allowed_stages=("shape",),
        max_horizon=2,
        max_memory_mb=100,
        max_predicted_error=0.2,
        audit_budget=1,
    )
    runtime = CacheBudgetRuntime(budget, _identity())
    audit = runtime.decide("shape", horizon=1, audit=True)
    forecast = runtime.decide("shape", horizon=1, predicted_error=0.1, memory_mb=50)
    horizon = runtime.decide("shape", horizon=3)
    memory = runtime.decide("shape", horizon=1, memory_mb=101)
    assert audit.mode == "full" and audit.reason == "budgeted_forward_audit"
    assert forecast.mode == "forecast" and forecast.method == "hermite"
    assert horizon.mode == "fallback" and horizon.reason == "horizon_exceeds_budget"
    assert memory.mode == "fallback" and memory.reason == "memory_cap_exceeded"
    assert runtime.manifest.as_dict()["counts"] == {"full": 1, "forecast": 1, "fallback": 2}


def test_unsupported_or_auto_without_controller_falls_back_and_manifest_has_no_private_payload():
    budget = CacheBudget(backend="auto", allowed_stages=("shape",), fallback="full")
    runtime = CacheBudgetRuntime(budget, _identity())
    unsupported = runtime.decide("texture", horizon=1, supported=False)
    auto = runtime.decide("shape", horizon=1)
    selected = runtime.decide("shape", horizon=1, controller_selected=True)
    assert unsupported.method == "full"
    assert auto.reason == "controller_selection_required"
    assert selected.mode == "forecast" and selected.method == "auto"
    with pytest.raises(ValueError):
        runtime.record_measurement("shape", "hermite", wall_time_ms=1, quality={"image_path": "secret"})


def test_manifest_json_is_deterministic_and_only_contains_digests_for_inputs():
    budget = CacheBudget(backend="hermite", allowed_stages=("shape",))
    runtime = CacheBudgetRuntime(
        budget,
        _identity(),
        source_digest={"commit": "abc"},
        input_digest={"seed": 7},
    )
    runtime.decide("shape", horizon=1)
    runtime.record_measurement(
        "shape",
        "hermite",
        wall_time_ms=12.5,
        peak_memory_mb=512,
        quality={"geometry_l1": 0.03},
        output_digest={"mesh": "abc"},
        validation={"passed": True},
    )
    payload = runtime.manifest.to_json()
    decoded = json.loads(payload)
    assert decoded["source"]["input_digest"]
    assert "seed" not in payload
    assert "tensor" not in payload.lower()
    assert payload == runtime.manifest.to_json()
