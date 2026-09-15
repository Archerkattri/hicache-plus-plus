"""Explicit quality/latency/memory budgets and portable run manifests.

The forecasting modules answer *how* to produce a predicted feature.  This
module answers the deployment question: when is that prediction allowed to be
served for a particular stage and run identity?  It is deliberately small and
does not import a GPU framework, so adapters can use it during configuration,
CPU tests and installed-workflow startup.

The contract is operational rather than a theorem.  A budget can force a full
compute fallback when a stage, horizon, memory estimate or quality estimate is
outside the calibrated envelope.  ``RunManifest`` records the actual decision
and cost events; it never stores tensors, prompts, images or local paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FORECAST_METHODS = ("hermite", "dmd", "auto")
DECISION_MODES = ("full", "forecast", "fallback")
_HEX_DIGEST = re.compile(r"^(?:sha(?:1|224|256|384|512):)?[0-9a-fA-F]{32,128}$")
_PRIVATE_KEYS = ("path", "filename", "filepath", "prompt", "image", "tensor", "latent")


def _canonical(value: Any) -> Any:
    """Convert supported values to deterministic JSON-compatible primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest values must be finite")
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical(item) for item in value]
    raise TypeError(f"unsupported manifest value type: {type(value).__name__}")


def stable_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON-like value."""
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _public_token(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    token = str(value).strip()
    if not token:
        return ""
    if any(character in token for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{field} must not contain control characters")
    if "\\" in token or "/" in token:
        raise ValueError(f"{field} must be an opaque identifier, not a local path")
    return token


def _digest_token(value: Any, *, field: str) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str) and _HEX_DIGEST.fullmatch(value.strip()):
        return value.strip().lower()
    return stable_digest(value)


def _finite_nonnegative(value: Any, *, field: str, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


@dataclass(frozen=True)
class CacheBudget:
    """Immutable deployment envelope for one acceleration configuration.

    ``max_horizon`` is the maximum number of future steps a forecast may be
    served away from its paid anchor.  ``max_memory_mb`` and
    ``max_predicted_error`` are optional guardrails learned from a locked
    model/sampler envelope.  ``audit_budget`` is the maximum number of explicit
    paid audit decisions a run may request; it is not a guarantee that an audit
    will happen automatically.
    """

    backend: str = "hermite"
    allowed_stages: tuple[str, ...] = ("default",)
    max_horizon: int = 1
    quality_preset: str = "balanced"
    max_memory_mb: float | None = None
    audit_budget: int = 0
    fallback: str = "full"
    max_predicted_error: float | None = None

    def __post_init__(self) -> None:
        backend = str(self.backend).strip().lower()
        fallback = str(self.fallback).strip().lower()
        if backend not in (*FORECAST_METHODS, "full"):
            raise ValueError("backend must be full, hermite, dmd or auto")
        if fallback not in (*FORECAST_METHODS, "full"):
            raise ValueError("fallback must be full, hermite, dmd or auto")
        if not self.allowed_stages:
            raise ValueError("allowed_stages must contain at least one stage")
        stages = tuple(dict.fromkeys(_public_token(stage, field="stage") for stage in self.allowed_stages))
        if any(not stage for stage in stages):
            raise ValueError("allowed_stages must contain non-empty stage identifiers")
        if int(self.max_horizon) != self.max_horizon or int(self.max_horizon) < 1:
            raise ValueError("max_horizon must be a positive integer")
        quality_preset = _public_token(self.quality_preset, field="quality_preset")
        if not quality_preset:
            raise ValueError("quality_preset must be non-empty")
        if int(self.audit_budget) != self.audit_budget or int(self.audit_budget) < 0:
            raise ValueError("audit_budget must be a non-negative integer")
        memory = _finite_nonnegative(self.max_memory_mb, field="max_memory_mb")
        if memory == 0:
            raise ValueError("max_memory_mb must be positive when provided")
        error = _finite_nonnegative(self.max_predicted_error, field="max_predicted_error")
        if error == 0:
            raise ValueError("max_predicted_error must be positive when provided")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "fallback", fallback)
        object.__setattr__(self, "allowed_stages", stages)
        object.__setattr__(self, "max_horizon", int(self.max_horizon))
        object.__setattr__(self, "quality_preset", quality_preset)
        object.__setattr__(self, "audit_budget", int(self.audit_budget))
        object.__setattr__(self, "max_memory_mb", memory)
        object.__setattr__(self, "max_predicted_error", error)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CacheBudget":
        """Build a budget while accepting the common adapter aliases."""
        stages = values.get("allowed_stages", values.get("stages", ("default",)))
        # A single stage is a useful config shorthand.  Treat it as one
        # identifier instead of iterating over its characters.
        if isinstance(stages, str):
            stages = (stages,)
        return cls(
            backend=values.get("backend", values.get("method", "hermite")),
            allowed_stages=tuple(stages),
            max_horizon=values.get("max_horizon", values.get("max_interval", values.get("horizon", 1))),
            quality_preset=values.get("quality_preset", values.get("quality", "balanced")),
            max_memory_mb=values.get("max_memory_mb", values.get("memory_cap_mb")),
            audit_budget=values.get("audit_budget", 0),
            fallback=values.get("fallback", "full"),
            max_predicted_error=values.get("max_predicted_error", values.get("error_cap")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "allowed_stages": list(self.allowed_stages),
            "max_horizon": self.max_horizon,
            "quality_preset": self.quality_preset,
            "max_memory_mb": self.max_memory_mb,
            "audit_budget": self.audit_budget,
            "fallback": self.fallback,
            "max_predicted_error": self.max_predicted_error,
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.as_dict())


@dataclass(frozen=True)
class RunIdentity:
    """Identity fields that must match before cached state can be reused.

    Schedules, conditioning and layouts are represented by opaque identifiers
    or content digests.  This prevents a portable report from accidentally
    persisting a private prompt, tensor, image or local path.
    """

    model_id: str
    run_id: str = ""
    weights_digest: str = ""
    schedule_digest: str = ""
    cfg_branch: str = ""
    conditioning_id: str = ""
    stage: str = "default"
    token_layout_digest: str = ""
    dtype: str = ""
    device: str = ""
    object_id: str = ""
    batch_id: str = ""

    def __post_init__(self) -> None:
        fields = {
            "model_id": _public_token(self.model_id, field="model_id"),
            "run_id": _public_token(self.run_id, field="run_id"),
            "weights_digest": _digest_token(self.weights_digest, field="weights_digest"),
            "schedule_digest": _digest_token(self.schedule_digest, field="schedule_digest"),
            "cfg_branch": _public_token(self.cfg_branch, field="cfg_branch"),
            "conditioning_id": _digest_token(self.conditioning_id, field="conditioning_id"),
            "stage": _public_token(self.stage, field="stage"),
            "token_layout_digest": _digest_token(self.token_layout_digest, field="token_layout_digest"),
            "dtype": _public_token(self.dtype, field="dtype"),
            "device": _public_token(self.device, field="device"),
            "object_id": _public_token(self.object_id, field="object_id"),
            "batch_id": _public_token(self.batch_id, field="batch_id"),
        }
        if not fields["model_id"] or not fields["stage"]:
            raise ValueError("model_id and stage must be non-empty")
        for field, value in fields.items():
            object.__setattr__(self, field, value)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RunIdentity":
        return cls(
            model_id=values.get("model_id", values.get("model", "unknown")),
            run_id=values.get("run_id", ""),
            weights_digest=values.get("weights_digest", values.get("weights", "")),
            schedule_digest=values.get("schedule_digest", values.get("schedule", "")),
            cfg_branch=values.get("cfg_branch", values.get("branch", "")),
            conditioning_id=values.get("conditioning_id", values.get("conditioning", "")),
            stage=values.get("stage", "default"),
            token_layout_digest=values.get("token_layout_digest", values.get("token_layout", "")),
            dtype=values.get("dtype", ""),
            device=values.get("device", ""),
            object_id=values.get("object_id", values.get("object", "")),
            batch_id=values.get("batch_id", values.get("batch", "")),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "run_id": self.run_id,
            "weights_digest": self.weights_digest,
            "schedule_digest": self.schedule_digest,
            "cfg_branch": self.cfg_branch,
            "conditioning_id": self.conditioning_id,
            "stage": self.stage,
            "token_layout_digest": self.token_layout_digest,
            "dtype": self.dtype,
            "device": self.device,
            "object_id": self.object_id,
            "batch_id": self.batch_id,
        }

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.as_dict())


@dataclass(frozen=True)
class BudgetDecision:
    """One auditable full/forecast/fallback decision."""

    mode: str
    method: str
    reason: str
    stage: str
    horizon: int
    identity_fingerprint: str
    predicted_error: float | None = None
    predicted_cost_ms: float | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        method = _public_token(self.method, field="method")
        reason = _public_token(self.reason, field="reason")
        stage = _public_token(self.stage, field="stage")
        if mode not in DECISION_MODES:
            raise ValueError(f"mode must be one of {DECISION_MODES}")
        if not method or not reason or not stage:
            raise ValueError("method, reason and stage must be non-empty")
        if int(self.horizon) != self.horizon or int(self.horizon) < 0:
            raise ValueError("horizon must be a non-negative integer")
        _public_token(self.identity_fingerprint, field="identity_fingerprint")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "predicted_error", _finite_nonnegative(self.predicted_error, field="predicted_error"))
        object.__setattr__(self, "predicted_cost_ms", _finite_nonnegative(self.predicted_cost_ms, field="predicted_cost_ms"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "method": self.method,
            "reason": self.reason,
            "stage": self.stage,
            "horizon": self.horizon,
            "identity_fingerprint": self.identity_fingerprint,
            "predicted_error": self.predicted_error,
            "predicted_cost_ms": self.predicted_cost_ms,
        }


def _validate_public_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """Reject obvious private payloads before they enter a portable manifest."""
    output: dict[str, Any] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        lower = key.lower()
        if any(term in lower for term in _PRIVATE_KEYS) and not lower.endswith("_digest"):
            raise ValueError(f"private payload key is not allowed in a manifest: {key}")
        if isinstance(raw_value, Mapping):
            output[key] = _validate_public_mapping(raw_value)
        elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes, bytearray)):
            output[key] = [_canonical(item) for item in raw_value]
        else:
            output[key] = _canonical(raw_value)
    return output


class RunManifest:
    """Portable report containing actual decisions and measured costs."""

    schema = "hicache-pp.run-manifest.v1"

    def __init__(
        self,
        budget: CacheBudget,
        identity: RunIdentity,
        *,
        source_digest: Any = "",
        model_digest: Any = "",
        config_digest: Any = "",
        input_digest: Any = "",
    ) -> None:
        if not isinstance(budget, CacheBudget) or not isinstance(identity, RunIdentity):
            raise TypeError("RunManifest requires CacheBudget and RunIdentity")
        self.budget = budget
        self.identity = identity
        self.source = {
            "source_digest": _digest_token(source_digest, field="source_digest"),
            "model_digest": _digest_token(model_digest, field="model_digest"),
            "config_digest": _digest_token(config_digest, field="config_digest"),
            "input_digest": _digest_token(input_digest, field="input_digest"),
        }
        self.events: list[dict[str, Any]] = []
        self.measurements: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()

    def record_decision(self, decision: BudgetDecision) -> None:
        if decision.identity_fingerprint != self.identity.fingerprint:
            raise ValueError("decision identity does not match manifest identity")
        self.events.append(decision.as_dict())
        self.counts[decision.mode] += 1

    def record_measurement(
        self,
        stage: str,
        method: str,
        *,
        wall_time_ms: float,
        peak_memory_mb: float | None = None,
        quality: Mapping[str, Any] | None = None,
        output_digest: Any = None,
        validation: Mapping[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "stage": _public_token(stage, field="stage"),
            "method": _public_token(method, field="method"),
            "wall_time_ms": _finite_nonnegative(wall_time_ms, field="wall_time_ms", allow_none=False),
        }
        if not row["stage"] or not row["method"]:
            raise ValueError("measurement stage and method must be non-empty")
        if peak_memory_mb is not None:
            row["peak_memory_mb"] = _finite_nonnegative(peak_memory_mb, field="peak_memory_mb")
        if quality is not None:
            row["quality"] = _validate_public_mapping(quality)
        if output_digest is not None:
            row["output_digest"] = _digest_token(output_digest, field="output_digest")
        if validation is not None:
            row["validation"] = _validate_public_mapping(validation)
        self.measurements.append(row)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "budget": self.budget.as_dict(),
            "budget_digest": self.budget.digest,
            "identity": self.identity.as_dict(),
            "identity_fingerprint": self.identity.fingerprint,
            "source": dict(self.source),
            "counts": {mode: int(self.counts.get(mode, 0)) for mode in DECISION_MODES},
            "events": list(self.events),
            "measurements": list(self.measurements),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=indent, ensure_ascii=True, allow_nan=False)

    def write_json(self, destination: str | Path) -> None:
        Path(destination).write_text(self.to_json() + "\n", encoding="utf-8")


class CacheBudgetRuntime:
    """Apply a ``CacheBudget`` and emit a manifest event for every decision."""

    def __init__(
        self,
        budget: CacheBudget,
        identity: RunIdentity,
        *,
        source_digest: Any = "",
        model_digest: Any = "",
        config_digest: Any = "",
        input_digest: Any = "",
    ) -> None:
        self.budget = budget
        self.manifest = RunManifest(
            budget,
            identity,
            source_digest=source_digest,
            model_digest=model_digest,
            config_digest=config_digest,
            input_digest=input_digest,
        )
        self.audit_used = 0

    @property
    def identity(self) -> RunIdentity:
        return self.manifest.identity

    @property
    def audits_remaining(self) -> int:
        return max(0, self.budget.audit_budget - self.audit_used)

    def start(self, identity: RunIdentity) -> None:
        """Begin a new run and discard the previous run's budget state."""
        self.manifest = RunManifest(self.budget, identity)
        self.audit_used = 0

    def _decision(
        self,
        mode: str,
        method: str,
        reason: str,
        stage: str,
        horizon: int,
        predicted_error: float | None,
        predicted_cost_ms: float | None,
    ) -> BudgetDecision:
        decision = BudgetDecision(
            mode=mode,
            method=method,
            reason=reason,
            stage=stage,
            horizon=horizon,
            identity_fingerprint=self.identity.fingerprint,
            predicted_error=predicted_error,
            predicted_cost_ms=predicted_cost_ms,
        )
        self.manifest.record_decision(decision)
        return decision

    def decide(
        self,
        stage: str,
        *,
        horizon: int,
        method: str | None = None,
        supported: bool = True,
        predicted_error: float | None = None,
        predicted_cost_ms: float | None = None,
        memory_mb: float | None = None,
        force_full: bool = False,
        audit: bool = False,
        controller_selected: bool = False,
    ) -> BudgetDecision:
        """Return and record the operational decision for one requested step."""
        stage = _public_token(stage, field="stage")
        if not stage:
            raise ValueError("stage must be non-empty")
        if int(horizon) != horizon or int(horizon) < 0:
            raise ValueError("horizon must be a non-negative integer")
        horizon = int(horizon)
        requested = self.budget.backend if method is None else str(method).strip().lower()
        if requested not in (*FORECAST_METHODS, "full"):
            raise ValueError("method must be full, hermite, dmd or auto")
        if memory_mb is not None:
            memory_mb = _finite_nonnegative(memory_mb, field="memory_mb")
        if predicted_error is not None:
            predicted_error = _finite_nonnegative(predicted_error, field="predicted_error")
        if predicted_cost_ms is not None:
            predicted_cost_ms = _finite_nonnegative(predicted_cost_ms, field="predicted_cost_ms")

        if force_full or requested == "full":
            return self._decision("full", "full", "full_compute_reference", stage, horizon, predicted_error, predicted_cost_ms)
        if audit and self.audits_remaining:
            self.audit_used += 1
            return self._decision("full", "full", "budgeted_forward_audit", stage, horizon, predicted_error, predicted_cost_ms)

        reason = None
        if not supported:
            reason = "unsupported_stage_or_schedule"
        elif stage not in self.budget.allowed_stages:
            reason = "stage_outside_budget"
        elif horizon == 0:
            reason = "zero_horizon_requires_full_compute"
        elif horizon > self.budget.max_horizon:
            reason = "horizon_exceeds_budget"
        elif self.budget.max_memory_mb is not None and memory_mb is not None and memory_mb > self.budget.max_memory_mb:
            reason = "memory_cap_exceeded"
        elif self.budget.max_predicted_error is not None and predicted_error is not None and predicted_error > self.budget.max_predicted_error:
            reason = "predicted_error_exceeds_budget"
        elif requested == "auto" and not controller_selected:
            reason = "controller_selection_required"

        if reason is not None:
            fallback = self.budget.fallback
            if fallback == "full":
                return self._decision("fallback", "full", reason, stage, horizon, predicted_error, predicted_cost_ms)
            return self._decision("fallback", fallback, reason, stage, horizon, predicted_error, predicted_cost_ms)
        return self._decision("forecast", requested, "within_budget", stage, horizon, predicted_error, predicted_cost_ms)

    def record_measurement(self, *args: Any, **kwargs: Any) -> None:
        self.manifest.record_measurement(*args, **kwargs)
