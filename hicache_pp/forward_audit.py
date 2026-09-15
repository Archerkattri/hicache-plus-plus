"""Forward-observation audit primitives for cache-controller experiments.

The ordinary ``auto`` backend chooses from a backcast inside the already
observed history.  This module records a different kind of evidence: a
prediction issued at an earlier compute anchor is scored only when the future
feature is actually computed.  It deliberately handles flat tensors first;
the tree adapter can wrap the same event contract without making nested
payloads part of the persistence format.

This is an empirical audit ledger, not a conformal or output-quality
guarantee.  A prediction is never scored against a value from the same
history window, and changing run/branch/stage identity invalidates pending
events rather than silently mixing trajectories.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


def _relative_l2(prediction: torch.Tensor, truth: torch.Tensor, eps: float) -> float:
    if prediction.shape != truth.shape:
        raise ValueError(
            f"prediction/truth shape mismatch: {tuple(prediction.shape)} vs {tuple(truth.shape)}"
        )
    if not torch.isfinite(prediction).all() or not torch.isfinite(truth).all():
        raise ValueError("forward-audit tensors must be finite")
    denominator = truth.detach().to(torch.float64).norm().clamp_min(eps)
    numerator = (prediction.detach().to(torch.float64) - truth.detach().to(torch.float64)).norm()
    return float((numerator / denominator).item())


@dataclass(frozen=True)
class ForwardEvent:
    """One prediction issued before its target was observed."""

    run_id: str
    branch_id: str | None
    stage: str
    method: str
    anchor_step: int
    target_step: int
    horizon: int
    fit_cost_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardObservation:
    """A scored forward event."""

    event: ForwardEvent
    relative_l2: float

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self.event)
        row["relative_l2"] = self.relative_l2
        return row


class ForwardAuditLedger:
    """Collect forward-only cache evidence for one trajectory identity.

    ``issue`` is called at a paid anchor after a candidate prediction has been
    produced.  ``observe`` is called at a later paid anchor with the actual
    feature.  Pending predictions from another run identity are discarded by
    ``start``; they are never scored against a new request or a changed stage.
    """

    def __init__(
        self,
        run_id: str,
        *,
        branch_id: str | None = None,
        stage: str = "default",
        eps: float = 1e-12,
        max_pending: int = 4096,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if not stage:
            raise ValueError("stage must be non-empty")
        if eps <= 0 or not math.isfinite(eps):
            raise ValueError("eps must be finite and positive")
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self.run_id = str(run_id)
        self.branch_id = None if branch_id is None else str(branch_id)
        self.stage = str(stage)
        self.eps = float(eps)
        self.max_pending = int(max_pending)
        self._pending: list[tuple[ForwardEvent, torch.Tensor]] = []
        self.observations: list[ForwardObservation] = []
        self.dropped_pending = 0

    def start(
        self,
        run_id: str,
        *,
        branch_id: str | None = None,
        stage: str | None = None,
    ) -> None:
        """Switch trajectory identity and invalidate all pending predictions."""
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self.dropped_pending += len(self._pending)
        self._pending.clear()
        self.run_id = str(run_id)
        self.branch_id = None if branch_id is None else str(branch_id)
        if stage is not None:
            if not stage:
                raise ValueError("stage must be non-empty")
            self.stage = str(stage)

    def issue(
        self,
        method: str,
        *,
        anchor_step: int,
        target_step: int,
        prediction: torch.Tensor,
        fit_cost_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ForwardEvent:
        """Record a prediction for a strictly later full-compute target."""
        if not method:
            raise ValueError("method must be non-empty")
        if target_step <= anchor_step:
            raise ValueError("target_step must be strictly later than anchor_step")
        if not torch.is_tensor(prediction):
            raise TypeError("forward-audit predictions must be torch tensors")
        if not torch.isfinite(prediction).all():
            raise ValueError("forward-audit predictions must be finite")
        if not math.isfinite(float(fit_cost_ms)) or float(fit_cost_ms) < 0:
            raise ValueError("fit_cost_ms must be finite and non-negative")
        if len(self._pending) >= self.max_pending:
            self._pending.pop(0)
            self.dropped_pending += 1
        event = ForwardEvent(
            run_id=self.run_id,
            branch_id=self.branch_id,
            stage=self.stage,
            method=str(method),
            anchor_step=int(anchor_step),
            target_step=int(target_step),
            horizon=int(target_step - anchor_step),
            fit_cost_ms=float(fit_cost_ms),
            metadata=dict(metadata or {}),
        )
        self._pending.append((event, prediction.detach().clone()))
        return event

    def observe(self, target_step: int, truth: torch.Tensor) -> list[ForwardObservation]:
        """Score all pending predictions whose future target is now observed."""
        if not torch.is_tensor(truth):
            raise TypeError("forward-audit truths must be torch tensors")
        matched: list[ForwardObservation] = []
        keep: list[tuple[ForwardEvent, torch.Tensor]] = []
        for event, prediction in self._pending:
            if event.target_step != int(target_step):
                keep.append((event, prediction))
                continue
            score = _relative_l2(prediction, truth, self.eps)
            observation = ForwardObservation(event=event, relative_l2=score)
            self.observations.append(observation)
            matched.append(observation)
        self._pending = keep
        return matched

    def pending(self) -> tuple[ForwardEvent, ...]:
        """Return pending event metadata without exposing stored tensors."""
        return tuple(event for event, _ in self._pending)

    def summary(self) -> dict[str, Any]:
        """Return deterministic method/horizon aggregates for reports."""
        buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
        costs: dict[tuple[str, int], list[float]] = defaultdict(list)
        for obs in self.observations:
            key = (obs.event.method, obs.event.horizon)
            buckets[key].append(obs.relative_l2)
            costs[key].append(obs.event.fit_cost_ms)
        rows = []
        for (method, horizon), values in sorted(buckets.items()):
            vals = torch.tensor(values, dtype=torch.float64)
            rows.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "n": len(values),
                    "mean_relative_l2": float(vals.mean().item()),
                    "p95_relative_l2": float(torch.quantile(vals, 0.95).item()),
                    "mean_fit_cost_ms": sum(costs[(method, horizon)]) / len(values),
                }
            )
        return {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "stage": self.stage,
            "observations": len(self.observations),
            "pending": len(self._pending),
            "dropped_pending": self.dropped_pending,
            "rows": rows,
        }

    def records(self) -> list[dict[str, Any]]:
        """Return scored observations in issue/observation order."""
        return [obs.as_dict() for obs in self.observations]

