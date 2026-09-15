"""Small cost-aware controller driven by *forward* audit observations.

The controller is intentionally conservative: before a candidate has enough
future observations it returns the configured fallback.  Once candidates are
eligible it minimizes mean audited feature error plus a normalized fit-cost
penalty.  This is an empirical runtime policy, not an output-quality or formal
coverage guarantee.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .forward_audit import ForwardObservation


@dataclass(frozen=True)
class ControllerChoice:
    method: str
    reason: str
    score: float | None
    evidence_n: int


class ForwardRiskController:
    """Select a forecast method from delayed, held-out future evidence."""

    def __init__(
        self,
        methods: list[str] | tuple[str, ...],
        *,
        fallback: str,
        min_observations: int = 3,
        cost_weight: float = 0.0,
        max_forecast_norm_ratio: float = float("inf"),
    ) -> None:
        unique = tuple(dict.fromkeys(str(m) for m in methods if str(m)))
        if not unique:
            raise ValueError("at least one method is required")
        if fallback not in unique:
            raise ValueError("fallback must be one of methods")
        if min_observations < 1:
            raise ValueError("min_observations must be >= 1")
        if cost_weight < 0 or not math.isfinite(float(cost_weight)):
            raise ValueError("cost_weight must be finite and non-negative")
        if max_forecast_norm_ratio <= 0 or not (
            math.isfinite(float(max_forecast_norm_ratio))
            or math.isinf(float(max_forecast_norm_ratio))
        ):
            raise ValueError("max_forecast_norm_ratio must be positive and finite or inf")
        self.methods = unique
        self.fallback = fallback
        self.min_observations = int(min_observations)
        self.cost_weight = float(cost_weight)
        self.max_forecast_norm_ratio = float(max_forecast_norm_ratio)
        self._scores: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)

    def update(self, observations: list[ForwardObservation] | tuple[ForwardObservation, ...]) -> None:
        for obs in observations:
            method = obs.event.method
            if method in self.methods:
                self._scores[(method, obs.event.horizon)].append(
                    (float(obs.relative_l2), float(obs.event.fit_cost_ms))
                )

    def choose(
        self, horizon: int, *, health: dict[str, float] | None = None
    ) -> ControllerChoice:
        """Choose from audited candidates, applying a current-step health guard.

        ``health`` is computed from the currently observed anchor history (for
        example forecast norm divided by the newest anchor norm).  It is not
        future truth.  Candidates with missing/non-finite or excessive health
        are excluded, except for the configured fallback, which remains the
        last-resort operational arm.
        """
        eligible = []
        guarded: list[str] = []
        costs = [
            cost
            for method in self.methods
            for error, cost in self._scores.get((method, int(horizon)), [])
        ]
        cost_scale = max(max(costs, default=0.0), 1e-12)
        for order, method in enumerate(self.methods):
            if method != self.fallback and health is not None:
                ratio = health.get(method)
                if ratio is None or not math.isfinite(float(ratio)) or ratio > self.max_forecast_norm_ratio:
                    guarded.append(method)
                    continue
            samples = self._scores.get((method, int(horizon)), [])
            if len(samples) < self.min_observations:
                continue
            mean_error = sum(error for error, _ in samples) / len(samples)
            mean_cost = sum(cost for _, cost in samples) / len(samples)
            score = mean_error + self.cost_weight * mean_cost / cost_scale
            eligible.append((score, order, method, len(samples)))
        if not eligible:
            health_blocked = False
            if health is not None:
                health_blocked = any(
                    method != self.fallback
                    and (
                        health.get(method) is None
                        or not math.isfinite(float(health[method]))
                        or health[method] > self.max_forecast_norm_ratio
                    )
                    for method in self.methods
                )
            reason = "current_forecast_health_guard" if health_blocked else "insufficient_forward_evidence"
            return ControllerChoice(
                method=self.fallback,
                reason=reason,
                score=None,
                evidence_n=len(self._scores.get((self.fallback, int(horizon)), [])),
            )
        score, _, method, n = min(eligible)
        reason = (
            "current_forecast_health_guard"
            if method == self.fallback and guarded
            else "minimum_cost_adjusted_forward_error"
        )
        return ControllerChoice(
            method=method,
            reason=reason,
            score=float(score),
            evidence_n=n,
        )

    def report(self) -> dict[str, object]:
        return {
            "methods": list(self.methods),
            "fallback": self.fallback,
            "min_observations": self.min_observations,
            "cost_weight": self.cost_weight,
            "max_forecast_norm_ratio": self.max_forecast_norm_ratio,
            "evidence": [
                {
                    "method": method,
                    "horizon": horizon,
                    "n": len(values),
                    "mean_relative_l2": sum(error for error, _ in values) / len(values),
                    "mean_fit_cost_ms": sum(cost for _, cost in values) / len(values),
                }
                for (method, horizon), values in sorted(self._scores.items())
                if values
            ],
        }
