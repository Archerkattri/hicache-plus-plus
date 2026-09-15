#!/usr/bin/env python3
"""CPU forward-audit pilot for the HiCache++ controller.

This pilot uses deterministic synthetic trajectories to validate the protocol,
not the end-to-end diffusion claim.  At each anchor it issues candidate
forecasts, chooses using only evidence from earlier future observations, and
scores them when the target feature is later "computed".  The artifact keeps
that distinction explicit so it cannot be mistaken for a GPU/output-quality
benchmark.

Run from the repository root::

    python benchmarks/forward_audit_pilot.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from forecast_microbench import (  # noqa: E402
    forecast_dmd,
    forecast_hermite,
    make_traj_switch,
)
from hicache_pp import ForwardAuditLedger, ForwardRiskController  # noqa: E402


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "hicache_pp").glob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _predict(method: str, snapshots: list[torch.Tensor], horizon: int) -> torch.Tensor:
    if method == "reuse":
        return snapshots[-1].clone()
    if method == "hermite":
        return forecast_hermite(snapshots, horizon)
    if method == "dmd":
        return forecast_dmd(snapshots, horizon, rank=0)
    raise ValueError(method)


def run(*, seeds: int = 12, history: int = 8, horizon: int = 3, total: int = 48) -> dict:
    methods = ["reuse", "hermite", "dmd"]
    all_rows: list[dict] = []
    controller_rows: list[dict] = []
    for seed in range(seeds):
        traj = make_traj_switch(total, d=24, n_modes=2, seed=seed, t_switch=total // 2)
        ledger = ForwardAuditLedger(
            f"pilot-{seed}", branch_id="synthetic-switch", stage="feature"
        )
        controller = ForwardRiskController(
            methods,
            fallback="hermite",
            min_observations=3,
            cost_weight=0.01,
            max_forecast_norm_ratio=4.0,
        )
        for anchor in range(history - 1, total - horizon):
            snapshots = traj[anchor - history + 1 : anchor + 1]
            predictions: dict[str, torch.Tensor] = {}
            costs: dict[str, float] = {}
            for method in methods:
                start = time.perf_counter()
                predictions[method] = _predict(method, snapshots, horizon)
                costs[method] = (time.perf_counter() - start) * 1000.0
                ledger.issue(
                    method,
                    anchor_step=anchor,
                    target_step=anchor + horizon,
                    prediction=predictions[method],
                    fit_cost_ms=costs[method],
                    metadata={"evidence_scope": "observed_trajectory_feature"},
                )
            health = {
                method: float(predictions[method].norm() / snapshots[-1].norm().clamp_min(1e-12))
                for method in methods
            }
            choice = controller.choose(horizon, health=health)
            truth = traj[anchor + horizon]
            observed = ledger.observe(anchor + horizon, truth)
            controller.update(observed)
            selected_error = next(
                obs.relative_l2
                for obs in observed
                if obs.event.method == choice.method
            )
            controller_rows.append(
                {
                    "seed": seed,
                    "anchor_step": anchor,
                    "horizon": horizon,
                    "choice": choice.method,
                    "choice_reason": choice.reason,
                    "choice_evidence_n_before_observation": choice.evidence_n,
                    "selected_relative_l2": selected_error,
                }
            )
            all_rows.extend(obs.as_dict() | {"seed": seed} for obs in observed)
    by_method: dict[str, list[float]] = {}
    for row in all_rows:
        by_method.setdefault(row["method"], []).append(float(row["relative_l2"]))
    controller_errors = [float(row["selected_relative_l2"]) for row in controller_rows]
    return {
        "protocol": {
            "kind": "synthetic_forward_feature_audit",
            "seeds": seeds,
            "history": history,
            "horizon": horizon,
            "total_steps": total,
            "trajectory": "piecewise exponential regime switch",
            "counterfactual_output_quality": False,
            "real_model_or_gpu": False,
        },
        "source_digest": _source_digest(),
        "observations": len(all_rows),
        "methods": {
            method: {
                "n": len(values),
                "mean_relative_l2": sum(values) / len(values),
                "p95_relative_l2": float(torch.quantile(torch.tensor(values), 0.95).item()),
            }
            for method, values in sorted(by_method.items())
        },
            "controller": {
            "n": len(controller_errors),
            "mean_selected_relative_l2": sum(controller_errors) / len(controller_errors),
            "p95_selected_relative_l2": float(
                torch.quantile(torch.tensor(controller_errors), 0.95).item()
            ),
            "choices": {
                method: sum(row["choice"] == method for row in controller_rows)
                for method in methods
            },
            "insufficient_evidence_choices": sum(
                row["choice_reason"] == "insufficient_forward_evidence"
                for row in controller_rows
            ),
            "health_guard_choices": sum(
                row["choice_reason"] == "current_forecast_health_guard"
                for row in controller_rows
            ),
        },
        "rows": all_rows,
        "controller_rows": controller_rows,
    }


def main() -> int:
    result = run()
    out = ROOT / "results" / "forward_audit_pilot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "source_digest": result["source_digest"],
        "methods": result["methods"],
        "controller": result["controller"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
