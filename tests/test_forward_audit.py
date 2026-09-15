"""CPU tests for the forward-only evidence and controller contract."""

import pytest
import torch

from hicache_pp import ForwardAuditLedger, ForwardRiskController


def test_future_prediction_is_scored_only_at_matching_target():
    ledger = ForwardAuditLedger("run-a", branch_id="b", stage="shape")
    ledger.issue("reuse", anchor_step=0, target_step=2, prediction=torch.tensor([1.0, 1.0]))
    assert ledger.observe(1, torch.tensor([2.0, 2.0])) == []
    scored = ledger.observe(2, torch.tensor([2.0, 2.0]))
    assert len(scored) == 1
    assert scored[0].event.horizon == 2
    assert scored[0].relative_l2 == pytest.approx(0.5)
    assert ledger.summary()["pending"] == 0


def test_identity_reset_drops_pending_predictions():
    ledger = ForwardAuditLedger("old")
    ledger.issue("dmd", anchor_step=2, target_step=4, prediction=torch.ones(2))
    ledger.start("new", branch_id="retry", stage="texture")
    assert ledger.pending() == ()
    assert ledger.dropped_pending == 1
    assert ledger.observe(4, torch.ones(2)) == []


def test_bad_event_inputs_are_rejected_without_mutating_pending():
    ledger = ForwardAuditLedger("run")
    with pytest.raises(ValueError):
        ledger.issue("reuse", anchor_step=3, target_step=3, prediction=torch.ones(1))
    with pytest.raises(ValueError):
        ledger.issue("reuse", anchor_step=0, target_step=1, prediction=torch.tensor([float("nan")]))
    assert ledger.pending() == ()


def test_controller_falls_back_until_forward_evidence_then_chooses_best():
    ledger = ForwardAuditLedger("run")
    controller = ForwardRiskController(
        ["hermite", "dmd"], fallback="hermite", min_observations=2, cost_weight=0.0
    )
    assert controller.choose(1).method == "hermite"
    for step in range(2):
        ledger.issue("hermite", anchor_step=step, target_step=step + 1,
                     prediction=torch.tensor([1.0]), fit_cost_ms=0.1)
        ledger.issue("dmd", anchor_step=step, target_step=step + 1,
                     prediction=torch.tensor([1.01]), fit_cost_ms=1.0)
        obs = ledger.observe(step + 1, torch.tensor([1.0]))
        controller.update(obs)
    choice = controller.choose(1)
    assert choice.method == "hermite"
    assert choice.evidence_n == 2
    assert controller.report()["evidence"][0]["n"] == 2


def test_cost_penalty_can_prefer_cheaper_method_at_equal_error():
    ledger = ForwardAuditLedger("run")
    controller = ForwardRiskController(
        ["expensive", "cheap"], fallback="cheap", min_observations=1, cost_weight=0.25
    )
    for method, cost in (("expensive", 10.0), ("cheap", 1.0)):
        ledger.issue(method, anchor_step=0, target_step=1,
                     prediction=torch.tensor([1.0]), fit_cost_ms=cost)
    controller.update(ledger.observe(1, torch.tensor([1.0])))
    assert controller.choose(1).method == "cheap"


def test_current_forecast_health_guard_rejects_an_exploding_candidate():
    ledger = ForwardAuditLedger("run")
    controller = ForwardRiskController(
        ["hermite", "dmd"], fallback="hermite", min_observations=1,
        max_forecast_norm_ratio=4.0,
    )
    for method, value in (("hermite", 1.0), ("dmd", 1.0)):
        ledger.issue(method, anchor_step=0, target_step=1,
                     prediction=torch.tensor([value]))
    controller.update(ledger.observe(1, torch.tensor([1.0])))
    choice = controller.choose(1, health={"hermite": 1.0, "dmd": 25.0})
    assert choice.method == "hermite"
    assert choice.reason == "current_forecast_health_guard"
