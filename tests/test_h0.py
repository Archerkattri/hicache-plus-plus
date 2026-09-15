"""Focused CPU regressions for the H0 state, finiteness, ownership, and telemetry contract."""

import torch

from hicache_pp import (
    dmd_forecast_state,
    hicache_decide,
    hicache_init,
    hicache_reset,
    hicache_telemetry,
    hicache_update_derivatives,
)
from hicache_pp.dmd import dmd_eval, dmd_update_snapshots
from hicache_pp import tree


def test_zero_warmup_requires_first_anchor():
    state = hicache_init(num_steps=8, interval=4, first_enhance=0)
    assert hicache_decide(state) == "full"
    assert state["activated_steps"] == [0]
    feature = torch.tensor([1.0, -2.0])
    hicache_update_derivatives(state, feature)
    state["step"] = 1
    assert hicache_decide(state) == "forecast"
    assert state["telemetry"]["decisions"] == {"full": 1, "forecast": 1}


def test_flat_dmd_rejects_dtype_overflow_after_cast():
    fit = (
        torch.tensor([[100000.0 + 0j]], dtype=torch.complex128),
        torch.tensor([1.0 + 0j], dtype=torch.complex128),
        torch.tensor([1.0 + 0j], dtype=torch.complex128),
        torch.Size([1]),
        torch.float16,
    )
    # The complex/float64 intermediate is finite, but float16 cannot represent it.
    assert dmd_eval(fit, 0) is None


def test_tree_dmd_snapshots_own_detached_leaves():
    state = {"activated_steps": [0], "history": 5, "dmd_snapshots": []}
    source = {"x": torch.tensor([1.0], requires_grad=True),
              "nested": {"y": torch.tensor([[2.0]])}}
    dmd_update_snapshots_tree = tree.dmd_update_snapshots_tree
    dmd_update_snapshots_tree(state, source)
    source["x"].data.fill_(9.0)
    source["nested"]["y"].fill_(8.0)
    stored = state["dmd_snapshots"][0][1]
    assert stored["x"].item() == 1.0
    assert stored["nested"]["y"].item() == 2.0
    assert not stored["x"].requires_grad
    assert stored["x"].data_ptr() != source["x"].data_ptr()


def test_telemetry_reports_fallback_and_actual_dmd():
    state = hicache_init(num_steps=16, interval=3, first_enhance=0, backend="dmd")
    feature = torch.tensor([1.0, 2.0], dtype=torch.float64)
    state["step"] = 0
    state["activated_steps"].append(0)
    hicache_update_derivatives(state, feature)
    dmd_update_snapshots(state, feature)
    state["step"] = 1
    assert torch.equal(dmd_forecast_state(state), feature)
    assert state["telemetry"]["method_counts"]["hermite"] == 1
    assert state["telemetry"]["fallbacks"]["dmd_insufficient_uniform_history"] == 1

    dmd_state = hicache_init(num_steps=16, interval=1, first_enhance=0, backend="dmd")
    snapshots = [torch.tensor([0.9 ** i, 0.7 ** i], dtype=torch.float64)
                 for i in range(4)]
    dmd_state["dmd_snapshots"] = list(enumerate(snapshots))
    dmd_state["step"] = 4
    assert dmd_forecast_state(dmd_state) is not None
    assert dmd_state["telemetry"]["method_counts"]["dmd"] == 1


def test_reset_starts_new_run_without_changing_configuration():
    state = hicache_init(num_steps=8, interval=4, first_enhance=0, backend="dmd")
    state["branch_id"] = "test-branch"
    old_run = state["run_id"]
    state["step"] = 7
    state["dmd_snapshots"].append((0, torch.ones(1)))
    assert hicache_reset(state) is state
    assert state["run_id"] != old_run
    assert state["branch_id"] == "test-branch"
    assert state["step"] == 0 and state["activated_steps"] == []
    assert state["dmd_snapshots"] == [] and state["telemetry"]["fallbacks"] == {}
    assert state["interval"] == 4 and state["backend"] == "dmd"
    report = hicache_telemetry(state)
    report["decisions"]["full"] = 99
    assert state["telemetry"]["decisions"]["full"] == 0


def test_tree_zero_warmup_and_reset_contract():
    state = tree.hicache_init(num_steps=8, interval=4, first_enhance=0)
    assert tree.hicache_decide(state) == "full"
    tree.hicache_update_tree(state, {"x": torch.ones(1)})
    state["step"] = 1
    assert tree.hicache_decide(state) == "forecast"
    old_run = state["run_id"]
    tree.hicache_reset(state)
    assert state["run_id"] != old_run and state["activated_steps"] == []


def main() -> int:
    tests = [value for name, value in globals().items()
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"\n{len(tests)} H0 TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
