"""Deterministic CPU adapter-contract fixture for the central HiCache API."""

import json
from pathlib import Path

import torch

from hicache_pp import (
    dmd_forecast_state,
    hicache_decide,
    hicache_init,
    hicache_reset,
    hicache_telemetry,
    hicache_update_derivatives,
    dmd_update_snapshots,
)
from hicache_pp import tree


FIXTURE = Path(__file__).with_name("fixtures") / "adapter_contract_trace.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _flat_window(steps, values):
    state = hicache_init(
        num_steps=max(steps) + 2, interval=2, first_enhance=0, backend="dmd"
    )
    tensors = [torch.tensor(value, dtype=torch.float64) for value in values]
    state["run_id"] = "fixture-run"
    state["branch_id"] = "conditional"
    state["dmd_snapshots"] = list(zip(steps, tensors))
    state["step"] = steps[-1] + 1
    state["activated_steps"] = [steps[-1]]
    state["derivatives"] = {0: tensors[-1]}
    return state


def _tree_window(steps, values):
    state = tree.hicache_init(
        num_steps=max(steps) + 2, interval=2, first_enhance=0, backend="dmd"
    )
    trees = [
        {"x": torch.tensor(value, dtype=torch.float64), "nested": (torch.tensor([value[0]], dtype=torch.float64),)}
        for value in values
    ]
    state["run_id"] = "fixture-tree-run"
    state["branch_id"] = "conditional"
    state["dmd_snapshots"] = list(zip(steps, trees))
    state["step"] = steps[-1] + 1
    state["activated_steps"] = [steps[-1]]
    state["derivatives"] = {0: trees[-1]}
    return state


def test_fixture_is_deterministic_and_flat_contract_is_visible():
    data = _fixture()

    def run_once():
        state = hicache_init(num_steps=8, interval=2, first_enhance=0, backend="dmd")
        state["run_id"] = "deterministic-fixture"
        state["branch_id"] = data["trace"][0]["branch_id"]
        events = []
        for row in data["trace"]:
            state["step"] = row["step"]
            feature = torch.tensor(row["feature"], dtype=torch.float64)
            decision = hicache_decide(state)
            if decision == "full":
                hicache_update_derivatives(state, feature)
                dmd_update_snapshots(state, feature)
                output = feature
            else:
                output = dmd_forecast_state(state)
            events.append({"step": row["step"], "decision": decision, "output": output.tolist()})
        return {"events": events, "telemetry": hicache_telemetry(state)}

    first = run_once()
    second = run_once()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [event["decision"] for event in first["events"]] == ["full", "forecast", "full", "forecast"]
    assert first["telemetry"]["decisions"] == {"full": 2, "forecast": 2}
    assert first["telemetry"]["fallbacks"]["dmd_insufficient_uniform_history"] == 2
    assert first["telemetry"]["method_counts"]["hermite"] == 2


def test_interval_one_is_full_compute_bypass():
    data = _fixture()
    state = hicache_init(num_steps=8, interval=1, first_enhance=0, backend="dmd")
    for row in data["trace"][:3]:
        state["step"] = row["step"]
        assert hicache_decide(state) == "full"
        feature = torch.tensor(row["feature"], dtype=torch.float64)
        hicache_update_derivatives(state, feature)
        dmd_update_snapshots(state, feature)
    report = hicache_telemetry(state)
    assert report["decisions"] == {"full": 3, "forecast": 0}
    assert report["method_counts"] == {"hermite": 0, "dmd": 0, "reuse": 0}


def test_uniform_and_nonuniform_windows_report_actual_method():
    data = _fixture()
    uniform = _flat_window(data["uniform_steps"], data["dmd_values"])
    assert dmd_forecast_state(uniform) is not None
    assert uniform["telemetry"]["method_counts"]["dmd"] == 1
    assert uniform["telemetry"]["fallbacks"] == {}

    nonuniform = _flat_window(data["nonuniform_steps"], data["dmd_values"])
    assert dmd_forecast_state(nonuniform) is not None
    assert nonuniform["telemetry"]["method_counts"]["hermite"] == 1
    assert nonuniform["telemetry"]["fallbacks"]["dmd_nonuniform_or_short_tail"] == 1

    tree_state = _tree_window(data["uniform_steps"], data["dmd_values"])
    tree_result = tree.dmd_forecast_tree(tree_state)
    assert isinstance(tree_result, dict) and set(tree_result) == {"x", "nested"}
    assert tree_state["telemetry"]["method_counts"]["dmd"] == 1


def test_run_branch_reset_isolation_for_flat_and_tree():
    state = hicache_init(num_steps=8, interval=2, first_enhance=0, backend="dmd")
    state["branch_id"] = "conditional"
    old_run = state["run_id"]
    state["step"] = 0
    assert hicache_decide(state) == "full"
    feature = torch.tensor([1.0, 2.0])
    hicache_update_derivatives(state, feature)
    dmd_update_snapshots(state, feature)
    state["step"] = 1
    assert hicache_decide(state) == "forecast"
    hicache_reset(state, run_id="retry-run")
    assert old_run != state["run_id"] == "retry-run"
    assert state["branch_id"] == "conditional"
    assert state["dmd_snapshots"] == [] and state["telemetry"]["decisions"] == {"full": 0, "forecast": 0}
    assert hicache_decide(state) == "full"

    tree_state = tree.hicache_init(num_steps=8, interval=2, first_enhance=0, backend="dmd")
    tree_state["branch_id"] = "unconditional"
    tree_run = tree_state["run_id"]
    assert tree.hicache_reset(tree_state, run_id="tree-retry") is tree_state
    assert tree_state["run_id"] != tree_run and tree_state["branch_id"] == "unconditional"
    assert tree.hicache_decide(tree_state) == "full"


def main() -> int:
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"\n{len(tests)} H1 ADAPTER-CONTRACT TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
