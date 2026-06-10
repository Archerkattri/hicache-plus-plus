#!/usr/bin/env python3
"""CPU tests for the DiT benchmark harness forecast paths (no GPU, no model).

Covers the sign convention of bench_dit.taylor_forecast: forward finite
differences must be evaluated at distance +k (a degree-1 monomial forecast is
EXACT on a linear series; the pre-fix (-k)**o version lands on the wrong side
of the anchor and loses to plain reuse).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks", "dit_imagenet"))
from bench_dit import taylor_forecast  # noqa: E402

from hicache_pp.hermite import hicache_init, hicache_update_derivatives  # noqa: E402


def _linear_state(a, b, anchors=(0, 4), step=6, max_order=1):
    """Real state machinery fed a linear velocity series F_s = a + b*s."""
    st = hicache_init(num_steps=64, interval=anchors[1] - anchors[0], max_order=max_order,
                      first_enhance=0, end_enhance=64, sigma=0.5)
    for s in anchors:
        st["step"] = s
        st["activated_steps"].append(s)
        hicache_update_derivatives(st, a + b * float(s))
    st["step"] = step
    return st


def test_taylor_forecast_exact_on_linear_series():
    """Degree-1 monomial forecast at +k is EXACT on a linear series:
    F_hat = F_4 + b*k = a + b*6. The (-k)**o bug gives a + b*2 instead."""
    a = torch.tensor([1.0, -2.0, 0.5])
    b = torch.tensor([0.3, -0.7, 1.1])
    st = _linear_state(a, b)                      # anchors 0,4; forecast step 6 (k=2)
    pred = taylor_forecast(st)
    assert torch.allclose(pred, a + b * 6.0, atol=1e-6), \
        f"taylor_forecast not forward-exact on linear series: {pred} != {a + b * 6.0}"


def test_taylor_forecast_beats_reuse():
    """At any k>0 on a linear series, a forward monomial forecast strictly beats
    plain reuse of the cached anchor; the -k version strictly loses to it."""
    a = torch.zeros(4)
    b = torch.tensor([1.0, 2.0, -1.0, 0.5])
    st = _linear_state(a, b, step=7)              # k=3
    truth = a + b * 7.0
    err_fc = (taylor_forecast(st) - truth).norm()
    err_reuse = ((a + b * 4.0) - truth).norm()
    assert err_fc < err_reuse, f"forecast ({err_fc}) must beat reuse ({err_reuse})"


def main() -> int:
    ok = True
    for fn in (test_taylor_forecast_exact_on_linear_series, test_taylor_forecast_beats_reuse):
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            ok = False
            print(f"[FAIL] {fn.__name__}: {e}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
