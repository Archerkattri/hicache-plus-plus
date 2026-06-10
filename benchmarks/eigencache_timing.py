#!/usr/bin/env python3
"""CPU timing microbench for the per-window DMD eigendecomposition cache.

The stateful DMD paths cache (Phi, lambda, b) per compute window: the fit inputs
cannot change between two compute steps, so the pre-1.2 fit-per-skip-step was pure
waste. This script measures the per-forecast wall-clock of the stateful path with
the cache active vs the same path with the cache key cleared before every call
(which reproduces the pre-1.2 fit-every-call behavior exactly), at DiT-like feature
sizes. Correctness of the cache (<1e-12 vs the uncached path) is asserted in the
module self-tests; this file is timing only. GPU timing happens on the GPU resume.

CPU only; runs in well under a minute.  Usage:  python benchmarks/eigencache_timing.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from hicache_pp.dmd import dmd_forecast_state  # noqa: E402

torch.set_grad_enabled(False)


def bench(d, history=6, interval=4, windows=20, dtype=torch.float32):
    """Mean per-forecast seconds for (cached, uncached) over `windows` compute
    windows of `interval-1` skip-step forecasts each, feature dimension `d`."""
    g = torch.Generator().manual_seed(0)
    base = torch.randn(d, generator=g, dtype=dtype)
    drift = torch.randn(d, generator=g, dtype=dtype) * 0.01
    snaps = [(i * interval, base + i * drift) for i in range(history)]
    times = {"cached": 0.0, "uncached": 0.0}
    calls = 0
    for w in range(windows):
        newest = snaps[-1][0]
        st = {"history": history, "dmd_snapshots": list(snaps), "step": 0}
        for k in range(1, interval):
            st["step"] = newest + k
            t0 = time.perf_counter()
            dmd_forecast_state(st)
            times["cached"] += time.perf_counter() - t0
            calls += 1
        st_u = {"history": history, "dmd_snapshots": list(snaps), "step": 0}
        for k in range(1, interval):
            st_u["step"] = newest + k
            st_u.pop("_dmd_fit_key", None)            # pre-1.2 behavior: refit every call
            t0 = time.perf_counter()
            dmd_forecast_state(st_u)
            times["uncached"] += time.perf_counter() - t0
        # advance one window (new snapshot arrives, oldest drops)
        snaps = snaps[1:] + [(newest + interval, base + (history + w) * drift)]
    return times["cached"] / calls, times["uncached"] / calls


if __name__ == "__main__":
    print("Per-forecast wall-clock of the stateful DMD path, eigendecomposition cache")
    print("ON vs OFF (fit-per-call, the pre-1.2 behavior). history=6, interval=4")
    print(f"(3 forecasts/window), 20 windows, float32, CPU ({torch.get_num_threads()} threads).\n")
    print("| feature dim d | uncached (fit/call) | cached (fit/window) | per-forecast speedup |")
    print("|---:|---:|---:|---:|")
    for d in (8192, 65536, 262144, 524288):
        # warm-up pass so lazy LAPACK init is not billed to either arm
        bench(d, windows=2)
        c, u = bench(d)
        print(f"| {d:,} | {u*1e3:.3f} ms | {c*1e3:.3f} ms | {u/c:.1f}x |", flush=True)
    print("\nThe cached fit amortizes one SVD+eig+lstsq over the interval-1 skip steps of")
    print("a window; the remaining per-forecast cost is one Phi @ (lambda**k * b).")
