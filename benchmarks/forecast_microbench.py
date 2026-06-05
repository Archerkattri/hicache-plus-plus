#!/usr/bin/env python3
"""Controlled forecast microbenchmark — HiCache++ (DMD) vs HiCache (Hermite) vs TaylorSeer.

Isolates the mechanism behind the skip-interval ceiling. A diffusion feature trajectory
across timesteps is (locally) the solution of a linear feature-ODE, i.e. a sum of slowly
damped / mildly oscillatory **exponentials**. We generate exactly such trajectories and ask
each method to forecast the feature ``H`` steps past a window of cached anchors — ``H`` is the
skip distance, the furthest a cache extrapolates at interval ``H+1``. We report the mean
relative L2 error over seeds and anchor positions.

  * **TaylorSeer** — the degree-``order`` monomial extrapolant (the unique polynomial through
    the last ``order+1`` cached anchors): the canonical polynomial-family cache. HiCache's
    scaled-Hermite is a *stabilised* member of the same family (it bounds the divergence at the
    cost of a damping bias); the fair HiCache head-to-head is the per-model A/B in the README,
    on real diffusion features its basis is tuned for. This microbench isolates the basis
    question — polynomial vs exponential — with the cleanest polynomial representative.
  * **HiCache++** — the exponential DMD / Prony forecast (``hicache_pp.dmd``): identify the
    propagator from a rank-truncated window, extrapolate by eigenvalue powers.

The polynomial basis is a *local truncation* of the exponential and diverges as the horizon
grows; DMD is the exact basis, so its error stays flat. A second pass adds Gaussian snapshot
noise: the small exact-interpolating polynomial window amplifies it, while DMD's SVD-rank
truncation rejects the noise subspace.

CPU only; runs in a few seconds.   Usage:  python benchmarks/forecast_microbench.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from hicache_pp.dmd import dmd_forecast  # noqa: E402

torch.set_grad_enabled(False)


def make_traj(T, d=64, n_modes=2, seed=0):
    """A real-valued sum-of-exponentials trajectory: F_t = Re(sum_j a_j z_j^t),
    poles z_j = r_j e^{i w_j} near the unit circle (a smooth, slowly varying, low-rank
    signal — like a diffusion feature, which changes only a few percent per step). Each
    complex pole costs 2 real DOF, so n_modes complex poles need rank 2*n_modes -> at least
    2*n_modes+1 snapshots for DMD to identify them."""
    g = torch.Generator().manual_seed(seed)
    r = 0.85 + 0.14 * torch.rand(n_modes, generator=g, dtype=torch.float64)       # decay in [.85,.99]
    w = (torch.rand(n_modes, generator=g, dtype=torch.float64) * 2 - 1) * 0.5     # frequency in [-.5,.5]
    z = (r * torch.exp(1j * w)).to(torch.complex128)
    a = torch.randn(d, n_modes, generator=g, dtype=torch.complex128)              # complex amplitudes
    return [(a @ (z ** float(t))).real for t in range(T)]


def forecast_taylor(snaps, horizon, order):
    """Unique degree-``order`` polynomial through the last order+1 anchors (monomial basis),
    evaluated ``horizon`` steps ahead. Centred at the last anchor for conditioning."""
    m = min(order, len(snaps) - 1)
    X = torch.arange(-m, 1, dtype=torch.float64)                  # -m .. 0 (local coords)
    Y = torch.stack(snaps[-(m + 1):], dim=0)                       # [m+1, d]
    V = torch.stack([X ** i for i in range(m + 1)], dim=1)         # [m+1, m+1]
    coeffs = torch.linalg.solve(V, Y)                            # exact through the points
    powers = torch.tensor([float(horizon) ** i for i in range(m + 1)], dtype=torch.float64)
    return powers @ coeffs


def forecast_dmd(snaps, horizon, rank):
    """HiCache++ exponential forecast ``horizon`` steps ahead (integer eigenvalue power)."""
    return dmd_forecast(snaps, horizon, rank=rank)


def run(noise=0.0, seeds=20, horizons=(1, 2, 3, 4, 6, 8), order=3, history=8, n_modes=2):
    # auto-rank on clean data finds the true mode count; under noise we cap to the physical
    # signal rank (2 real DOF per complex pole) so the SVD truncation rejects the noise subspace.
    rank = 0 if noise == 0 else 2 * n_modes
    out = {"TaylorSeer": {}, "HiCache++": {}}
    for Hz in horizons:
        acc = {k: [] for k in out}
        for seed in range(seeds):
            traj = make_traj(history + Hz + 2, n_modes=n_modes, seed=seed)
            snaps = [traj[s] for s in range(history)]                 # cached window t=0..history-1
            if noise:
                snaps = [F + noise * F.norm() / F.numel() ** 0.5 * torch.randn_like(F)
                         for F in snaps]
            truth = traj[(history - 1) + Hz]
            preds = {
                "TaylorSeer": forecast_taylor(snaps, Hz, order),
                "HiCache++": forecast_dmd(snaps, Hz, rank),
            }
            den = truth.norm() + 1e-12
            for name, p in preds.items():
                acc[name].append(((p - truth).norm() / den).item())
        for name in out:
            out[name][Hz] = sum(acc[name]) / len(acc[name])
    return out


def md_table(res, horizons):
    cols = "".join(f" H={h} |" for h in horizons)
    lines = [f"| method (rel. L2 error ↓) |{cols}", "|---|" + "".join("---:|" for _ in horizons)]
    label = {"TaylorSeer": "TaylorSeer (polynomial)", "HiCache++": "HiCache++ (exponential)"}
    for name in ("TaylorSeer", "HiCache++"):
        cells = "".join(f" {res[name][h]:.2e} |" for h in horizons)
        bold = "**" if name == "HiCache++" else ""
        lines.append(f"| {bold}{label[name]}{bold} |{cells}")
    return "\n".join(lines)


if __name__ == "__main__":
    horizons = (1, 2, 3, 4, 6, 8)
    print("Forecast rel. L2 error vs skip horizon H on the exponential (feature-ODE) class.\n"
          "H = steps past the cached window (the reach of interval H+1). Lower is better.\n")
    print("### Clean trajectories (20 seeds, 64-channel, 2 modes)\n")
    print(md_table(run(noise=0.0, horizons=horizons), horizons))
    print("\n### + 1% snapshot noise\n")
    print(md_table(run(noise=0.01, horizons=horizons), horizons))
    print("\nHiCache++ (DMD) is exact on the solution class, so its error stays flat as H grows;")
    print("the polynomial basis diverges with H, and the small Taylor window amplifies noise.")
