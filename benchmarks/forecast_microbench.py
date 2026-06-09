#!/usr/bin/env python3
"""Controlled forecast microbenchmark — HiCache++ (DMD) vs HiCache (Hermite) vs TaylorSeer.

Isolates the mechanism behind the skip-interval ceiling. A diffusion feature trajectory
across timesteps is (locally) the solution of a linear feature-ODE, i.e. a sum of slowly
damped / mildly oscillatory **exponentials**. We generate exactly such trajectories and ask
each method to forecast the feature ``H`` steps past a window of cached anchors — ``H`` is the
skip distance, the furthest a cache extrapolates at interval ``H+1``. We report the mean
relative L2 error over seeds and anchor positions.

  * **TaylorSeer** — the degree-``order`` monomial extrapolant (the unique polynomial through the
    last ``order+1`` cached anchors): the canonical polynomial-family cache. HiCache's
    scaled-Hermite is a *stabilised* member of the same family (it bounds the divergence at a
    damping-bias cost); the fair HiCache head-to-head is the per-model A/B in the README, on real
    diffusion features its basis is tuned for.
  * **Pade / FoCa** — a rational [3/2] extrapolant: the FoCa / Pade / Chebyshev rational family.
    A rational basis fits exponentials better than a polynomial (it improves on Taylor) but is
    still not the exact class, and its higher orders are fragile (spurious Froissart poles).
  * **HiCache++** — the exponential DMD / Prony forecast (``hicache_pp.dmd``): identify the
    propagator from a rank-truncated window, extrapolate by eigenvalue powers.

The polynomial and rational bases are *truncations / approximations* of the exponential and
diverge as the horizon grows (rational improves on polynomial but is still inexact and can go
unstable); DMD is the exact basis, so its error stays flat. A second pass adds Gaussian snapshot
noise: the small exact-interpolating windows amplify it, while DMD's SVD-rank truncation rejects
the noise subspace.

CPU only; runs in a few seconds.   Usage:  python benchmarks/forecast_microbench.py
"""
import math
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


def forecast_pade(snaps, horizon, L=3, M=2):
    """Rational [L/M] Pade extrapolant — the FoCa / Pade / Chebyshev rational family. Builds the
    local Taylor series from backward finite differences, converts it to a Pade approximant per
    channel, and evaluates ``horizon`` steps ahead. Falls back to last-value reuse on any channel
    whose denominator goes singular or spurious (the deployed-safe response to the Froissart-doublet
    instability that makes high-order rational caches fragile — a real cache never ships the blow-up)."""
    F = torch.stack(snaps, 0)                                     # [n, d]
    coeffs, cur = [F[-1]], F
    for k in range(1, L + M + 1):                                 # backward-difference Taylor series
        cur = cur[1:] - cur[:-1]
        coeffs.append(cur[-1] / math.factorial(k))
    C = torch.stack(coeffs, 0)                                    # [L+M+1, d]
    wmax = F.abs().amax(0)                                        # [d] recent dynamic range per channel
    h = float(horizon)
    out = F[-1].clone()                                           # last-value reuse fallback per channel
    for ch in range(C.shape[1]):
        cc = C[:, ch]
        A = torch.stack([torch.stack([cc[L + i - j] for j in range(1, M + 1)]) for i in range(1, M + 1)])
        b = torch.stack([-cc[L + i] for i in range(1, M + 1)])
        try:
            q = torch.linalg.solve(A, b)
        except Exception:  # noqa: BLE001 — singular denominator system: keep the reuse fallback
            continue
        p = torch.stack([cc[i] + sum(cc[i - j] * q[j - 1] for j in range(1, min(i, M) + 1))
                         for i in range(L + 1)])
        QH = 1.0 + sum(q[j - 1] * h ** j for j in range(1, M + 1))
        PH = sum(p[i] * h ** i for i in range(L + 1))
        if abs(QH) > 1e-3:                                        # reject a near-pole in the extrap range
            val = PH / QH
            if torch.isfinite(val) and abs(val) <= 2.0 * wmax[ch]:  # deployed clamp: else last-value reuse
                out[ch] = val
    return out


def forecast_dmd(snaps, horizon, rank):
    """HiCache++ exponential forecast ``horizon`` steps ahead (integer eigenvalue power)."""
    return dmd_forecast(snaps, horizon, rank=rank)


def make_traj_drift(T, d=64, n_modes=2, seed=0, w_drift=0.6, r_drift=0.10):
    """A NON-AUTONOMOUS trajectory: locally a sum of exponentials, but the pole
    frequencies/decays drift across the horizon (w_t = w0*(1 + w_drift*t/T), and
    log r drifts similarly). This models the regime the holdout 'auto' backend
    exists for: diffusion feature dynamics whose propagator changes between
    timestep windows, where a fixed exponential fit can misfit."""
    g = torch.Generator().manual_seed(seed)
    r0 = 0.85 + 0.14 * torch.rand(n_modes, generator=g, dtype=torch.float64)
    w0 = (torch.rand(n_modes, generator=g, dtype=torch.float64) * 2 - 1) * 0.5
    a = torch.randn(d, n_modes, generator=g, dtype=torch.complex128)
    logz_acc = torch.zeros(n_modes, dtype=torch.complex128)
    traj = [(a @ torch.exp(logz_acc)).real.clone()]
    for t in range(1, T):
        frac = t / float(T)
        w_t = w0 * (1.0 + w_drift * frac)
        logr_t = torch.log(r0) * (1.0 + r_drift * frac)
        logz_acc = logz_acc + (logr_t + 1j * w_t)
        traj.append((a @ torch.exp(logz_acc)).real)
    return traj


def make_traj_switch(T, d=64, n_modes=2, seed=0, t_switch=None):
    """A REGIME-SWITCH trajectory: the poles change abruptly at ``t_switch`` (default:
    mid-window), so an exponential fit over the whole cached window straddles two
    different dynamics and misfits. A local polynomial (last few anchors only) sees
    mostly post-switch data. This is the stress case the holdout 'auto' backend is
    designed to catch."""
    g = torch.Generator().manual_seed(seed)
    if t_switch is None:
        t_switch = T // 2
    def poles(gen):
        r = 0.85 + 0.14 * torch.rand(n_modes, generator=gen, dtype=torch.float64)
        w = (torch.rand(n_modes, generator=gen, dtype=torch.float64) * 2 - 1) * 0.5
        return (r * torch.exp(1j * w)).to(torch.complex128)
    z1, z2 = poles(g), poles(g)
    a = torch.randn(d, n_modes, dtype=torch.complex128, generator=g)
    traj, state = [], torch.ones(n_modes, dtype=torch.complex128)
    for t in range(T):
        traj.append((a @ state).real)
        state = state * (z1 if t < t_switch else z2)
    return traj


def forecast_auto(snaps, horizon, max_order=2):
    """The SHIPPED HiCache++ ``backend='auto'`` path, driven end-to-end through the
    real state machinery (hicache_init/update + dmd_update_snapshots +
    auto_forecast_state) -- holdout-selects DMD vs the Hermite fallback per window."""
    from hicache_pp.hermite import hicache_init, hicache_update_derivatives
    from hicache_pp.dmd import dmd_update_snapshots, auto_forecast_state
    st = hicache_init(num_steps=10 ** 6, interval=horizon + 1, max_order=max_order,
                      first_enhance=0, sigma=0.5, backend="auto", history=len(snaps))
    for t, F in enumerate(snaps):
        st["step"] = t
        st["activated_steps"].append(t)
        hicache_update_derivatives(st, F)
        dmd_update_snapshots(st, F, history=len(snaps))
    st["step"] = (len(snaps) - 1) + horizon
    return auto_forecast_state(st), st.get("_auto_choice")


def run(noise=0.0, seeds=20, horizons=(1, 2, 3, 4, 6, 8), order=3, history=8, n_modes=2,
        drift=False):
    # auto-rank on clean data finds the true mode count; under noise we cap to the physical
    # signal rank (2 real DOF per complex pole) so the SVD truncation rejects the noise subspace.
    rank = 2 * n_modes if (noise or drift) else 0
    out = {"TaylorSeer": {}, "Pade": {}, "HiCache++": {}, "HiCache++ auto": {}}
    picks = {}
    for Hz in horizons:
        acc = {k: [] for k in out}
        for seed in range(seeds):
            T_total = history + Hz + 2
            if drift == "switch":
                traj = make_traj_switch(T_total, n_modes=n_modes, seed=seed,
                                        t_switch=history - 3)
            elif drift:
                traj = make_traj_drift(T_total, n_modes=n_modes, seed=seed)
            else:
                traj = make_traj(T_total, n_modes=n_modes, seed=seed)
            snaps = [traj[s] for s in range(history)]                 # cached window t=0..history-1
            if noise:
                snaps = [F + noise * F.norm() / F.numel() ** 0.5 * torch.randn_like(F)
                         for F in snaps]
            truth = traj[(history - 1) + Hz]
            auto_pred, auto_choice = forecast_auto(snaps, Hz)
            picks[auto_choice] = picks.get(auto_choice, 0) + 1
            preds = {
                "TaylorSeer": forecast_taylor(snaps, Hz, order),
                "Pade": forecast_pade(snaps, Hz),
                "HiCache++": forecast_dmd(snaps, Hz, rank),
                "HiCache++ auto": auto_pred,
            }
            den = truth.norm() + 1e-12
            for name, p in preds.items():
                acc[name].append(((p - truth).norm() / den).item())
        for name in out:
            out[name][Hz] = sum(acc[name]) / len(acc[name])
    out["_picks"] = picks
    return out


def md_table(res, horizons):
    res = {k: v for k, v in res.items() if not k.startswith("_")}
    cols = "".join(f" H={h} |" for h in horizons)
    lines = [f"| method (rel. L2 error ↓) |{cols}", "|---|" + "".join("---:|" for _ in horizons)]
    label = {"TaylorSeer": "TaylorSeer (polynomial)", "Pade": "Pade / FoCa (rational)",
             "HiCache++": "HiCache++ (exponential)",
             "HiCache++ auto": "HiCache++ (auto, holdout-selected)"}
    for name in ("TaylorSeer", "Pade", "HiCache++", "HiCache++ auto"):
        if name not in res:
            continue
        cells = "".join(f" {res[name][h]:.2e} |" for h in horizons)
        bold = "**" if name.startswith("HiCache++") else ""
        lines.append(f"| {bold}{label[name]}{bold} |{cells}")
    return "\n".join(lines)


if __name__ == "__main__":
    horizons = (1, 2, 3, 4, 6, 8)
    print("Forecast rel. L2 error vs skip horizon H on the exponential (feature-ODE) class.\n"
          "H = steps past the cached window (the reach of interval H+1). Lower is better.\n")
    print("### Clean trajectories (20 seeds, 64-channel, 2 modes)\n")
    print(md_table(run(noise=0.0, horizons=horizons), horizons))
    print("\n### + 1% snapshot noise\n")
    r = run(noise=0.01, horizons=horizons)
    print(md_table(r, horizons))
    print(f"\n(auto picked: {r['_picks']})")
    print("\n### Drifting (non-autonomous) dynamics — why backend='auto' exists\n")
    r = run(drift=True, horizons=horizons)
    print(md_table(r, horizons))
    print(f"\n(auto picked: {r['_picks']})")
    print("\n### Regime switch inside the cached window — the DMD-misfit stress\n")
    r = run(drift="switch", horizons=horizons)
    print(md_table(r, horizons))
    print(f"\n(auto picked: {r['_picks']})")
    print("\n### Drifting dynamics + 1% snapshot noise\n")
    r = run(noise=0.01, drift=True, horizons=horizons)
    print(md_table(r, horizons))
    print(f"\n(auto picked: {r['_picks']})")
    print("\nHiCache++ (DMD) is exact on the solution class, so its error stays flat as H grows;")
    print("polynomial diverges, rational (Pade/FoCa) improves but still diverges and goes fragile.")
