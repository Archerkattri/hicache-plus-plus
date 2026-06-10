"""Exponential (DMD / Prony) velocity forecasting (HiCache++) — the exponential forecaster.

Existing feature caches (TaylorSeer, HiCache, FoCa, Padé / Chebyshev variants) extrapolate
the cached velocity with a POLYNOMIAL or RATIONAL basis. But a diffusion feature trajectory
across timesteps is the solution of a (near-)linear feature-ODE, whose *exact* solution
class is a sum of (damped / oscillatory) EXPONENTIALS — not polynomials. A polynomial is
only a local truncation of that exponential and diverges under extrapolation, which is why
every polynomial cache caps out at a modest skip interval; a sum of exponentials has the
correct asymptotics and stays bounded.

This module forecasts the next cached velocity with **Dynamic Mode Decomposition**
(Schmid 2010) — the multivariate, SVD-regularised generalisation of **Prony's method**
(1795) and the **Matrix-Pencil** method (Hua & Sarkar 1990): identify the linear
propagator ``A`` from snapshots (``F_{t+1} ≈ A F_t``), eigendecompose it once, and predict
any (fractional) horizon ``k`` by eigenvalue powers::

    F_{t+k} ≈ Phi @ (lambda**k * b),     b = Phi^+ F_t

One economy SVD of a ``[d, n]`` snapshot matrix (``d >> n``, ``n`` = number of cached
steps) makes it cheap relative to a single network forward. It is *exact* on exponential
trajectories (the solution class) — the property polynomials lack — so it stays lossless
at larger skip intervals than the polynomial basis.

To our knowledge DMD / Prony has not previously been applied to diffusion feature caching;
this is the novel contribution.
"""
from __future__ import annotations

import math

import torch


def dmd_fit(snapshots, rank: int = 0, ridge: float = 1e-8):
    """Fit the DMD eigendecomposition once for a snapshot window.

    Returns ``(Phi, evals, b, shape, dtype)`` -- everything a forecast at ANY
    horizon needs -- or ``None`` on a degenerate fit (caller falls back to
    last-value reuse). The fit depends only on the snapshot window, not on the
    horizon, so between two compute steps it can be computed once and reused by
    every skip-step forecast (see :func:`dmd_forecast_state`)."""
    if len(snapshots) < 3:
        return None
    shp, dt = snapshots[-1].shape, snapshots[-1].dtype
    V = torch.stack([s.reshape(-1) for s in snapshots], dim=1).to(torch.float64)  # [d, n+1]
    X, Xp = V[:, :-1], V[:, 1:]                                                   # [d, n]
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)                       # U[d,n] S[n] Vh[n,n]
    except Exception:  # noqa: BLE001 — numerically degenerate fit: fall back to last-value reuse
        return None
    if rank <= 0:
        rank = int((S > S[0] * 1e-4).sum().clamp(min=1).item())
    rank = max(1, min(rank, S.numel()))
    Ur, Sr, Vr = U[:, :rank], S[:rank], Vh[:rank].mH                              # Vr [n, r]
    Sinv = (1.0 / (Sr + ridge)).to(torch.complex128)
    Atil = (Ur.mH @ Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)            # [r, r] (= Ur^H Xp Vr Sr^-1)
    try:
        evals, W = torch.linalg.eig(Atil)                                        # poles lambda, [r,r]
        Phi = ((Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)) @ W           # DMD modes [d, r]
        b = torch.linalg.lstsq(Phi, V[:, -1].to(torch.complex128).unsqueeze(1)).solution.squeeze(1)
    except Exception:  # noqa: BLE001 — numerically degenerate fit: fall back to last-value reuse
        return None
    return (Phi, evals, b, shp, dt)


def dmd_eval(fit, k):
    """Evaluate a cached :func:`dmd_fit` at (fractional, signed) horizon ``k``:
    ``Phi @ (lambda**k * b)``. Returns ``None`` if the prediction is non-finite
    (caller falls back to last-value reuse)."""
    Phi, evals, b, shp, dt = fit
    pred = (Phi @ (evals.pow(float(k)) * b)).real                                # [d]
    if not torch.isfinite(pred).all():
        return None
    return pred.to(dt).reshape(shp)


def dmd_forecast(snapshots, k: int, rank: int = 0, ridge: float = 1e-8) -> torch.Tensor:
    """Forecast the feature ``k`` steps past the newest snapshot via DMD.

    snapshots : list of >=3 tensors (same shape), OLDEST..NEWEST, the cached
                (CFG-combined) velocities at the recent compute steps.
    k         : horizon (number of steps past the newest snapshot).
    Returns a tensor of the snapshot shape. Falls back to last-value reuse if the
    history is too short or the fit is degenerate. (Stateless: fits on every call;
    the stateful paths cache the fit per compute window instead.)
    """
    fit = dmd_fit(snapshots, rank=rank, ridge=ridge)
    if fit is None:
        return snapshots[-1].clone()
    pred = dmd_eval(fit, k)
    if pred is None:
        return snapshots[-1].clone()
    return pred


# ---------------------------------------------------------------------------
# Stateful integration with the HiCache loop (shares its compute/skip schedule).
# At a compute step we record the raw (CFG-combined) velocity snapshot; at a skip
# step we forecast it via DMD on the UNIFORMLY-SPACED tail of those snapshots.
# ---------------------------------------------------------------------------
def dmd_update_snapshots(state, feature, history: int = 5) -> None:
    """Record the CFG-combined velocity at a compute step for the DMD forecaster.

    Stores ``(compute_step_index, velocity)`` and keeps only the most recent
    ``history`` snapshots — a short, *local* window, because the diffusion
    feature dynamics are non-autonomous (the propagator drifts across timesteps),
    so a long window would average over changing dynamics."""
    snaps = state.setdefault("dmd_snapshots", [])
    # detach().clone(): a bare detached view shares storage with the pipeline
    # tensor, so buffer-reusing inference (torch.compile + CUDA graphs) would
    # silently overwrite the snapshot history in place. The clone costs nothing
    # net — a detached view pins the whole original storage anyway.
    snaps.append((int(state["activated_steps"][-1]), feature.detach().clone()))
    h = int(state.get("history", history))
    if len(snaps) > h:
        del snaps[: len(snaps) - h]


def dmd_forecast_state(state) -> torch.Tensor:
    """DMD forecast of the velocity at the current skip step.

    Uses the longest *uniformly spaced* suffix of the cached snapshots — DMD's
    propagator advances exactly one fixed snapshot-spacing per application, so a
    mixed-spacing window (e.g. across the first-enhance boundary, where the
    compute cadence changes) would corrupt the fit. The skip horizon is expressed
    in snapshot-spacing units, i.e. a *fractional* power of the DMD eigenvalues
    ``lambda**(k/spacing)``. Falls back to the Hermite forecast during warm-up or
    when the uniform window is shorter than 4 — so DMD acts only where it is valid
    and the polynomial path covers the rest.

    The window floor is **4 snapshots (3 pairs)**, not 3: a real-valued trajectory
    spends two real degrees of freedom on every *complex* pole (a conjugate pair
    ``r e^{+-i w}`` -> ``r^t cos(wt), r^t sin(wt)``), so even a single oscillatory
    mode needs rank 3 to identify, which needs 3 snapshot-pairs. With only 2 pairs
    the fit aliases (empirically ~2e-1 vs ~5e-9 at 3 pairs).

    The eigendecomposition ``(Phi, lambda, b)`` is CACHED per compute window: the
    fit inputs cannot change between two compute steps, so refitting on every
    skip step (the pre-1.2 behavior) was pure waste. The cache is keyed by
    (newest compute step, window length, spacing) and recomputed exactly when a
    new snapshot arrives; the per-skip cost drops from one SVD+eig+lstsq to one
    ``Phi @ (lambda**k * b)``."""
    snaps = state.get("dmd_snapshots", [])
    if len(snaps) >= 4:
        steps = [s for s, _ in snaps]
        spacing = steps[-1] - steps[-2]
        if spacing > 0:
            # longest uniform-spacing suffix (walk back while the gap stays equal)
            tail = [snaps[-1], snaps[-2]]
            j = len(snaps) - 2
            while j - 1 >= 0 and steps[j] - steps[j - 1] == spacing:
                tail.append(snaps[j - 1])
                j -= 1
            if len(tail) >= 4:
                vels = [v for _, v in reversed(tail)]            # oldest..newest
                fit_key = (steps[-1], len(vels), spacing)
                if state.get("_dmd_fit_key") != fit_key:
                    state["_dmd_fit"] = dmd_fit(vels)
                    state["_dmd_fit_key"] = fit_key
                fit = state["_dmd_fit"]
                k = (state["step"] - steps[-1]) / spacing        # fractional horizon
                pred = dmd_eval(fit, k) if fit is not None else None
                return vels[-1].clone() if pred is None else pred
    try:                                                         # lazy: keep standalone-testable
        from .hermite import hicache_forecast
    except ImportError:
        from hermite import hicache_forecast
    return hicache_forecast(state)


def _poly2_backcast(prefix, k: float) -> torch.Tensor:
    """Degree-2 Newton-forward extrapolation from the last three of ``prefix``
    --- the polynomial analogue of the Hermite basis, used only as the holdout
    yardstick of the ``holdout="1step"`` mode in :func:`auto_forecast_state`."""
    f0, f1, f2 = prefix[-3], prefix[-2], prefix[-1]
    d1 = f2 - f1
    d2 = (f2 - f1) - (f1 - f0)
    x = float(k)
    return f2 + d1 * x + 0.5 * d2 * x * (x - 1)


def _hermite_holdout_eval(prefix, k: float, sigma: float, max_order: int) -> torch.Tensor:
    """The SERVED Hermite arm rebuilt from a snapshot list: damped scaled-Hermite
    forecast from the newest ``max_order+1`` of ``prefix`` (unit anchor spacing),
    evaluated at SIGNED distance ``k`` past ``prefix[-1]`` (negative = backward).
    Used as the holdout yardstick of the ``holdout="horizon"`` mode, so the
    selection compares DMD against what the fallback arm would actually serve,
    not against an undamped polynomial proxy."""
    try:                                                         # lazy: keep standalone-testable
        from .hermite import scaled_hermite
    except ImportError:
        from hermite import scaled_hermite
    m = max(1, min(int(max_order), len(prefix) - 1))
    diffs = [prefix[-1]]
    cur = list(prefix[-(m + 1):])
    for _ in range(m):
        cur = [cur[i + 1] - cur[i] for i in range(len(cur) - 1)]
        diffs.append(cur[-1])
    x = torch.tensor(float(k), dtype=prefix[-1].dtype, device=prefix[-1].device)
    out = diffs[0]
    for order in range(1, m + 1):
        out = out + diffs[order] / math.factorial(order) * scaled_hermite(order, x, sigma)
    return out


def auto_forecast_state(state) -> torch.Tensor:
    """Holdout-selected forecast: serve DMD only when it *demonstrably* beats
    the polynomial arm on the data at hand (``backend="auto"``).

    Diffusion feature dynamics are non-autonomous --- the propagator drifts
    across timesteps --- so the exponential fit can occasionally misfit a
    window where the polynomial would have been fine. Rather than trusting
    either basis a priori, backcast a held-out snapshot with both arms and
    serve whichever reproduces it better for the current skip window. Two
    holdout modes (``state["holdout"]``):

    ``"1step"`` (default)
        Fit on the tail minus the newest snapshot, backcast it at distance 1
        against the degree-2 Newton polynomial. Cheap and robust: the target
        sits one gap from the fit, so the test is low-variance, but it cannot
        see multi-gap divergence.

    ``"horizon"`` (opt-in)
        Distance-matched selection: backcast at the ACTUAL skip distance of the
        window, ``h ~ (interval-1)/spacing`` snapshot gaps, against the SERVED
        Hermite arm (damped scaled-Hermite, the real fallback) instead of an
        undamped polynomial proxy. For ``h >= 4`` the DMD fit uses the NEWEST
        ``h`` snapshots and backcasts the snapshot ``h`` gaps older (a fresh
        fit, backward extrapolation ``lambda**(-h)``); a 1-gap backcast cannot
        see multi-gap divergence, which makes ``"1step"`` mispredict on
        trajectories whose short-range structure (e.g. a decaying or
        frequency-drifting oscillation) is forecastable one gap out but stale
        at the served distance. For ``h < 4`` (not enough snapshots to hold
        ``h`` out, or a sub-spacing skip distance as in a typical interval-N
        cache where the served horizon is fractional) it degrades to the
        forward prefix backcast at distance ``h``. Evidence (microbench,
        full tables in benchmarks/MICROBENCH_RESULTS.md): horizon-matching
        picks correctly in the oscillatory-with-trend misprediction regime
        where 1step serves the losing arm, but its single far-out target is
        higher-variance under noise and regime switches, so it is NOT
        strictly better and stays opt-in.

    The selection is cached per compute step (it can only change when a new
    snapshot arrives), so the extra cost is one small SVD per *compute* step,
    amortized over all the skip steps that follow it.

    Falls back exactly like :func:`dmd_forecast_state` (Hermite) when the
    uniform tail is shorter than 5 (4 to fit + 1 held out)."""
    snaps = state.get("dmd_snapshots", [])
    if len(snaps) >= 5:
        steps = [s for s, _ in snaps]
        spacing = steps[-1] - steps[-2]
        if spacing > 0:
            tail = [snaps[-1], snaps[-2]]
            j = len(snaps) - 2
            while j - 1 >= 0 and steps[j] - steps[j - 1] == spacing:
                tail.append(snaps[j - 1])
                j -= 1
            if len(tail) >= 5:
                vels = [v for _, v in reversed(tail)]            # oldest..newest
                mode = str(state.get("holdout", "1step"))
                if mode == "horizon":
                    iv = int(state.get("interval", 0))
                    kf_max = ((iv - 1.0) / spacing if iv > 1
                              else (state["step"] - steps[-1]) / spacing)
                    h = max(1, min(int(math.ceil(kf_max)), len(vels) - 1))
                    if h < 4 and len(vels) - h < 4:
                        h = max(1, len(vels) - 4)
                else:
                    h = 1
                cache_key = (steps[-1], len(vels), mode, h)
                choice = state.get("_auto_choice") \
                    if state.get("_auto_choice_key") == cache_key else None
                if choice is None:
                    sigma = float(state.get("sigma", 0.5))
                    max_order = int(state.get("max_order", 2))
                    if mode == "horizon" and h >= 4:
                        # fresh fit on the newest h snapshots, held-out target h
                        # gaps OLDER, matched backward distance -h
                        fit, held = vels[-h:], vels[-1 - h]
                        denom = held.norm().clamp_min(1e-12)
                        e_dmd = (dmd_forecast(fit, -h) - held).norm() / denom
                        e_arm = (_hermite_holdout_eval(fit, -h, sigma, max_order)
                                 - held).norm() / denom
                    elif mode == "horizon":
                        # forward prefix backcast at the matched distance h
                        fit, held = vels[:-h], vels[-1]
                        denom = held.norm().clamp_min(1e-12)
                        e_dmd = (dmd_forecast(fit, h) - held).norm() / denom
                        e_arm = (_hermite_holdout_eval(fit, float(h), sigma, max_order)
                                 - held).norm() / denom
                    else:
                        fit, held = vels[:-1], vels[-1]
                        denom = held.norm().clamp_min(1e-12)
                        e_dmd = (dmd_forecast(fit, 1) - held).norm() / denom
                        e_arm = (_poly2_backcast(fit, 1.0) - held).norm() / denom
                    choice = "dmd" if float(e_dmd) <= float(e_arm) else "hermite"
                    state["_auto_choice"] = choice
                    state["_auto_choice_key"] = cache_key
                if choice == "dmd":
                    # serve through the same per-window eigendecomposition cache
                    # as dmd_forecast_state (refit only when a snapshot arrives)
                    fit_key = (steps[-1], len(vels), spacing)
                    if state.get("_dmd_fit_key") != fit_key:
                        state["_dmd_fit"] = dmd_fit(vels)
                        state["_dmd_fit_key"] = fit_key
                    fit = state["_dmd_fit"]
                    k = (state["step"] - steps[-1]) / spacing
                    pred = dmd_eval(fit, k) if fit is not None else None
                    return vels[-1].clone() if pred is None else pred
                # fall through to the Hermite forecast below
    try:                                                         # lazy: keep standalone-testable
        from .hermite import hicache_forecast
    except ImportError:
        from hermite import hicache_forecast
    return hicache_forecast(state)


# ---------------------------------------------------------------------------
# CPU unit test: DMD is EXACT on an exponential trajectory; a polynomial drifts.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        global ok
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    # Synthetic feature trajectory = sum of 2 damped/oscillatory exponentials (the
    # feature-ODE solution class). d=16 channels, shared poles, per-channel amplitudes.
    d, T = 16, 14
    z = torch.tensor([0.92 * torch.exp(torch.tensor(0.35j)), torch.tensor(0.70 + 0j)],
                     dtype=torch.complex128)                       # 2 poles |z|<1
    A = torch.randn(d, 2, dtype=torch.complex128)                 # per-channel amplitudes
    def true_v(t):
        return (A @ (z ** t)).real.to(torch.float64)              # F_t = Re(sum a_j z_j^t)
    traj = [true_v(t) for t in range(T)]

    # cache the first 5 steps, forecast steps 5,6,7 (k=1,2,3 past the newest)
    hist = traj[:5]
    for k in (1, 2, 3):
        pred = dmd_forecast(hist, k)
        tgt = traj[4 + k]
        rel = (pred - tgt).norm() / tgt.norm()
        check(f"DMD exact on exponential traj @ k={k} (rel err {rel:.2e} < 1e-6)", rel < 1e-6)

    # contrast: a degree-2 polynomial (Taylor/finite-diff) extrapolation DRIFTS on the
    # same exponential signal — the failure mode that caps HiCache.
    def poly2_forecast(h, k):
        f0, f1, f2 = h[-3], h[-2], h[-1]                          # last 3, uniform spacing
        d1 = f2 - f1                                              # 1st diff
        d2 = (f2 - f1) - (f1 - f0)                                # 2nd diff
        x = float(k)
        return f2 + d1 * x + 0.5 * d2 * x * (x - 1)               # Newton forward, degree 2
    rel_dmd = (dmd_forecast(hist, 3) - traj[7]).norm() / traj[7].norm()
    rel_poly = (poly2_forecast(hist, 3) - traj[7]).norm() / traj[7].norm()
    check(f"DMD beats degree-2 poly on exponential extrap @k=3 "
          f"(dmd {rel_dmd:.2e} << poly {rel_poly:.2e})", rel_dmd < 0.01 * rel_poly)

    # robustness: short history -> graceful last-value fallback (no crash)
    check("short history -> fallback (no crash)",
          torch.equal(dmd_forecast(traj[:2], 2), traj[1]))

    # stateful: forecast a SUB-step from 4 snapshots spaced 3 apart. DMD identifies the
    # 3-step propagator (poles z^3); the fractional horizon k=(11-10)/3 takes the
    # principal 1/3-power back to z, advancing exactly one step -> traj[11]. (3 pairs
    # are needed because the complex pole costs 2 real DOF -- see the floor in
    # dmd_forecast_state; 2 pairs would alias.)
    st_uni = {"step": 11, "history": 5,
              "dmd_snapshots": [(1, traj[1]), (4, traj[4]), (7, traj[7]), (10, traj[10])]}
    rel_sub = (dmd_forecast_state(st_uni) - traj[11]).norm() / traj[11].norm()
    check(f"DMD sub-step via uniform tail @ spacing 3 (rel {rel_sub:.2e} < 1e-5)", rel_sub < 1e-5)

    # the uniform-tail walk drops a non-uniform leading snapshot (step 0, spacing 1)
    st_mix = {"step": 11, "history": 6,
              "dmd_snapshots": [(0, traj[0]), (1, traj[1]), (4, traj[4]), (7, traj[7]), (10, traj[10])]}
    rel_mix = (dmd_forecast_state(st_mix) - traj[11]).norm() / traj[11].norm()
    check(f"DMD ignores non-uniform leading snapshot (rel {rel_mix:.2e} < 1e-5)", rel_mix < 1e-5)

    # below the 4-snapshot floor -> Hermite fallback (here only order-0 cached -> last value)
    st_short = {"step": 8, "history": 5, "sigma": 0.5,
                "dmd_snapshots": [(4, traj[4]), (7, traj[7])],
                "derivatives": {0: traj[7]}, "activated_steps": [7]}
    check("DMD < 4 snapshots -> Hermite fallback (last value)",
          torch.allclose(dmd_forecast_state(st_short), traj[7]))

    # ------------------------------------------------------------------
    # auto backend: 1step vs horizon holdout selection
    # ------------------------------------------------------------------
    try:
        from hicache_pp.hermite import hicache_init, hicache_update_derivatives
    except ImportError:
        from hermite import hicache_init, hicache_update_derivatives
    import math as _math

    def _drive(snaps, horizon, holdout):
        st = hicache_init(num_steps=10 ** 6, interval=horizon + 1, max_order=2,
                          first_enhance=0, sigma=0.5, backend="auto",
                          history=len(snaps), holdout=holdout)
        for t, F in enumerate(snaps):
            st["step"] = t
            st["activated_steps"].append(t)
            hicache_update_derivatives(st, F)
            dmd_update_snapshots(st, F, history=len(snaps))
        st["step"] = (len(snaps) - 1) + horizon
        return st

    # the holdout-misprediction regime (oscillatory-with-trend, see
    # benchmarks/forecast_microbench.py::make_traj_osc_trend, seed 0): a smooth
    # curved trend + a decaying, frequency-drifting oscillation. The 1-gap
    # backcast scores DMD well (the oscillation is forecastable one gap out) but
    # at the served distance H=4 the oscillation is decayed/stale and the damped
    # Hermite arm wins -- the distance-matched horizon holdout catches this.
    g = torch.Generator().manual_seed(0)
    Amp = torch.randn(64, generator=g, dtype=torch.float64)
    Cur = torch.randn(64, generator=g, dtype=torch.float64)
    Osc = torch.randn(64, generator=g, dtype=torch.float64)
    H_mis, hist_mis, T_mis = 4, 8, 14
    traj_mis, phase = [], 0.0
    for t in range(T_mis):
        frac = t / float(T_mis)
        phase += 2.2 * (1.0 + 0.5 * frac)
        traj_mis.append(Amp * (2.0 * _math.tanh(2.5 * (frac - 0.45)))
                        + Cur * (1.5 * frac * frac)
                        + Osc * (0.8 * (0.85 ** t) * _math.cos(phase))
                        + 0.01 * torch.randn(64, generator=g, dtype=torch.float64))
    snaps_mis = traj_mis[:hist_mis]
    truth_mis = traj_mis[hist_mis - 1 + H_mis]
    st_1s = _drive(snaps_mis, H_mis, "1step")
    out_1s = auto_forecast_state(st_1s)
    st_hz = _drive(snaps_mis, H_mis, "horizon")
    out_hz = auto_forecast_state(st_hz)
    check("misprediction regime: 1step holdout picks dmd (the inversion)",
          st_1s.get("_auto_choice") == "dmd")
    check("misprediction regime: horizon holdout picks hermite (correct)",
          st_hz.get("_auto_choice") == "hermite")
    err_1s = (out_1s - truth_mis).norm()
    err_hz = (out_hz - truth_mis).norm()
    check(f"horizon-selected forecast beats 1step-selected at the served distance "
          f"({err_hz:.3f} < {err_1s:.3f})", err_hz < err_1s)

    # sanity: on the clean exponential class the horizon holdout still picks DMD
    snaps_exp = traj[:8]
    st_exp = _drive(snaps_exp, 4, "horizon")
    pred_exp = auto_forecast_state(st_exp)
    check("horizon holdout still picks dmd on clean exponentials",
          st_exp.get("_auto_choice") == "dmd")
    rel_exp = (pred_exp - traj[11]).norm() / traj[11].norm()
    check(f"horizon-auto exact on exponential @H=4 (rel {rel_exp:.2e} < 1e-5)", rel_exp < 1e-5)

    # selection is cached per (window, mode, distance); switching the mode re-selects
    check("auto choice cache key includes holdout mode",
          st_1s["_auto_choice_key"] != st_hz["_auto_choice_key"])

    # short-horizon horizon mode (h < 4 -> forward prefix path) stays valid
    st_h2 = _drive(snaps_mis, 2, "horizon")
    out_h2 = auto_forecast_state(st_h2)
    check("horizon holdout h<4 (prefix path) returns a finite forecast and a choice",
          torch.isfinite(out_h2).all().item()
          and st_h2.get("_auto_choice") in ("dmd", "hermite"))

    # ------------------------------------------------------------------
    # eigendecomposition cache: bit-identical to the uncached fit-per-call
    # path across skip steps, scenarios, and cache invalidations
    # ------------------------------------------------------------------
    for name_c, snaps_full in (("exponential", traj[:7]), ("osc-trend", traj_mis[:7])):
        snaps_c = snaps_full[:6]
        st_c = {"step": 0, "history": 8,
                "dmd_snapshots": [(i, v) for i, v in enumerate(snaps_c)]}
        max_dev = 0.0
        for k_skip in (1, 2, 3):                      # several forecasts, ONE window
            st_c["step"] = 5 + k_skip
            cached = dmd_forecast_state(st_c)
            uncached = dmd_forecast(snaps_c, k_skip)  # stateless refit every call
            max_dev = max(max_dev, float((cached - uncached).abs().max()))
        # tolerance, not bit-equality: CPU LAPACK svd/eig is run-to-run
        # nondeterministic at ~1e-16 even for two IDENTICAL stateless calls
        # (measured 3e-16 on this stack); 1e-12 bounds the cache discrepancy.
        check(f"eigencache matches uncached path ({name_c}, 3 horizons, "
              f"max dev {max_dev:.1e} < 1e-12)", max_dev < 1e-12)
        n_snaps = len(st_c["dmd_snapshots"])
        check("eigencache fitted exactly once per window",
              st_c.get("_dmd_fit_key") == (n_snaps - 1, n_snaps, 1))
        # a new snapshot must invalidate the cache and change the fit. Use the
        # trajectory's true next point: a synthetic snapshot (e.g. 1.5x the
        # last) makes the refit eigenproblem near-defective, and CPU LAPACK's
        # run-to-run nondeterminism then amplifies through the ill-conditioned
        # eigenvectors to ~1e-8 forecast jitter (observed), flaking the check.
        key_before = st_c["_dmd_fit_key"]
        st_c["dmd_snapshots"].append((6, snaps_full[6]))
        st_c["step"] = 7
        nxt = dmd_forecast_state(st_c)
        snaps_n = [v for _, v in st_c["dmd_snapshots"]]
        dev_n = float((nxt - dmd_forecast(snaps_n, 1)).abs().max())
        check(f"new snapshot invalidates the eigencache (refit matches uncached, "
              f"dev {dev_n:.1e})",
              st_c["_dmd_fit_key"] != key_before and dev_n < 1e-12)

    # auto serve path goes through the same cache and stays identical
    st_a = _drive(snaps_exp, 4, "horizon")            # picks dmd (checked above)
    served = auto_forecast_state(st_a)
    check("auto dmd serve via eigencache == uncached dmd_forecast (<1e-12)",
          float((served - dmd_forecast(snaps_exp, 4)).abs().max()) < 1e-12)

    print("\nALL PASS" if ok else "\nSOME FAILED")
    raise SystemExit(0 if ok else 1)
