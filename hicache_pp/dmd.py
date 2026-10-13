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


def _record_method(state, method: str) -> None:
    telemetry = state.setdefault("telemetry", {})
    counts = telemetry.setdefault("method_counts", {})
    counts[method] = int(counts.get(method, 0)) + 1


def _record_fallback(state, reason: str) -> None:
    telemetry = state.setdefault("telemetry", {})
    fallbacks = telemetry.setdefault("fallbacks", {})
    fallbacks[reason] = int(fallbacks.get(reason, 0)) + 1


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
    pred = pred.to(dt)
    if not torch.isfinite(pred).all():
        return None
    return pred.reshape(shp)


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
    fallback_reason = "dmd_insufficient_uniform_history"
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
                if pred is not None:
                    _record_method(state, "dmd")
                    return pred
                _record_fallback(state, "dmd_fit_failed" if fit is None else "dmd_nonfinite_output")
                _record_method(state, "reuse")
                return vels[-1].clone()
            fallback_reason = "dmd_nonuniform_or_short_tail"
    try:                                                         # lazy: keep standalone-testable
        from .hermite import hicache_forecast
    except ImportError:
        from hermite import hicache_forecast
    _record_fallback(state, fallback_reason)
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
                    if pred is not None:
                        _record_method(state, "dmd")
                        return pred
                    _record_fallback(state, "dmd_fit_failed" if fit is None else "dmd_nonfinite_output")
                    _record_method(state, "reuse")
                    return vels[-1].clone()
                # fall through to the Hermite forecast below
    try:                                                         # lazy: keep standalone-testable
        from .hermite import hicache_forecast
    except ImportError:
        from hermite import hicache_forecast
    return hicache_forecast(state)


# ---------------------------------------------------------------------------
