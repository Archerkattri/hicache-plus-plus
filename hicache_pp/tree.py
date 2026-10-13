"""PyTree-aware velocity forecasting — Hermite (HiCache), DMD/Prony (HiCache++), Adaptive-CFG.

The tree-aware counterpart of ``hermite`` + ``dmd``, for samplers whose velocity is a
``torch.utils._pytree`` structure (a dict / nested container of tensors) rather than a
single tensor — e.g. structured-latent or sparse generators. All three methods operate on
PyTrees via ``_pytree.tree_map``; the scalar Hermite / finite-difference coefficients (and
the DMD eigen-spectrum) are shared across leaves, so a forecast is one tree map.

  * **HiCache** (Hermite) — at a *skipped* solver step the (CFG-combined) velocity tree is
    forecast with a dual-scaled physicist's Hermite polynomial instead of calling the
    dynamics function, skipping ``(interval-1)/interval`` of the evaluations.

  * **HiCache++** (DMD / Prony) — the exponential forecaster (see :mod:`hicache_pp.dmd`),
    tree-aware: flatten the uniform-spacing snapshot tail to one vector per snapshot, run
    DMD, and unflatten the prediction back to the tree. Exact on the exponential solution
    class, so it stays lossless at larger skip intervals than the polynomial.

  * **Adaptive-CFG** (Adaptive Guidance, arXiv:2312.12487) — once the conditional and
    unconditional velocities align (cosine ≥ ``gamma_bar``) the unconditional pass is
    dropped and the guidance term is reconstructed from cached anchors. With the convention
    ``v_cfg = (1+w)·y_cond − w·y_uncond`` the guidance term is ``g = w·(y_cond − y_uncond)``
    and ``v_cfg = y_cond + g``.

Integrate by calling these helpers from the ODE-solver step (see ``integrations/``); there
is no runtime monkey-patching.
"""
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import uuid

import torch
from torch.utils import _pytree


def _new_telemetry() -> Dict[str, Any]:
    return {
        "decisions": {"full": 0, "forecast": 0},
        "method_counts": {"hermite": 0, "dmd": 0, "reuse": 0},
        "fallbacks": {},
    }


def _ensure_telemetry(state: Dict[str, Any]) -> Dict[str, Any]:
    telemetry = state.setdefault("telemetry", _new_telemetry())
    telemetry.setdefault("decisions", {})
    telemetry.setdefault("method_counts", {})
    telemetry.setdefault("fallbacks", {})
    return telemetry


def _record_method(state: Dict[str, Any], method: str) -> None:
    counts = _ensure_telemetry(state)["method_counts"]
    counts[method] = int(counts.get(method, 0)) + 1


def _record_fallback(state: Dict[str, Any], reason: str) -> None:
    fallbacks = _ensure_telemetry(state)["fallbacks"]
    fallbacks[reason] = int(fallbacks.get(reason, 0)) + 1


def hicache_telemetry(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a detached copy of additive decision/method/fallback telemetry."""
    return deepcopy(_ensure_telemetry(state))


# --------------------------------------------------------------------------- #
# Hermite basis (scalar coefficients, applied per leaf)                        #
# --------------------------------------------------------------------------- #
def physicists_hermite(n: int, x: float) -> float:
    """Physicist's Hermite ``H_n(x)`` via the stable recurrence (scalar)."""
    if n < 0:
        raise ValueError(f"Hermite order must be >= 0, got {n}")
    if n == 0:
        return 1.0
    h_prev, h_curr = 1.0, 2.0 * x
    if n == 1:
        return h_curr
    for k in range(1, n):
        h_prev, h_curr = h_curr, 2.0 * x * h_curr - 2.0 * k * h_prev
    return h_curr


def hermite_coeff(order: int, k: int, sigma: float) -> float:
    """Forecast coefficient ``Htilde_order(k) / order!`` (a scalar, same for all
    leaves). ``Htilde_n(x) = sigma^n H_n(sigma x)``. Evaluated at ``x = +k``: the
    finite differences are forward slopes, so forecasting ``k`` steps PAST the newest
    anchor evaluates at ``+k`` (upstream TaylorSeer convention); ``-k`` extrapolates
    backwards."""
    x = float(k)
    htilde = (sigma ** order) * physicists_hermite(order, sigma * x)
    return htilde / math.factorial(order)


# --------------------------------------------------------------------------- #
# tree helpers                                                                #
# --------------------------------------------------------------------------- #
def tree_axpy(c: float, a: Any, b: Any) -> Any:
    """``a + c * b`` leafwise (c scalar)."""
    return _pytree.tree_map(lambda av, bv: av + c * bv, a, b)


def tree_sub_div(a: Any, b: Any, d: float) -> Any:
    """``(a - b) / d`` leafwise."""
    return _pytree.tree_map(lambda av, bv: (av - bv) / d, a, b)


def tree_detach(a: Any) -> Any:
    return _pytree.tree_map(lambda v: v.detach() if isinstance(v, torch.Tensor) else v, a)


def _tree_snapshot(a: Any) -> Any:
    """detach + clone leafwise. Anything STORED in the cache state across steps
    must own its storage: a bare detached view shares the pipeline tensor's
    buffer, which buffer-reusing inference (torch.compile + CUDA graphs) would
    silently overwrite in place. One memcpy per compute step — negligible next
    to a network forward."""
    return _pytree.tree_map(
        lambda v: v.detach().clone() if isinstance(v, torch.Tensor) else v, a)


def tree_cosine(a: Any, b: Any, eps: float = 1e-12) -> float:
    """Cosine similarity over the concatenated leaves of two velocity trees."""
    la = [t.reshape(-1).float() for t in _pytree.tree_leaves(a) if isinstance(t, torch.Tensor)]
    lb = [t.reshape(-1).float() for t in _pytree.tree_leaves(b) if isinstance(t, torch.Tensor)]
    fa, fb = torch.cat(la), torch.cat(lb)
    return float((torch.dot(fa, fb) / (fa.norm() * fb.norm() + eps)).item())


# --------------------------------------------------------------------------- #
# HiCache (tree-aware velocity forecast)                                       #
# --------------------------------------------------------------------------- #
def hicache_init(num_steps, interval=4, max_order=1, first_enhance=2,
                 end_enhance=None, sigma=0.5, backend="hermite", history=5) -> Dict[str, Any]:
    if interval < 1 or max_order < 1:
        raise ValueError("interval and max_order must be >= 1")
    if not (0.0 < sigma < 1.0):
        raise ValueError(f"sigma must be in (0,1), got {sigma}")
    if backend not in ("hermite", "dmd", "auto"):
        raise ValueError(f"backend must be 'hermite', 'dmd', or 'auto', got {backend!r}")
    return {
        "num_steps": int(num_steps), "interval": int(interval), "max_order": int(max_order),
        "first_enhance": int(first_enhance),
        "end_enhance": int(end_enhance if end_enhance is not None else num_steps),
        "sigma": float(sigma), "backend": str(backend), "history": int(history),
        "run_id": uuid.uuid4().hex, "branch_id": None,
        "telemetry": _new_telemetry(),
        "step": 0, "counter": 0,
        "type": None, "activated_steps": [], "derivatives": {}, "prev_derivatives": {},
        "dmd_snapshots": [],   # [(compute_step, velocity_tree), ...] for the DMD backend
    }


def hicache_decide(state: Dict[str, Any]) -> str:
    step = state["step"]
    no_anchor = not state.get("derivatives")
    if step < state["first_enhance"] or step >= state["end_enhance"] \
            or no_anchor \
            or state["counter"] >= state["interval"] - 1:
        state["counter"] = 0
        state["activated_steps"].append(step)
        state["type"] = "full"
    else:
        state["counter"] += 1
        state["type"] = "forecast"
    decision = state["type"]
    decisions = _ensure_telemetry(state)["decisions"]
    decisions[decision] = int(decisions.get(decision, 0)) + 1
    return decision


def hicache_reset(state: Dict[str, Any], *, run_id: Optional[str] = None,
                  branch_id: Optional[str] = None) -> Dict[str, Any]:
    """Reset mutable tree sampling state in place and return the same object."""
    state["run_id"] = str(run_id) if run_id is not None else uuid.uuid4().hex
    if branch_id is not None:
        state["branch_id"] = str(branch_id)
    state["step"] = 0
    state["counter"] = 0
    state["type"] = None
    state["activated_steps"] = []
    state["derivatives"] = {}
    state["prev_derivatives"] = {}
    state["dmd_snapshots"] = []
    for key in ("_dmd_tree_fit", "_dmd_tree_fit_key", "_auto_choice", "_auto_choice_key"):
        state.pop(key, None)
    state["telemetry"] = _new_telemetry()
    return state


def hicache_update_tree(state: Dict[str, Any], velocity_tree: Any) -> None:
    prev = state["derivatives"]
    new_deriv = {0: _tree_snapshot(velocity_tree)}
    if len(prev) > 0:
        acts = state["activated_steps"]
        dist = max(int(acts[-1] - acts[-2]) if len(acts) >= 2 else state["interval"], 1)
        for order in range(state["max_order"]):
            if order not in prev:
                break
            new_deriv[order + 1] = tree_sub_div(new_deriv[order], prev[order], dist)
    state["prev_derivatives"] = prev
    state["derivatives"] = new_deriv


def hicache_forecast_tree(state: Dict[str, Any]) -> Any:
    deriv = state["derivatives"]
    if 0 not in deriv:
        raise RuntimeError("hicache_forecast_tree called before any compute step")
    _record_method(state, "hermite")
    k = state["step"] - state["activated_steps"][-1]
    result = deriv[0]
    order = 1
    while order in deriv:
        result = tree_axpy(hermite_coeff(order, k, state["sigma"]), result, deriv[order])
        order += 1
    return result


# --------------------------------------------------------------------------- #
# DMD / Prony exponential velocity forecast (tree-aware)                       #
# The exponential analogue of HiCache: a diffusion feature trajectory solves a #
# near-linear feature-ODE whose exact solution class is a sum of (damped/      #
# oscillatory) EXPONENTIALS, not polynomials. DMD (Schmid 2010) is the SVD-     #
# regularised generalisation of Prony's method (1795) / Matrix-Pencil: identify #
# the linear propagator from raw velocity snapshots and advance it by          #
# eigenvalue powers. Trees are flattened to one vector per snapshot, DMD'd, and #
# unflattened back. Exact on the exponential class where the Hermite polynomial #
# drifts (the failure mode that caps HiCache at a modest skip interval).        #
# --------------------------------------------------------------------------- #
def _dmd_fit_flat(snapshots, rank=0, ridge=1e-8):
    """Fit the DMD eigendecomposition once for a window of >=4 equal-length 1-D
    snapshot vectors. Returns ``(Phi, evals, b, dtype)`` or ``None`` on a
    degenerate fit. Horizon-independent, so :func:`dmd_forecast_tree` caches it
    per compute window and reuses it for every skip-step forecast."""
    if len(snapshots) < 4:
        return None
    dt = snapshots[-1].dtype
    V = torch.stack(snapshots, dim=1).to(torch.float64)          # [d, n]
    X, Xp = V[:, :-1], V[:, 1:]
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except Exception:  # noqa: BLE001 — numerically degenerate fit: fall back to last-value reuse
        return None
    if rank <= 0:
        rank = int((S > S[0] * 1e-4).sum().clamp(min=1).item())
    rank = max(1, min(rank, S.numel()))
    Ur, Sr, Vr = U[:, :rank], S[:rank], Vh[:rank].mH
    Sinv = (1.0 / (Sr + ridge)).to(torch.complex128)
    Atil = (Ur.mH @ Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)
    try:
        evals, W = torch.linalg.eig(Atil)
        Phi = ((Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)) @ W
        b = torch.linalg.lstsq(Phi, V[:, -1].to(torch.complex128).unsqueeze(1)).solution.squeeze(1)
    except Exception:  # noqa: BLE001 — numerically degenerate fit: fall back to last-value reuse
        return None
    return (Phi, evals, b, dt)


def _dmd_eval_flat(fit, k):
    """Evaluate a cached :func:`_dmd_fit_flat` at (fractional) horizon ``k``.
    Returns ``None`` on a non-finite prediction (caller falls back to reuse).
    The finite check is post-cast: a finite float64 forecast can still overflow
    to ``inf`` once cast to an fp16 output dtype, so the cast result is the thing
    that must be guarded before it reaches the denoiser."""
    Phi, evals, b, dt = fit
    out = (Phi @ (evals.pow(float(k)) * b)).real.to(dt)
    if not torch.isfinite(out).all():
        return None
    return out


def _dmd_forecast_flat(snapshots, k, rank=0, ridge=1e-8):
    """DMD forecast of a flat vector ``k`` (fractional) steps past the newest of a
    list of >=4 equal-length 1-D snapshot vectors. Falls back to last-value reuse.
    (Stateless fit-per-call wrapper; the stateful tree path caches the fit.)"""
    fit = _dmd_fit_flat(snapshots, rank=rank, ridge=ridge)
    if fit is None:
        return snapshots[-1].clone()
    pred = _dmd_eval_flat(fit, k)
    if pred is None:
        return snapshots[-1].clone()
    return pred


def dmd_update_snapshots_tree(state: Dict[str, Any], velocity_tree: Any, history: int = 5) -> None:
    """Record the (CFG-combined) velocity TREE at a compute step (short local sliding
    window — the diffusion dynamics are non-autonomous, so a long window averages over
    changing dynamics)."""
    snaps = state.setdefault("dmd_snapshots", [])
    snaps.append((int(state["activated_steps"][-1]), _tree_snapshot(velocity_tree)))
    h = int(state.get("history", history))
    if len(snaps) > h:
        del snaps[: len(snaps) - h]


def dmd_forecast_tree(state: Dict[str, Any]) -> Any:
    """DMD forecast of the velocity tree at the current skip step. Flattens the longest
    uniformly-spaced tail (>=4 snapshots) to vectors, runs DMD with a fractional horizon
    ``lambda**(k/spacing)``, and unflattens to the tree. Floor is 4 snapshots (3 pairs):
    a real trajectory spends two real DOF per COMPLEX pole, so even one oscillatory mode
    needs rank 3. Below that / on a non-uniform window, falls back to Hermite (warm-up).

    The eigendecomposition (and the tree flattening that feeds it) is CACHED per
    compute window: the snapshot window cannot change between two compute steps, so
    refitting on every skip step (the pre-1.2 behavior) was pure waste. The cache is
    keyed by (newest compute step, window length, spacing) and recomputed exactly
    when a new snapshot arrives."""
    snaps = state.get("dmd_snapshots", [])
    fallback_reason = "dmd_insufficient_uniform_history"
    if len(snaps) >= 4:
        steps = [s for s, _ in snaps]
        spacing = steps[-1] - steps[-2]
        if spacing > 0:
            tail = [snaps[-1], snaps[-2]]
            j = len(snaps) - 2
            while j - 1 >= 0 and steps[j] - steps[j - 1] == spacing:
                tail.append(snaps[j - 1]); j -= 1
            if len(tail) >= 4:
                fit_key = (steps[-1], len(tail), spacing)
                if state.get("_dmd_tree_fit_key") != fit_key:
                    trees = [v for _, v in reversed(tail)]            # oldest..newest
                    flat = [_pytree.tree_flatten(t) for t in trees]   # [(leaves, spec), ...]
                    leaves_new, spec = flat[-1]
                    shapes = [l.shape for l in leaves_new]
                    vecs = [torch.cat([l.reshape(-1) for l in lv]) for lv, _ in flat]
                    state["_dmd_tree_fit"] = (_dmd_fit_flat(vecs), spec, shapes, vecs[-1])
                    state["_dmd_tree_fit_key"] = fit_key
                fit, spec, shapes, newest_vec = state["_dmd_tree_fit"]
                kf = (state["step"] - steps[-1]) / spacing
                pred = _dmd_eval_flat(fit, kf) if fit is not None else None
                if pred is None:
                    _record_fallback(state, "dmd_fit_failed" if fit is None else "dmd_nonfinite_output")
                    _record_method(state, "reuse")
                    pred = newest_vec.clone()
                else:
                    _record_method(state, "dmd")
                out, i = [], 0
                for sh in shapes:
                    n = 1
                    for d in sh:
                        n *= int(d)
                    out.append(pred[i:i + n].reshape(sh)); i += n
                return _pytree.tree_unflatten(out, spec)
            fallback_reason = "dmd_nonuniform_or_short_tail"
    _record_fallback(state, fallback_reason)
    return hicache_forecast_tree(state)


# --------------------------------------------------------------------------- #
# Adaptive-CFG (tree-aware guidance forecast)                                  #
# --------------------------------------------------------------------------- #
def adaptive_cfg_init(num_steps, gamma_bar=0.94, warmup=2, max_order=1) -> Dict[str, Any]:
    if not (0.0 <= gamma_bar <= 1.0):
        raise ValueError(f"gamma_bar must be in [0,1], got {gamma_bar}")
    return {"num_steps": int(num_steps), "gamma_bar": float(gamma_bar), "warmup": int(warmup),
            "max_order": int(max_order), "step": 0, "anchors": [], "last_gamma": None,
            "n_full": 0, "n_skip": 0}


def adaptive_cfg_decide(state: Dict[str, Any], gamma: Optional[float]) -> bool:
    """True -> run the full (uncond) pass this step."""
    step = state["step"]
    if step < state["warmup"] or step >= state["num_steps"] - 1:
        return True
    if len(state["anchors"]) == 0 or gamma is None:
        return True
    return gamma < state["gamma_bar"]


def guidance_term_tree(y_cond: Any, y_uncond: Any, strength: float) -> Any:
    """Guidance term ``g = strength * (y_cond - y_uncond)`` (leafwise)."""
    return _pytree.tree_map(lambda c, u: strength * (c - u), y_cond, y_uncond)


def forecast_guidance_tree(anchors: List[Tuple[int, Any]], step: int, max_order: int = 1) -> Any:
    """Newton divided-difference forecast of the guidance-term TREE at ``step``.
    Handles non-uniform anchor spacing (robust to HiCache-skipped steps)."""
    if len(anchors) == 0:
        raise ValueError("forecast_guidance_tree requires at least one anchor")
    if len(anchors) == 1 or max_order < 1:
        return anchors[-1][1]
    used = anchors[-(max_order + 1):]
    xs = [float(s) for s, _ in used]
    col = [g for _, g in used]
    n = len(used)
    coeffs = [col[0]]
    for k in range(1, n):
        col = [tree_sub_div(col[i + 1], col[i], xs[i + k] - xs[i]) for i in range(n - k)]
        coeffs.append(col[0])
    x = float(step)
    result = coeffs[-1]
    for k in range(n - 2, -1, -1):
        xk = xs[k]
        result = _pytree.tree_map(lambda r, c: r * (x - xk) + c, result, coeffs[k])
    return result


def reconstruct_cfg_tree(y_cond: Any, g: Any) -> Any:
    """``v_cfg = y_cond + g`` (leafwise) — the skip-step reconstruction."""
    return _pytree.tree_map(lambda c, gv: c + gv, y_cond, g)


# --------------------------------------------------------------------------- #
