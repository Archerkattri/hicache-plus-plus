"""Hermite-polynomial velocity forecasting (HiCache) — the polynomial forecaster.

On a *skipped* step of a flow-matching / diffusion denoise loop, the final
(CFG-combined) velocity is forecast with a **dual-scaled physicist's Hermite
polynomial** basis instead of evaluating the network, skipping
``(interval-1)/interval`` of the forward passes. The dual scaling keeps the
high-order terms bounded, giving a more stable forecast than the equivalent Taylor
(monomial) series — TaylorSeer is the special case where the basis is the plain
monomial ``k^i``.

The basis and finite-difference core are model-agnostic; integrate it by calling
``hicache_init`` / ``hicache_decide`` / ``hicache_update_derivatives`` /
``hicache_forecast`` from your sampling loop (see ``integrations/``).

Reference
---------
HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial
Feature Forecasting (arXiv:2508.16984).

Method
------
Let ``F_t`` be the cached velocity at the most recent compute ("full") step and
``N = interval`` the spacing between compute steps. At a compute step we update
backward finite differences::

    Delta^0 F_t = F_t
    Delta^i F_t = (Delta^{i-1} F_t - Delta^{i-1} F_{t-N}) / N

At a skipped step ``k`` steps past the last compute step the velocity is::

    F_hat_{t+k} = F_t + sum_{i=1}^{m} (Delta^i F_t / i!) * Htilde_i(k)

with the dual-scaled physicist's Hermite polynomial (contraction ``sigma in (0,1)``)::

    Htilde_n(x) = sigma^n * H_n(sigma * x)
    H_0(x) = 1,  H_1(x) = 2x,  H_{n+1}(x) = 2 x H_n(x) - 2 n H_{n-1}(x)

TaylorSeer is the special case where ``Htilde_i(k)`` is the monomial ``k^i``.
The basis is evaluated at ``x = +k``: the finite differences are forward slopes on
increasing step indices, so the forecast ``k`` steps PAST the newest anchor must
evaluate at ``+k`` (the upstream TaylorSeer convention, distance ``+k``); evaluating
at ``-k`` extrapolates backwards (odd-order terms flip sign).
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, Optional
import uuid

import torch


def _new_telemetry() -> Dict[str, Any]:
    """Return the intentionally small, additive state telemetry schema."""
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


# ---------------------------------------------------------------------------
# Hermite basis
# ---------------------------------------------------------------------------
def physicists_hermite(n: int, x: torch.Tensor) -> torch.Tensor:
    """Physicist's Hermite ``H_n(x)`` via the stable recurrence.

    ``H_0 = 1``, ``H_1 = 2x``, ``H_{k+1} = 2 x H_k - 2 k H_{k-1}``.
    """
    if n < 0:
        raise ValueError(f"Hermite order must be >= 0, got {n}")
    if n == 0:
        return torch.ones_like(x)
    h_prev = torch.ones_like(x)          # H_0
    h_curr = 2.0 * x                     # H_1
    if n == 1:
        return h_curr
    for k in range(1, n):
        h_next = 2.0 * x * h_curr - 2.0 * k * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr


def scaled_hermite(n: int, x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Dual-scaled Hermite ``Htilde_n(x) = sigma^n * H_n(sigma * x)``."""
    return (sigma ** n) * physicists_hermite(n, sigma * x)


# ---------------------------------------------------------------------------
# HiCache state
# ---------------------------------------------------------------------------
def hicache_init(
    num_steps: int,
    interval: int = 4,
    max_order: int = 1,
    first_enhance: int = 2,
    end_enhance: Optional[int] = None,
    sigma: float = 0.5,
    backend: str = "hermite",
    history: int = 5,
    holdout: str = "1step",
) -> Dict[str, Any]:
    """Create a fresh HiCache state dict for one sampling run.

    interval   : ``N`` -- one compute step then ``interval-1`` forecasts.
    max_order  : highest finite-difference / Hermite order ``m`` (>= 1).
    first_enhance : always compute the first ``first_enhance`` steps.
    end_enhance   : always compute steps with index ``>= end_enhance``
                    (defaults to ``num_steps`` -> only the schedule applies).
    sigma      : Hermite contraction factor in ``(0, 1)``.
    backend    : forecast basis -- ``"hermite"`` (polynomial, default) or
                 ``"dmd"`` (exponential / Prony-DMD; uses the same schedule but
                 forecasts from raw velocity snapshots, with Hermite as warm-up
                 fallback -- see ``hicache_dmd.py``).
    history    : DMD snapshot window length (ignored by the Hermite backend).
    holdout    : selection test of the ``"auto"`` backend (ignored otherwise) --
                 ``"1step"`` (default): 1-gap backcast of the newest held-out
                 snapshot against a degree-2 polynomial proxy; ``"horizon"``
                 (opt-in): backcast a held-out snapshot at the actual skip
                 distance of the window against the served Hermite arm.
                 Horizon-matching fixes regimes where the 1-gap ranking inverts
                 at the served distance, but its single far-out target makes the
                 selection noisier elsewhere -- it is not strictly better on the
                 microbench, hence opt-in. See ``auto_forecast_state``.
    """
    if interval < 1:
        raise ValueError("interval must be >= 1")
    if max_order < 1:
        raise ValueError("max_order must be >= 1")
    if not (0.0 < sigma < 1.0):
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")
    if backend not in ("hermite", "dmd", "auto"):
        raise ValueError(f"backend must be 'hermite', 'dmd', or 'auto', got {backend!r}")
    if holdout not in ("horizon", "1step"):
        raise ValueError(f"holdout must be 'horizon' or '1step', got {holdout!r}")
    return {
        "num_steps": int(num_steps),
        "interval": int(interval),
        "max_order": int(max_order),
        "first_enhance": int(first_enhance),
        "end_enhance": int(end_enhance if end_enhance is not None else num_steps),
        "sigma": float(sigma),
        "backend": str(backend),
        "history": int(history),
        "holdout": str(holdout),
        "run_id": uuid.uuid4().hex,
        "branch_id": None,
        "telemetry": _new_telemetry(),
        "step": 0,
        "counter": 0,            # forecasts since last compute
        "type": None,            # "full" | "forecast"
        "activated_steps": [],   # indices of compute steps
        "derivatives": {},       # {0: F_t, 1: Delta^1 F_t, ...}
        "prev_derivatives": {},
        "dmd_snapshots": [],     # [(compute_step, velocity), ...] for the DMD backend
    }


def hicache_decide(state: Dict[str, Any]) -> str:
    """Decide whether the current step is computed or forecast. Mirrors the
    paper's ``t mod N`` schedule plus the first/last enhance-window guards.

    A state with no anchor always returns ``"full"``.  This makes
    ``first_enhance=0`` safe: the first decision cannot select a forecast before
    the caller has supplied any feature to forecast.
    """
    step = state["step"]
    first = step < state["first_enhance"]
    last = step >= state["end_enhance"]
    interval_hit = state["counter"] >= state["interval"] - 1
    no_anchor = not state.get("derivatives")

    if first or last or interval_hit or no_anchor:
        state["type"] = "full"
        state["counter"] = 0
        state["activated_steps"].append(step)
    else:
        state["type"] = "forecast"
        state["counter"] += 1
    _ensure_telemetry(state)["decisions"][state["type"]] = (
        int(_ensure_telemetry(state)["decisions"].get(state["type"], 0)) + 1
    )
    return state["type"]


def hicache_reset(state: Dict[str, Any], *, run_id: Optional[str] = None,
                  branch_id: Optional[str] = None) -> Dict[str, Any]:
    """Reset mutable sampling state in place and return the same state object.

    Configuration keys and the existing public state-dict shape are preserved.
    A reset starts a fresh run identity unless one is supplied; ``branch_id`` is
    retained unless a replacement is supplied.  This wrapper is additive and
    does not change any existing helper signature.
    """
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
    for key in ("_dmd_fit", "_dmd_fit_key", "_auto_choice", "_auto_choice_key"):
        state.pop(key, None)
    state["telemetry"] = _new_telemetry()
    return state


def hicache_update_derivatives(state: Dict[str, Any], feature: torch.Tensor) -> None:
    """Backward finite-difference derivatives at a compute step.

    ``Delta^0 = feature``; ``Delta^i = (Delta^{i-1}_now - Delta^{i-1}_prev) / dist``.
    With only one anchor so far, only the 0th-order term is kept, so the forecast
    degenerates to plain reuse of the cached velocity (the correct zero-information
    forecast).
    """
    interval = state["interval"]
    prev = state["derivatives"]
    have_prev = len(prev) > 0

    new_deriv: Dict[int, torch.Tensor] = {0: feature}
    if have_prev:
        acts = state["activated_steps"]
        dist = acts[-1] - acts[-2] if len(acts) >= 2 else interval
        dist = max(int(dist), 1)
        for order in range(state["max_order"]):
            if order not in prev:
                break
            new_deriv[order + 1] = (new_deriv[order] - prev[order]) / dist

    state["prev_derivatives"] = prev
    state["derivatives"] = new_deriv


def hicache_forecast(state: Dict[str, Any]) -> torch.Tensor:
    """Scaled-Hermite forecast of the velocity at the current skip step:
    ``F_hat = F_t + sum_{i>=1} (Delta^i F_t / i!) * Htilde_i(k)`` where ``k`` is the
    number of steps since the last compute step. With <2 anchors this returns the
    cached velocity unchanged (the correct degenerate forecast)."""
    deriv = state["derivatives"]
    if 0 not in deriv:
        raise RuntimeError("hicache_forecast called before any compute step")
    _record_method(state, "hermite")

    k = state["step"] - state["activated_steps"][-1]
    sigma = state["sigma"]
    base = deriv[0]
    x = torch.tensor(float(k), dtype=base.dtype, device=base.device)

    result = base
    order = 1
    while order in deriv:
        coeff = deriv[order] / math.factorial(order)
        result = result + coeff * scaled_hermite(order, x, sigma)
        order += 1
    return result


# ---------------------------------------------------------------------------
