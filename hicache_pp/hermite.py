"""HiCache: Hermite-polynomial velocity forecasting for Hunyuan3D-2.1.

Training-free inference acceleration for the flow-matching denoise loop in
``Hunyuan3DDiTFlowMatchingPipeline``. The final (CFG-combined) flow-matching
velocity at *skipped* sampling steps is forecast with a **dual-scaled
(physicist's) Hermite polynomial** basis instead of a DiT forward pass, so the
expensive ``self.model(...)`` call is skipped on ``(interval-1)/interval`` of the
steps. The dual scaling keeps the high-order terms bounded, giving a more stable
forecast than the equivalent Taylor (monomial) series.

This is a first-class part of the pipeline (the denoise loop calls these helpers
directly — there is NO runtime monkey-patching). Enable it with
``pipe.enable_hicache(...)``.

Reference
---------
HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial
Feature Forecasting (arXiv:2508.16984). Ported from the TRELLIS-v1 implementation
in ``faster-trellis`` — the Hermite/finite-difference core is model-agnostic; only
the loop wiring (in ``pipelines.py``) is Hunyuan-specific.

Method
------
Let ``F_t`` be the cached velocity at the most recent compute ("full") step and
``N = interval`` the spacing between compute steps. At a compute step we update
backward finite differences::

    Delta^0 F_t = F_t
    Delta^i F_t = (Delta^{i-1} F_t - Delta^{i-1} F_{t-N}) / N

At a skipped step ``k`` steps past the last compute step the velocity is::

    F_hat_{t-k} = F_t + sum_{i=1}^{m} (Delta^i F_t / i!) * Htilde_i(-k)

with the dual-scaled physicist's Hermite polynomial (contraction ``sigma in (0,1)``)::

    Htilde_n(x) = sigma^n * H_n(sigma * x)
    H_0(x) = 1,  H_1(x) = 2x,  H_{n+1}(x) = 2 x H_n(x) - 2 n H_{n-1}(x)

TaylorSeer is the special case where ``Htilde_i(-k)`` is the monomial ``(-k)^i``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


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
    """
    if interval < 1:
        raise ValueError("interval must be >= 1")
    if max_order < 1:
        raise ValueError("max_order must be >= 1")
    if not (0.0 < sigma < 1.0):
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")
    if backend not in ("hermite", "dmd"):
        raise ValueError(f"backend must be 'hermite' or 'dmd', got {backend!r}")
    return {
        "num_steps": int(num_steps),
        "interval": int(interval),
        "max_order": int(max_order),
        "first_enhance": int(first_enhance),
        "end_enhance": int(end_enhance if end_enhance is not None else num_steps),
        "sigma": float(sigma),
        "backend": str(backend),
        "history": int(history),
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
    paper's ``t mod N`` schedule plus the first/last enhance-window guards."""
    step = state["step"]
    first = step < state["first_enhance"]
    last = step >= state["end_enhance"]
    interval_hit = state["counter"] >= state["interval"] - 1

    if first or last or interval_hit:
        state["type"] = "full"
        state["counter"] = 0
        state["activated_steps"].append(step)
    else:
        state["type"] = "forecast"
        state["counter"] += 1
    return state["type"]


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
    ``F_hat = F_t + sum_{i>=1} (Delta^i F_t / i!) * Htilde_i(-k)`` where ``k`` is the
    number of steps since the last compute step. With <2 anchors this returns the
    cached velocity unchanged (the correct degenerate forecast)."""
    deriv = state["derivatives"]
    if 0 not in deriv:
        raise RuntimeError("hicache_forecast called before any compute step")

    k = state["step"] - state["activated_steps"][-1]
    sigma = state["sigma"]
    base = deriv[0]
    x = torch.tensor(float(-k), dtype=base.dtype, device=base.device)

    result = base
    order = 1
    while order in deriv:
        coeff = deriv[order] / math.factorial(order)
        result = result + coeff * scaled_hermite(order, x, sigma)
        order += 1
    return result


# ---------------------------------------------------------------------------
# CPU unit test (no GPU, no model): validates the Hermite/forecast core.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True

    def check(name: str, cond: bool) -> None:
        global ok
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    xs = torch.tensor([-1.5, 0.0, 0.7, 2.0])
    check("H_0 == 1", torch.allclose(physicists_hermite(0, xs), torch.ones_like(xs)))
    check("H_1 == 2x", torch.allclose(physicists_hermite(1, xs), 2 * xs))
    check("H_2 == 4x^2-2", torch.allclose(physicists_hermite(2, xs), 4 * xs**2 - 2))
    check("H_3 == 8x^3-12x", torch.allclose(physicists_hermite(3, xs), 8 * xs**3 - 12 * xs))

    sig = 0.5
    check("scaled_hermite == sigma^n H_n(sigma x)",
          torch.allclose(scaled_hermite(2, xs, sig), (sig**2) * (4 * (sig * xs) ** 2 - 2)))

    # finite differences exact on a linear velocity series F_s = a + b*s
    a = torch.tensor([1.0, -2.0, 0.5]); b = torch.tensor([0.3, 0.3, 0.3]); interval = 4
    st = hicache_init(num_steps=12, interval=interval, max_order=1,
                      first_enhance=0, end_enhance=12, sigma=sig)
    st["step"] = 0; st["activated_steps"].append(0)
    hicache_update_derivatives(st, a + b * 0.0)
    check("1 anchor -> only order-0 derivative", set(st["derivatives"].keys()) == {0})
    st["step"] = 1
    check("<2 anchors -> forecast == cached velocity", torch.allclose(hicache_forecast(st), a))
    st["step"] = interval; st["activated_steps"].append(interval)
    hicache_update_derivatives(st, a + b * float(interval))
    check("finite-diff order-1 == b (exact on linear series)",
          torch.allclose(st["derivatives"][1], b))

    # schedule cadence
    sched = hicache_init(num_steps=12, interval=4, max_order=1,
                         first_enhance=2, end_enhance=10, sigma=sig)
    types = []
    for s in range(12):
        sched["step"] = s; types.append(hicache_decide(sched))
    check("steps 0,1 full (first_enhance)", types[0] == "full" and types[1] == "full")
    check("forecast then full at interval boundary", types[2] == "forecast" and types[5] == "full")
    check("steps >= end_enhance full", types[10] == "full" and types[11] == "full")

    # constant series -> forecast is exact (all higher diffs vanish)
    stc = hicache_init(num_steps=8, interval=4, max_order=3,
                       first_enhance=0, end_enhance=8, sigma=sig)
    const = torch.tensor([2.0, -1.0, 4.0])
    for idx in (0, 4):
        stc["step"] = idx; stc["activated_steps"].append(idx)
        hicache_update_derivatives(stc, const.clone())
    stc["step"] = 6
    check("constant series -> forecast == constant (exact)",
          torch.allclose(hicache_forecast(stc), const))

    # far-horizon stability (sigma contraction keeps it finite)
    stb = hicache_init(num_steps=64, interval=32, max_order=3,
                       first_enhance=0, end_enhance=64, sigma=0.4)
    f0 = torch.randn(5)
    for idx in (0, 32):
        stb["step"] = idx; stb["activated_steps"].append(idx)
        hicache_update_derivatives(stb, f0 + 0.01 * idx)
    stb["step"] = 60
    check("far-horizon forecast finite", torch.isfinite(hicache_forecast(stb)).all().item())

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    raise SystemExit(0 if ok else 1)
