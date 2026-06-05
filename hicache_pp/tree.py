# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Training-free inference acceleration for SAM 3D flow-matching.

Two methods, ported from the TRELLIS / Hunyuan-2.1 acceleration work and
generalised to SAM3D's **PyTree velocities** (the solver's ``x_t`` and the
backbone velocity are ``torch.utils._pytree`` structures, not single tensors):

  * **HiCache** — the (CFG-combined) velocity at *skipped* solver steps is
    forecast with a dual-scaled physicist's Hermite polynomial instead of calling
    the dynamics function, skipping ``(interval-1)/interval`` of the model
    evaluations. Hooked at the ODE-solver level (it wraps ``dynamics_fn``).

  * **Adaptive-CFG** (Adaptive Guidance, arXiv:2312.12487) — once the conditional
    and unconditional velocities align (cosine >= ``gamma_bar``) the unconditional
    backbone pass is dropped and the guidance term is reconstructed from cached
    anchors. Hooked inside ``ClassifierFreeGuidance.inner_forward``. SAM3D's CFG is
    ``(1+w)*y_cond - w*y_uncond`` (the TRELLIS convention), so the guidance term is
    ``g = w*(y_cond - y_uncond)`` and ``v_cfg = y_cond + g``.

Everything here is model-agnostic except that it operates on PyTrees via
``_pytree.tree_map``. The solver / CFG modules call these helpers directly — there
is NO runtime monkey-patching. The Hermite/finite-difference scalar coefficients
are identical across all leaves, so a forecast is one ``tree_map`` per order.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils import _pytree


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
    """Forecast coefficient ``Htilde_order(-k) / order!`` (a scalar, same for all
    leaves). ``Htilde_n(x) = sigma^n H_n(sigma x)``."""
    x = -float(k)
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
    if backend not in ("hermite", "dmd"):
        raise ValueError(f"backend must be 'hermite' or 'dmd', got {backend!r}")
    return {
        "num_steps": int(num_steps), "interval": int(interval), "max_order": int(max_order),
        "first_enhance": int(first_enhance),
        "end_enhance": int(end_enhance if end_enhance is not None else num_steps),
        "sigma": float(sigma), "backend": str(backend), "history": int(history),
        "step": 0, "counter": 0,
        "activated_steps": [], "derivatives": {}, "prev_derivatives": {},
        "dmd_snapshots": [],   # [(compute_step, velocity_tree), ...] for the DMD backend
    }


def hicache_decide(state: Dict[str, Any]) -> str:
    step = state["step"]
    if step < state["first_enhance"] or step >= state["end_enhance"] \
            or state["counter"] >= state["interval"] - 1:
        state["counter"] = 0
        state["activated_steps"].append(step)
        return "full"
    state["counter"] += 1
    return "forecast"


def hicache_update_tree(state: Dict[str, Any], velocity_tree: Any) -> None:
    prev = state["derivatives"]
    new_deriv = {0: velocity_tree}
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
def _dmd_forecast_flat(snapshots, k, rank=0, ridge=1e-8):
    """DMD forecast of a flat vector ``k`` (fractional) steps past the newest of a
    list of >=4 equal-length 1-D snapshot vectors. Falls back to last-value reuse."""
    if len(snapshots) < 4:
        return snapshots[-1].clone()
    dt = snapshots[-1].dtype
    V = torch.stack(snapshots, dim=1).to(torch.float64)          # [d, n]
    X, Xp = V[:, :-1], V[:, 1:]
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except Exception:  # noqa: BLE001
        return snapshots[-1].clone()
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
    except Exception:  # noqa: BLE001
        return snapshots[-1].clone()
    pred = (Phi @ (evals.pow(float(k)) * b)).real
    if not torch.isfinite(pred).all():
        return snapshots[-1].clone()
    return pred.to(dt)


def dmd_update_snapshots_tree(state: Dict[str, Any], velocity_tree: Any, history: int = 5) -> None:
    """Record the (CFG-combined) velocity TREE at a compute step (short local sliding
    window — the diffusion dynamics are non-autonomous, so a long window averages over
    changing dynamics)."""
    snaps = state.setdefault("dmd_snapshots", [])
    snaps.append((int(state["activated_steps"][-1]), velocity_tree))
    h = int(state.get("history", history))
    if len(snaps) > h:
        del snaps[: len(snaps) - h]


def dmd_forecast_tree(state: Dict[str, Any]) -> Any:
    """DMD forecast of the velocity tree at the current skip step. Flattens the longest
    uniformly-spaced tail (>=4 snapshots) to vectors, runs DMD with a fractional horizon
    ``lambda**(k/spacing)``, and unflattens to the tree. Floor is 4 snapshots (3 pairs):
    a real trajectory spends two real DOF per COMPLEX pole, so even one oscillatory mode
    needs rank 3. Below that / on a non-uniform window, falls back to Hermite (warm-up)."""
    snaps = state.get("dmd_snapshots", [])
    if len(snaps) >= 4:
        steps = [s for s, _ in snaps]
        spacing = steps[-1] - steps[-2]
        if spacing > 0:
            tail = [snaps[-1], snaps[-2]]
            j = len(snaps) - 2
            while j - 1 >= 0 and steps[j] - steps[j - 1] == spacing:
                tail.append(snaps[j - 1]); j -= 1
            if len(tail) >= 4:
                trees = [v for _, v in reversed(tail)]                # oldest..newest
                flat = [_pytree.tree_flatten(t) for t in trees]       # [(leaves, spec), ...]
                leaves_new, spec = flat[-1]
                shapes = [l.shape for l in leaves_new]
                vecs = [torch.cat([l.reshape(-1) for l in lv]) for lv, _ in flat]
                kf = (state["step"] - steps[-1]) / spacing
                pred = _dmd_forecast_flat(vecs, kf)
                out, i = [], 0
                for sh in shapes:
                    n = 1
                    for d in sh:
                        n *= int(d)
                    out.append(pred[i:i + n].reshape(sh)); i += n
                return _pytree.tree_unflatten(out, spec)
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
    """SAM3D guidance term ``g = strength * (y_cond - y_uncond)`` (leafwise)."""
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
# CPU unit test (no GPU, no SAM3D model): trees are plain dict-of-tensors      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        global ok
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    def tree(a, b):
        return {"u": a, "nested": {"v": b}}

    def tclose(t1, t2, atol=1e-4):
        return all(torch.allclose(l1, l2, atol=atol)
                   for l1, l2 in zip(_pytree.tree_leaves(t1), _pytree.tree_leaves(t2)))

    # 1) Hermite coeff scalars match H_2 etc.
    check("H_2(0.5) == 4*0.25-2 == -1", abs(physicists_hermite(2, 0.5) - (-1.0)) < 1e-9)

    # 2) HiCache forecast EXACT on a constant velocity tree (all higher diffs vanish).
    st = hicache_init(num_steps=8, interval=4, max_order=2, first_enhance=0, end_enhance=8, sigma=0.5)
    const = tree(torch.tensor([2.0, -1.0]), torch.tensor([[3.0, 0.0]]))
    for idx in (0, 4):
        st["step"] = idx; st["activated_steps"].append(idx)
        hicache_update_tree(st, const)
    st["step"] = 6
    check("HiCache tree forecast == constant (exact)", tclose(hicache_forecast_tree(st), const))

    # 3) HiCache forecast applies the Hermite coeff leafwise. Linear series F(s)=A+sB,
    #    anchors at 0,4 -> Delta^1 = B; forecast at step 6 (k=2) = F4 + coeff*Delta^1.
    #    (Hermite is NOT Taylor, so this is NOT A+6B — the check is the tree plumbing.)
    A = tree(torch.randn(3), torch.randn(2, 2)); B = tree(torch.randn(3), torch.randn(2, 2))
    lin = lambda s: tree_axpy(float(s), A, B)            # A + s*B
    st2 = hicache_init(num_steps=12, interval=4, max_order=1, first_enhance=0, end_enhance=12, sigma=0.5)
    for idx in (0, 4):
        st2["step"] = idx; st2["activated_steps"].append(idx)
        hicache_update_tree(st2, lin(idx))
    st2["step"] = 6
    expected = tree_axpy(hermite_coeff(1, 2, 0.5), lin(4), B)   # F4 + coeff * (Delta^1 == B)
    check("HiCache tree forecast matches explicit Hermite formula leafwise",
          tclose(hicache_forecast_tree(st2), expected, atol=1e-4))

    # 4) Adaptive-CFG: guidance term + reconstruction match full CFG (SAM3D conv).
    w = 3.0
    yc = tree(torch.randn(4), torch.randn(2, 3)); yu = tree(torch.randn(4), torch.randn(2, 3))
    g = guidance_term_tree(yc, yu, w)
    v_cfg = reconstruct_cfg_tree(yc, g)
    v_true = _pytree.tree_map(lambda c, u: (1 + w) * c - w * u, yc, yu)   # (1+w)cond - w*uncond
    check("Adaptive-CFG reconstruction == (1+w)cond - w*uncond", tclose(v_cfg, v_true))

    # 5) guidance-term forecast EXACT on a linear-in-step guidance tree.
    gA = tree(torch.randn(4), torch.randn(2, 3)); gB = tree(torch.randn(4), torch.randn(2, 3))
    gfun = lambda s: tree_axpy(float(s), gA, gB)
    anchors = [(2, gfun(2)), (4, gfun(4))]               # non-uniform-friendly
    check("guidance tree forecast exact (linear)", tclose(forecast_guidance_tree(anchors, 7, 1), gfun(7), atol=1e-3))

    # 6) decisions + cosine
    check("cosine self == 1", abs(tree_cosine(yc, yc) - 1.0) < 1e-5)
    sa = adaptive_cfg_init(num_steps=10, gamma_bar=0.9, warmup=2)
    sa["step"] = 0; check("warmup full", adaptive_cfg_decide(sa, 0.99) is True)
    sa["step"] = 3; sa["anchors"].append((2, g))
    check("aligned -> skip", adaptive_cfg_decide(sa, 0.95) is False)
    check("misaligned -> full", adaptive_cfg_decide(sa, 0.8) is True)

    # 7) HiCache schedule cadence
    sc2 = hicache_init(num_steps=12, interval=4, max_order=1, first_enhance=2, end_enhance=10, sigma=0.5)
    seq = []
    for s in range(12):
        sc2["step"] = s; seq.append(hicache_decide(sc2))
    check("cadence: 0,1 full; 2 forecast; 5 full; 10,11 full",
          seq[0] == "full" and seq[1] == "full" and seq[2] == "forecast" and seq[5] == "full"
          and seq[10] == "full" and seq[11] == "full")

    # 8) DMD tree forecast EXACT on an exponential tree trajectory (the solution class
    #    where the Hermite polynomial drifts). Snapshots spaced 3 apart; the fractional
    #    horizon k=(11-10)/3 takes the principal 1/3-power of the 3-step poles back to z.
    z = torch.tensor([0.92 * torch.exp(torch.tensor(0.35j)), torch.tensor(0.70 + 0j)], dtype=torch.complex128)
    Au = torch.randn(6, 2, dtype=torch.complex128); Av = torch.randn(4, 2, dtype=torch.complex128)
    def vtree(s):
        return tree((Au @ (z ** s)).real.to(torch.float64),
                    (Av @ (z ** s)).real.to(torch.float64).reshape(2, 2))
    st_dmd = {"step": 11, "history": 5,
              "dmd_snapshots": [(1, vtree(1)), (4, vtree(4)), (7, vtree(7)), (10, vtree(10))]}
    rel = max((l1 - l2).norm() / (l2.norm() + 1e-12)
              for l1, l2 in zip(_pytree.tree_leaves(dmd_forecast_tree(st_dmd)),
                                _pytree.tree_leaves(vtree(11))))
    check(f"DMD tree forecast exact on exponential traj (rel {rel:.2e} < 1e-4)", rel < 1e-4)

    # 9) DMD below the 4-snapshot floor -> Hermite fallback (== constant here, no crash).
    st_fb = hicache_init(num_steps=8, interval=3, max_order=1, first_enhance=0, end_enhance=8,
                         sigma=0.5, backend="dmd")
    cst = tree(torch.tensor([2.0, -1.0]), torch.tensor([[3.0, 0.0]]))
    for idx in (0, 3):
        st_fb["step"] = idx; st_fb["activated_steps"].append(idx)
        hicache_update_tree(st_fb, cst); dmd_update_snapshots_tree(st_fb, cst, 5)
    st_fb["step"] = 5
    check("DMD < 4 snapshots -> Hermite fallback == constant", tclose(dmd_forecast_tree(st_fb), cst))

    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    import sys
    sys.exit(0 if ok else 1)
