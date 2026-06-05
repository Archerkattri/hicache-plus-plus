<div align="center">

# HiCache++

**Training-free diffusion inference acceleration by *exponential* velocity forecasting.**

*A drop-in upgrade to TaylorSeer / HiCache: replace the polynomial feature-cache basis with
a Dynamic-Mode-Decomposition (Prony) **exponential** basis — exact on the class diffusion
features actually live in, so it stays lossless at larger skip intervals than the polynomial.*

![training&#8209;free](https://img.shields.io/badge/training--free-%E2%9C%93-2e8f5c)
&nbsp;![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
&nbsp;![CPU tests](https://img.shields.io/badge/CPU%20tests-passing-2e8f5c)
&nbsp;![license MIT](https://img.shields.io/badge/license-MIT-2e6db0)
&nbsp;![python](https://img.shields.io/badge/python-%E2%89%A53.9-3776ab)

</div>

---

## TL;DR

On a flow-matching / diffusion denoise loop you can skip the network on most steps and
*forecast* the velocity from cached anchors. The state of the art (TaylorSeer, HiCache)
forecasts with a **polynomial** basis (monomial / scaled-Hermite). But a diffusion feature
trajectory is the solution of a near-linear feature-ODE whose **exact** solution class is a
sum of (damped/oscillatory) **exponentials** — not polynomials. Polynomials diverge under
extrapolation, which is exactly why every polynomial cache caps out at a modest skip.

**HiCache++** forecasts with **Dynamic Mode Decomposition** (Schmid 2010) — the
SVD-regularised generalisation of **Prony's method** (1795): identify the linear propagator
`A` from raw velocity snapshots (`F_{t+1} ≈ A F_t`), eigendecompose it once, and predict any
(fractional) horizon `k` by eigenvalue powers:

```
F_{t+k} ≈ Φ (λ**k ⊙ b),     b = Φ⁺ F_t
```

It is **exact on exponential trajectories** (the solution class) — the property polynomials
lack — so it holds quality at skip intervals where Hermite/Taylor drift.

> **Headline:** on Hunyuan3D-2.1, HiCache (Hermite) is lossless only up to interval-3
> (1.81×); at interval-5 it collapses (F-score 0.68). **HiCache++ (DMD) is near-lossless at
> interval-5 (0.86 ≈ the 0.87 uncached baseline)** — it breaks the polynomial ceiling.

---

## How it compares

Every modern feature cache skips the network on most steps and *forecasts* the velocity;
they differ in the **basis** used to extrapolate. The basis is what sets the skip ceiling,
because a diffusion feature trajectory is (locally) a sum of exponentials, not a polynomial:

| Method | Forecast basis | Exact on the feature-ODE class | Extrapolation | Max lossless skip\* |
|---|---|:--:|:--:|:--:|
| TaylorSeer | monomial (Taylor) | ✗ | diverges | small |
| **HiCache** | scaled-Hermite | ✗ | drifts | interval&#8209;3 |
| FoCa · Padé · Chebyshev | rational / orthogonal poly | ✗ | drifts | small–moderate |
| **HiCache++** _(this work)_ | **exponential (DMD / Prony)** | **✓ exact** | **bounded, correct asymptotics** | **interval&#8209;5–6** |

<sub>\*measured on Hunyuan3D-2.1 / SAM3D-slat (see Results). A polynomial basis is only a
local truncation of the exponential, so it is accurate for a tiny skip and diverges as the
horizon grows; the exponential basis *is* the exact solution class, so it stays lossless
further out — and DMD admits *fractional* horizons, so it forecasts sub-steps between
compute steps exactly.</sub>

---

## Why exponentials (the math)

A diffusion/flow-matching sampler integrates `dx/dt = v_θ(x, t)`. Across timesteps the
cached feature `F_t` (the CFG-combined velocity) evolves under a slowly-varying, near-linear
operator. The exact solution of a linear ODE `Ḟ = M F` is `F_t = Σ_j a_j e^{μ_j t}` — a sum
of exponentials with poles `μ_j` (damped if `Re μ_j < 0`, oscillatory if `Im μ_j ≠ 0`).

- **Polynomial basis** (Taylor monomials, Hermite): a *local* Taylor truncation of that
  exponential. Accurate for a tiny skip, **diverges** as the horizon grows → modest skip cap.
- **Exponential basis** (DMD / Prony): the *exact* function class. Fit the poles `λ_j = e^{μ_j Δ}`
  from snapshots and extrapolate with bounded, correct asymptotics.

**The ≥4-snapshot floor.** A *real-valued* trajectory spends **two** real degrees of freedom
on every **complex** pole (a conjugate pair `r e^{±iω}` → `r^t cos ωt, r^t sin ωt`). So even a
single oscillatory mode needs rank 3 to identify, i.e. **3 snapshot-pairs = 4 snapshots**. With
only 2 pairs the fit aliases (empirically ~2e-1 error vs ~5e-9 at 3 pairs). Below the floor (or
across a non-uniform window) HiCache++ falls back to the Hermite forecast for warm-up.

---

## Results (A/B, geometry-preserving)

All accelerators are *training-free and geometry-preserving*; the right A/B is **how far the
output drifts from the uncached/baseline geometry vs how much faster it runs**.

### Hunyuan3D (flat DiT velocities) — Toys4K F-score@0.05, n=10

| interval | Hermite (HiCache) | **DMD (HiCache++)** |
|---:|---:|---:|
| baseline (uncached) | 0.867 | 0.867 |
| i3 | **0.888** (1.81× lossless) | 0.867 |
| i4 | 0.773 | **0.807** |
| **i5** | 0.683 | **0.864**  ← near-lossless |
| i6 | 0.292 | 0.732 |

On the **deployed Hunyuan3D-2-mini**, DMD is **exactly lossless at i5** (0.794 = baseline 0.794).

### SAM3D (PyTree velocities, slat FlowMatching) — real weights, F1 vs baseline

| config | speedup | CD_vs_base | F1_vs_base |
|---|---:|---:|---:|
| vanilla | 1.00× | 0.000 | **1.000** |
| HiCache i3 | 1.44× | 0.013 | **1.000** |
| DMD i5 | 1.47× | 0.013 | **1.000** |
| **DMD i6** | **1.56×** | 0.013 | **1.000** |

Both are geometry-lossless (F1=1.000); **DMD stays lossless to interval-6**, where it gives the
best speedup — past Hermite's lossless i3.

### Fast-SAM3D (SS-stage TaylorSeer)
Hermite ≈ Taylor (a wash): both run the same stride-3 schedule, so the basis swap doesn't
change latency — TaylorSeer caching (the default) is what gives the ~3×, not the basis.

---

## Install / use

```python
import torch
from hicache_pp import hicache_init, hicache_decide, hicache_update_derivatives, hicache_forecast
from hicache_pp import dmd_update_snapshots, dmd_forecast_state   # the exponential forecaster

# in your denoise loop (flat tensor velocities):
state = hicache_init(num_steps=N, interval=5, first_enhance=4, backend="dmd", history=6)
for i, t in enumerate(timesteps):
    if hicache_decide(state) == "forecast":
        v = dmd_forecast_state(state)            # skip the network — forecast the velocity
        state["step"] += 1
    else:
        v = model(x, t, ...)                     # the expensive forward
        hicache_update_derivatives(state, v.detach())
        dmd_update_snapshots(state, v.detach(), state["history"])
        state["step"] += 1
    x = scheduler.step(v, t, x)
```

For **PyTree / structured** velocities (e.g. SAM3D), use `hicache_pp.tree` — the same API but
tree-aware (`hicache_forecast_tree`, `dmd_forecast_tree`, plus tree Adaptive-CFG).

See [`integrations/`](integrations/) for the exact wiring into Hunyuan3D-2.1, Hunyuan3D-2-mini,
SAM3D and Fast-SAM3D, and [`results/`](results/) for the full tables.

---

## Tests

```bash
python -m hicache_pp.hermite     # Hermite basis + schedule (CPU, no GPU/model)
python -m hicache_pp.dmd         # DMD exact-on-exponential + ≥4-snapshot floor
python -m hicache_pp.tree        # tree-aware Hermite + DMD + Adaptive-CFG
python tests/run_tests.py        # all of the above
```

---

## Lineage & attribution

- **TaylorSeer** — feature caching with a monomial (Taylor) basis.
- **HiCache** (arXiv:2508.16984) — the scaled-Hermite polynomial upgrade. `hicache_pp.hermite`
  is a clean reimplementation.
- **HiCache++ (this work)** — the **DMD/Prony exponential** forecaster (`hicache_pp.dmd`). DMD
  (Schmid 2010) / Prony (1795) / Matrix-Pencil (Hua–Sarkar 1990) are classical spectral
  estimation; their application to **diffusion feature caching** is, to our knowledge, new.
- **Adaptive-CFG** (Adaptive Guidance, arXiv:2312.12487) — composable uncond-skip, included in
  the tree module.
