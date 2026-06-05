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

## When to use this repo

These repos are **complementary accelerators, not competing solutions** — each speeds up a *different*
base generator, and the `+` / `++` suffix is a **method choice**, not a rival product. Pick by
**(1) which base model you run**, then **(2) which forecast basis you want**:

| base generator | `+` = HiCache (Hermite) | `++` = HiCache++ (DMD) |
|---|---|---|
| Hunyuan3D-2.1 | `hunyuan2.1-plus` | `hunyuan2.1-plus-plus` |
| Hunyuan3D-2 mini | `hunyuan2-plus` | `hunyuan2-plus-plus` |
| SAM 3D Objects | `sam3d-plus` | `sam3d-plus-plus` |
| Fast-SAM3D | `fastsam3d-plus` | `fastsam3d-plus-plus` |
| DiT-XL/2 (ImageNet) | `dit-plus` | `dit-plus-plus` |
| TRELLIS (v1) | `faster-trellis` | `faster-trellis-plus-plus` |
| TRELLIS.2-4B (v2) | `hermit-trellis2` | `hermit-trellis2-plus-plus` |

- **`+` (HiCache / scaled-Hermite):** the *published* polynomial velocity-forecast basis — conservative, reproduces the HiCache paper. Use it to deploy the established method.
- **`++` (HiCache++ / DMD exponential):** our Dynamic-Mode-Decomposition basis — *the same near-lossless quality at wider skip intervals*, where the polynomial diverges. Use it when you push the cache interval for more speed.
- **standalone / model-agnostic:** [`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus) — the forecaster itself, to add DMD caching to *your own* diffusion/flow model.
- **`fast-trellis2`** = the TaylorSeer baseline fork (the upstream "Fast" accel) — the v2 reference point, not a HiCache variant.

> **This repo:** `hicache-plus-plus` — the **standalone HiCache++ forecaster** (DMD/Prony exponential velocity cache) + the Hermite baseline — model-agnostic; the per-model integrations are the sibling repos above.

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

> **Headline:** on Hunyuan3D-2.1, as the skip interval grows the polynomial (Hermite) decays
> fast — 0.88 → 0.74 → 0.38 at interval 3 / 5 / 6 — while the exponential (DMD) holds: 0.85 →
> 0.86 → 0.62 (baseline 0.91). **DMD's lead grows with the skip — +0.13 at i5, +0.24 at i6** —
> the exponential basis is what extends the lossless skip range.

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

### Mechanism — controlled, no model

Forecasting `H` steps past an 8-step cached window on synthetic trajectories from the exact
feature-ODE class — three forecast bases, rel. L2 error (↓):

| basis | H=1 | H=2 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 1.5e-2 | 8.0e-2 | 6.2e-1 | 2.3e0 | 6.5e0 |
| Padé / FoCa (rational) | 4.9e-2 | 1.1e-1 | 2.4e-1 | 5.3e-1 | 1.2e0 |
| **HiCache++ (exponential)** | **4.7e-9** | **1.4e-8** | **5.3e-8** | **1.2e-7** | **2.2e-7** |

The exponential basis is **exact** (~1e-8, flat in `H`); the polynomial **diverges**, and the
rational (Padé / FoCa) improves on it but still diverges — 6-to-9 orders of magnitude behind DMD,
and under noise the rational basis turns fragile (Froissart poles). That gap *is* the skip ceiling.
Reproduce: `python benchmarks/forecast_microbench.py`.

### Hunyuan3D-2.1 (flat DiT velocities) — Toys4K F-score@0.05

Excludes `ball_000` (a sphere — Go-ICP alignment is rotationally degenerate on it; two runs
otherwise agree to ±0.01). Speedup is solo / uncontended.

| interval | Hermite (HiCache) | **DMD (HiCache++)** | speedup |
|---:|---:|---:|---:|
| baseline (uncached) | 0.911 | 0.911 | 1.00× |
| i3 | **0.876** | 0.852 | 1.72× |
| i4 | 0.776 | **0.827** | 1.80× |
| **i5** | 0.735 | **0.860** | 1.79× |
| i6 | 0.375 | **0.616** | ~2.0× |

DMD degrades *gracefully* where Hermite collapses, and its lead grows with the interval. On the
**deployed Hunyuan3D-2-mini**, DMD is **exactly lossless at i5** (0.794 = baseline 0.794).

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

### TRELLIS v1 (sparse-structure stage) — Toys4K F-score@0.05, n=31
Swapping *only* the SS forecast basis Hermite→DMD in `faster-trellis` (same carved-hybrid schedule):

| variant | F@0.05 | speedup | vs vanilla |
|---|---:|---:|---:|
| vanilla (uncached) | 0.839 | 1.00× | — |
| HiCache (Hermite) | 0.825 | 2.82× | −0.014 |
| **HiCache++ (DMD)** | **0.829** | **2.76×** | **−0.010** |

At the deployed ~interval-3 (2.8×), DMD is the most lossless accelerator (beats Hermite by +0.005
at matched speed); the margin widens at higher intervals. *(TRELLIS.2 v2 and the DiT-XL/2 ImageNet
FID-vs-latency table are still pending.)*

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
SAM3D and Fast-SAM3D, [`benchmarks/`](benchmarks/) for the controlled forecast microbenchmark,
and [`results/`](results/) for the full tables.

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

## Citation

If you use this library, please cite HiCache++ (this work) and the methods it builds on:

```bibtex
@misc{hicachepp2026,
  title  = {HiCache++: Training-free Diffusion Inference Acceleration via Exponential (DMD/Prony) Velocity Forecasting},
  author = {Attri, Krishi},
  year   = {2026},
  note   = {https://github.com/Archerkattri/hicache-plus-plus}
}

@misc{hicache2025,
  title  = {HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial Feature Forecasting},
  eprint = {2508.16984}, archivePrefix = {arXiv}, primaryClass = {cs.CV}, year = {2025}
}

@misc{taylorseer2025,
  title  = {From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers},
  eprint = {2503.06923}, archivePrefix = {arXiv}, year = {2025}
}

@article{schmid2010dmd,
  title   = {Dynamic mode decomposition of numerical and experimental data},
  author  = {Schmid, Peter J.},
  journal = {Journal of Fluid Mechanics}, volume = {656}, pages = {5--28}, year = {2010}
}

@article{hua1990matrixpencil,
  title   = {Matrix pencil method for estimating parameters of exponentially damped/undamped sinusoids in noise},
  author  = {Hua, Yingbo and Sarkar, Tapan K.},
  journal = {IEEE Transactions on Acoustics, Speech, and Signal Processing},
  volume  = {38}, number = {5}, pages = {814--824}, year = {1990}
}

@misc{adaptiveguidance2023,
  title  = {Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models},
  eprint = {2312.12487}, archivePrefix = {arXiv}, year = {2023}
}
```
