<div align="center">

<img src="https://raw.githubusercontent.com/Archerkattri/hicache-plus-plus/master/assets/banner.png" alt="HiCache++" width="680">

# HiCache++

**A drop-in basis upgrade for TaylorSeer / HiCache: forecast cached diffusion features with a
Dynamic Mode Decomposition (Prony) *exponential* basis instead of a polynomial —
same schedule, same API, near-lossless at wider skip intervals.**

[![PyPI](https://img.shields.io/pypi/v/hicache-pp)](https://pypi.org/project/hicache-pp/)
&nbsp;[![License](https://img.shields.io/badge/license-MIT-2e6db0.svg)](LICENSE)
&nbsp;[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-3776ab.svg)](pyproject.toml)
&nbsp;[![arXiv](https://img.shields.io/badge/arXiv-2508.16984-b31b1b.svg)](https://arxiv.org/abs/2508.16984)

</div>

Feature caches (TaylorSeer, HiCache) skip the network on most denoising steps and *forecast*
the velocity from cached anchors — with a **polynomial** basis. But a diffusion feature
trajectory solves a near-linear feature-ODE whose exact solution class is a sum of damped /
oscillatory **exponentials**; polynomials only locally truncate that class and diverge under
extrapolation, which is exactly why every polynomial cache caps out at a modest skip interval.
**HiCache++** swaps in the exponential basis — Dynamic Mode Decomposition (Prony) — and keeps
quality at skip intervals where the polynomial collapses. One loop, no training, no model edits:

```python
import torch
from hicache_pp import hicache_init, hicache_decide, hicache_update_derivatives
from hicache_pp import dmd_update_snapshots, dmd_forecast_state   # the exponential forecaster

state = hicache_init(num_steps=N, interval=5, first_enhance=4, backend="dmd", history=6)
for i, t in enumerate(timesteps):
    if hicache_decide(state) == "forecast":
        v = dmd_forecast_state(state)            # skip the network — forecast the velocity
    else:
        v = model(x, t, ...)                     # the expensive forward
        hicache_update_derivatives(state, v.detach())
        dmd_update_snapshots(state, v.detach(), state["history"])
    state["step"] += 1
    x = scheduler.step(v, t, x)
```

If you already run TaylorSeer or HiCache, this is a *basis swap*, not a new pipeline: the
compute/skip schedule, warm-up and API stay identical — only the per-skip forecast formula
changes (`backend="dmd"`, or `backend="auto"` to let a holdout test pick the basis per window).

<div align="center">

*DiT-XL/2 ImageNet FID-50k vs latency Pareto plot — in progress
(`benchmarks/dit_imagenet/`); plot lands here.*
<!-- TODO(pareto): <img src="assets/pareto_dit_imagenet.png" width="620"> -->

> **Headline so far:** on Hunyuan3D-2.1, as the skip interval grows the polynomial (Hermite)
> decays fast — 0.88 → 0.74 → 0.38 F-score at interval 3 / 5 / 6 — while the exponential holds:
> 0.85 → 0.86 → 0.62 (baseline 0.91). **The exponential lead grows with the skip — +0.13 at i5,
> +0.24 at i6.**

</div>

> **Name note.** *HiCache* here refers to the diffusion **feature-forecasting** method (Hermite
> polynomial feature caching, [arXiv:2508.16984](https://arxiv.org/abs/2508.16984)), which
> HiCache++ upgrades. It is **unrelated to SGLang / Mooncake's "HiCache"**, a hierarchical
> **KV cache** for LLM serving. Likewise, in this repo "DMD" always abbreviates **Dynamic Mode
> Decomposition (Prony)** — classical spectral estimation — and never Distribution Matching
> Distillation.

---

## TL;DR

On a flow-matching / diffusion denoise loop you can skip the network on most steps and
*forecast* the velocity from cached anchors. The state of the art (TaylorSeer, HiCache)
forecasts with a **polynomial** basis (monomial / scaled-Hermite). But a diffusion feature
trajectory is the solution of a near-linear feature-ODE whose **exact** solution class is a
sum of (damped/oscillatory) **exponentials** — not polynomials. Polynomials diverge under
extrapolation, which is exactly why every polynomial cache caps out at a modest skip.

**HiCache++** forecasts with **Dynamic Mode Decomposition (Prony)** — DMD (Schmid 2010) is the
SVD-regularised generalisation of **Prony's method** (1795): identify the linear propagator
`A` from raw velocity snapshots (`F_{t+1} ≈ A F_t`), eigendecompose it once, and predict any
(fractional) horizon `k` by eigenvalue powers:

```
F_{t+k} ≈ Φ (λ**k ⊙ b),     b = Φ⁺ F_t
```

It is **exact on exponential trajectories** (the solution class) — the property polynomials
lack — so it holds quality at skip intervals where Hermite/Taylor drift.

---

## How it compares

Every modern feature cache skips the network on most steps and *forecasts* the velocity;
they differ in the **basis** used to extrapolate. The basis is what sets the skip ceiling,
because a diffusion feature trajectory is (locally) a sum of exponentials, not a polynomial.
HiCache++'s basis is the Dynamic Mode Decomposition (Prony) exponential:

| Method | Forecast basis | Exact on the feature-ODE class | Extrapolation | Max lossless skip\* |
|---|---|:--:|:--:|:--:|
| TaylorSeer | monomial (Taylor) | ✗ | diverges | small |
| **HiCache** | scaled-Hermite | ✗ | drifts | interval&#8209;3 |
| FoCa · Padé · Chebyshev | rational / orthogonal poly | ✗ | drifts | small–moderate |
| **HiCache++** _(this work)_ | **exponential (DMD / Prony)** | **✓ exact** | **bounded, correct asymptotics** | **interval&#8209;5–6** |

<sub>\*measured on Hunyuan3D-2.1 / SAM3D-slat (see Results). A polynomial basis is only a
local truncation of the exponential, so it is accurate for a tiny skip and diverges as the
horizon grows; the exponential basis *is* the exact solution class, so it stays lossless
further out — and the exponential forecaster admits *fractional* horizons, so it forecasts
sub-steps between compute steps exactly.</sub>

---

## Why exponentials (the math)

A diffusion/flow-matching sampler integrates `dx/dt = v_θ(x, t)`. Across timesteps the
cached feature `F_t` (the CFG-combined velocity) evolves under a slowly-varying, near-linear
operator. The exact solution of a linear ODE `Ḟ = M F` is `F_t = Σ_j a_j e^{μ_j t}` — a sum
of exponentials with poles `μ_j` (damped if `Re μ_j < 0`, oscillatory if `Im μ_j ≠ 0`).

- **Polynomial basis** (Taylor monomials, Hermite): a *local* Taylor truncation of that
  exponential. Accurate for a tiny skip, **diverges** as the horizon grows → modest skip cap.
- **Exponential basis** — Dynamic Mode Decomposition (Prony): the *exact* function class. Fit
  the poles `λ_j = e^{μ_j Δ}` from snapshots and extrapolate with bounded, correct asymptotics.

**The ≥4-snapshot floor.** A *real-valued* trajectory spends **two** real degrees of freedom
on every **complex** pole (a conjugate pair `r e^{±iω}` → `r^t cos ωt, r^t sin ωt`). So even a
single oscillatory mode needs rank 3 to identify, i.e. **3 snapshot-pairs = 4 snapshots**. With
only 2 pairs the fit aliases (empirically ~2e-1 error vs ~5e-9 at 3 pairs). Below the floor (or
across a non-uniform window) HiCache++ falls back to the Hermite forecast for warm-up.

---

## Results (A/B, geometry-preserving)

All accelerators are *training-free and geometry-preserving*; the right A/B is **how far the
output drifts from the uncached/baseline geometry vs how much faster it runs**. "DMD" below
is the HiCache++ Dynamic Mode Decomposition (Prony) exponential basis.

### Mechanism — controlled, no model

Forecasting `H` steps past an 8-step cached window on synthetic trajectories from the exact
feature-ODE class — three forecast bases, rel. L2 error (↓):

| basis | H=1 | H=2 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 1.5e-2 | 8.0e-2 | 6.2e-1 | 2.3e0 | 6.5e0 |
| Padé / FoCa (rational) | 4.9e-2 | 1.1e-1 | 2.4e-1 | 5.3e-1 | 1.2e0 |
| **HiCache++ (exponential)** | **4.7e-9** | **1.4e-8** | **5.3e-8** | **1.2e-7** | **2.2e-7** |

The exponential basis is **exact** (~1e-8, flat in `H`); the polynomial **diverges**, and the
rational (Padé / FoCa) improves on it but still diverges — 6-to-9 orders of magnitude behind the
exponential, and under noise the rational basis turns fragile (Froissart poles). That gap *is*
the skip ceiling. And when the dynamics are NOT clean — an abrupt regime switch inside the
cached window, where a whole-window exponential fit misfits — the holdout-selected
`backend="auto"` catches it every time (it backcasts the newest snapshot with both bases and
serves the winner): on the switch stress it picks the safe fallback in 120/120 windows and cuts
the long-horizon error ~3x vs a forced exponential fit (H=8: 3.1 vs 9.4 rel. error, with the
polynomial at 106), while on clean/drifting/noisy trajectories it picks the exponential basis
120/120 and matches it exactly. Full five-scenario tables:
[`benchmarks/MICROBENCH_RESULTS.md`](benchmarks/MICROBENCH_RESULTS.md). Reproduce:
`python benchmarks/forecast_microbench.py`.

### DiT-XL/2 ImageNet — FID-50k / IS vs latency

*In progress* — the class-conditional ImageNet-256 sweep (FID-50k + Inception Score across
intervals, Hermite vs exponential) is running in [`benchmarks/dit_imagenet/`](benchmarks/dit_imagenet/);
the table and Pareto plot land here.

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

The exponential basis degrades *gracefully* where Hermite collapses, and its lead grows with the
interval. On the **deployed Hunyuan3D-2-mini**, it is **exactly lossless at i5** (0.794 =
baseline 0.794).

### SAM3D (PyTree velocities, slat FlowMatching) — real weights, F1 vs baseline

| config | speedup | CD_vs_base | F1_vs_base |
|---|---:|---:|---:|
| vanilla | 1.00× | 0.000 | **1.000** |
| HiCache i3 | 1.44× | 0.013 | **1.000** |
| DMD i5 | 1.47× | 0.013 | **1.000** |
| **DMD i6** | **1.56×** | 0.013 | **1.000** |

Both are geometry-lossless (F1=1.000); **the exponential basis stays lossless to interval-6**,
where it gives the best speedup — past Hermite's lossless i3.

### TRELLIS v1 (sparse-structure stage) — Toys4K F-score@0.05, n=31

Swapping *only* the SS forecast basis Hermite→exponential in `faster-trellis` (same
carved-hybrid schedule):

| variant | F@0.05 | speedup | vs vanilla |
|---|---:|---:|---:|
| vanilla (uncached) | 0.839 | 1.00× | — |
| HiCache (Hermite) | 0.825 | 2.82× | −0.014 |
| **HiCache++ (DMD)** | **0.829** | **2.76×** | **−0.010** |

At the deployed ~interval-3 (2.8×), the exponential basis is the most lossless accelerator
(beats Hermite by +0.005 at matched speed); the margin widens at higher intervals. The same
holds on **TRELLIS.2-4B (v2)** — it ties Hermite at the deployed interval and pulls
**+0.03–0.04 F-score ahead at intervals 3–4**
(see [`hermit-trellis2-plus-plus`](https://github.com/Archerkattri/hermit-trellis2-plus-plus#results)).

Full tables: [`results/RESULTS.md`](results/RESULTS.md).

---

## Install / use

```bash
pip install hicache-pp
```

The one-loop snippet at the top is the whole integration for **flat tensor** velocities
(e.g. a DiT). For **PyTree / structured** velocities (e.g. SAM3D), use `hicache_pp.tree` —
the same API but tree-aware (`hicache_forecast_tree`, `dmd_forecast_tree`, plus tree
Adaptive-CFG). Backends:

- `backend="hermite"` — the published HiCache scaled-Hermite polynomial (clean reimplementation).
- `backend="dmd"` — the HiCache++ Dynamic Mode Decomposition (Prony) exponential basis.
- `backend="auto"` — holdout selection: per compute step, backcast the newest held-out
  snapshot with both bases and serve whichever demonstrably wins on the data at hand.

See [`integrations/`](integrations/) for the exact wiring into Hunyuan3D-2.1,
Hunyuan3D-2-mini, SAM3D and Fast-SAM3D, and
[`integrations/pr_drafts/`](integrations/pr_drafts/) for prepared patches that add this
exponential basis to **cache-dit**, **Hugging Face diffusers** (`TaylorSeerCacheConfig`)
and **Cache4Diffusion** in each project's native conventions.

### Tuning notes

- **Hermite**: lossless up to a modest interval (Hunyuan-2.1: i3/order-2). Higher order does
  *not* rescue bigger intervals — the polynomial ceiling.
- **Exponential**: push the interval further (i5–i6) for more skip while staying lossless.
  `history` is the snapshot window (5–6); needs ≥4 *uniformly-spaced* snapshots before it
  engages (Hermite covers warm-up automatically).
- `first_enhance` always computes the first few steps (high curvature); keep it ≥ 3.

---

## Tests

```bash
python -m hicache_pp.hermite     # Hermite basis + schedule (CPU, no GPU/model)
python -m hicache_pp.dmd         # exponential basis exact-on-exponential + ≥4-snapshot floor
python -m hicache_pp.tree        # tree-aware Hermite + exponential + Adaptive-CFG
python tests/run_tests.py        # all of the above
```

---

## 3D generator integrations (sibling repos)

The forecaster in this repo is model-agnostic; it has also been wired natively into a family
of 3D-generator forks. These are **complementary accelerators, not competing solutions** —
each speeds up a *different* base generator, and the `+` / `++` suffix is a **method choice**
(`+` = HiCache Hermite polynomial, `++` = HiCache++ Dynamic Mode Decomposition (Prony)
exponential), not a rival product. Pick by **(1) which base model you run**, then **(2) which
forecast basis you want**:

| base generator | `+` = HiCache (Hermite) | `++` = HiCache++ (DMD) |
|---|---|---|
| Hunyuan3D-2.1 | `hunyuan2.1-plus` | `hunyuan2.1-plus-plus` |
| Hunyuan3D-2 mini | `hunyuan2-plus` | `hunyuan2-plus-plus` |
| SAM 3D Objects | `sam3d-plus` | `sam3d-plus-plus` |
| Fast-SAM3D | `fastsam3d-plus` | `fastsam3d-plus-plus` |
| DiT-XL/2 (ImageNet) | `dit-plus` | `dit-plus-plus` |
| TRELLIS (v1) | `faster-trellis` | `faster-trellis-plus-plus` |
| TRELLIS.2-4B (v2) | `hermit-trellis2` | `hermit-trellis2-plus-plus` |

- **`+` (HiCache / scaled-Hermite):** the *published* polynomial velocity-forecast basis —
  conservative, reproduces the HiCache paper. Use it to deploy the established method.
- **`++` (HiCache++ / exponential):** our Dynamic Mode Decomposition (Prony) basis — *the same
  near-lossless quality at wider skip intervals*, where the polynomial diverges. Use it when
  you push the cache interval for more speed.
- **standalone / model-agnostic:** [`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus)
  (this repo) — the forecaster itself, to add exponential caching to *your own* diffusion/flow model.
- **`fast-trellis2`** = the TaylorSeer baseline fork (the upstream "Fast" accel) — the v2
  reference point, not a HiCache variant.

---

## Lineage & attribution

- **TaylorSeer** — feature caching with a monomial (Taylor) basis.
- **HiCache** (arXiv:2508.16984) — the scaled-Hermite polynomial upgrade. `hicache_pp.hermite`
  is a clean reimplementation.
- **HiCache++ (this work)** — the **Dynamic Mode Decomposition (Prony) exponential** forecaster
  (`hicache_pp.dmd`). DMD (Schmid 2010) / Prony (1795) / Matrix-Pencil (Hua–Sarkar 1990) are
  classical spectral estimation; their application to **diffusion feature caching** is, to our
  knowledge, new.
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
