<div align="center">

<img src="https://raw.githubusercontent.com/Archerkattri/hicache-plus-plus/master/assets/banner.png" alt="HiCache++" width="680">

# HiCache++

**No single forecast basis wins across diffusion families. HiCache++ ships the exponential
(Dynamic Mode Decomposition / Prony) basis that wins on flow-matching 3D generators, a
corrected (sign-fixed) Hermite polynomial that dominates on DiT-class denoising, and a
training-free holdout selector (`backend="auto"`) that solves regime switches but, per our
own pre-registered DiT A/B, does NOT solve the domain split. Pick the basis by model
family; the numbers for both families are below.**

[![PyPI](https://img.shields.io/pypi/v/hicache-pp)](https://pypi.org/project/hicache-pp/)
[![DOI](https://zenodo.org/badge/1260475033.svg)](https://zenodo.org/badge/latestdoi/1260475033)
&nbsp;[![Paper](https://img.shields.io/badge/engrXiv-10.31224%2F7309-009E73.svg)](https://doi.org/10.31224/7309)
&nbsp;[![License](https://img.shields.io/badge/license-MIT-2e6db0.svg)](LICENSE)
&nbsp;[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-3776ab.svg)](pyproject.toml)
&nbsp;[![arXiv](https://img.shields.io/badge/arXiv-2508.16984-b31b1b.svg)](https://arxiv.org/abs/2508.16984)

</div>

Feature caches (TaylorSeer, HiCache) skip the network on most denoising steps and
*forecast* the cached features from recent compute-step anchors. The literature treats the
forecast basis as a ladder to climb: monomial, Hermite, rational, Chebyshev, each pitched
as a better basis. We built the natural endpoint of that ladder, the exponential solution
class of the local feature-ODE fitted by Dynamic Mode Decomposition (Prony), benchmarked
it across two diffusion families, and found the honest answer:

## The domain split (read this first)

| workload | winning basis | evidence |
|---|---|---|
| **Flow-matching 3D generators** (Hunyuan3D-2.1 / 2-mini, SAM3D, TRELLIS v1/v2) | **exponential (DMD)** | +0.13 / +0.24 F-score over the deployed Hermite arm at intervals 5 / 6 on Hunyuan3D-2.1; exactly lossless at i5 on 2-mini; geometry-lossless (F1 = 1.000) through i6 at 1.56x on SAM3D |
| **DiT-class denoising** (DiT-XL/2, ImageNet-256, 250-step DDPM) | **polynomial** | corrected TaylorSeer is near-lossless (paired-noise FID drift 2.27 at 3.81x) and the corrected Hermite sweeps the interval ladder (3.54 / 6.46 / 10.74 at i4 / i6 / i8). The exponential basis drifts 5-9x more than the corrected Hermite at every interval tested |

Neither basis transfers, and on the Pareto front each family is owned outright: corrected
TaylorSeer / Hermite dominate DiT at every interval, DMD dominates the flow-matching 3D
family at every interval past i3. Any single-basis claim you have read (including our own
earlier framing) is family-conditional.

**The selection verdict, stated plainly.** We built a training-free selector
(`backend="auto"`) that backcasts a held-out snapshot with both bases per compute window
and serves the winner, in two modes (1-step and horizon-matched backcast). It is exact on
synthetic regime switches (120/120), harmless on the 3D generators, and **wrong on DiT in
both modes**: auto reads FID drift 18.11 in the pre-registered A/B (both modes,
identical) where the corrected Hermite sits at 3.54. The reason is characterized below:
on DiT's real features DMD's richer parameterization fits and backcasts the snapshot
history better while extrapolating forward worse, so in-window backcast fit carries no
signal about forward quality. Therefore:

- **Pick the basis by model family** (the table above). That is the recommendation.
- Use `backend="auto"` only as a safety net for intra-run regime switches, which it
  demonstrably solves; do not expect it to discover the family-level winner.
- Per-window basis selection from real feature trajectories is an **open problem**.

## Quickstart

```python
import torch
from hicache_pp import hicache_init, hicache_decide, hicache_update_derivatives
from hicache_pp import dmd_update_snapshots, dmd_forecast_state

# backend: pick by model family -- "dmd" for flow-matching 3D generators,
# "hermite" for DiT-class denoising, "auto" only as a regime-switch safety net.
state = hicache_init(num_steps=N, interval=5, first_enhance=4, backend="dmd", history=6)
for i, t in enumerate(timesteps):
    if hicache_decide(state) == "forecast":
        v = dmd_forecast_state(state)            # skip the network, forecast the velocity
    else:
        v = model(x, t, ...)                     # the expensive forward
        hicache_update_derivatives(state, v.detach())
        dmd_update_snapshots(state, v.detach(), state["history"])
    state["step"] += 1
    x = scheduler.step(v, t, x)
```

If you already run TaylorSeer or HiCache this is a basis swap, not a new pipeline: the
compute/skip schedule, warm-up and API stay identical. Only the per-skip forecast formula
changes. Backends: `"hermite"` (corrected HiCache polynomial; pick this for DiT-class
denoising), `"dmd"` (exponential; pick this for flow-matching 3D generators), `"auto"`
(holdout selection; regime-switch safety net, see the verdict above).

> **Name note.** *HiCache* here refers to the diffusion **feature-forecasting** method
> (Hermite polynomial feature caching, [arXiv:2508.16984](https://arxiv.org/abs/2508.16984)),
> which HiCache++ extends. It is **unrelated to SGLang / Mooncake's "HiCache"**, a
> hierarchical **KV cache** for LLM serving. Likewise, in this repo "DMD" always
> abbreviates **Dynamic Mode Decomposition (Prony)**, classical spectral estimation, and
> never Distribution Matching Distillation.

---

## Why exponentials (the hypothesis, and where it holds)

A diffusion/flow-matching sampler integrates `dx/dt = v_theta(x, t)`. If, across a short
window of steps, the cached feature `F_t` (the CFG-combined velocity) evolves under a
slowly varying near-linear operator, the trajectory locally solves a linear feature-ODE
`F' = M F`, whose exact solution class is a sum of damped/oscillatory exponentials
`F_t = sum_j a_j e^{mu_j t}`. Polynomials only locally truncate that class and diverge
under extrapolation; the exponential basis is exact on it.

DMD (Schmid 2010) is the SVD-regularised multivariate generalisation of Prony's method
(1795): identify the linear propagator `A` from raw velocity snapshots
(`F_{t+1} ~ A F_t`), eigendecompose once per compute window, and predict any (fractional)
horizon `k` by eigenvalue powers:

```
F_{t+k} ~ Phi (lambda**k * b),     b = pinv(Phi) F_t
```

**The >= 4-snapshot floor.** A real-valued trajectory spends two real degrees of freedom
per complex pole (a conjugate pair `r e^{+-iw}` gives `r^t cos wt, r^t sin wt`), so a
single oscillatory mode already needs 3 snapshot pairs = 4 snapshots. Below the floor, or
across a non-uniform window, HiCache++ serves the Hermite forecast instead (warm-up is
covered automatically).

**Where the hypothesis binds.** Whether a real model's feature stream is in this class at
the horizons actually served is an empirical question per family. On flow-matching 3D
velocity streams it is (the exponential basis wins). On the 250-step DDPM DiT stream it is
not: the stream is locally so smooth that low-order polynomial truncation error is
negligible at the served horizons, while the exponential fit pays pole-estimation error
that compounds with the horizon. That is the domain split. And note that holdout
selection cannot see it from inside the window: DMD fits (and therefore backcasts) the
snapshot history better on DiT even while extrapolating forward worse, which is exactly
why `auto` picks the wrong arm there and why the family default is the recommendation.

---

## Results

All accelerators here are training-free; the right A/B is how far the output drifts from
the uncached baseline vs how much faster it runs. "DMD" below is the HiCache++ exponential
basis.

### Mechanism (controlled, no model)

Forecasting `H` steps past an 8-anchor window on synthetic trajectories from the exact
feature-ODE class, rel. L2 error (lower is better), all rows sign-correct:

| basis | H=1 | H=2 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|
| TaylorSeer (monomial) | 1.5e-2 | 8.0e-2 | 6.2e-1 | 2.3e0 | 6.5e0 |
| Pade / FoCa (rational) | 4.9e-2 | 1.1e-1 | 2.4e-1 | 5.3e-1 | 1.2e0 |
| HiCache (Hermite, fixed +k) | 1.5e-1 | 3.0e-1 | 6.3e-1 | 1.0e0 | 1.7e0 |
| **HiCache++ (exponential)** | **4.7e-9** | **1.4e-8** | **5.3e-8** | **1.2e-7** | **2.2e-7** |

On the class the exponential basis is exact and flat in `H`; every polynomial-family basis
diverges. Under 1% snapshot noise DMD's rank truncation contains the error at ~0.3 where
the monomial amplifies to 15. When the dynamics break the fit (an abrupt regime switch
inside the window), `auto` detects the misfit in 120/120 windows, serves the Hermite arm,
and contains the long-horizon error at 1.56 vs 9.40 for a forced exponential fit (the
plain monomial sits at 106); on clean/noisy/drifting trajectories it picks the exponential
arm 120/120 and matches it exactly. Full six-scenario tables for both holdout modes:
[`benchmarks/MICROBENCH_RESULTS.md`](benchmarks/MICROBENCH_RESULTS.md). Reproduce:
`python benchmarks/forecast_microbench.py`.

### Flow-matching 3D generators (where the exponential basis wins)

**Hunyuan3D-2.1** (flat DiT velocities), Toys4K F-score@0.05 vs the uncached baseline
(0.911 = self-score). Excludes `ball_000` (a sphere; Go-ICP alignment is rotationally
degenerate on it; other cells reproduce to +-0.01). The Hermite arm is the fork's deployed
(as-released, pre-sign-fix) forecaster; see the honesty note below. Speedup is solo.

| interval | Hermite arm (as released) | **DMD (HiCache++)** | speedup |
|---:|---:|---:|---:|
| baseline (uncached) | 0.911 | 0.911 | 1.00x |
| i3 | **0.876** | 0.852 | 1.72x |
| i4 | 0.776 | **0.827** | 1.80x |
| **i5** | 0.735 | **0.860** | 1.79x |
| i6 | 0.375 | **0.616** | ~2.0x |

The exponential basis degrades gracefully where the polynomial arm collapses, and its lead
grows with the interval. On the deployed **Hunyuan3D-2-mini** it is **exactly lossless at
i5** (0.794 = baseline 0.794).

**SAM3D** (PyTree velocities, slat FlowMatching), real weights, vs baseline:

| config | speedup | CD_vs_base | F1_vs_base |
|---|---:|---:|---:|
| vanilla | 1.00x | 0.000 | **1.000** |
| HiCache i3 | 1.44x | 0.013 | **1.000** |
| DMD i5 | 1.47x | 0.013 | **1.000** |
| **DMD i6** | **1.56x** | 0.013 | **1.000** |

Geometry-lossless to interval 6, past the Hermite arm's lossless i3; the extra speed is
entirely the wider lossless interval.

**TRELLIS v1** (sparse-structure stage), Toys4K F-score@0.05, n=31, same carved-hybrid
schedule, only the SS forecast basis swapped:

| variant | F@0.05 | speedup | vs vanilla |
|---|---:|---:|---:|
| vanilla (uncached) | 0.839 | 1.00x | n/a |
| Hermite arm | 0.825 | 2.82x | -0.014 |
| **HiCache++ (DMD)** | **0.829** | **2.76x** | **-0.010** |

The same holds on **TRELLIS.2-4B (v2)**: ties at the deployed interval, +0.03-0.04 F-score
at intervals 3-4
(see [`hermit-trellis2-plus-plus`](https://github.com/Archerkattri/hermit-trellis2-plus-plus#results)).

Full tables: [`results/RESULTS.md`](results/RESULTS.md).

### DiT-XL/2 ImageNet-256 (where the polynomial basis wins)

Paired-noise FID-10k ladder, 250-step DDPM, cfg 1.5: the per-step noise is re-seeded
identically across cells, so FID vs the uncached baseline measures pure cache-induced
drift (a lossless cache reads ~0; the interval-1 control reads -0.00). The ladder is
COMPLETE: both eras are labeled ("as released" = the pre-sign-fix near-reuse Hermite and
the auto cell measured alongside it; "corrected" = sign-fixed +k re-runs). FID values are
from the 10k runs; latencies marked \* were re-timed post-eigencache at n=512 because
their FID runs predate the per-window eigendecomposition cache. Full protocol and ledger:
[`benchmarks/dit_imagenet/RESULTS_DIT.md`](benchmarks/dit_imagenet/RESULTS_DIT.md).

| cell | speedup | FID drift (as released) | FID drift (corrected) | FID vs ImageNet-10k ref |
|---|---:|---:|---:|---:|
| baseline | 1.00x | 0.00 | 0.00 | 8.89 |
| taylor_i4 (TaylorSeer, +k) | 3.81x | n/a | **2.27** | 8.95 |
| hermite_i4 (HiCache, +k) | 3.79x | 10.57 | **3.54** | 9.60 |
| dmd_i4 (HiCache++) | 3.71x\* | 18.02 | 18.02 | 21.47 |
| auto_i4 (1-step holdout) | 2.59x\* | 18.08 | 18.11 | 21.57 |
| auto_i4 (horizon holdout) | 2.59x\* | n/a | 18.11 | 21.57 |
| hermite_i6 (+k) | 5.46x | 28.06 | **6.46** | 11.61 |
| dmd_i6 | 5.31x\* | 54.24 | 54.24 | 55.57 |
| hermite_i8 (+k) | 7.21x | 57.79 | **10.74** | 15.10 |
| dmd_i8 | 6.98x | 100.65 | 100.65 | 100.99 |

(Speedup shown for the corrected run where one exists; the as-released hermite cells ran
at 4.02x / 5.80x / 7.66x. Absolute FID for as-released hermite/auto cells: 15.09 / 31.06 /
59.73 and 21.54.)

What this table says, honestly:

1. **Polynomial forecasting is near-lossless on DiT.** The sign-correct TaylorSeer cell
   drifts 2.27 FID at 3.81x (absolute 8.95 vs the baseline's 8.89), and the corrected
   Hermite sweeps the ladder: 3.54 / 6.46 / 10.74 at i4 / i6 / i8. The corrected
   polynomial at i8 (7.2x) drifts less than the as-released one did at i4.
2. **The exponential basis loses here at every interval**: 1.7-1.9x more drift than even
   the as-released (near-reuse) control, 5-9x more than the corrected Hermite. Do not
   deploy `backend="dmd"` on DiT-class denoising; use the polynomial.
3. **Holdout selection failed in the wild, in BOTH modes.** `auto_i4` tracks DMD (18.11
   corrected, vs the corrected Hermite's 3.54) because the 1-gap backcast ranks DMD
   ahead. `holdout="horizon"` (backcast at the actual skip distance), which fixes the
   reconstructed inversion 20/20 on the microbench, reads 18.11 on DiT, identical: on
   real DiT features the horizon holdout ALSO serves the DMD arm. The microbench win did
   not transfer. Why: DMD's richer parameterization fits and backcasts the snapshot
   history better at any in-window distance while extrapolating forward worse, so
   backcast fit carries no signal about forward quality there. Recommendation: family
   defaults (above); `auto` only as a regime-switch safety net; real-feature basis
   selection = open problem.

Not run: the FID-50k trio (~40 h GPU; under the paired-noise differential protocol it
can only tighten an estimator bias that is identical across cells and cancels in the
drift metric).

> **Sign-convention fix (2026-06-10), and why you should care even if you never use this
> repo.** Versions up to and including v1.1.0 evaluated the Hermite basis at `x = -k`
> instead of `x = +k` (`k` = steps past the newest anchor). One character, ours,
> introduced in porting; upstream TaylorSeer uses `+k`, but step-index conventions differ
> across caching codebases. Odd-order terms flipped sign, so the shipped Hermite forecast
> extrapolated backwards: strictly worse than plain reuse in every cell of the controlled
> suite. It survived every end-to-end benchmark because near-reuse fails safe (images
> render, F-scores degrade gracefully), and tuning concealed it (the buggy variant's best
> sigma is the one that mutes it toward reuse). The fixed (+k) Hermite beats reuse in 29
> of 30 cells, is 1.45-2.47x more accurate than shipped, and keeps sigma = 0.5 optimal.
> The DMD and `auto`-selection paths were never affected (forward eigenvalue powers, a
> forward polynomial yardstick). Fixed in **v1.2.0** (`hicache_pp/hermite.py` and `hicache_pp/tree.py`)
> with closed-form directional regression tests (exact extrapolation of a linear series at
> sigma = sqrt(1/2)); we recommend the same one-line test to every caching codebase. Full
> shipped-vs-fixed-vs-reuse A/B:
> [`benchmarks/MICROBENCH_RESULTS.md`](benchmarks/MICROBENCH_RESULTS.md). As-released
> numbers stay published in labeled columns next to corrected ones; the 3D-generator
> results were measured through the forks' deployed (as-released) Hermite arm and stand as
> measured.

---

## Install / use

```bash
pip install hicache-pp
```

The one-loop snippet at the top is the whole integration for **flat tensor** velocities
(e.g. a DiT). For **PyTree / structured** velocities (e.g. SAM3D), use `hicache_pp.tree`,
the same API but tree-aware (`hicache_forecast_tree`, `dmd_forecast_tree`, plus tree
Adaptive-CFG). Backends:

- `backend="hermite"`: the published HiCache scaled-Hermite polynomial (clean,
  sign-correct reimplementation). The basis to pick for DiT-class denoising.
- `backend="dmd"`: the exponential basis. Wins on flow-matching 3D generators; loses on
  DiT-class denoising (see the domain split above).
- `backend="auto"`: holdout selection. Per compute step, backcast a held-out snapshot
  with both arms and serve whichever wins. `holdout="1step"` is the default;
  `holdout="horizon"` is the opt-in distance-matched test. Solves intra-run regime
  switches; does NOT recover the family-level winner on DiT (both modes serve the DMD arm
  there, FID drift 18.11 vs the corrected Hermite's 3.54). Use family defaults first;
  evidence in [`benchmarks/MICROBENCH_RESULTS.md`](benchmarks/MICROBENCH_RESULTS.md) and
  [`benchmarks/dit_imagenet/RESULTS_DIT.md`](benchmarks/dit_imagenet/RESULTS_DIT.md).

See [`integrations/`](integrations/) for the exact wiring into Hunyuan3D-2.1,
Hunyuan3D-2-mini, SAM3D and Fast-SAM3D, and
[`integrations/pr_drafts/`](integrations/pr_drafts/) for prepared patches that add the
exponential basis to **cache-dit**, **Hugging Face diffusers** (`TaylorSeerCacheConfig`)
and **Cache4Diffusion** in each project's native conventions.

#### cache-dit FLUX A/B (the DMD-calibrator patch, end-to-end)

The cache-dit patch was run as a three-arm A/B on real FLUX-class DiTs, one fresh
subprocess per arm, identical prompts / seed 42 / 28 steps / 1024². No visible quality
regression vs vanilla.

![cache-dit FLUX.1-dev A/B: vanilla vs DBCache+TaylorSeer vs DBCache+DMD, same seed](integrations/pr_drafts/cache-dit/flux_ab/side_by_side_dev.png)

*On the real target, FLUX.1-dev: vanilla / TaylorSeer / DMD, same seed. Speedups 1.00x /
1.15x / 1.10x (RTX 5090, n=2 prompts), parity images. Honest reading: on dev TaylorSeer
edges DMD, the reverse of the Chroma ordering below, and the 0.05x gap is within n=2
run-to-run noise. DMD is a competitive complementary basis whose edge is model-dependent;
the per-window auto selector picks it where it wins (flow-matching 3D generators). Full
record: [`integrations/pr_drafts/cache-dit/flux_ab/`](integrations/pr_drafts/cache-dit/flux_ab/).*

![cache-dit FLUX A/B on Chroma1-HD: vanilla vs DBCache+TaylorSeer vs DBCache+DMD, same seed](integrations/pr_drafts/cache-dit/flux_ab/side_by_side.png)

*Chroma1-HD (ungated de-distilled FLUX architecture), vanilla / TaylorSeer / DMD, same
seed. Mean s/image 33.89 / 32.18 / 30.47, speedups 1.00x / 1.05x / **1.11x** (RTX 5090,
n=2 prompts). The patched DMD calibrator is the fastest of the three with parity images.*

### Tuning notes

- **Pick the basis by family.** Flow-matching 3D velocity streams: `"dmd"`, push the
  interval to i5-i6. DiT-class denoising: `"hermite"` (or a plain TaylorSeer); never
  `"dmd"`, and do not rely on `"auto"` to save you (it serves the DMD arm there).
- **Hermite**: lossless up to a modest interval (Hunyuan-2.1: i3/order-2). Higher order
  does not rescue bigger intervals.
- **Exponential**: `history` is the snapshot window (5-6); needs >= 4 *uniformly spaced*
  snapshots before it engages (Hermite covers warm-up automatically).
- `first_enhance` always computes the first few steps (high curvature); keep it >= 3.
- The DMD eigendecomposition is cached per compute window (refit exactly when a new
  snapshot arrives), so the per-skip forecast cost is one `Phi @ (lambda**k * b)`:
  2.5-3.1x cheaper per forecast than fit-per-call on CPU
  (`benchmarks/eigencache_timing.py`). On the DiT GPU re-time it cuts dmd_i4 from 566
  to 483 ms/img (3.17x -> 3.71x) and dmd_i6 from 434 to 337 ms/img (4.13x -> 5.31x).

---

## Tests

```bash
python -m hicache_pp.hermite     # Hermite basis + schedule + sign-convention regressions
python -m hicache_pp.dmd         # exponential basis + auto holdout modes + eigencache
python -m hicache_pp.tree        # tree-aware Hermite + exponential + Adaptive-CFG + eigencache
python tests/run_tests.py        # all of the above + DiT-harness tests (taylor sign,
                                 # FID checkpoint/resume) + version sync; also
                                 # pytest-discoverable
```

---

## 3D generator integrations (sibling repos)

The forecaster in this repo is model-agnostic; it has also been wired natively into a
family of 3D-generator forks. These are **complementary accelerators, not competing
solutions**: each speeds up a *different* base generator, and the `+` / `++` suffix is a
**method choice** (`+` = HiCache Hermite polynomial, `++` = HiCache++ exponential), not a
rival product. Pick by (1) which base model you run, then (2) which forecast basis you
want:

| base generator | `+` = HiCache (Hermite) | `++` = HiCache++ (DMD) |
|---|---|---|
| Hunyuan3D-2.1 | `hunyuan2.1-plus` | `hunyuan2.1-plus-plus` |
| Hunyuan3D-2 mini | `hunyuan2-plus` | `hunyuan2-plus-plus` |
| SAM 3D Objects | `sam3d-plus` | `sam3d-plus-plus` |
| Fast-SAM3D | `fastsam3d-plus` | `fastsam3d-plus-plus` |
| TRELLIS (v1) | `faster-trellis` | `faster-trellis-plus-plus` |
| TRELLIS.2-4B (v2) | `hermit-trellis2` | `hermit-trellis2-plus-plus` |

All of these are flow-matching 3D generators, i.e. the side of the domain split where the
exponential basis wins. `fast-trellis2` is the TaylorSeer baseline fork (the upstream
"Fast" accel), the v2 reference point, not a HiCache variant.

**Sign-convention note (1.2.0 wave, 2026-06-10).** The family forks above vendor their own
copy of the Hermite forecast, and all of them inherited the `x = -k` porting bug fixed here
in 1.2.0; every fork has been updated to the corrected `x = +k` convention. The numbers
published in each fork were measured with the as-released code and remain valid as-measured.
Re-validated so far on the same protocols as published: SAM 3D Objects (slat stage) and
Hunyuan3D-2 mini; see each fork's README for the corrected-vs-as-released tables. The DMD
(`++`) arms are unaffected by the sign fix (it touches only their Hermite comparison arm and
warm-up fallback), and they still vendor the pre-eigencache DMD fit, so 1.2.0's per-window
eigendecomposition cache is a latency win they do not yet ship.

---

## Lineage & attribution

- **TaylorSeer**: feature caching with a monomial (Taylor) basis.
- **HiCache** (arXiv:2508.16984): the scaled-Hermite polynomial upgrade.
  `hicache_pp.hermite` is a clean reimplementation (sign-corrected as of v1.2).
- **HiCache++ (this work)**: the exponential (DMD/Prony) forecaster (`hicache_pp.dmd`),
  the cross-family domain-split benchmark, and the `auto` holdout selector with both
  holdout modes. DMD (Schmid 2010) / Prony (1795) / Matrix-Pencil (Hua-Sarkar 1990) are
  classical spectral estimation; their application to diffusion feature caching is, to
  our knowledge, new.
- **Adaptive-CFG** (Adaptive Guidance, arXiv:2312.12487): composable uncond-skip, included
  in the tree module.

## Citation

If you use this library, please cite HiCache++ (this work) and the methods it builds on:

```bibtex
@article{attri2026hicachepp,
  title   = {No Single Basis Wins: A Cross-Family Study of Diffusion Feature
             Forecasting and the Limits of Training-Free Basis Selection},
  author  = {Attri, Krishi},
  journal = {engrXiv},
  doi     = {10.31224/7309},
  year    = {2026}
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
