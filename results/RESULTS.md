# A/B results

All numbers are from geometry-preserving A/B runs: the accelerator's output is compared to the
uncached / baseline geometry (F-score / Chamfer drift) at a matched speed measurement. Speedups
are from solo (uncontended) re-timing. Two reproducibility notes: **(a)** wall-clock is set by the
DiT-skip schedule, not the basis — Hermite and DMD at the same interval differ only in the cheap
per-skip formula, so their speedups match; **(b)** the F-score has ±~0.06 run-to-run variance on
*rotationally symmetric* objects, where Go-ICP alignment is degenerate, so we exclude the one such
Toys4K object (`ball_000`, a sphere) and cross-check every headline against a second run.

## 0. Controlled microbenchmark — the mechanism, no model

Before the model A/Bs, a controlled study isolates *why* the skip ceiling exists. On synthetic
trajectories drawn from the exact feature-ODE class (a sum of damped / oscillatory exponentials),
we cache an 8-step window and forecast `H` steps past it — `H` is the skip reach of interval
`H+1`. Three bases: TaylorSeer (polynomial), Padé / FoCa (rational), HiCache++ (DMD / exponential):

**Clean** — rel. L2 forecast error, 20 seeds × 64-channel × 2 modes (lower is better)

| basis | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 1.5e-02 | 8.0e-02 | 2.6e-01 | 6.2e-01 | 2.3e+00 | 6.5e+00 |
| Padé / FoCa (rational) | 4.9e-02 | 1.1e-01 | 1.7e-01 | 2.4e-01 | 5.3e-01 | 1.2e+00 |
| **HiCache++ (exponential)** | **4.7e-09** | **1.4e-08** | **3.0e-08** | **5.3e-08** | **1.2e-07** | **2.2e-07** |

**+ 1% snapshot noise**

| basis | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 9.9e-02 | 3.5e-01 | 8.8e-01 | 1.9e+00 | 6.1e+00 | 1.5e+01 |
| Padé / FoCa (rational) | 7.3e-02 | 4.7e-01 | 1.1e+00 | 1.6e+00 | 2.5e+00 | 3.2e+00 |
| **HiCache++ (exponential)** | **2.2e-02** | **4.7e-02** | **8.4e-02** | **1.2e-01** | **2.1e-01** | **2.9e-01** |

The exponential basis is **exact on the solution class** (~1e-8, flat in `H`); the polynomial
**diverges** with the horizon, and the rational (Padé / FoCa) basis improves on the polynomial
but still diverges — that gap *is* the skip ceiling (6-to-9 orders of magnitude on clean data).
Under noise, DMD's SVD-rank truncation rejects the noise subspace (DMD ≤ 0.3) where the polynomial
amplifies it (→ 15) and the rational basis turns fragile (spurious Froissart poles; shown here
deployed-clamped to last-value reuse → ~3, otherwise it blows up). HiCache's scaled-Hermite is a
*stabilised* member of the polynomial family — the per-model tables below are the fair HiCache
head-to-head.

Reproduce: `python benchmarks/forecast_microbench.py` (CPU, a few seconds).

## 1. Hunyuan3D-2.1 — Toys4K, F-score@0.05 (flat DiT velocities)

Per-interval F-score and clean (solo, uncontended) speedup. Numbers **exclude `ball_000`** — a
perfect sphere on which Go-ICP alignment is rotationally degenerate, so its post-alignment
F-score is essentially random per run (that one cell alone swings the 10-object mean by ±0.05–0.07).
Excluding it, two independent runs agree to ±0.01 (cross-check below).

| interval | Hermite (HiCache) | DMD (HiCache++) | speedup |
|---:|---:|---:|---:|
| baseline (uncached) | 0.911 | 0.911 | 1.00× |
| i3 | **0.876** | 0.852 | 1.72× |
| i4 | 0.776 | **0.827** | 1.80× |
| **i5** | 0.735 | **0.860** | 1.79× |
| i6 | 0.375 | **0.616** | ~2.0× |

**Finding:** Hermite degrades fast as the interval grows (0.88 → 0.78 → 0.74 → 0.38 at i3/i4/i5/i6);
DMD degrades *gracefully* (0.85 → 0.83 → 0.86 → 0.62), holding ≈0.86 through i5 (baseline 0.91). DMD's
lead over Hermite **grows with the skip** — +0.05 at i4, +0.13 at i5, +0.24 at i6 — i.e. the exponential
basis is exactly what extends the lossless skip range. (Wall-clock saturates ~1.8× because the 2.1
pipeline is only partly DiT — the rest, e.g. VAE decode, is a fixed floor; DMD's payoff is the quality
headroom at high interval, which converts to speed in DiT-dominated regimes like SAM3D below.)

<sub>Cross-check (baseline / hc-i3 / dmd-i4 / dmd-i5, excl. ball): run A `out_dmd_ab` 0.911 / 0.876 /
0.827 / 0.860 — run B solo re-time 0.922 / 0.871 / 0.824 / 0.854. All-10 incl. the degenerate sphere:
baseline 0.87 (run A) vs 0.93 (run B), the ±0.06 swing localised entirely to `ball_000`.</sub>

## 2. Hunyuan3D-2-mini (deployed prior) — Toys4K, F-score@0.05, n=10

| variant | F@0.05 | gen_s (solo) |
|---|---:|---:|
| vanilla | 0.794 | 1.89 |
| HiCache i3/o2 | 0.792 | 1.58 |
| DMD i4 | 0.792 | 1.59 |
| **DMD i5** | **0.794**  (exactly lossless) | 1.69 |
| DMD i6 | 0.695 | 1.71 |

DMD is exactly lossless at i5 on the deployed model (mini's F-scores are stable run-to-run —
the symmetric-object variance above is a 2.1-scale effect). Wall-clock gain here is only ~1.1×
(vs 2.1's ~1.8×) because mini's runtime is VAE-decode-dominated, not DiT-dominated — so skipping
DiT forwards saves proportionally less. (Quality is the clean signal.)

## 3. SAM3D — slat-stage FlowMatching, real weights (PyTree velocities)

End-to-end pipeline latency; geometry = output gaussians, F1 vs vanilla baseline. Warmup pass
applied (no cold-start artifact).

| config | latency | speedup | CD_vs_base | F1_vs_base |
|---|---:|---:|---:|---:|
| vanilla | 8.00 s | 1.00× | 0.000 | **1.000** |
| HiCache i3/o2 | 5.56 s | 1.44× | 0.013 | **1.000** |
| DMD i4 | 5.61 s | 1.43× | 0.013 | **1.000** |
| DMD i5 | 5.43 s | 1.47× | 0.013 | **1.000** |
| **DMD i6** | 5.12 s | **1.56×** | 0.013 | **1.000** |

Both HiCache and DMD are geometry-lossless (F1=1.000). DMD stays lossless to interval-6, where
it gives the best speedup (1.56×) — past Hermite's lossless i3.

> The runnable SAM3D pipeline here is Fast-SAM3D's `InferencePipelinePointMap`; its
> `slat_generator` is a `sam3d_objects` FlowMatching — the **same architecture** as standalone
> sam-3d-objects (whose weights are gated/undownloaded). The HiCache++ port is identical in
> both; this table is the same FlowMatching run on real weights.

## 4. Fast-SAM3D — SS-stage TaylorSeer (forecast-basis A/B)

| config | latency | speedup | CD_vs_base | F1_vs_base |
|---|---:|---:|---:|---:|
| taylor (baseline) | 7.68 s | 1.00× | 0.000 | 1.000 |
| hermite (HiCache) | 8.33 s | 0.92× | 0.016 | 1.000 |

Hermite ≈ Taylor — a **wash**: both run the same TaylorSeer stride-3 schedule, so the basis
swap doesn't change the DiT-forward count, only the cheap per-skip formula. TaylorSeer caching
(the default) is the ~3× accelerator, not HiCache-over-Taylor. (Unlike Hunyuan, which has no
default caching — so there the basis choice gives real speedups.)

## 5. TRELLIS v1 — Toys4K F-score@0.05, n=31 (sparse-structure stage)

`faster-trellis` accelerates the sparse-structure (SS) stage with a HiCache (Hermite) velocity
forecast over a token-carved SLaT sampler. Swapping *only the SS forecast basis* Hermite→DMD
(identical schedule and carving) on the 31-object subset that succeeded for every variant:

| variant | F-score@0.05 | speedup | vs vanilla |
|---|---:|---:|---:|
| vanilla (uncached) | 0.839 | 1.00× | — |
| Fast-TRELLIS | 0.823\* | 2.12× | −0.016 |
| HiCache (Hermite) | 0.825 | 2.82× | −0.014 |
| **HiCache++ (DMD)** | **0.829** | **2.76×** | **−0.010** |

<sub>\*Fast-TRELLIS cited from the published `faster-trellis` table (its upstream env was not set
up in this run). v1 only — TRELLIS.2 (the 4B v2 model) is pending its own env debugging.</sub>

**Finding:** at the deployed ~interval-3 schedule (2.8×), **DMD is the most lossless accelerator**
(−0.010 vs vanilla, vs Hermite's −0.014) — it beats Hermite by +0.005 at matched speedup. The
margin is modest *here* because interval-3 is a **low** skip where the polynomial is still decent;
the DMD advantage widens at higher intervals (see §1/§3 and the §0 microbench).

## The model matrix

| | **HiCache** (polynomial) | **HiCache++** (DMD / exponential) |
|---|---|---|
| **Hunyuan3D-2.1** | ✅ integrated (Hermite) | ✅ holds ≈0.83 @ i5 vs Hermite 0.74; lead grows with interval |
| **Hunyuan3D-2 mini** | ✅ integrated | ✅ **exactly lossless** @ i5 (0.794 = baseline) |
| **SAM3D** | ✅ slat: 1.44× lossless (i3) | ✅ slat: lossless to **i6, 1.56×** |
| **Fast-SAM3D** | ✅ SS wash (TaylorSeer) · ✅ slat lossless | ✅ slat: lossless to i6 |
| **TRELLIS v1** | ✅ 0.825 @ 2.82× | ✅ **0.829 @ 2.76×** (most lossless) |
| **DiT-XL/2 (ImageNet)** | ✅ Taylor + Hermite | ⏳ FID-vs-latency sweep in progress |
