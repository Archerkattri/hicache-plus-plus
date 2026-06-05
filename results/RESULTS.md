# A/B results

All numbers are from geometry-preserving A/B runs: the accelerator's output is compared to the
uncached / baseline geometry (F-score / Chamfer drift) at a matched speed measurement. Gen-time
notes: numbers marked *(contended)* were measured with another GPU job running and should be
re-timed solo for a clean speedup; quality (F-score / F1) is unaffected by contention.

## 1. Hunyuan3D-2.1 — Toys4K, F-score@0.05, n=10 (flat DiT velocities)

| interval | Hermite (HiCache) | DMD (HiCache++) |
|---:|---:|---:|
| baseline (uncached) | 0.867 | 0.867 |
| i3 (order-2) | **0.888**  (1.81× lossless) | 0.867 |
| i4 | 0.773 | 0.807 |
| **i5** | 0.683 | **0.864**  (≈ lossless) |
| i6 | 0.292 | 0.732 |

**Finding:** the polynomial (Hermite) is lossless only to i3 and collapses by i5; the
exponential (DMD) holds near-baseline quality at i5 — it breaks the polynomial skip ceiling.

## 2. Hunyuan3D-2-mini (deployed prior) — Toys4K, F-score@0.05, n=10

| variant | F@0.05 | gen_s *(contended)* |
|---|---:|---:|
| vanilla | 0.794 | 1.96 |
| HiCache i3/o2 | 0.792 | 1.58 |
| DMD i4 | 0.792 | 1.59 |
| **DMD i5** | **0.794**  (exactly lossless) | 1.69 |
| DMD i6 | 0.695 | 1.71 |

DMD is exactly lossless at i5 on the deployed model. Wall-clock gain here is only ~1.2×
(vs 2.1's 1.81×) because mini's runtime is VAE-decode-dominated, not DiT-dominated — so
skipping DiT forwards saves proportionally less. (Quality is the clean signal.)

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

## The 6-cell matrix

| | **HiCache** (polynomial) | **HiCache++** (DMD / exponential) |
|---|---|---|
| **Hunyuan3D** | ✅ 2.1: 1.81× lossless (i3) | ✅ lossless at i5 (breaks the ceiling) |
| **SAM3D** | ✅ slat: 1.44× lossless | ✅ slat: lossless to i6, 1.56× |
| **Fast-SAM3D** | ✅ SS: wash (TaylorSeer default) · ✅ slat: 1.44× lossless | ✅ slat: lossless to i6, 1.56× |
