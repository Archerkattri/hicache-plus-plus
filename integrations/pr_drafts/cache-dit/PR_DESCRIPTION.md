# Add a Dynamic Mode Decomposition (Prony) exponential-basis calibrator (`calibrator_type="dmd"`)

## Motivation

cache-dit's calibrators currently forecast cached hidden states / residuals with the
TaylorSeer **polynomial** expansion. A polynomial is only a local truncation of the
trajectory the cached features actually follow, so its error compounds with the cache
interval — which is exactly what caps how far `Fn/Bn` caching can be pushed before quality
drops.

This PR adds a second, drop-in calibrator backend with an **exponential** forecast basis:
**Dynamic Mode Decomposition** (Schmid 2010), the SVD-regularised multivariate
generalisation of **Prony's method** (1795). (To avoid the common collision: this is *not*
Distribution Matching Distillation.) Across denoising steps each cached feature stream
evolves under a slowly varying, near-linear operator; the exact solution class of a linear
feature-ODE is a sum of damped/oscillatory exponentials. The exponential basis is exact on
that class, where any polynomial diverges under extrapolation — so it degrades much more
gracefully as the cache interval grows.

It plugs into the existing `CalibratorConfig` pattern, exactly like
`TaylorSeerCalibratorConfig`:

```python
import cache_dit
from cache_dit import DMDCalibratorConfig

cache_dit.enable_cache(
    pipe,
    calibrator_config=DMDCalibratorConfig(dmd_history=6),
)
```

## What the calibrator does (math summary)

At each full-compute step the calibrator records the computed tensor as a snapshot
(per named stream, like the TaylorSeer states). At an approximation step it:

1. takes the longest **uniformly spaced** suffix of the snapshot history (the identified
   propagator advances exactly one snapshot-spacing per application, and DBCache's dynamic
   decisions can make the compute cadence non-uniform — mixed spacings would corrupt the
   fit);
2. identifies the linear propagator `A` with `Y_{t+1} ≈ A Y_t` via one economy SVD of the
   `[d, n]` snapshot matrix (`n` = history ≤ 6, so this is cheap relative to a forward
   pass) with spectrum-based rank truncation (this is what rejects noise);
3. eigendecomposes once and forecasts the (fractional) horizon `k` by eigenvalue powers:
   `Y_{t+k} ≈ Φ (λ^k ⊙ b)`, `b = Φ⁺ Y_t`.

Below the 4-snapshot identifiability floor (a real-valued trajectory spends two real
degrees of freedom per complex pole, so one oscillatory mode already needs three snapshot
pairs), or whenever the fit is degenerate/non-finite, it transparently falls back to the
TaylorSeer expansion it also maintains — warm-up behaves exactly like the existing
calibrator.

## Changes

- `caching/cache_contexts/calibrators/dmd.py` — new: `DMDCalibrator` + `DMDState`,
  mirroring the `TaylorSeerCalibrator` / `TaylorSeerState` API (`mark_step_begin`,
  `update`, `approximate`, `step`, `reset_cache`; per-stream states keyed by name).
- `caching/cache_contexts/calibrators/__init__.py` — new `DMDCalibratorConfig`
  dataclass (`dmd_history`, `dmd_rank`, `dmd_ridge`), registered in the `Calibrator`
  factory and `_supported_calibrators`.
- Export chain: `DMDCalibratorConfig` re-exported from `cache_contexts`, `caching`, and
  the top-level `cache_dit` namespace, alongside `TaylorSeerCalibratorConfig`.

No new dependencies (torch-only), no behavior change unless `calibrator_type="dmd"` is
selected.

## Validation so far

- Unit-level: on synthetic trajectories from the exponential solution class, the
  calibrator's post-warm-up forecast error is ~5e-8 relative L2 where the order-1 Taylor
  expansion sits at ~0.4-1.9 (same snapshots, same schedule).
- Method-level (standalone implementation of the same forecaster,
  [hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus)): on
  Hunyuan3D-2.1 the Hermite/Taylor polynomial basis decays 0.88 → 0.74 → 0.38 F-score at
  cache interval 3/5/6 while the exponential basis holds 0.85 → 0.86 → 0.62
  (baseline 0.91); on SAM3D it stays geometry-lossless (F1 = 1.000) to interval 6.
- **Placeholder:** DiT-XL/2 ImageNet-256 FID-50k / IS vs latency table (Taylor vs
  exponential at matched intervals) is in progress at
  `hicache-plus-plus/benchmarks/dit_imagenet/` and will be added here, plus a FLUX.1-dev
  A/B with this exact calibrator, before marking the PR ready for review.

Honest scoping: the exponential basis is not uniformly better at *small* intervals — at
interval 3 the polynomial can edge it out. Its value is the graceful degradation at the
larger intervals users actually want for speed; the head-to-head tables above are reported
per interval so reviewers can judge that trade-off directly.
