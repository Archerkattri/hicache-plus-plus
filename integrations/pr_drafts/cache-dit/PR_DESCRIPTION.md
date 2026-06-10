# Add a Dynamic Mode Decomposition (Prony) exponential-basis calibrator (`calibrator_type="dmd"`)

## Motivation

cache-dit's calibrators currently forecast cached hidden states / residuals with the
TaylorSeer **polynomial** expansion. This PR adds a second, drop-in calibrator backend
with an **exponential** forecast basis: **Dynamic Mode Decomposition** (Schmid 2010), the
SVD-regularised multivariate generalisation of **Prony's method** (1795). (To avoid the
common collision: this is *not* Distribution Matching Distillation.)

Honest, family-conditional pitch. We benchmarked both bases across two diffusion
families, and **no single basis wins**:

- On **flow-matching 3D generators** the exponential basis wins clearly and the lead
  grows with the cache interval (numbers below). This is the regime this calibrator is
  for.
- On **DiT-class denoising** (DiT-XL/2 ImageNet-256, 250-step DDPM) the ranking inverts:
  the sign-correct TaylorSeer polynomial is near-lossless (paired-noise FID drift 2.27 vs
  the uncached baseline at 3.81x), while the exponential basis drifts 1.7-1.9x more than
  even a near-reuse Hermite control at every interval tested. We therefore do **not**
  claim DMD as a better default; it is an additional basis for the workloads where it
  wins, default behavior unchanged.

The mechanism behind the 3D win: across denoising steps each cached feature stream
evolves under a slowly varying, near-linear operator; the exact solution class of a
linear feature-ODE is a sum of damped/oscillatory exponentials, and the exponential basis
is exact on that class where any polynomial diverges under extrapolation. Whether a given
model family's stream is in that class at the served horizons is empirical, hence the
per-family numbers below.

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

The reference implementation
([hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus)) also ships a
training-free holdout selector (`backend="auto"`) that backcasts a held-out snapshot with
both bases per compute window and serves the winner. We benchmarked it on this exact
split and report the honest verdict: it solves intra-run regime switches, but it does
not recover the family-level winner on DiT (both holdout modes served the exponential
arm there, FID drift 18.11 vs the corrected polynomial's 3.54), so the recommended way
to consume this calibrator is a **per-family default** (DMD for flow-matching
generators; TaylorSeer for DiT-class denoising), not a selector. This PR keeps the
surface minimal: one new basis.

## What the calibrator does (math summary)

At each full-compute step the calibrator records the computed tensor as a snapshot
(per named stream, like the TaylorSeer states). At an approximation step it:

1. takes the longest **uniformly spaced** suffix of the snapshot history (the identified
   propagator advances exactly one snapshot-spacing per application, and DBCache's dynamic
   decisions can make the compute cadence non-uniform; mixed spacings would corrupt the
   fit);
2. identifies the linear propagator `A` with `Y_{t+1} ~ A Y_t` via one economy SVD of the
   `[d, n]` snapshot matrix (`n` = history <= 6, so this is cheap relative to a forward
   pass) with spectrum-based rank truncation (this is what rejects noise);
3. eigendecomposes once per compute window (cached; refit only when a new snapshot
   arrives) and forecasts the (fractional) horizon `k` by eigenvalue powers:
   `Y_{t+k} ~ Phi (lambda^k * b)`, `b = pinv(Phi) Y_t`.

Below the 4-snapshot identifiability floor (a real-valued trajectory spends two real
degrees of freedom per complex pole, so one oscillatory mode already needs three snapshot
pairs), or whenever the fit is degenerate/non-finite, it transparently falls back to the
TaylorSeer expansion it also maintains; warm-up behaves exactly like the existing
calibrator.

## Changes

- `caching/cache_contexts/calibrators/dmd.py`: new `DMDCalibrator` + `DMDState`,
  mirroring the `TaylorSeerCalibrator` / `TaylorSeerState` API (`mark_step_begin`,
  `update`, `approximate`, `step`, `reset_cache`; per-stream states keyed by name).
- `caching/cache_contexts/calibrators/__init__.py`: new `DMDCalibratorConfig`
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
- Method-level (reference implementation,
  [hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus)), flow-matching
  3D generators: on Hunyuan3D-2.1 (Toys4K, F-score@0.05 vs uncached baseline 0.911) the
  deployed polynomial arm decays 0.88 / 0.74 / 0.38 at cache interval 3 / 5 / 6 while the
  exponential basis holds 0.85 / 0.86 / 0.62; exactly lossless at interval 5 on
  Hunyuan3D-2-mini; on SAM3D geometry-lossless (F1 = 1.000) through interval 6 at 1.56x.
- DiT-class denoising, reported for honesty (the regime where you should NOT pick this
  calibrator). DiT-XL/2 ImageNet-256, 250-step DDPM, cfg 1.5, paired-noise FID-10k drift
  vs the uncached baseline (lossless cache reads ~0; full ledger and protocol:
  `hicache-plus-plus/benchmarks/dit_imagenet/RESULTS_DIT.md`):

  | basis | i4 | i6 | i8 |
  |---|---:|---:|---:|
  | TaylorSeer (corrected, +k) | **2.27** (3.81x) | - | - |
  | Hermite (corrected, +k) | 3.54 (3.79x) | **6.46** (5.46x) | **10.74** (7.21x) |
  | exponential (DMD) | 18.02 | 54.24 | 100.65 |

  Holdout selection does not rescue DiT either: in our pre-registered A/B both holdout
  modes of the reference selector served the exponential arm (drift 18.11), because the
  richer exponential fit backcasts the snapshot history better even where it
  extrapolates forward worse. Hence the per-family default recommendation above.
- Remaining before marking ready for review: a FLUX.1-dev A/B with this exact
  calibrator.

Scoping summary for reviewers: this adds an opt-in basis that wins on flow-matching
generators and is reported, with numbers, as losing on DiT-class denoising. The
per-interval tables are included so the trade-off is judged directly, not from a single
operating point.
