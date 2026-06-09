# Add an exponential forecast basis to TaylorSeer — Dynamic Mode Decomposition (Prony)

## Motivation

TaylorSeer's "cache then forecast" paradigm showed that diffusion features are
predictable; this PR upgrades *what predicts them*. The Taylor expansion is a polynomial,
i.e. a local truncation of the trajectory the features actually follow: across timesteps
each cached feature evolves under a slowly varying, near-linear operator, so locally it is
a sum of damped/oscillatory **exponentials** (the exact solution class of a linear
feature-ODE). Polynomials diverge under extrapolation — that is what caps
`fresh_threshold` — while an exponential basis stays bounded with the correct asymptotics.

The exponential fit used here is **Dynamic Mode Decomposition** (Schmid 2010), the
SVD-regularised multivariate generalisation of **Prony's method** (1795). To preempt the
usual collision: this is *not* Distribution Matching Distillation.

Following the framework's spirit of unifying forecast methods, the basis slots in as a
**per-run switch inside the existing TaylorSeer method** rather than a duplicated method
directory: all `forwards/` files are untouched, and every model port that already calls
`derivative_approximation` / `taylor_formula` inherits the new basis for free.

```python
# flux/taylorseer_flux — after cache_init:
joint_attention_kwargs['cache_dic']['forecast_basis'] = 'dmd'   # default: 'taylor'
joint_attention_kwargs['cache_dic']['fresh_threshold'] = 8      # push the interval
```

## Math summary

At a `'full'` step, `derivative_approximation` additionally records the raw feature as a
snapshot per `(stream, layer, module)` (window of `dmd_history = 6`; the dynamics drift
across timesteps, so a longer window would average over changing dynamics). At a
`'Taylor'` step, `taylor_formula` first tries the exponential forecast:

1. longest **uniformly spaced** suffix of the snapshot history (the identified propagator
   advances exactly one snapshot-spacing per application; the `first_enhance` boundary
   makes early spacings non-uniform, so they are excluded automatically);
2. one economy SVD of the `[d, n]` snapshot matrix (`n ≤ 6` — negligible vs a forward
   pass) with spectrum-based rank truncation identifies the propagator `A` from
   `Y_{t+1} ≈ A Y_t`;
3. one eigendecomposition; the (fractional) horizon `k` in spacing units is forecast by
   eigenvalue powers `Y_{t+k} ≈ Φ (λ^k ⊙ b)`, `b = Φ⁺ Y_t`.

Below the 4-snapshot identifiability floor (a real trajectory spends two real degrees of
freedom per complex pole → three snapshot pairs minimum), or on a degenerate/non-finite
fit, it returns to the unchanged Taylor formula — warm-up behaves exactly as today.

## Changes (FLUX first; other models follow the same 3 hooks)

- `flux/taylorseer_flux/dmd_utils/__init__.py` — new: `_dmd_forecast`,
  `dmd_snapshot_update`, `dmd_formula`, keyed by the same
  `cache_dic` / `current['stream']/['layer']/['module']` convention as
  `taylorseer_utils`.
- `flux/taylorseer_flux/taylorseer_utils/__init__.py` — `derivative_approximation`
  records snapshots and `taylor_formula` dispatches when
  `cache_dic['forecast_basis'] == 'dmd'`; with the default `'taylor'` both functions are
  byte-identical in behavior.
- `flux/taylorseer_flux/cache_functions/cache_init.py` — `forecast_basis`,
  `dmd_history`, `dmd_rank`, `dmd_ridge` knobs in the `'Taylor'` mode block.

Torch-only, no new dependencies. Qwen-Image / HunyuanVideo / HunyuanImage-2.1 / DiT ports
are the same three hooks; happy to add them in this PR or a follow-up, whichever the
maintainers prefer.

## Benchmarks

- Controlled (synthetic trajectories from the exponential solution class, identical
  schedule/snapshots): Taylor order-1 rel. L2 error 0.4-1.9 past warm-up; exponential
  basis ~5e-8.
- Method-level, from the standalone implementation
  ([hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus)): on
  Hunyuan3D-2.1 the polynomial basis decays 0.88 → 0.74 → 0.38 F-score at interval 3/5/6
  (baseline 0.91) while the exponential holds 0.85 → 0.86 → 0.62 — the lead grows with the
  interval, which is the regime this framework's >4-6x speedups live in.
- **Placeholder:** DiT-XL/2 ImageNet-256 FID-50k / IS vs latency (both bases, matched
  intervals) is in progress at `hicache-plus-plus/benchmarks/dit_imagenet/` and will be
  added here, together with a FLUX-dev ImageReward/CLIP A/B using this exact patch, before
  the PR is marked ready.

Honest scoping: at the small `fresh_threshold` values where Taylor is already
near-lossless, the exponential basis is a tie or slightly behind; its value is graceful
degradation at larger thresholds. Numbers are reported per interval so the trade-off is
explicit.
