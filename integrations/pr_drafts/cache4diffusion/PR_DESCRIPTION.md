# Add an exponential forecast basis to TaylorSeer: Dynamic Mode Decomposition (Prony)

## Motivation

TaylorSeer's "cache then forecast" paradigm showed that diffusion features are
predictable; this PR adds a second predictor for them. The exponential fit used here is
**Dynamic Mode Decomposition** (Schmid 2010), the SVD-regularised multivariate
generalisation of **Prony's method** (1795). To preempt the usual collision: this is
*not* Distribution Matching Distillation.

The pitch is deliberately family-conditional, because our cross-family benchmark shows
**no single forecast basis wins**:

- On **flow-matching generators** the exponential basis wins and the lead grows with
  `fresh_threshold` (numbers below): across timesteps each cached feature evolves under a
  slowly varying, near-linear operator, so locally it is a sum of damped/oscillatory
  exponentials (the exact solution class of a linear feature-ODE), which the exponential
  basis fits directly where a polynomial truncation diverges.
- On **DiT-class denoising** (DiT-XL/2 ImageNet-256, 250-step DDPM) the ranking inverts:
  the sign-correct Taylor polynomial is near-lossless (paired-noise FID drift 2.27 vs the
  uncached baseline at 3.81x) while the exponential basis drifts 1.7-1.9x more than even
  a near-reuse Hermite control at every interval tested. We do **not** claim the
  exponential basis as a better default; the default `'taylor'` path is byte-identical in
  behavior, and the numbers below say per family when to flip the switch.

The reference implementation
([hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus)) also ships a
training-free per-window holdout selector (backcast a held-out snapshot with both bases,
serve the winner) that automates this choice; offered as a follow-up if the maintainers
want it. This PR keeps the surface to one dispatch switch.

Following the framework's spirit of unifying forecast methods, the basis slots in as a
**per-run switch inside the existing TaylorSeer method** rather than a duplicated method
directory: all `forwards/` files are untouched, and every model port that already calls
`derivative_approximation` / `taylor_formula` inherits the new basis for free.

```python
# flux/taylorseer_flux, after cache_init:
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
2. one economy SVD of the `[d, n]` snapshot matrix (`n <= 6`, negligible vs a forward
   pass) with spectrum-based rank truncation identifies the propagator `A` from
   `Y_{t+1} ~ A Y_t`;
3. one eigendecomposition per compute window; the (fractional) horizon `k` in spacing
   units is forecast by eigenvalue powers `Y_{t+k} ~ Phi (lambda^k * b)`,
   `b = pinv(Phi) Y_t`.

Below the 4-snapshot identifiability floor (a real trajectory spends two real degrees of
freedom per complex pole, so three snapshot pairs minimum), or on a degenerate/non-finite
fit, it returns to the unchanged Taylor formula; warm-up behaves exactly as today.

## Changes (FLUX first; other models follow the same 3 hooks)

- `flux/taylorseer_flux/dmd_utils/__init__.py`: new `_dmd_forecast`,
  `dmd_snapshot_update`, `dmd_formula`, keyed by the same
  `cache_dic` / `current['stream']/['layer']/['module']` convention as
  `taylorseer_utils`.
- `flux/taylorseer_flux/taylorseer_utils/__init__.py`: `derivative_approximation`
  records snapshots and `taylor_formula` dispatches when
  `cache_dic['forecast_basis'] == 'dmd'`; with the default `'taylor'` both functions are
  byte-identical in behavior.
- `flux/taylorseer_flux/cache_functions/cache_init.py`: `forecast_basis`,
  `dmd_history`, `dmd_rank`, `dmd_ridge` knobs in the `'Taylor'` mode block.

Torch-only, no new dependencies. Qwen-Image / HunyuanVideo / HunyuanImage-2.1 / DiT ports
are the same three hooks; happy to add them in this PR or a follow-up, whichever the
maintainers prefer.

## Benchmarks

- Controlled (synthetic trajectories from the exponential solution class, identical
  schedule/snapshots): Taylor order-1 rel. L2 error 0.4-1.9 past warm-up; exponential
  basis ~5e-8.
- Flow-matching generators (reference implementation,
  [hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus)): Hunyuan3D-2.1
  (Toys4K F-score@0.05, uncached baseline 0.911): deployed polynomial arm 0.88 / 0.74 /
  0.38 at interval 3 / 5 / 6, exponential 0.85 / 0.86 / 0.62, the lead growing with the
  interval, which is the regime this framework's >4-6x speedups live in; exactly lossless
  at interval 5 on Hunyuan3D-2-mini; geometry-lossless (F1 = 1.000) to interval 6 on
  SAM3D.
- DiT-class denoising, included for honesty (do not flip the switch there): DiT-XL/2
  ImageNet-256 paired-noise FID-10k drift, exponential 18.02 / 54.24 / 100.65 at interval
  4/6/8 vs corrected Taylor 2.27 at interval 4. Full ledger:
  `hicache-plus-plus/benchmarks/dit_imagenet/RESULTS_DIT.md`.
- **Placeholder:** the final DiT table (corrected-Hermite cells, selector A/B, GPU
  re-timing, FID-50k trio) plus a FLUX-dev ImageReward/CLIP A/B using this exact patch
  will be added before the PR is marked ready.

Numbers are reported per interval and per family so the trade-off is explicit.
