# [TaylorSeer cache] Add `basis="dmd"`: Dynamic Mode Decomposition (Prony) exponential forecasting

Extends the TaylorSeer cache hook merged in #12648 (tracking issue #12569) with an optional
**exponential** forecast basis, selected via one new `TaylorSeerCacheConfig` field:

```python
import torch
from diffusers import FluxPipeline, TaylorSeerCacheConfig

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

config = TaylorSeerCacheConfig(
    cache_interval=5,
    disable_cache_before_step=4,
    taylor_factors_dtype=torch.float32,
    basis="dmd",        # NEW: exponential (Dynamic Mode Decomposition / Prony) basis
    dmd_history=6,      # snapshots retained per cached stream
)
pipe.transformer.enable_cache(config)
```

Default behavior (`basis="taylor"`) is bit-for-bit unchanged.

## Motivation (honest, family-conditional)

We benchmarked polynomial vs exponential forecast bases across two diffusion families and
found that **no single basis wins**; this PR adds the exponential basis for the family
where it wins, without touching the default.

- **Flow-matching generators** (the family FLUX belongs to): across four 3D
  flow-matching generators the exponential basis wins and its lead grows with
  `cache_interval` (numbers below). The mechanism: across denoising steps a cached
  output stream evolves under a slowly varying, near-linear operator, so locally it is a
  sum of damped/oscillatory exponentials (the exact solution class of a linear
  feature-ODE); **Dynamic Mode Decomposition** (Schmid 2010; the SVD-regularised
  generalisation of Prony's method, *not* Distribution Matching Distillation) fits that
  class directly where a polynomial truncation diverges.
- **DiT-class denoising** (DiT-XL/2 ImageNet-256, 250-step DDPM): the ranking inverts.
  The sign-correct Taylor polynomial is near-lossless (paired-noise FID drift 2.27 vs the
  uncached baseline at 3.81x) and the exponential basis drifts 1.7-1.9x more than even a
  near-reuse Hermite control at every interval tested. So we explicitly do **not** pitch
  `basis="dmd"` as a better default; it is an opt-in basis, and the docstring carries the
  per-family guidance.

The reference implementation
([hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus)) also ships a
training-free per-window holdout selector that backcasts a held-out snapshot with both
bases and serves the winner. We tested it on this exact split and report the honest
verdict: it solves intra-run regime switches but does not recover the family-level
winner on DiT (both holdout modes serve the exponential arm there), so the per-family
default in this PR's docstring, not a selector, is the recommendation. This PR keeps the
surface to one config field.

## Math summary

At full-compute steps the state additionally records the computed output as a snapshot
(per output stream, capped at `dmd_history`, stored in `taylor_factors_dtype`). At
prediction steps, per stream:

1. take the longest **uniformly spaced** suffix of the snapshot history (mixed spacings,
   e.g. across the warmup boundary, would corrupt the fit, since the identified
   propagator advances exactly one spacing per application);
2. one economy SVD of the `[d, n]` snapshot matrix (`n <= dmd_history`, so negligible next
   to a transformer forward) with spectrum-based rank truncation identifies the linear
   propagator `A` from `Y_{t+1} ~ A Y_t`;
3. eigendecompose once per compute window and predict the (fractional) horizon `k` in
   spacing units by eigenvalue powers: `Y_{t+k} ~ Phi (lambda^k * b)`,
   `b = pinv(Phi) Y_t`.

Below the 4-snapshot identifiability floor (one complex pole costs two real degrees of
freedom, so three snapshot pairs minimum) or on any degenerate/non-finite fit, the state
transparently falls back to the existing Taylor prediction, so warm-up and edge cases
behave exactly like current `main`.

## Changes

- `src/diffusers/hooks/taylorseer_cache.py` only:
  - `TaylorSeerCacheConfig`: new fields `basis` (`"taylor"` | `"dmd"`), `dmd_history`,
    `dmd_rank`, `dmd_ridge` (+ validation, docstring, `__repr__`).
  - `TaylorSeerState`: per-stream snapshot history; `_predict_dmd()` with Taylor fallback.
  - module-level `_dmd_forecast()` helper (torch-only; the
    `@torch.compiler.disable`d path is unchanged; prediction already runs outside
    compiled regions).
- No new dependencies, no new public classes; one config object, per reviewer guidance on
  #12569 to keep cache backends behind their config.

## Benchmarks

- Controlled (synthetic trajectories from the exponential solution class, same schedule
  and snapshots for both bases): Taylor order-1 rel. L2 forecast error ~4.6e-1 past
  warm-up; exponential basis ~4.7e-8.
- Flow-matching generators (reference implementation,
  [hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus)): Hunyuan3D-2.1
  (Toys4K F-score@0.05, uncached baseline 0.911): deployed polynomial arm 0.88 / 0.74 /
  0.38 at interval 3 / 5 / 6, exponential 0.85 / 0.86 / 0.62; exactly lossless at
  interval 5 on Hunyuan3D-2-mini; geometry-lossless (F1 = 1.000) to interval 6 on SAM3D
  at 1.56x.
- DiT-class denoising, included for honesty (where the polynomial should stay the
  choice). DiT-XL/2 ImageNet-256, 250-step DDPM, cfg 1.5, paired-noise FID-10k drift vs
  the uncached baseline (lossless cache reads ~0; full ledger and protocol:
  `hicache-plus-plus/benchmarks/dit_imagenet/RESULTS_DIT.md`):

  | basis | i4 | i6 | i8 |
  |---|---:|---:|---:|
  | TaylorSeer (corrected, +k) | **2.27** (3.81x) | - | - |
  | Hermite (corrected, +k) | 3.54 (3.79x) | **6.46** (5.46x) | **10.74** (7.21x) |
  | exponential (DMD) | 18.02 | 54.24 | 100.65 |

  So the explicit recommendation this PR encodes in its docstring: **polynomial default
  for DiT-class models; `basis="dmd"` for flow-matching generators**. A holdout selector
  cannot substitute for that choice: in our pre-registered A/B both holdout modes of the
  reference selector served the exponential arm on DiT (FID drift 18.11 vs the corrected
  polynomial's 3.54), because the richer exponential fit backcasts the snapshot history
  better even where it extrapolates forward worse.
- Remaining before marking ready for review: FLUX.1-dev side-by-side images generated
  with this exact hook.

Memory cost: `dmd_history` snapshots per cached stream (vs `max_order + 1` Taylor
factors), so `use_lite_mode` pairs well with `basis="dmd"`.
