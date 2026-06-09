# [TaylorSeer cache] Add `basis="dmd"` — Dynamic Mode Decomposition (Prony) exponential forecasting

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

## Motivation

The TaylorSeer hook forecasts skipped module outputs with a polynomial (Taylor) expansion.
Across denoising steps a cached output stream evolves under a slowly varying, near-linear
operator, so locally it is a sum of damped/oscillatory **exponentials** — the exact
solution class of a linear feature-ODE. A polynomial is only a local truncation of that
class and diverges as `cache_interval` grows, which is what limits how aggressively the
cache can be pushed. **Dynamic Mode Decomposition** (Schmid 2010; the SVD-regularised
generalisation of Prony's method — *not* Distribution Matching Distillation) fits the
exponential class directly, so the forecast stays bounded and accurate at larger intervals.

## Math summary

At full-compute steps the state additionally records the computed output as a snapshot
(per output stream, capped at `dmd_history`, stored in `taylor_factors_dtype`). At
prediction steps, per stream:

1. take the longest **uniformly spaced** suffix of the snapshot history (mixed spacings —
   e.g. across the warmup boundary — would corrupt the fit, since the identified
   propagator advances exactly one spacing per application);
2. one economy SVD of the `[d, n]` snapshot matrix (`n ≤ dmd_history`, so negligible next
   to a transformer forward) with spectrum-based rank truncation identifies the linear
   propagator `A` from `Y_{t+1} ≈ A Y_t`;
3. eigendecompose once and predict the (fractional) horizon `k` in spacing units by
   eigenvalue powers: `Y_{t+k} ≈ Φ (λ^k ⊙ b)`, `b = Φ⁺ Y_t`.

Below the 4-snapshot identifiability floor (one complex pole costs two real degrees of
freedom → three snapshot pairs minimum) or on any degenerate/non-finite fit, the state
transparently falls back to the existing Taylor prediction, so warm-up and edge cases
behave exactly like current `main`.

## Changes

- `src/diffusers/hooks/taylorseer_cache.py` only:
  - `TaylorSeerCacheConfig`: new fields `basis` (`"taylor"` | `"dmd"`), `dmd_history`,
    `dmd_rank`, `dmd_ridge` (+ validation, docstring, `__repr__`).
  - `TaylorSeerState`: per-stream snapshot history; `_predict_dmd()` with Taylor fallback.
  - module-level `_dmd_forecast()` helper (torch-only; `@torch.compiler.disable`d path
    unchanged — prediction already runs outside compiled regions).
- No new dependencies, no new public classes; one config object, per reviewer guidance on
  #12569 to keep cache backends behind their config.

## Benchmarks

- Controlled (synthetic trajectories from the exponential solution class, same schedule
  and snapshots for both bases): Taylor order-1 rel. L2 forecast error ~4.6e-1 past
  warm-up; exponential basis ~4.7e-8.
- Method-level results from the standalone implementation
  ([hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus)): on
  Hunyuan3D-2.1 (a flow-matching DiT) the polynomial basis decays 0.88 → 0.74 → 0.38
  F-score at interval 3/5/6 vs baseline 0.91, while the exponential basis holds
  0.85 → 0.86 → 0.62; geometry-lossless to interval 6 on SAM3D.
- **Placeholder:** the DiT-XL/2 ImageNet-256 FID-50k / IS vs latency table (both bases at
  matched intervals) is in progress at `hicache-plus-plus/benchmarks/dit_imagenet/`;
  it, plus FLUX.1-dev side-by-side images with this exact hook, will be attached before
  the PR is marked ready for review.

Honest scoping: at small intervals (≤3) the polynomial is already near-lossless and can
edge out the exponential fit; the win is at the larger intervals (4-6) users choose for
speed, where the polynomial collapses and the exponential degrades gracefully. Memory
cost: `dmd_history` snapshots per cached stream (vs `max_order + 1` Taylor factors), so
`use_lite_mode` pairs well with `basis="dmd"`.
