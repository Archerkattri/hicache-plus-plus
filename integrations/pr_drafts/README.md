# Prepared upstream PR drafts (not yet submitted)

Locally prepared patches that add the HiCache++ **Dynamic Mode Decomposition (Prony)
exponential forecast basis** to the three upstream feature-caching hubs, each in that
project's own conventions. Nothing here has been forked, pushed, or opened as a PR;
these are review-ready drafts.

The pitch in every draft is deliberately family-conditional, matching the repo's
cross-family findings: the exponential basis **wins on flow-matching 3D generators**
(Hunyuan3D-2.1 / 2-mini, SAM3D, TRELLIS) and **loses on DiT-class denoising** (DiT-XL/2
ImageNet-256, where the corrected TaylorSeer polynomial is near-lossless). No draft
claims the exponential basis as a better default; each adds it as an opt-in basis with
per-family numbers, and points at the `auto` holdout selector in this repo as the
recommended follow-up for automating the choice.

| target | patch | pattern followed | base commit |
|---|---|---|---|
| [vipshop/cache-dit](https://github.com/vipshop/cache-dit) | `cache-dit/0001-dmd-calibrator.patch` | `Calibrator` plugin + `CalibratorConfig` dataclass (like `TaylorSeerCalibratorConfig`) | `95b08005` |
| [huggingface/diffusers](https://github.com/huggingface/diffusers) | `diffusers/0001-taylorseer-dmd-basis.patch` | `basis="dmd"` option on the merged TaylorSeer hook (`TaylorSeerCacheConfig`, PR #12648) | `2c7efb95` |
| [Shenyi-Z/Cache4Diffusion](https://github.com/Shenyi-Z/Cache4Diffusion) | `cache4diffusion/0001-flux-dmd-basis.patch` | `forecast_basis` dispatch inside `taylorseer_utils` + a `dmd_utils` module (FLUX first) | `91a1949f` |

Each subdirectory contains the patch (`git apply` / `git am`-able unified diff against the
listed base commit) and a `PR_DESCRIPTION.md` ready to paste into the PR body.

All three patches embed the same self-contained forecaster (one economy SVD + one
eigendecomposition per compute window; no new dependencies) rather than importing
`hicache_pp`, so each upstream stays dependency-free. Reference implementation and tests:
[`hicache_pp/dmd.py`](../../hicache_pp/dmd.py).

**Before submitting each PR:**

1. fill the benchmark placeholder in each `PR_DESCRIPTION.md` from the final DiT table
   (`benchmarks/dit_imagenet/RESULTS_DIT.md`: corrected-Hermite cells, the holdout
   selector A/B, GPU re-timing, and the FID-50k trio are queued on GPU);
2. re-validate the patch against upstream HEAD (these repos move fast);
3. run one end-to-end A/B on the target model (FLUX for cache-dit/diffusers) with the
   exact patched code.
