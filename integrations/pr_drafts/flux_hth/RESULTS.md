# FLUX.1-dev head-to-head: DMD vs Spectrum vs TaylorSeer

Forecast-basis comparison on FLUX.1-dev (a flow-matching MMDiT image model),
50 steps, seed 42, 12 DrawBench prompts, 1024x1024, one RTX 5090.

Each method is scored against its **own** uncached output (the fidelity a cache
must hit): LPIPS (AlexNet) and PSNR vs that vanilla, plus an absolute CLIP
(ViT-B-32) prompt-alignment score. Speedup is wall-time vs the same-env vanilla,
with the text encoders dropped and the transformer + VAE resident, so the timing
is the denoising cost rather than PCIe transfer.

| method | basis / fit | speedup | LPIPS (down) | PSNR (up) | CLIP (up) |
|---|---|---|---|---|---|
| **Spectrum** (CVPR'26) | Chebyshev, global error-bounded | **3.46x** | **0.072** | **28.2** | 0.322 |
| DMD (this work) | exponential (Prony), local | 3.14x | 0.378 | 19.8 | 0.323 |
| TaylorSeer | polynomial, local | 3.21x | 0.784 | 11.8 | 0.274 |
| DMD (lower interval) | exponential, local | 2.45x | 0.160 | 23.0 | 0.320 |
| TaylorSeer (lower interval) | polynomial, local | 2.55x | 0.201 | 21.0 | 0.323 |

Uncached CLIP reference: 0.325. Spectrum computes 14 / 50 forwards; DMD@2.45x
also computes 14.

## Findings (honest)

1. **On FLUX, Spectrum wins, clearly.** At matched ~3.3x it is both faster and
   far more faithful (LPIPS 0.072 vs DMD 0.378), and the win holds at equal
   compute (both compute 14 / 50 forwards, yet Spectrum is more faithful and
   faster). On this flow-matching image model a global, error-bounded fit beats
   local forecasting of any basis.

2. **The exponential basis beats the polynomial.** Within the same cache-dit
   DBCache skip mechanism, DMD beats TaylorSeer at every operating point and the
   gap widens with speedup: at ~3.2x TaylorSeer collapses (LPIPS 0.78, CLIP
   0.27, visible noise) while DMD stays prompt-faithful (LPIPS 0.38, CLIP 0.32).
   That is the polynomial-vs-exponential question, isolated.

3. **The confound.** Spectrum's edge is global, error-bounded fitting, not the
   Chebyshev basis itself. The constructive direction is an exponential basis
   *with* a global error-bounded fit, not a choice between the two.

**Caveat:** FLUX is a flow-matching *image* model. HiCache++'s reported DMD wins
are on flow-matching *3D* generators (TRELLIS / Hunyuan3D / SAM-3D), which are
not in this comparison; the polynomial-wins regime in HiCache++ is DiT-class DDPM
denoising. So this result adds a third data point, it does not overturn the
cross-family finding.

## Panels
- `hth_panel.png` — uncached | Spectrum | DMD | TaylorSeer at matched ~3.3x.
- `cachedit_dmd_vs_taylorseer.png` — in-framework, DMD vs the existing TaylorSeer
  calibrator (same DBCache).
- `dmd_vs_vanilla.png` — the DMD calibrator with / without, at a faithful ~2.45x.

## Reproduce
`flux_hth.py` (one arm per process), `eval_hth.py` (metrics), `make_viz.py`
(panels). Spectrum runs in diffusers 0.34 (its pinned version), the cache-dit
arms in 0.38; each method is scored against its own-env uncached run, and the two
uncached runs differ by < 1 s/img and are visually identical.
