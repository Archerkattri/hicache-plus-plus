# FLUX-family A/B evidence for the DMD-calibrator PR (#642)

Driver: `/home/krishi/workspace/cache_dit_ab/flux_ab.py` (one fresh subprocess per arm —
pipelines do not release ~30 GiB between arms in-process). Patch under test:
`../0001-dmd-calibrator.patch` applied to `vipshop/cache-dit@95b0800`.

Three arms, identical prompts/seed(42)/28 steps/1024², guidance 4.0:
`vanilla` (no cache) vs `taylorseer` (DBCache + existing best calibrator) vs `dmd` (the
PR's exponential-basis calibrator).

## Model: the FLUX gate forced a stand-in

BOTH `FLUX.1-dev` and `FLUX.1-schnell` are HF click-gated (2026-06; kattri token gets 403
on file access). The A/B therefore runs on **Chroma1-HD** (`lodestones/Chroma1-HD`,
Apache-2.0) — the ungated **de-distilled FLUX architecture** (same DiT double/single block
structure, 8.9B, CFG, many-step), a pipeline cache-dit officially supports. This validates
the patch end-to-end on a FLUX-class DiT now; re-run with `--model dev` (script unchanged)
once the one-click license is accepted. See `../../../../../../cache_dit_ab/README.md`.

## Artifacts (this dir)

- `timings.json` — per-arm mean seconds + speedup-vs-vanilla (full record)
- `side_by_side.png` — vanilla / taylorseer / dmd × 2 prompts
- `vanilla_p0.png` / `taylorseer_p0.png` / `dmd_p0.png` — per-arm images for parity check

## Result (Chroma1-HD, 28 steps, 1024², seed 42, RTX 5090, n=2 prompts)

| arm | mean s/image | speedup vs vanilla |
|---|---:|---:|
| vanilla (no cache) | 33.89 | 1.00x |
| DBCache + TaylorSeer | 32.18 | 1.05x |
| DBCache + **DMD** (this PR) | **30.47** | **1.11x** |

The patched DMD calibrator runs end-to-end on a real FLUX-class DiT and is the fastest of
the three, no visible quality regression vs vanilla (`side_by_side.png`). Modest absolute
speedup is expected: DBCache defaults (`residual_diff_threshold=0.08`, 8-step warm-up)
cache only ~8/28 steps at 28 total steps; calibrator quality matters most as the cache
interval grows (the PR's interval-3/5/6 tables). Integration check, not the headline bench.

## Caveat

DiT-class denoising is the regime the PR itself flags as NOT DMD's strong suit (TaylorSeer
near-lossless, DMD drifts) — see the PR's DiT table. Chroma is FLUX-architecture
flow-matching, closer to the regime DMD targets, but the headline 3D flow-matching win
lives in hicache-plus-plus's Hunyuan3D/SAM3D benches, not here. This A/B is an
end-to-end "the patched calibrator runs and produces parity images on a real FLUX-class
pipeline at a measured speedup" integration check.

## On-target re-run: FLUX.1-dev (license accepted 2026-06-11)

Same driver, `--model dev`, FLUX.1-dev, 28 steps, seed 42, n=2 prompts, RTX 5090:

| arm | mean s/image | speedup vs vanilla |
|---|---:|---:|
| vanilla | 29.71 | 1.00x |
| taylorseer | 25.90 | 1.15x |
| dmd (this PR) | 26.94 | 1.10x |

Honest reading: on FLUX.1-dev specifically, TaylorSeer edges DMD (1.15x vs 1.10x), the
reverse of the Chroma1-HD ordering above (DMD 1.11x vs TaylorSeer 1.05x). FLUX is a
DiT-class flow-matching transformer, the regime where our broader study finds polynomial
forecasting competitive-to-better; at n=2 prompts the 0.05x gap is within run-to-run noise.
The honest claim this evidence supports is therefore NOT "DMD is faster on FLUX" but
"DMD is a competitive complementary forecast basis whose advantage is model-dependent,
and the per-window auto selector chooses it where it wins (e.g. flow-matching 3D generators)."
The PR pitch is updated to match: DMD as an added basis + the auto selector, never a
universal-default claim.
