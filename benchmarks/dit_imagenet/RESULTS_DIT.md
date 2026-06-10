# DiT-XL/2 ImageNet-256 FID — credibility-gate benchmark (HiCache++)

**Status: IN PROGRESS** — cells fill in as the queue (`results/queue_all.sh`) completes.
Started 2026-06-09.

## Protocol

Literature-standard feature-cache setting (TaylorSeer / HiCache / FoCa / Spectrum):

- Model: **DiT-XL/2, 256×256** (official facebookresearch/DiT checkpoint
  `data/weights/DiT/DiT-XL-2-256x256.pt`), fp32 + TF32, eager.
- Sampler: **250-step DDPM** (DiT `create_diffusion("250")`), **cfg-scale 1.5**,
  class-conditional with classes drawn uniformly from a seeded generator (seed 0).
- Batch 64 (128 with CFG doubling), identical across every cell.
- **Paired-noise FID** (commit 92c5c78): the per-step ancestral noise is re-seeded
  identically per batch across cells, so the *primary* metric — FID of a cached cell
  **vs the uncached baseline** — measures pure cache-induced drift with no RNG floor.
  A lossless cache reads ~0. Validity control: `hermite --interval 1` (zero forecast
  steps) must read FID ≈ 0 vs baseline (Phase 0 below).
- Absolute **FID vs ImageNet** is also reported, against Inception stats computed from
  the 10k images in ADM's reference batch `VIRTUAL_imagenet256_labeled.npz` with the
  **same pytorch_fid InceptionV3** the harness uses
  (`results/ref_stats_imagenet256_10k_pytorchfid.npz`). The npz's *built-in* `mu/sigma`
  were produced by ADM's TF-Inception evaluator and are NOT feature-compatible with
  pytorch_fid; mixing extractors skews FID, so they are not used.
- **IS is not reported**: the harness stores only 2048-d Inception pool stats
  (mu, sigma), not logits, so Inception Score cannot be computed from the saved cells.
- Latency = wall-clock of `p_sample_loop` only (CUDA-synchronized, same GPU, same
  batch size for every cell; VAE decode + Inception featurization excluded equally
  for all cells). Speedup = baseline ms/img ÷ cell ms/img.
- Cache hyperparameters (harness defaults): order 2, sigma 0.5, history 6,
  first_enhance 4.

Methods: `none` = uncached baseline · `taylor` = TaylorSeer monomial forecast ·
`hermite` = HiCache scaled-Hermite · `dmd` = HiCache++ exponential (DMD) ·
`auto` = HiCache++ holdout-selected DMD-vs-Hermite backend.

## Sample sizes (be explicit)

FID-50k per config is infeasible in a few hours on this GPU (baseline alone ≈ 25 h:
1.80 s/img × 50k). Per the agreed fallback:

- **Phase 0** — control, N=512: baseline + `hermite_i1` (paired-noise validity check).
- **Phase 1** — **FID-10k full ladder** (`results/fid10k/`): baseline, hermite_i4,
  dmd_i4, auto_i4, hermite_i6, dmd_i6, hermite_i8, dmd_i8, taylor_i4.
- **Phase 2** — **FID-50k headline trio** (`results/fid50k/`): baseline, hermite_i4,
  dmd_i4. Queued after Phase 1 (≈ 40 h GPU; resumable per cell, NOT mid-cell).

N is in the table header of every results block below. 10k-vs-10k FID carries the usual
small-N estimator bias; it is the *same* bias for every cell, and the paired-noise
vs-baseline column is differential, so the ladder ordering is meaningful at 10k.

## Phase 0 — paired-noise control (N=512) — PASSED

| cell | compute / steps | ms/img | speedup | FID↓ vs baseline |
|---|---:|---:|---:|---:|
| baseline | 250/250 | 1814 | 1.00× | 0.00 |
| hermite_i1 | 250/250 | 1817 | 1.00× | −0.00 |

`hermite --interval 1` executes the full cache machinery but never forecasts; FID vs
baseline ≈ 0 confirms trajectories are exactly noise-paired across separate processes —
any nonzero FID in Phase 1 is genuine cache-induced drift, not an RNG floor.

## Phase 1 — FID-10k ladder

PENDING

## Phase 2 — FID-50k headline trio

QUEUED (not started)

## Reproduction

```bash
# environment: gim_env (torch 2.12.0.dev20260407+cu128), GPU0 = RTX 5090 32 GB,
# driver 570.211.01, CUDA_VISIBLE_DEVICES=0, OMP_NUM_THREADS=2
cd third_party/hicache-plus-plus/benchmarks/dit_imagenet
bash results/queue_all.sh                      # everything, resumable
# one cell:
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 ../../../../gim_env/bin/python bench_dit.py \
  --method dmd --interval 4 --n 10000 --batch 64 --steps 250 --cfg-scale 1.5 \
  --seed 0 --gpu 0 --out results/fid10k
# tables:
../../../../gim_env/bin/python compute_fid.py results/fid10k \
  --ref results/ref_stats_imagenet256_10k_pytorchfid.npz
```

## Git state (exact code under test)

- Repo `third_party/hicache-plus-plus` @ **HEAD 92c5c78**
  ("bench(dit-imagenet): paired-noise FID harness improvement"), **dirty**:
  uncommitted working-tree changes in `hicache_pp/` add the `auto` backend
  (`auto_forecast_state`, holdout DMD-vs-poly selection in `dmd.py`;
  `backend="auto"` accepted in `hermite.py`; snapshot `detach().clone()` fix;
  `tree.py` tweak). `hicache_pp/` was NOT modified by this benchmark run.
  Pin: `git diff hicache_pp/ | sha256sum` =
  `87178ec0cb967caa4d4abe1de68d90786eda7d4747919628eea536e76bb03c0a`.
- Benchmark-side edits made for this run (harness only, outside `hicache_pp/`):
  `bench_dit.py` gained `--method auto` (4-line additive change wiring the existing
  dirty-tree `auto` backend through the CLI; `dmd`/`hermite`/`taylor`/`none` paths
  byte-identical in behavior).
- DiT model code: `third_party/DiT` was **missing**; cloned fresh from
  facebookresearch/DiT (depth-1) for this run.

## Hardware / environment

- GPU0: NVIDIA GeForce RTX 5090 (32 GB), driver 570.211.01. GPU0 only.
- Python: `gim_env` (torch 2.12.0.dev20260407+cu128, diffusers, pytorch_fid 2048-d
  block, timm 1.0.26). VAE: `stabilityai/sd-vae-ft-mse` (HF cache).
- OMP_NUM_THREADS=2 for every process.

## Caveats / harness notes (documented, core not fixed)

1. `compute_fid.py` docstring mentions a `--build-ref` flag that does not exist;
   reference stats were built manually (script inlined; output
   `results/ref_stats_imagenet256_10k_pytorchfid.npz`, N=10,000 reference images).
2. Absolute-FID column uses a 10k reference (the images bundled in ADM's npz), not the
   full-training-set 50k+ reference of the ADM TF evaluator — absolute numbers are
   therefore not directly comparable to published ADM-protocol FIDs; the vs-baseline
   drift column is the primary, protocol-clean metric.
3. `--batch 128` OOMs on 32 GB in the VAE decode; no throughput gain over 64 anyway
   (GPU saturated: 1804 vs 1841 ms/img at batch 128 vs 32).
4. `bench_dit.py` has no mid-cell checkpointing: a killed 50k cell restarts from 0.
5. Smoke observation to verify in Phase 1: at batch 32, `auto_i4` ran 837 ms/img vs
   `hermite_i4` 486 ms/img at identical compute calls (65/250) — DMD/auto forecast
   overhead (per-skip-step SVD fit) is material and the speedup column reflects it.
