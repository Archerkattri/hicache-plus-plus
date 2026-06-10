# DiT-XL/2 ImageNet-256 FID — credibility-gate benchmark (HiCache++)

**Status: Phase 1 (as-released ladder) COMPLETE; Phase 1b (corrected re-run) QUEUED** —
cells fill in as the queues (`results/queue_all.sh`, `results/queue_resume.sh`) complete.
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

## Phase 1 — FID-10k ladder (COMPLETE — as released)

Provenance note (sign-convention fix, 2026-06-10): the `hermite_*` and `auto_i4` cells
below were measured with the as-released (−k, anti-extrapolative, near-reuse) Hermite —
see `benchmarks/MICROBENCH_RESULTS.md` for the bug A/B. The `dmd_*` cells are
sign-independent and `taylor_i4` was measured post-fix (+k), so those rows are already
the corrected values. Corrected re-runs of the affected cells are Phase 1b below.

| cell | compute / steps | ms/img | speedup | FID↓ vs baseline | FID↓ vs ImageNet-10k ref |
|---|---:|---:|---:|---:|---:|
| baseline | 250/250 | 1791 | 1.00× | 0.00 | 8.89 |
| hermite_i4 (HiCache) | 65/250 | 445 | 4.02× | 10.57 | 15.09 |
| dmd_i4 (HiCache++) | 65/250 | 566 | 3.17× | 18.02 | 21.47 |
| auto_i4 (HiCache++ auto) | 65/250 | 766 | 2.34× | 18.08 | 21.54 |
| hermite_i6 (HiCache) | 45/250 | 309 | 5.80× | 28.06 | 31.06 |
| dmd_i6 (HiCache++) | 45/250 | 434 | 4.13× | 54.24 | 55.57 |
| hermite_i8 (HiCache) | 34/250 | 234 | 7.66× | 57.79 | 59.73 |
| dmd_i8 (HiCache++) | 34/250 | 256 | 6.98× | 100.65 | 100.99 |
| taylor_i4 (TaylorSeer, +k) | 65/250 | 470 | 3.81× | 2.27 | 8.95 |

Note on the absolute column: published DiT-XL/2 cfg-1.5 FID-50k is 2.27 under the ADM
TF evaluator with the full reference; 8.89 here reflects N=10k-vs-10k estimator bias,
the 10k reference, and the pytorch_fid extractor — consistent across all cells, so
within-table comparisons remain valid.

### Headline reads of Phase 1 (paper §DiT)

1. **Polynomial forecasting is near-lossless on this workload**: corrected TaylorSeer at
   i4 drifts 2.27 FID at 3.81× (absolute 8.95 vs baseline 8.89).
2. **The exponential (DMD) basis loses at every interval**, drifting 1.7–1.9× more than
   even the as-released near-reuse Hermite control (18.02/54.24/100.65 vs
   10.57/28.06/57.79 at i4/i6/i8).
3. **The 1-step holdout failed in the wild**: `auto_i4` (18.08) tracked DMD (18.02) while
   multi-step reality favored the polynomial arm (10.57). This motivated
   `holdout="horizon"` (Phase 1b A/B cell).

## Phase 1b — corrected re-run + holdout A/B (`results/fid10k_fix/`) — QUEUED

Queue: `results/queue_resume.sh`. Same protocol, code at the sign-fixed HEAD; latency
columns additionally pick up the per-window eigendecomposition cache (so dmd/auto ms/img
re-times here too). Placeholders — only numbers change when cells land:

| cell | compute / steps | ms/img | speedup | FID↓ vs baseline | FID↓ vs ImageNet-10k ref |
|---|---:|---:|---:|---:|---:|
| hermite_i4_fix (+k) | 65/250 | TBD | TBD | TBD | TBD |
| hermite_i6_fix (+k) | 45/250 | TBD | TBD | TBD | TBD |
| hermite_i8_fix (+k) | 34/250 | TBD | TBD | TBD | TBD |
| auto_i4_fix (1step holdout) | 65/250 | TBD | TBD | TBD | TBD |
| auto_i4_horizon (horizon holdout) | 65/250 | TBD | TBD | TBD | TBD |

Pre-registered analysis (written before any Phase 1b cell ran):

- The Phase 1 conclusions above do NOT depend on these cells (they rest on the
  sign-correct taylor_i4 cell and on DMD losing to a *weaker* near-reuse control); the
  corrected-Hermite cells can shift the quantitative, not the qualitative, ladder.
- auto_i4_horizon vs auto_i4_fix is the live A/B of the horizon-matched holdout on the
  workload where the 1-step holdout demonstrably mispredicted; the microbench
  (oscillatory-with-trend, `benchmarks/MICROBENCH_RESULTS.md`) predicts horizon should
  track the better (Hermite) arm here.
- Both Phase-1 and Phase-1b numbers stay published side by side (as released / corrected).

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
4. ~~`bench_dit.py` has no mid-cell checkpointing: a killed 50k cell restarts from 0.~~
   FIXED 2026-06-10: per-1k-image atomic checkpointing (`<cell>.partial.npz`, resumed
   automatically; final npz format unchanged; resumed stats bit-identical to an
   uninterrupted run, see `tests/test_bench_checkpoint.py`). Cells generated before
   the fix were single-shot runs and are unaffected.
5. Smoke observation to verify in Phase 1: at batch 32, `auto_i4` ran 837 ms/img vs
   `hermite_i4` 486 ms/img at identical compute calls (65/250) — DMD/auto forecast
   overhead (per-skip-step SVD fit) is material and the speedup column reflects it.
