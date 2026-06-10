# DiT-XL/2 ImageNet-256 FID — credibility-gate benchmark (HiCache++)

**Status: COMPLETE (2026-06-10).** Phase 0 (paired-noise control), Phase 1 (as-released
FID-10k ladder), Phase 1b (corrected re-run + holdout A/B) and the post-eigencache
latency re-time are all banked. Phase 2 (FID-50k trio) was **not run** (see below);
nothing in the conclusions depends on it.

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
- **Latency provenance (re-time, 2026-06-10):** the `dmd_i4`, `dmd_i6` and `auto` FID
  cells were generated *before* the per-window eigendecomposition cache (commit
  46b6548), so their original ms/img mixed eras with the rest of the ladder. Those four
  cells were re-timed with the current code at **n=512, latency only** (same GPU,
  same protocol; `results/latency_retime/`, driver `results/queue_retime.sh`) and the
  tables below carry the re-timed figures, marked `*`. FID values are always from the
  10k runs. `dmd_i8` and every corrected (`fid10k_fix`) cell were generated after the
  eigencache landed, so their FID-run latencies are already post-optimization.
- Cache hyperparameters (harness defaults): order 2, sigma 0.5, history 6,
  first_enhance 4.

Methods: `none` = uncached baseline · `taylor` = TaylorSeer monomial forecast ·
`hermite` = HiCache scaled-Hermite · `dmd` = HiCache++ exponential (DMD) ·
`auto` = HiCache++ holdout-selected DMD-vs-Hermite backend (1step / horizon modes).

## Sample sizes (be explicit)

FID-50k per config is infeasible in a few hours on this GPU (baseline alone ≈ 25 h:
1.80 s/img × 50k). Per the agreed fallback:

- **Phase 0** — control, N=512: baseline + `hermite_i1` (paired-noise validity check).
- **Phase 1** — **FID-10k full ladder** (`results/fid10k/`): baseline, hermite_i4,
  dmd_i4, auto_i4, hermite_i6, dmd_i6, hermite_i8, dmd_i8, taylor_i4.
- **Phase 1b** — **corrected re-run + holdout A/B** (`results/fid10k_fix/`):
  hermite_i4/i6/i8 (+k), auto_i4 (1step), auto_i4_horizon.
- **Phase 2** — FID-50k headline trio: **not run.** ≈ 40 h GPU for a result that, under
  the paired-noise differential protocol, can only tighten the estimator bias shared by
  all cells; every conclusion below is drawn from within-10k comparisons. Documented as
  a limitation in the paper.

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

## Phase 1 — FID-10k ladder (as released) — COMPLETE

Provenance note (sign-convention fix, 2026-06-10): the `hermite_*` and `auto_i4` cells
below were measured with the as-released (−k, anti-extrapolative, near-reuse) Hermite —
see `benchmarks/MICROBENCH_RESULTS.md` for the bug A/B. The `dmd_*` cells are
sign-independent and `taylor_i4` was measured post-fix (+k), so those rows are already
the corrected values. Corrected re-runs of the affected cells are Phase 1b below.
Latencies marked `*` are the post-eigencache re-time (n=512); see Protocol.

| cell | compute / steps | ms/img | speedup | FID↓ vs baseline | FID↓ vs ImageNet-10k ref |
|---|---:|---:|---:|---:|---:|
| baseline | 250/250 | 1791 | 1.00× | 0.00 | 8.89 |
| hermite_i4 (HiCache) | 65/250 | 445 | 4.02× | 10.57 | 15.09 |
| dmd_i4 (HiCache++) | 65/250 | 483\* | 3.71× | 18.02 | 21.47 |
| auto_i4 (HiCache++ auto) | 65/250 | 691\* | 2.59× | 18.08 | 21.54 |
| hermite_i6 (HiCache) | 45/250 | 309 | 5.80× | 28.06 | 31.06 |
| dmd_i6 (HiCache++) | 45/250 | 337\* | 5.31× | 54.24 | 55.57 |
| hermite_i8 (HiCache) | 34/250 | 234 | 7.66× | 57.79 | 59.73 |
| dmd_i8 (HiCache++) | 34/250 | 256 | 6.98× | 100.65 | 100.99 |
| taylor_i4 (TaylorSeer, +k) | 65/250 | 470 | 3.81× | 2.27 | 8.95 |

Pre-eigencache originals for the re-timed rows (superseded, kept for the record):
dmd_i4 566 ms (3.17×), dmd_i6 434 ms (4.13×), auto_i4 766 ms (2.34×). The eigencache
cuts the DMD forecast overhead by ~15% of cell wall-clock at i4 and ~22% at i6
(amortization grows with the number of skipped steps per window served per fit).

Note on the absolute column: published DiT-XL/2 cfg-1.5 FID-50k is 2.27 under the ADM
TF evaluator with the full reference; 8.89 here reflects N=10k-vs-10k estimator bias,
the 10k reference, and the pytorch_fid extractor — consistent across all cells, so
within-table comparisons remain valid.

## Phase 1b — corrected re-run + holdout A/B (`results/fid10k_fix/`) — COMPLETE

Same protocol, code at the sign-fixed HEAD (+ eigencache, so the FID-run latencies are
already consistent; the auto rows additionally re-timed at n=512, marked `*`, agreeing
within ~4%: 718/722 ms in the 10k runs).

| cell | compute / steps | ms/img | speedup | FID↓ vs baseline | FID↓ vs ImageNet-10k ref |
|---|---:|---:|---:|---:|---:|
| hermite_i4 (+k) | 65/250 | 472 | 3.79× | **3.54** | 9.60 |
| hermite_i6 (+k) | 45/250 | 328 | 5.46× | **6.46** | 11.61 |
| hermite_i8 (+k) | 34/250 | 248 | 7.21× | **10.74** | 15.10 |
| auto_i4 (1step holdout) | 65/250 | 691\* | 2.59× | 18.11 | 21.57 |
| auto_i4_horizon (horizon holdout) | 65/250 | 692\* | 2.59× | 18.11 | 21.57 |

### Pre-registered analysis — RESOLVED

The analysis below was written before any Phase 1b cell ran; each point is now resolved
against the measured cells:

- *"The Phase 1 conclusions do NOT depend on these cells ... the corrected-Hermite cells
  can shift the quantitative, not the qualitative, ladder."* → **Held.** The corrected
  Hermite improved 3.0–5.4× over as-released (3.54/6.46/10.74 vs 10.57/28.06/57.79) and
  every qualitative conclusion stands: polynomials win DiT at every interval, DMD loses
  at every interval (now by 5–9× vs the corrected control), taylor_i4 remains the best
  i4 cell.
- *"auto_i4_horizon vs auto_i4_fix is the live A/B ... the microbench predicts horizon
  should track the better (Hermite) arm here."* → **REFUTED — the key negative result.**
  auto_i4_horizon reads 18.11, identical to auto_i4 (18.11): the horizon holdout ALSO
  serves the DMD arm on real DiT features. The microbench misprediction-regime win
  (20/20 correct picks) did not transfer.
- *"Both Phase-1 and Phase-1b numbers stay published side by side."* → Done (this file,
  README, paper Table dit).

### The selection-failure finding (key negative result)

Backcast-based holdout selection — at ANY backcast distance — fails on DiT. Mechanism:
DMD's parameterization is strictly richer than the polynomial yardstick at matched
window (complex poles + amplitudes vs three real coefficients), so it fits, and
therefore *backcasts*, the snapshot history better at any holdout distance inside the
window — while extrapolating *forward* worse, because forward error is dominated by
pole-estimation error invisible to the in-window fit. A holdout drawn from the same
short history the model was fit on rewards in-sample flexibility: it is model selection
by training error, locally disguised as validation. On the synthetic suites the holdout
is informative because the generating dynamics genuinely separate the two bases; on
DiT's smooth real features both bases fit the history near-perfectly and the backcast
margin carries no signal about forward extrapolation quality.

Consequences (propagated to README, paper, PR drafts):
1. **Recommendation = basis by model family.** Polynomial (TaylorSeer/corrected Hermite)
   for DiT-class denoising; DMD for the flow-matching 3D family (where the family repos
   show it winning).
2. `auto` remains shipped as a **regime-switch safety net** (microbench: 120/120 correct
   on regime switches) — not as a way to discover the family winner.
3. **Open problem:** per-window basis selection from real feature trajectories needs a
   signal that correlates with forward extrapolation error; in-window backcast fit
   demonstrably is not one on this family.

## Phase 2 — FID-50k headline trio

**Not run** (decision 2026-06-10): ≈ 40 h GPU; under the paired-noise differential
protocol the 50k run can only shrink an estimator bias that is identical across cells
and cancels in the primary drift metric. Documented as limitation (i) in the paper.

## Reproduction

```bash
# environment: gim_env (torch 2.12.0.dev20260407+cu128), GPU0 = RTX 5090 32 GB,
# driver 570.211.01, CUDA_VISIBLE_DEVICES=0, OMP_NUM_THREADS=2
cd third_party/hicache-plus-plus/benchmarks/dit_imagenet
bash results/queue_all.sh                      # Phase 0 + 1, resumable
bash results/queue_resume.sh                   # Phase 1b (corrected + holdout A/B)
bash results/queue_retime.sh                   # latency re-time (n=512, post-eigencache)
# one cell:
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 ../../../../gim_env/bin/python bench_dit.py \
  --method dmd --interval 4 --n 10000 --batch 64 --steps 250 --cfg-scale 1.5 \
  --seed 0 --gpu 0 --out results/fid10k
# tables:
../../../../gim_env/bin/python compute_fid.py results/fid10k \
  --ref results/ref_stats_imagenet256_10k_pytorchfid.npz
# (for results/fid10k_fix, copy results/fid10k/baseline.npz in first — the corrected
#  cells are compared against the same paired-noise baseline.)
```

## Git state (exact code under test)

- Phase 1 cells: repo @ 92c5c78 + the then-uncommitted `auto` backend working tree
  (pin: `git diff hicache_pp/ | sha256sum` =
  `87178ec0cb967caa4d4abe1de68d90786eda7d4747919628eea536e76bb03c0a`), except
  `dmd_i8`/`taylor_i4` (banked at 3adc012, post sign-fix + eigencache + checkpointing).
- Phase 1b cells + latency re-time: repo @ 8b767e6 (sign-fixed `taylor_forecast`,
  eigencache 46b6548, horizon holdout aba0462, checkpointing 6302547) — all committed,
  clean tree.
- DiT model code: `third_party/DiT` cloned fresh from facebookresearch/DiT (depth-1).

## Hardware / environment

- GPU0: NVIDIA GeForce RTX 5090 (32 GB), driver 570.211.01. GPU0 only, solo (no
  concurrent GPU or heavy CPU jobs during any timed cell).
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
5. The Phase-1 smoke observation (auto 837 vs hermite 486 ms/img at batch 32) was the
   pre-eigencache DMD/auto per-skip SVD overhead; the eigencache + re-time above is the
   current truth (auto_i4 691 ms/img at batch 64).
