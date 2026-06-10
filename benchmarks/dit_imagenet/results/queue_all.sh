#!/bin/bash
# Credibility-gate DiT-XL/2 ImageNet-256 FID queue (GPU0 only, OMP=2).
# Resumable per cell (bench_dit.py skips existing .npz); NO mid-cell resume.
# Phase 0: paired-noise control (N=512)  Phase 1: FID-10k full ladder
# Phase 2: FID-50k headline trio (long; safe to kill, finished cells kept)
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # benchmarks/dit_imagenet
PY=/home/krishi/workspace/gaussianfeels/gim_env/bin/python
RES="$HERE/results"
run(){ # method interval n outdir
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 "$PY" "$HERE/bench_dit.py" \
    --method "$1" --interval "$2" --n "$3" --batch 64 --steps 250 \
    --cfg-scale 1.5 --seed 0 --gpu 0 --out "$4"
}
echo "=== PHASE 0: control N=512 (paired-noise validity) ==="
run none    0 512 "$RES/control"
run hermite 1 512 "$RES/control"
"$PY" "$HERE/compute_fid.py" "$RES/control"

echo "=== PHASE 1: FID-10k ladder ==="
for cell in "none 0" "hermite 4" "dmd 4" "auto 4" "hermite 6" "dmd 6" \
            "hermite 8" "dmd 8" "taylor 4"; do
  run $cell 10000 "$RES/fid10k"
  "$PY" "$HERE/compute_fid.py" "$RES/fid10k" \
        --ref "$RES/ref_stats_imagenet256_10k_pytorchfid.npz" || true
done

echo "=== PHASE 2: FID-50k headline trio ==="
for cell in "none 0" "hermite 4" "dmd 4"; do
  run $cell 50000 "$RES/fid50k"
done
"$PY" "$HERE/compute_fid.py" "$RES/fid50k" \
      --ref "$RES/ref_stats_imagenet256_10k_pytorchfid.npz" || true
echo "=== QUEUE DONE ==="
