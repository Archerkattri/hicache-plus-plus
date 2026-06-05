#!/bin/bash
# DiT-XL/2 ImageNet-256 FID sweep — runs every (method, interval) cell sequentially into one
# output dir. Resumable: bench_dit.py skips a cell whose .npz already exists.
#   usage: run_sweep.sh [N_SAMPLES] [STEPS] [GPU] [OUTDIR]
set -e
N=${1:-5000}; STEPS=${2:-250}; GPU=${3:-0}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
OUT=${4:-$ROOT/manipulation/dit_fid_out}
PY="$ROOT/gim_env/bin/python"
B=64
run(){ CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 "$PY" "$HERE/bench_dit.py" \
        --method "$1" --interval "$2" --n "$N" --batch "$B" --steps "$STEPS" \
        --gpu 0 --out "$OUT"; }

echo "=== DiT FID sweep: N=$N steps=$STEPS gpu=$GPU out=$OUT ==="
run none 0          # uncached baseline
run taylor 3        # TaylorSeer (monomial)
run hermite 2; run hermite 3; run hermite 4          # HiCache (Hermite)
run dmd 2; run dmd 3; run dmd 4; run dmd 5            # HiCache++ (DMD)
echo "=== SWEEP DONE ==="
"$PY" "$HERE/compute_fid.py" "$OUT"
