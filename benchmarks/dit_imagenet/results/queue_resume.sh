#!/bin/bash
# RESUME script for the interrupted FID-10k ladder (written 2026-06-10, queue stopped
# mid-dmd_i8 to free GPU0 for splatreg benchmarks).
#
# DONE (npz present, rows in RESULTS_DIT.md): baseline, hermite_i4, dmd_i4, auto_i4,
#   hermite_i6, dmd_i6, hermite_i8.   LOST: dmd_i8 partial (~1.5k/10k, no checkpoint).
#
# SIGN FIX STATUS (2026-06-10): ALREADY APPLIED AND COMMITTED at HEAD
# (hermite.py/tree.py commit 0c94ce0, bench_dit.py taylor commit 3ecd13d). The CHECK
# block below still verifies it. bench_dit.py also now checkpoints every 1k images
# (<cell>.partial.npz, auto-resumed), so a killed cell loses at most 1k images.
#
# Cells below: the 2 never-run cells + the 4 sign-fixed re-runs (suffix _fix keeps the
# buggy rows distinguishable in RESULTS_DIT.md; the paper table reports the _fix rows
# as "HiCache (corrected)" and keeps the shipped rows as "HiCache (as released)").
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/krishi/workspace/gaussianfeels/gim_env/bin/python
RES="$HERE/results"
run(){ CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 "$PY" "$HERE/bench_dit.py" \
  --method "$1" --interval "$2" --n 10000 --batch 64 --steps 250 \
  --cfg-scale 1.5 --seed 0 --gpu 0 --out "$3"; }

echo "=== resume: remaining original cells ==="
[ -f "$RES/fid10k/dmd_i8.npz" ]    || run dmd    8 "$RES/fid10k"
[ -f "$RES/fid10k/taylor_i4.npz" ] || run taylor 4 "$RES/fid10k"   # run AFTER taylor sign-fix

echo "=== sign-fixed re-runs (require signfix.patch applied) ==="
python3 - <<'CHECK'
import sys, pathlib
src = pathlib.Path(__file__).resolve()
herm = pathlib.Path("/home/krishi/workspace/gaussianfeels/third_party/hicache-plus-plus/hicache_pp/hermite.py").read_text()
if "float(-k)" in herm:
    sys.exit("ABORT: sign fix NOT applied (hermite.py still evaluates at -k). git apply /tmp/hicache_signfix/signfix.patch first.")
print("sign fix detected, proceeding")
CHECK
# guards point at the ACTUAL output paths (cells inside fid10k_fix/ keep plain names);
# bench_dit.py itself also skips finished cells, so these are belt-and-suspenders
[ -f "$RES/fid10k_fix/hermite_i4.npz" ] || run hermite 4 "$RES/fid10k_fix"
[ -f "$RES/fid10k_fix/hermite_i6.npz" ] || run hermite 6 "$RES/fid10k_fix"
[ -f "$RES/fid10k_fix/hermite_i8.npz" ] || run hermite 8 "$RES/fid10k_fix"
[ -f "$RES/fid10k_fix/auto_i4.npz" ]    || run auto    4 "$RES/fid10k_fix"
# horizon-matched selection A/B (paper-critical: 1-step holdout failed on DiT):
[ -f "$RES/fid10k_fix/auto_i4_horizon.npz" ] || CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
  "$PY" "$HERE/bench_dit.py" --method auto --interval 4 --holdout horizon --n 10000 \
  --batch 64 --steps 250 --cfg-scale 1.5 --seed 0 --gpu 0 --out "$RES/fid10k_fix"
"$PY" "$HERE/compute_fid.py" "$RES/fid10k_fix" --ref "$RES/ref_stats_imagenet256_10k_pytorchfid.npz" || true
echo "=== RESUME QUEUE DONE ==="
