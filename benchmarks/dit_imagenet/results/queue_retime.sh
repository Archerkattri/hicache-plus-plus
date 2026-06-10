#!/bin/bash
# Latency-only re-time (post-eigencache, n=512). FID is NOT read from these cells.
set -u
PY=../../../../gim_env/bin/python
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
OUT=results/latency_retime
$PY bench_dit.py --method dmd  --interval 4 --n 512 --batch 64 --steps 250 --cfg-scale 1.5 --seed 0 --gpu 0 --out $OUT
$PY bench_dit.py --method dmd  --interval 6 --n 512 --batch 64 --steps 250 --cfg-scale 1.5 --seed 0 --gpu 0 --out $OUT
$PY bench_dit.py --method auto --interval 4 --n 512 --batch 64 --steps 250 --cfg-scale 1.5 --seed 0 --gpu 0 --out $OUT
$PY bench_dit.py --method auto --interval 4 --holdout horizon --n 512 --batch 64 --steps 250 --cfg-scale 1.5 --seed 0 --gpu 0 --out $OUT
echo "=== RETIME QUEUE DONE ==="
