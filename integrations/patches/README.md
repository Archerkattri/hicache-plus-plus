# Integration patches

Reproducible `git apply`-able patches that wire HiCache++ into each model's denoise /
solver loop. The core forecaster files the patches reference are this repo's `hicache_pp/`
modules — copy those in, then apply the patch.

| Patch | Target repo / file | Core to copy in | Forecaster |
|---|---|---|---|
| `hunyuan3d-2.1-loop.patch` | Hunyuan3D-2.1 `hy3dshape/hy3dshape/pipelines.py` | `hermite.py`→`hicache.py`, `dmd.py`→`hicache_dmd.py` | flat (`hicache_pp.dmd`) |
| `hunyuan3d-2-mini-worker.patch` | mini subprocess worker (`h3d_runner/worker_protocol.py`); apply the **same** loop wiring as 2.1 to the vendored `hy3dgen/shapegen/pipelines.py` | hermite.py/dmd.py into `hy3dgen/shapegen/` | flat |
| `sam3d-flow-matching.patch` | sam-3d-objects `.../flow_matching/{model,solver}.py` | `tree.py`→`accel.py` | PyTree (`hicache_pp.tree`) |
| `fast-sam3d-slat.patch` | Fast-SAM3D `.../flow_matching/{model,solver}.py` (slat stage) | `tree.py`→`accel.py` | PyTree |

Each patch adds `enable_hicache(...)` / `enable_dmd(...)` and the forecast/update branches
in the per-step loop (skip the network on forecast steps; cache the CFG-combined velocity
on compute steps). The mini worker additionally reads `MANIPULATION_H3D_ACCEL`
(`hc_i3o2` / `dmd_i4` / `dmd_i5` / `dmd_i6`).

A/B-verified results for every integration are in [`../../results/RESULTS.md`](../../results/RESULTS.md):
the 6-cell matrix (HiCache + HiCache++ across Hunyuan3D, SAM3D, Fast-SAM3D) is all green —
DMD lossless at interval-5 on Hunyuan (breaks the polynomial ceiling) and lossless to
interval-6 on the SAM3D slat stage (1.56×).
