# Integrations

HiCache++ is model-agnostic — the forecaster only needs the per-step (CFG-combined) velocity.
It has been wired natively (no monkey-patching; the denoise loop calls the helpers directly)
into four pipelines. Two velocity types:

- **flat tensor** velocities (Hunyuan3D DiT) → `hicache_pp.hermite` + `hicache_pp.dmd`
- **PyTree / structured** velocities (SAM3D) → `hicache_pp.tree`

| Model | Velocity | Where wired | Enable |
|---|---|---|---|
| **Hunyuan3D-2.1** | flat DiT | `hy3dshape/pipelines.py` (loop) + `hicache.py` + `hicache_dmd.py` | `pipe.enable_dmd(interval=5, first_enhance=4, history=6)` or `pipe.enable_hicache(...)` |
| **Hunyuan3D-2-mini** (deployed) | flat DiT | `manipulation/vendor/Hunyuan3D-2/hy3dgen/shapegen/pipelines.py` | worker env `MANIPULATION_H3D_ACCEL=dmd_i5` (hc_i3o2 / dmd_i4 / dmd_i5 / dmd_i6) |
| **SAM3D** (sam-3d-objects) | PyTree | `sam3d_objects/.../flow_matching/{accel,solver,model}.py` | `slat_generator.enable_dmd(...)` / `.enable_hicache(...)` |
| **Fast-SAM3D** (slat stage) | PyTree | `sam3d_objects/.../flow_matching/{accel,solver,model}.py` | `slat_generator.enable_dmd(...)` |
| **Fast-SAM3D** (SS stage) | TaylorSeer | `forecast_basis.py` (pluggable basis) | env `GF_FORECAST_BASIS=hermite` |

## The native loop pattern (flat)

On a *forecast* step, skip the network and forecast the velocity; on a *compute* step, run the
network and update the cache:

```python
state = hicache_init(num_steps=N, interval=5, first_enhance=4, backend="dmd", history=6)
for i, t in enumerate(timesteps):
    if hicache_decide(state) == "forecast":
        v = dmd_forecast_state(state) if state["backend"] == "dmd" else hicache_forecast(state)
        state["step"] += 1
    else:
        v = model(x, t, ...)                       # the expensive forward
        hicache_update_derivatives(state, v.detach())
        if state["backend"] == "dmd":
            dmd_update_snapshots(state, v.detach(), state["history"])
        state["step"] += 1
    x = scheduler.step(v, t, x)
```

The PyTree path (`hicache_pp.tree`) is identical with `hicache_forecast_tree` /
`dmd_forecast_tree` / `hicache_update_tree` / `dmd_update_snapshots_tree`, hooked at the ODE
solver's `Euler.step`.

## Tuning notes

- **Hermite**: lossless up to a modest interval (Hunyuan-2.1: i3/order-2). Higher order does
  *not* rescue bigger intervals — the polynomial ceiling.
- **DMD**: push the interval further (i5–i6) for more skip while staying lossless. `history`
  is the snapshot window (5–6); needs ≥4 *uniformly-spaced* snapshots before it engages
  (Hermite covers warm-up automatically).
- `first_enhance` always computes the first few steps (high curvature); keep it ≥ 3.
