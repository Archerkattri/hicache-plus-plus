"""HiCache++ — training-free diffusion inference acceleration via velocity forecasting.

Two drop-in forecasters for a flow-matching / diffusion denoise loop. On a *skipped*
sampling step, instead of running the (expensive) network you forecast the CFG-combined
velocity from cached anchors at the recent *compute* steps:

  * **hermite** — HiCache (dual-scaled physicist's Hermite polynomial; arXiv:2508.16984).
    The polynomial basis. Generalises TaylorSeer (monomial) with bounded high-order terms.

  * **dmd** — HiCache++ (Dynamic Mode Decomposition / Prony). The EXPONENTIAL basis. A
    diffusion feature trajectory solves a near-linear feature-ODE whose exact solution
    class is a sum of (damped/oscillatory) exponentials — *not* polynomials. DMD (Schmid
    2010), the SVD-regularised generalisation of Prony's method (1795), identifies the
    linear propagator from raw velocity snapshots and advances it by eigenvalue powers,
    so it is *exact* on that class where the polynomial drifts.

The benchmark verdict is family-conditional (see README, "The domain split"): the
exponential basis wins on flow-matching 3D generators and LOSES on DiT-class denoising,
where the corrected polynomial is near-lossless. ``backend="auto"`` (holdout selection)
exists for exactly that reason; on DiT-class real features prefer the polynomial outright.

Flat-tensor velocities (e.g. Hunyuan3D DiT):     use ``hermite`` + ``dmd``.
PyTree / structured velocities (e.g. SAM3D):      use ``tree`` (tree-aware Hermite + DMD
                                                  + Adaptive-CFG).
"""
from . import hermite, dmd, tree

# flat-tensor API
from .hermite import (
    hicache_init, hicache_decide, hicache_update_derivatives, hicache_forecast,
    physicists_hermite, scaled_hermite,
)
from .dmd import dmd_forecast, dmd_update_snapshots, dmd_forecast_state, auto_forecast_state

__all__ = [
    "hermite", "dmd", "tree",
    "hicache_init", "hicache_decide", "hicache_update_derivatives", "hicache_forecast",
    "physicists_hermite", "scaled_hermite",
    "dmd_forecast", "dmd_update_snapshots", "dmd_forecast_state", "auto_forecast_state",
]
# Synced literal; pyproject.toml is the source of truth and tests/test_version.py
# asserts the two match. (importlib.metadata is NOT used on purpose: when this package
# is imported from a source tree on sys.path while an older wheel is also installed,
# the metadata lookup reports the wheel's version, not the code actually imported.)
__version__ = "1.2.0"
