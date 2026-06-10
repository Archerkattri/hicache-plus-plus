Forecast rel. L2 error vs skip horizon H on the exponential (feature-ODE) class.
H = steps past the cached window (the reach of interval H+1). Lower is better.

### Clean trajectories (20 seeds, 64-channel, 2 modes)

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 1.51e-02 | 8.04e-02 | 2.56e-01 | 6.23e-01 | 2.34e+00 | 6.48e+00 |
| Pade / FoCa (rational) | 4.85e-02 | 1.08e-01 | 1.71e-01 | 2.38e-01 | 5.28e-01 | 1.23e+00 |
| HiCache (scaled-Hermite, fixed +k) | 1.49e-01 | 2.98e-01 | 4.59e-01 | 6.30e-01 | 1.05e+00 | 1.72e+00 |
| **HiCache++ (exponential)** | 4.74e-09 | 1.42e-08 | 3.00e-08 | 5.28e-08 | 1.21e-07 | 2.21e-07 |
| **HiCache++ (auto, 1step holdout)** | 4.74e-09 | 1.42e-08 | 3.00e-08 | 5.28e-08 | 1.21e-07 | 2.21e-07 |
| **HiCache++ (auto, horizon holdout)** | 4.74e-09 | 1.42e-08 | 3.00e-08 | 5.28e-08 | 1.21e-07 | 2.21e-07 |

### + 1% snapshot noise

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 9.78e-02 | 3.66e-01 | 9.03e-01 | 1.90e+00 | 6.01e+00 | 1.53e+01 |
| Pade / FoCa (rational) | 7.35e-02 | 4.63e-01 | 1.19e+00 | 1.71e+00 | 2.50e+00 | 3.25e+00 |
| HiCache (scaled-Hermite, fixed +k) | 1.50e-01 | 3.00e-01 | 4.65e-01 | 6.37e-01 | 1.09e+00 | 1.81e+00 |
| **HiCache++ (exponential)** | 2.27e-02 | 4.54e-02 | 7.50e-02 | 1.18e-01 | 2.06e-01 | 3.01e-01 |
| **HiCache++ (auto, 1step holdout)** | 2.21e-02 | 4.54e-02 | 7.35e-02 | 1.16e-01 | 2.01e-01 | 3.02e-01 |
| **HiCache++ (auto, horizon holdout)** | 2.21e-02 | 4.54e-02 | 7.35e-02 | 2.38e-01 | 3.60e-01 | 5.22e-01 |

(auto picked: {'1step': {'dmd': 120}, 'horizon': {'dmd': 111, 'hermite': 9}})

### Drifting (non-autonomous) dynamics — why backend='auto' exists

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 4.69e-02 | 2.31e-01 | 6.64e-01 | 1.49e+00 | 5.24e+00 | 1.37e+01 |
| Pade / FoCa (rational) | 1.00e-01 | 2.21e-01 | 3.74e-01 | 5.93e-01 | 1.43e+00 | 2.47e+00 |
| HiCache (scaled-Hermite, fixed +k) | 2.50e-01 | 4.89e-01 | 7.45e-01 | 1.04e+00 | 1.89e+00 | 3.23e+00 |
| **HiCache++ (exponential)** | 8.30e-03 | 3.10e-02 | 7.09e-02 | 1.29e-01 | 3.02e-01 | 5.03e-01 |
| **HiCache++ (auto, 1step holdout)** | 8.30e-03 | 3.10e-02 | 7.09e-02 | 1.29e-01 | 3.02e-01 | 5.03e-01 |
| **HiCache++ (auto, horizon holdout)** | 8.30e-03 | 3.10e-02 | 7.09e-02 | 1.29e-01 | 3.02e-01 | 5.03e-01 |

(auto picked: {'1step': {'dmd': 120}, 'horizon': {'dmd': 120}})

### Regime switch inside the cached window — the DMD-misfit stress

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 5.02e-01 | 2.13e+00 | 5.70e+00 | 1.23e+01 | 4.16e+01 | 1.06e+02 |
| Pade / FoCa (rational) | 1.16e-01 | 1.57e+00 | 2.04e+00 | 1.85e+00 | 2.25e+00 | 2.63e+00 |
| HiCache (scaled-Hermite, fixed +k) | 1.38e-01 | 2.66e-01 | 3.94e-01 | 5.33e-01 | 9.21e-01 | 1.56e+00 |
| **HiCache++ (exponential)** | 2.27e-01 | 5.69e-01 | 8.91e-01 | 1.59e+00 | 3.60e+00 | 9.40e+00 |
| **HiCache++ (auto, 1step holdout)** | 1.38e-01 | 2.66e-01 | 3.94e-01 | 5.33e-01 | 9.21e-01 | 1.56e+00 |
| **HiCache++ (auto, horizon holdout)** | 1.47e-01 | 2.36e-01 | 3.54e-01 | 5.30e-01 | 3.07e+00 | 7.97e+00 |

(auto picked: {'1step': {'hermite': 120}, 'horizon': {'hermite': 84, 'dmd': 36}})

### Drifting dynamics + 1% snapshot noise

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 1.13e-01 | 4.64e-01 | 1.17e+00 | 2.55e+00 | 8.38e+00 | 2.16e+01 |
| Pade / FoCa (rational) | 1.17e-01 | 5.13e-01 | 1.29e+00 | 1.87e+00 | 2.95e+00 | 3.77e+00 |
| HiCache (scaled-Hermite, fixed +k) | 2.50e-01 | 4.89e-01 | 7.51e-01 | 1.04e+00 | 1.92e+00 | 3.29e+00 |
| **HiCache++ (exponential)** | 2.81e-02 | 6.62e-02 | 1.29e-01 | 2.02e-01 | 4.09e-01 | 6.01e-01 |
| **HiCache++ (auto, 1step holdout)** | 2.77e-02 | 6.20e-02 | 1.20e-01 | 1.83e-01 | 3.88e-01 | 5.45e-01 |
| **HiCache++ (auto, horizon holdout)** | 2.77e-02 | 6.20e-02 | 1.20e-01 | 4.77e-01 | 9.92e-01 | 8.49e-01 |

(auto picked: {'1step': {'dmd': 120}, 'horizon': {'dmd': 105, 'hermite': 15}})

### Oscillatory-with-trend -- the holdout-misprediction regime (1-step ranking inverts at multi-step)

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 1.02e+00 | 2.20e+00 | 2.27e+00 | 9.30e-01 | 1.34e+01 | 4.82e+01 |
| Pade / FoCa (rational) | 3.85e-01 | 8.19e-01 | 5.33e-01 | 5.19e-01 | 3.95e-01 | 8.03e-01 |
| HiCache (scaled-Hermite, fixed +k) | 1.21e-01 | 2.02e-01 | 4.17e-01 | 5.84e-01 | 1.72e+00 | 3.53e+00 |
| **HiCache++ (exponential)** | 2.36e-01 | 4.61e-01 | 7.69e-01 | 1.09e+00 | 1.98e+00 | 3.20e+00 |
| **HiCache++ (auto, 1step holdout)** | 2.23e-01 | 2.02e-01 | 4.17e-01 | 8.77e-01 | 1.96e+00 | 3.19e+00 |
| **HiCache++ (auto, horizon holdout)** | 1.25e-01 | 4.61e-01 | 7.69e-01 | 5.84e-01 | 1.83e+00 | 3.20e+00 |

(auto picked: {'1step': {'dmd': 68, 'hermite': 52}, 'horizon': {'hermite': 53, 'dmd': 67}})

HiCache++ (DMD) is exact on the solution class, so its error stays flat as H grows;
polynomial diverges, rational (Pade/FoCa) improves but still diverges and goes fragile.

Determinism note: the noise rows are seeded as of 2026-06-10 (the script previously
drew snapshot noise from the unseeded global RNG, so pre-2026-06-10 noise rows vary
by a few percent run to run; rankings were never affected). All tables in this file
are now exactly reproducible by `python benchmarks/forecast_microbench.py`.

### Holdout decision (2026-06-10, decided on this evidence)

`backend="auto"` selects DMD vs the Hermite arm by backcasting held-out snapshots.
Two modes were benchmarked above:

- **`holdout="1step"` (DEFAULT, unchanged):** 1-gap backcast of the newest held-out
  snapshot against a degree-2 polynomial proxy. Low variance, cannot see multi-gap
  divergence.
- **`holdout="horizon"` (NEW, opt-in):** backcast at the actual skip distance of the
  window (h ~ (interval-1)/spacing gaps) against the SERVED damped-Hermite arm. For
  h >= 4 the DMD fit uses the newest h snapshots and backcasts the snapshot h gaps
  older (fresh fit, backward extrapolation); for h < 4 it degrades to a forward
  prefix backcast at distance h.

Why horizon exists: on DiT-XL/2 FID-10k the 1-step backcast picked DMD while
multi-step reality favored Hermite (auto_i4 drift 18.08 vs hermite_i4 10.57). The
oscillatory-with-trend scenario reconstructs that inversion synthetically: at the
matched distance H=4 the horizon holdout picks the winning Hermite arm in 20/20
windows (error 5.84e-1, equal to the oracle Hermite row) where 1step picks DMD in
12/20 (error 8.77e-1). It also wins H=1 there (1.25e-1 vs 2.23e-1).

Why it is NOT the default: the single far-out backward target is higher-variance.
The same tables show horizon losing where 1step is fine: +1% noise H=4/H=8
(2.38e-1/5.22e-1 vs 1.16e-1/3.02e-1, 9/120 wrong picks), regime switch H=6/H=8
(3.07/7.97 vs 9.21e-1/1.56, 36/120 wrong picks), drift+noise H=4/H=6
(4.77e-1/9.92e-1 vs 1.83e-1/3.88e-1), and the misprediction regime's own H=2/H=3
(the h<4 prefix path inherits the dirty early window). Horizon-matched selection is
therefore not strictly better, and per the pre-registered decision rule the default
stays `1step`, with `horizon` shipped opt-in (`hicache_init(..., holdout="horizon")`)
for workloads whose failure mode is 1-step/multi-step ranking inversion.

---

## Sign-convention fix: shipped(-k) vs fixed(+k) vs reuse (2026-06-10)

The Hermite forecaster in this repo evaluated the scaled-Hermite basis at `x = -k`
(`hicache_pp/hermite.py`, `hicache_pp/tree.py::hermite_coeff`), where `k` is the number
of steps PAST the newest anchor. The finite differences are forward slopes, so the
forecast must evaluate at `x = +k` (the upstream TaylorSeer convention). Since
`Htilde_n(-x) = (-1)^n Htilde_n(x)`, every odd-order term flipped sign and the shipped
forecast extrapolated BACKWARDS by exactly the amount it should move forward. The bug
was ours, introduced in porting; it is fixed as of this section, with closed-form
regression tests (exact forward extrapolation on a linear series at `sigma = sqrt(1/2)`)
in the hermite and tree self-tests.

Provenance of the tables in this file: the scenario tables ABOVE this section were
REGENERATED after the fix (2026-06-10; they show the fixed +k Hermite, both auto
holdout modes, and the new oscillatory-with-trend scenario). The pre-fix versions of
those tables are in git history (v1.1.0 and earlier); their only fix-affected rows
were the ones where `auto` serves the Hermite arm (the regime-switch `auto` row).
The A/B tables in THIS section measure shipped(-k) against fixed(+k) directly.

A/B methodology: instrumented copies of the modules with an env-switched sign, driven
through the real state machinery; trajectory generators verbatim from
`forecast_microbench.py` (same seeds); `max_order=2`, `sigma=0.5`, 8 anchors, 20 seeds,
64-channel 2-mode float64 trajectories, CPU. Full evidence pack with per-table raw
numbers: sign-fix bench of 2026-06-10 (runner `bench_signfix.py`).

### Mean rel. L2 over horizons H=1..8 (lower is better)

| scenario | plain reuse | Hermite SHIPPED (-k) | Hermite FIXED (+k) | shipped/fixed | reuse/fixed |
|---|---:|---:|---:|---:|---:|
| clean exponential | 1.125 | 1.594 | **0.717** | 2.22x | 1.57x |
| +1% snapshot noise | 1.126 | 1.602 | **0.743** | 2.16x | 1.51x |
| drifting dynamics | 1.341 | 1.787 | **1.234** | 1.45x | 1.09x |
| regime switch | 1.040 | 1.566 | **0.635** | 2.47x | 1.64x |
| sub-step cadence (k=1..3) | 0.562 | 0.753 | **0.456** | 1.65x | 1.23x |

The shipped (-k) Hermite loses to plain reuse in every cell of every scenario: it is
strictly worse than doing nothing on this trajectory class. The fix makes it better
than reuse in every cell except drift H=8 (see honesty notes).

### Per-horizon detail (rel. L2, lower is better)

Clean exponential:

| method | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| plain reuse | 2.735e-01 | 5.614e-01 | 8.611e-01 | 1.158e+00 | 1.703e+00 | 2.193e+00 |
| Hermite SHIPPED (-k) | 4.184e-01 | 8.492e-01 | 1.275e+00 | 1.669e+00 | 2.334e+00 | 3.021e+00 |
| Hermite FIXED (+k) | **1.493e-01** | **2.981e-01** | **4.590e-01** | **6.300e-01** | **1.045e+00** | **1.718e+00** |

+1% snapshot noise:

| method | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| plain reuse | 2.740e-01 | 5.619e-01 | 8.617e-01 | 1.159e+00 | 1.704e+00 | 2.194e+00 |
| Hermite SHIPPED (-k) | 4.182e-01 | 8.491e-01 | 1.276e+00 | 1.671e+00 | 2.344e+00 | 3.056e+00 |
| Hermite FIXED (+k) | **1.509e-01** | **3.019e-01** | **4.671e-01** | **6.452e-01** | **1.086e+00** | **1.808e+00** |

Drifting (non-autonomous) dynamics:

| method | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| plain reuse | 3.476e-01 | 7.249e-01 | 1.094e+00 | 1.447e+00 | 2.053e+00 | **2.379e+00** |
| Hermite SHIPPED (-k) | 5.256e-01 | 1.047e+00 | 1.503e+00 | 1.872e+00 | 2.411e+00 | 3.365e+00 |
| Hermite FIXED (+k) | **2.111e-01** | **4.321e-01** | **6.747e-01** | **9.642e-01** | **1.827e+00** | 3.292e+00 |

Regime switch inside the cached window (the scenario where `auto` serves the Hermite arm,
so the shipped `auto` regime-switch row above embeds the (-k) numbers):

| method | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| plain reuse | 2.605e-01 | 5.196e-01 | 7.763e-01 | 1.034e+00 | 1.573e+00 | 2.078e+00 |
| Hermite SHIPPED (-k) | 4.002e-01 | 7.991e-01 | 1.188e+00 | 1.568e+00 | 2.332e+00 | 3.110e+00 |
| Hermite FIXED (+k) | **1.375e-01** | **2.656e-01** | **3.940e-01** | **5.329e-01** | **9.212e-01** | **1.556e+00** |

Sub-step cadence (anchors every 4 steps, forecast the skipped sub-steps k=1..3, the
real interval-4 cache geometry):

| method | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| plain reuse | 2.756e-01 | 5.638e-01 | 8.453e-01 |
| Hermite SHIPPED (-k) | 4.066e-01 | 7.712e-01 | 1.082e+00 |
| Hermite FIXED (+k) | **2.130e-01** | **4.427e-01** | **7.128e-01** |

### Linear-series closed form (anchors 0/4/8/12, abs L2)

| sigma | k | shipped(-k) | fixed(+k) | reuse |
|---:|---:|---:|---:|---:|
| 0.5 | 1 | 7.794e-01 | 2.598e-01 | 5.196e-01 |
| 0.5 | 2 | 1.559e+00 | 5.196e-01 | 1.039e+00 |
| 0.5 | 3 | 2.338e+00 | 7.794e-01 | 1.559e+00 |
| 0.7071 (sqrt(1/2)) | 1 | 1.039e+00 | **1.3e-15 (exact)** | 5.196e-01 |
| 0.7071 (sqrt(1/2)) | 2 | 2.078e+00 | **1.3e-15 (exact)** | 1.039e+00 |
| 0.7071 (sqrt(1/2)) | 3 | 3.118e+00 | **4.4e-16 (exact)** | 1.559e+00 |

Shipped error is exactly 2x the reuse error at `sigma = sqrt(1/2)` (the closed form);
this is the basis of the shipped regression tests.

### Sigma sweep and auto interaction

- No retune needed: `sigma = 0.5` stays optimal for the FIXED variant in all five
  scenarios. The tuned default does not encode the bug.
- For the SHIPPED variant the best sigma is 0.3 everywhere: smaller sigma shrinks the
  wrong-signed odd terms toward plain reuse, i.e. the only way to tune the buggy
  version is to mute it.
- `auto` holdout SELECTION does not change (the `_poly2_backcast` yardstick is already
  a forward +k Newton polynomial, sign-independent); `auto` output changes only where
  it serves the Hermite arm. On the regime-switch scenario that makes `auto` 2.0-2.9x
  better at every horizon.

### Honesty notes

1. Drift H=8 is the one cell where FIXED is worse than plain reuse (3.292 vs 2.379,
   1.38x): a correctly extrapolating polynomial overshoots at long horizons under
   drifting dynamics. It still beats SHIPPED there.
2. Noise did not vindicate the bug: under 1% noise FIXED beats SHIPPED 2.16x and reuse
   1.51x. The backwards term was not useful damping; it was anti-extrapolation.
3. The published DiT FID-10k hermite/auto cells (`benchmarks/dit_imagenet/RESULTS_DIT.md`)
   were measured with the SHIPPED (-k) Hermite. The fix makes Hermite genuinely
   extrapolative, which may move those FID numbers either way; the microbench cannot
   decide that. Those cells are queued for re-run; until then they stand as
   "as-released (buggy, near-reuse) Hermite" results.

---

## DMD eigendecomposition cache: per-forecast cost (CPU, 2026-06-10)

The stateful DMD paths (`dmd_forecast_state`, `dmd_forecast_tree`, the `auto` serve
path) now cache (Phi, lambda, b) per compute window and recompute only when a new
snapshot arrives; between snapshots the fit inputs cannot change, so the pre-1.2
fit-per-skip-step was pure waste. Correctness: the module self-tests assert the
cached path matches the uncached fit-per-call path to <1e-12 across scenarios,
horizons, and cache invalidations (bit-equality is unattainable because CPU LAPACK
svd/eig is run-to-run nondeterministic at ~1e-16 even for identical calls).

CPU timing (`python benchmarks/eigencache_timing.py`, history=6, interval=4 so one
fit amortizes over 3 skip-step forecasts, float32, 2 threads; uncached = the cache
key cleared before every call, reproducing the pre-1.2 behavior exactly):

| feature dim d | uncached (fit/call) | cached (fit/window) | per-forecast speedup |
|---:|---:|---:|---:|
| 8,192 | 0.852 ms | 0.323 ms | 2.6x |
| 65,536 | 3.977 ms | 1.610 ms | 2.5x |
| 262,144 | 27.855 ms | 8.946 ms | 3.1x |
| 524,288 | 57.369 ms | 20.013 ms | 2.9x |

The ceiling at interval N is ~(N-1)x when the eval cost is small next to the fit;
measured 2.5-3.1x at interval 4 is at that ceiling (the 3.1x cell is timing noise).
At larger intervals the amortization grows. GPU re-timing of the DiT dmd/auto cells
(where the RESULTS_DIT.md caveat 5 flagged the per-skip SVD overhead: dmd_i4 566
ms/img vs hermite_i4 445 ms/img at identical compute calls) happens on the resume.
