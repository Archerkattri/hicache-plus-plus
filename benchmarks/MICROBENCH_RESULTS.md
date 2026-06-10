Forecast rel. L2 error vs skip horizon H on the exponential (feature-ODE) class.
H = steps past the cached window (the reach of interval H+1). Lower is better.

### Clean trajectories (20 seeds, 64-channel, 2 modes)

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 1.51e-02 | 8.04e-02 | 2.56e-01 | 6.23e-01 | 2.34e+00 | 6.48e+00 |
| Pade / FoCa (rational) | 4.85e-02 | 1.08e-01 | 1.71e-01 | 2.38e-01 | 5.28e-01 | 1.23e+00 |
| **HiCache++ (exponential)** | 4.74e-09 | 1.42e-08 | 3.00e-08 | 5.28e-08 | 1.21e-07 | 2.21e-07 |
| **HiCache++ (auto, holdout-selected)** | 4.74e-09 | 1.42e-08 | 3.00e-08 | 5.28e-08 | 1.21e-07 | 2.21e-07 |

### + 1% snapshot noise

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 9.76e-02 | 3.67e-01 | 8.94e-01 | 1.86e+00 | 5.94e+00 | 1.55e+01 |
| Pade / FoCa (rational) | 7.10e-02 | 5.14e-01 | 1.13e+00 | 1.70e+00 | 2.49e+00 | 3.12e+00 |
| **HiCache++ (exponential)** | 2.35e-02 | 4.61e-02 | 7.86e-02 | 1.21e-01 | 2.22e-01 | 2.79e-01 |
| **HiCache++ (auto, holdout-selected)** | 2.26e-02 | 4.43e-02 | 7.59e-02 | 1.16e-01 | 2.16e-01 | 3.05e-01 |

(auto picked: {'dmd': 120})

### Drifting (non-autonomous) dynamics — why backend='auto' exists

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 4.69e-02 | 2.31e-01 | 6.64e-01 | 1.49e+00 | 5.24e+00 | 1.37e+01 |
| Pade / FoCa (rational) | 1.00e-01 | 2.21e-01 | 3.74e-01 | 5.93e-01 | 1.43e+00 | 2.47e+00 |
| **HiCache++ (exponential)** | 8.30e-03 | 3.10e-02 | 7.09e-02 | 1.29e-01 | 3.02e-01 | 5.03e-01 |
| **HiCache++ (auto, holdout-selected)** | 8.30e-03 | 3.10e-02 | 7.09e-02 | 1.29e-01 | 3.02e-01 | 5.03e-01 |

(auto picked: {'dmd': 120})

### Regime switch inside the cached window — the DMD-misfit stress

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 5.02e-01 | 2.13e+00 | 5.70e+00 | 1.23e+01 | 4.16e+01 | 1.06e+02 |
| Pade / FoCa (rational) | 1.16e-01 | 1.57e+00 | 2.04e+00 | 1.85e+00 | 2.25e+00 | 2.63e+00 |
| **HiCache++ (exponential)** | 2.27e-01 | 5.69e-01 | 8.91e-01 | 1.59e+00 | 3.60e+00 | 9.40e+00 |
| **HiCache++ (auto, holdout-selected)** | 4.00e-01 | 7.99e-01 | 1.19e+00 | 1.57e+00 | 2.33e+00 | 3.11e+00 |

(auto picked: {'hermite': 120})

### Drifting dynamics + 1% snapshot noise

| method (rel. L2 error ↓) | H=1 | H=2 | H=3 | H=4 | H=6 | H=8 |
|---|---:|---:|---:|---:|---:|---:|
| TaylorSeer (polynomial) | 1.18e-01 | 4.46e-01 | 1.21e+00 | 2.53e+00 | 8.65e+00 | 2.15e+01 |
| Pade / FoCa (rational) | 1.14e-01 | 4.58e-01 | 1.13e+00 | 1.91e+00 | 3.08e+00 | 3.49e+00 |
| **HiCache++ (exponential)** | 2.87e-02 | 6.91e-02 | 1.28e-01 | 1.96e-01 | 3.89e-01 | 5.84e-01 |
| **HiCache++ (auto, holdout-selected)** | 2.77e-02 | 6.52e-02 | 1.23e-01 | 1.89e-01 | 3.58e-01 | 5.42e-01 |

(auto picked: {'dmd': 120})

HiCache++ (DMD) is exact on the solution class, so its error stays flat as H grows;
polynomial diverges, rational (Pade/FoCa) improves but still diverges and goes fragile.

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

Provenance of the tables in this file: the five scenario tables ABOVE this section
were generated with the shipped (-k) Hermite (they predate the fix; only rows where
`auto` serves the Hermite arm are affected, i.e. the regime-switch `auto` row). The
tables BELOW (this section and later ones) postdate the fix.

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
