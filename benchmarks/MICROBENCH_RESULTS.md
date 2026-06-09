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
