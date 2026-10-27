# Voltage-Only FAS-MPC for Standalone LC-Filtered Inverters under Nonlinear Loads

Supplementary Material and code for the brief by Junxiang Zhang, Zhong Chu, Weijie Ren, and Guangren Duan.

## Contents

| Path | Description |
|---|---|
| `Supplementary_Material.pdf` | S1 derivations, S2 simulation models and metrics, S3 baseline implementations, S4 tuning protocol, S5 linear closed loop for the stability test, S6 extended results, S7 reproducibility |
| `code/` | Python scripts (run them from this folder) |
| `results/` | Result files behind every number of the brief and the Supplementary Material |
| `supp/` | Output folder of `supp_tables.py` |

`figures.py` writes its figures to `paper/fig/`.

## Requirements

Python 3.13, NumPy 2.1, SciPy 1.15, Numba 0.61, OSQP 1.0.5, Matplotlib 3.10.

## Scripts

| Script | Purpose |
|---|---|
| `robust_tune.py`, `select_params.py`, `sweep.py` | Tuning protocol -> `results/final_params*.json`, `results/sweep_*.json` |
| `experiments.py` | All reported runs -> `results/exp_all.json`, `results/waveforms.npz` |
| `summarize.py` | -> `results/summary.json` |
| `analysis.py`, `box_cert.py`, `small_gain.py`, `tube_check.py` | Stability test, small-gain test, tube check, QP check |
| `lc_extra.py`, `lc_fb.py` | Worst-case grids over L_f/L_0 and C_f/C_0 |
| `figures.py`, `supp_tables.py` | Figures and tables |
| `plant.py`, `sim.py`, `lti_scan.py` | Plant models, controllers and metrics, LTI scan |

All random seeds are fixed. With the included `results/` files, `figures.py` and `supp_tables.py` can be run directly.
