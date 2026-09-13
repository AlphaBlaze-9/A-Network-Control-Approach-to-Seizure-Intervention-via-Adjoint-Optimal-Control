# Code changes made for the revision (peer-review response)

All changes are additive or behaviour-preserving by default. Every number and
figure in the revised manuscript is produced by a script in this repository.

## Library (`src/`)

* `lqr_control.py` -- `LocalLQR(..., real_only=False)`: with `real_only=True`
  the regulator is designed for a single-input actuator (B = [1, 0]^T, i.e. the
  real component of z only). Default unchanged.
* `dpo_control.py` -- `DPOController(..., real_only=False)`: with
  `real_only=True` the actuation mask is zero on the imaginary half. Default
  unchanged.
* `closed_loop.py` -- `run_closed_loop(..., real_only=False)` forwards the flag
  to both levels. Default unchanged.
* `config.py` -- daily-energy accounting constants: `DETECTOR_FPR = None`
  (read the measured test-fold FPR), `ASSUMED_SEIZURES_DAY = 8`,
  `DETECTOR_RECORD_S = 120`, `DETECTOR_EVALS_DAY = 720`.

## Shared protocol (`scripts/_common.py`)

* `REP_SEED = config.MC_SEED0` (1000) and `CL_N_STEPS = 1600`.
* `closed_loop_kwargs(**overrides)` returns the closed-loop settings used by
  the Monte-Carlo scripts (detector window 100, threshold 0.5, horizon 200,
  50 Adam iterations, rho 0.01, q 1.0, L1 gain 0.25, refractory 20, poll
  stride 20, plant noise 0.02, U_MAX 0.5). Every representative-trial script
  (week03, week08-12, umax sweep) now uses it with `seed=REP_SEED`, so each
  single-trial figure is Monte-Carlo trial 0 under the identical protocol.

## Monte-Carlo benchmark (`scripts/monte_carlo_baselines.py`, rewritten)

* All four conditions run on the same 80-s continuously seizure-prone
  scenario (the open-loop surrogate previously used a 120-s intermittent
  scenario, which made its energy non-comparable).
* Per-trial CSVs gain `n_interventions` and `real_frac` (share of control
  energy in the real component). `--combine` also refreshes
  `results/monte_carlo_summary.csv`.

## New analysis scripts (`scripts/`)

| script | answers | outputs |
|---|---|---|
| `analysis_detector_weights.py` | R2 minor 1 (feature-weight ablation) | `results/analysis_detector_weights.csv`, `images/analysis_detector_weights.png` |
| `analysis_cost_weight_sensitivity.py` | R2 minor 2 (alpha/R2 sensitivity) | `results/analysis_cost_weight_sensitivity.csv`, `images/analysis_cost_weight_sensitivity.png` |
| `analysis_robustness_mc.py` | R3 9 (Monte-Carlo robustness to detector error) | `results/analysis_robustness_mc.csv`, `images/analysis_robustness_mc.png` |
| `analysis_real_only_actuation.py` | R2 major 4 (single-electrode actuation) | `results/analysis_real_only_actuation.csv`, `results/mc_realonly_per_trial.csv`, `images/analysis_real_only_actuation.png` |
| `analysis_surrogates.py` | Sec. 3.4/4.2 (Maslov-Sneppen surrogates, recruitment, Kuramoto R) | `results/analysis_surrogates.csv`, `images/analysis_surrogates.png` |
| `analysis_openloop_amplitude_mc.py` | R3 3 (open-loop amplitude sweep, both scenarios, Monte Carlo) | `results/analysis_openloop_amplitude_mc.csv`, `images/analysis_openloop_amplitude_mc.png` |

## Modified analysis scripts

* `analysis_ood_test.py` -- `--axis` / `--combine` so the three axes can run
  in parallel; uses the shared protocol.
* `analysis_complex_energy.py` -- reads the real-energy fraction from the
  Monte-Carlo per-trial CSV instead of re-running 100 deterministic trials.
* `analysis_fpr_energy.py` -- daily accounting with the measured per-record
  FPR, 120-s records, energy per Level-2 episode from the Monte-Carlo CSV,
  and the analytic open-loop daily energy (amp^2 x |zone| x 86400).
* `analysis_energy_significance.py` -- merges one row per comparison into the
  summary CSV instead of overwriting it.

## Run order used for the revision

```
python -m src.connectome
python scripts/week01_connectome.py                       # full scale
python scripts/week05_detector_validation.py
SCALE=reduced python scripts/week02_homeostasis.py ... week07_dpo.py
SCALE=reduced python scripts/monte_carlo_baselines.py --conditions proposed|ablation|openloop|pi
SCALE=reduced python scripts/monte_carlo_baselines.py --combine
SCALE=reduced python scripts/week08_closed_loop.py ... week12_final_report.py
SCALE=reduced python scripts/analysis_real_only_actuation.py --start 0 --count 50 / --start 50 --count 50 / --combine
SCALE=reduced python scripts/analysis_robustness_mc.py --condition <each> / --combine
SCALE=reduced python scripts/analysis_cost_weight_sensitivity.py --rho <each> / --combine
SCALE=reduced python scripts/analysis_ood_test.py --axis <each> / --combine
SCALE=reduced python scripts/analysis_surrogates.py
SCALE=reduced python scripts/analysis_openloop_amplitude_mc.py
python scripts/analysis_detector_weights.py
SCALE=reduced python scripts/analysis_umax_sweep.py, analysis_refractory.py, analysis_step_size.py
SCALE=reduced python scripts/analysis_scale_timing.py ; SCALE=full python scripts/analysis_scale_timing.py
SCALE=reduced python scripts/analysis_complex_energy.py, analysis_fpr_energy.py
python scripts/analysis_energy_significance.py --a proposed --b ablation ; --a proposed --b pi
python scripts/record_hardware.py
```
