# RUN COMMANDS — every figure and how to generate it

This file lists **every command** in the project and the **exact output
file(s)** each one writes. Images go to `images/`, tables/reports to `results/`.

## 0. One-time setup

```bash
pip install -r requirements.txt
```

All scripts are run from the **project root** (the folder containing this file).
Each script imports the shared code in `src/` automatically.

### Scale switch (important)

Every script honours a `SCALE` environment variable and **stamps each figure
with the scale it was actually produced at**:

* `SCALE=full` (default) — all 360 Glasser nodes. Use for the cheap weeks
  (1–6) and for the documented full-scale checkpoint.
* `SCALE=reduced` — a 60-node seizure-relevant sub-network around the onset
  zone. **Recommended for the heavy optimal-control weeks (7–12)**, which solve
  repeated optimal-control problems and are slow at full scale.

```bash
# examples
python scripts/week01_connectome.py            # full scale (default)
SCALE=reduced python scripts/week07_dpo.py      # reduced scale
```

You can run any week at either scale; the recommendations below are about speed,
not correctness.

---

## 1. Generate the normalised connectome cache (run once first)

```bash
python -m src.connectome
```
Writes: `data/brain_A_matrix.npy` (the cleaned, eigenvalue-normalised adjacency).
Reads: `data/brain_connectivity.mat` (your uploaded HCP-MMP connectome).

---

## 2. Week-by-week scripts

### Week 1 — Structural network acquisition & scale strategy
```bash
python scripts/week01_connectome.py
```
Images: `images/week01_connectome_matrix.png`,
`images/week01_degree_distribution.png`,
`images/week01_eigenvalue_spectrum.png`
Also writes: `results/week01_scale_plan.txt`

### Week 2 — Coupled-oscillator model & homeostasis baseline
```bash
python scripts/week02_homeostasis.py
```
Images: `images/week02_homeostasis_timeseries.png`,
`images/week02_homeostasis_meanfield.png`,
`images/week02_bifurcation_diagram.png`

### Week 3 — Uncontrolled seizure propagation
```bash
python scripts/week03_propagation.py
```
Images: `images/week03_propagation_heatmap.png`,
`images/week03_focus_vs_neighbours.png`,
`images/week03_connectome_dependence.png`,
`images/week03_recruitment_vs_connectivity.png`,
`images/week03_kuramoto_synchrony.png`
Also writes: `results/week03_propagation_note.txt`

### Week 4 — Unsafe-state (critical-slowing-down) detector
```bash
python scripts/week04_detector.py
```
Images: `images/week04_csd_features.png`,
`images/week04_risk_score.png`

### Week 5 — Independent detector validation & tuning
```bash
python scripts/week05_detector_validation.py
```
Images: `images/week05_roc_curve.png`,
`images/week05_leadtime_histogram.png`,
`images/week05_risk_traces.png`
Also writes: `results/week05_detector_metrics.csv`
(Note: Week 5 uses single-node ramp/stationary test trajectories, so it runs
quickly at either scale; the `SCALE` tag still appears on the figures.)

### Week 6 — Local feedback correction (Level-1 LQR)
```bash
python scripts/week06_lqr.py
```
Images: `images/week06_lqr_node_stabilisation.png`,
`images/week06_lqr_phase_space.png`,
`images/week06_lqr_effort_tradeoff.png`

### Week 7 — Optimal control via the Pontryagin adjoint (Level-2 DPO)
```bash
SCALE=reduced python scripts/week07_dpo.py
```
Images: `images/week07_dpo_gradient_check.png`,
`images/week07_dpo_convergence.png`,
`images/week07_dpo_suppression.png`,
`images/week07_dpo_control_schedule.png`

### Week 8 — Closed-loop system integration
```bash
SCALE=reduced python scripts/week08_closed_loop.py
```
Images: `images/week08_closed_loop_timeseries.png`,
`images/week08_closed_loop_risk.png`,
`images/week08_closed_loop_raster.png`,
`images/week08_detector_robustness.png`
Also writes: `results/week08_closed_loop_summary.csv`

### Week 9 — Ablation studies & clinical comparison
```bash
SCALE=reduced python scripts/week09_ablation.py
```
Images: `images/week09_targeting_ablation.png`,
`images/week09_targeting_energy.png`,
`images/week09_clinical_vs_closedloop.png`
Also writes: `results/week09_ablation_summary.csv`

### Week 10 — Literature benchmark (PI controller, Wang et al. 2016)
```bash
SCALE=reduced python scripts/week10_literature_benchmark.py
```
Images: `images/week10_pi_vs_ours_timeseries.png`,
`images/week10_pi_gain_sweep.png`,
`images/week10_benchmark_bars.png`
Also writes: `results/week10_benchmark_summary.csv`

### Week 11 — Visualisation & brain rendering
```bash
SCALE=reduced python scripts/week11_visualisation.py
```
Images: `images/week11_brain_recruitment_map.png`,
`images/week11_brain_control_map.png`,
`images/week11_focus_phase_portrait.png`,
`images/week11_connectivity_vs_control.png`

### Week 12 — Final integration, results tables & report
```bash
SCALE=reduced python scripts/week12_final_report.py
```
Images: `images/week12_headline_summary.png`,
`images/week12_results_table.png`
Also writes: `results/week12_master_results.csv`,
`results/week12_final_report.md`

---

## 3. Run everything at once

Reduced scale for the heavy weeks (fast, recommended for a first full pass):

```bash
python -m src.connectome
python scripts/week01_connectome.py
python scripts/week02_homeostasis.py
python scripts/week03_propagation.py
python scripts/week04_detector.py
python scripts/week05_detector_validation.py
python scripts/week06_lqr.py
SCALE=reduced python scripts/week07_dpo.py
SCALE=reduced python scripts/week08_closed_loop.py
SCALE=reduced python scripts/week09_ablation.py
SCALE=reduced python scripts/week10_literature_benchmark.py
SCALE=reduced python scripts/week11_visualisation.py
SCALE=reduced python scripts/week12_final_report.py
```

To also produce the full 360-node versions of the heavy weeks, re-run weeks
7–12 without the `SCALE=reduced` prefix (slower). Full-scale figures are written
with the same filenames, so move/rename the reduced-scale images first if you
want to keep both — the in-figure scale tag tells them apart.

---

## 4. Complete list of image filenames (39 figures)

```
week01_connectome_matrix.png
week01_degree_distribution.png
week01_eigenvalue_spectrum.png
week02_homeostasis_timeseries.png
week02_homeostasis_meanfield.png
week02_bifurcation_diagram.png
week03_propagation_heatmap.png
week03_focus_vs_neighbours.png
week03_connectome_dependence.png
week03_recruitment_vs_connectivity.png
week03_kuramoto_synchrony.png
week04_csd_features.png
week04_risk_score.png
week05_roc_curve.png
week05_leadtime_histogram.png
week05_risk_traces.png
week06_lqr_node_stabilisation.png
week06_lqr_phase_space.png
week06_lqr_effort_tradeoff.png
week07_dpo_gradient_check.png
week07_dpo_convergence.png
week07_dpo_suppression.png
week07_dpo_control_schedule.png
week08_closed_loop_timeseries.png
week08_closed_loop_risk.png
week08_closed_loop_raster.png
week08_detector_robustness.png
week09_targeting_ablation.png
week09_targeting_energy.png
week09_clinical_vs_closedloop.png
week10_pi_vs_ours_timeseries.png
week10_pi_gain_sweep.png
week10_benchmark_bars.png
week11_brain_recruitment_map.png
week11_brain_control_map.png
week11_focus_phase_portrait.png
week11_connectivity_vs_control.png
week12_headline_summary.png
week12_results_table.png
```

---

## 5. Revision (peer-review response) analyses

**Reproducibility note.** The closed-loop trigger is a threshold on a
noise-driven nonlinear signal, so a trial that ends near the ictal threshold
can flip outcome if the floating-point summation order changes (e.g. with a
multi-threaded BLAS). All numbers in the revised paper were produced with
single-threaded linear algebra; set the following before running anything:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

All representative-trial scripts (week03, week08-12) now use the shared
closed-loop protocol in `scripts/_common.py` (`closed_loop_kwargs`, seed
`REP_SEED = 1000`), so every single-trial figure is Monte-Carlo trial 0.

```bash
# Monte-Carlo benchmark (one condition per terminal, then combine)
SCALE=reduced python scripts/monte_carlo_baselines.py --conditions proposed
SCALE=reduced python scripts/monte_carlo_baselines.py --conditions ablation
SCALE=reduced python scripts/monte_carlo_baselines.py --conditions openloop
SCALE=reduced python scripts/monte_carlo_baselines.py --conditions pi
SCALE=reduced python scripts/monte_carlo_baselines.py --combine
python scripts/analysis_energy_significance.py --a proposed --b ablation
python scripts/analysis_energy_significance.py --a proposed --b pi

# Single-electrode (real-only) actuation             -> Table 2 column, Sec. 5.4
SCALE=reduced python scripts/analysis_real_only_actuation.py --start 0 --count 50
SCALE=reduced python scripts/analysis_real_only_actuation.py --start 50 --count 50
SCALE=reduced python scripts/analysis_real_only_actuation.py --combine

# Detector-error robustness, Monte Carlo             -> Figure 11
SCALE=reduced python scripts/analysis_robustness_mc.py --condition clean noise0.15 bias+0.20 bias-0.05 bias-0.10 bias-0.20
SCALE=reduced python scripts/analysis_robustness_mc.py --combine

# Level-2 cost-weight (alpha/R2) sensitivity         -> Figure 12
SCALE=reduced python scripts/analysis_cost_weight_sensitivity.py --rho 0.001 0.003 0.01 0.03 0.1
SCALE=reduced python scripts/analysis_cost_weight_sensitivity.py --combine

# Out-of-distribution sweep (one axis per terminal)  -> Figure 18
SCALE=reduced python scripts/analysis_ood_test.py --axis G_COUPLING
SCALE=reduced python scripts/analysis_ood_test.py --axis OMEGA_HZ
SCALE=reduced python scripts/analysis_ood_test.py --axis A_SEIZURE
SCALE=reduced python scripts/analysis_ood_test.py --combine

# Degree-preserving surrogates, recruitment, Kuramoto -> Sec. 4.2, Figure 4
SCALE=reduced python scripts/analysis_surrogates.py

# Open-loop amplitude sweep, both scenarios          -> Table 5
SCALE=reduced python scripts/analysis_openloop_amplitude_mc.py

# Detector feature-weight ablation                   -> Table 4
python scripts/analysis_detector_weights.py

# Real/imaginary energy split and FPR-aware daily energy (read the MC CSVs)
SCALE=reduced python scripts/analysis_complex_energy.py
SCALE=reduced python scripts/analysis_fpr_energy.py
```

Images: `analysis_robustness_mc.png`, `analysis_cost_weight_sensitivity.png`,
`analysis_real_only_actuation.png`, `analysis_surrogates.png`,
`analysis_openloop_amplitude_mc.png`, `analysis_detector_weights.png`.
See `REVISION_CHANGES.md` for what changed and why.
