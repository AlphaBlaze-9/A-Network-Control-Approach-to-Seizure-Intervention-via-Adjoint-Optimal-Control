# Connectome-Grounded Closed-Loop Seizure Control

A 12-week computational-neuroscience research project that builds, validates, and
benchmarks a **closed-loop controller that suppresses epileptic seizures on a
whole-brain structural connectome**. A mesial-temporal onset zone is driven past
a Hopf bifurcation; a critical-slowing-down detector raises an early warning;
and a two-level controller (always-on local LQR + triggered optimal control via
the Pontryagin adjoint) steers the network back to the healthy resting state.

> **Quick start:** install dependencies, then follow **`RUN_COMMANDS.md`**, which
> lists every command and the exact image file(s) each one produces.

```bash
pip install -r requirements.txt
python -m src.connectome                 # build the normalised connectome cache
python scripts/week01_connectome.py      # ... then run the week scripts
```

## The model

Each of the 360 Glasser (HCP-MMP) cortical regions is a **supercritical Hopf
(Stuart–Landau) oscillator**, coupled diffusively through the structural
connectome, following Deco et al. (2017):

```
r2_i = x_i^2 + y_i^2
dx_i = (a_i - r2_i) x_i - w_i y_i + G * sum_j A_ij (x_j - x_i) + u_i^x
dy_i = (a_i - r2_i) y_i + w_i x_i + G * sum_j A_ij (y_j - y_i) + u_i^y
```

The per-node bifurcation parameter `a_i` is the **excitability**: `a_i < 0` is a
stable (healthy, interictal) focus; `a_i > 0` is a limit cycle (a seizure) of
radius `sqrt(a_i)`. The supercritical Hopf bifurcation at `a_i = 0` is the
"edge" the detector watches and the controller keeps the system away from.

## What each phase does

| Phase | Weeks | Content |
|---|---|---|
| 1 | 1–3 | Connectome loading + normalisation; coupled-Hopf dynamics; homeostasis; uncontrolled connectome-mediated seizure propagation |
| 2 | 4–5 | Model-free critical-slowing-down detector; **independent** held-out validation (ROC, lead time) before it touches a controller |
| 3 | 6–8 | Level-1 LQR; Level-2 Pontryagin/adjoint optimal control (DPO); full closed loop with detector-robustness probes |
| 4 | 9–10 | Ablations (connectome-aware vs naive targeting; adaptive vs always-on); PI literature benchmark (Wang et al. 2016) |
| 5 | 11–12 | Brain rendering & phase portraits; results tables and final report |

## Project layout

```
seizure_control_project/
├── README.md                ← you are here
├── RUN_COMMANDS.md          ← every command + the exact image it produces
├── requirements.txt
├── data/
│   ├── brain_connectivity.mat   ← HCP-MMP structural connectome (input)
│   └── brain_A_matrix.npy       ← cleaned/normalised A (built by src.connectome)
├── src/                     ← the reusable library (imported by every script)
│   ├── config.py                ← paths, constants, scale strategy
│   ├── connectome.py            ← load / clean / normalise / reduce the network
│   ├── hopf_model.py            ← coupled-Hopf dynamics, RK4 + noise, analytic Jᵀλ
│   ├── metrics.py               ← amplitude, phase, Kuramoto, seizure load
│   ├── detector.py              ← critical-slowing-down detector
│   ├── lqr_control.py           ← per-node LQR (Level-1)
│   ├── dpo_control.py           ← Pontryagin-adjoint optimal control (Level-2)
│   ├── closed_loop.py           ← detector-triggered two-level closed loop
│   ├── baselines.py             ← always-on open-loop + PI literature baseline
│   └── viz.py                   ← shared plotting helpers
├── scripts/                 ← one driver per week (week01 … week12) + _common.py
├── images/                  ← all figures are written here (empty until you run)
└── results/                 ← CSV tables + text/markdown reports (built on run)
```

## Honest scale strategy

The design target is the full **360-node** network. Because the optimal-control
weeks solve repeated finite-horizon problems, they are validated on a dense
**60-node** sub-network around the onset zone for speed. This is made
mechanical and transparent:

* a single `SCALE` environment variable switches scale (`full` / `reduced`);
* **every figure title is stamped with the scale it was actually produced at**,
  so a reduced-scale result is never mistaken for a 360-node result.

See `RUN_COMMANDS.md` for the recommended scale per week.

## Key references

* Deco, Kringelbach, Jirsa & Ritter (2017), *The dynamics of resting fluctuations
  in the brain: metastability and its dynamical cortical core*, Sci. Rep. 7:3095
  — the whole-brain Hopf model.
* Wang, Niebur, Hu & Li (2016), *Suppressing epileptic activity in a neural mass
  model using a closed-loop proportional-integral controller*, Sci. Rep. 6:27344
  — the Week-10 literature baseline.
* Glasser et al. (2016), *A multi-modal parcellation of human cerebral cortex*,
  Nature 536:171 — the HCP-MMP atlas used for the connectome.

## Caveats

This is a phenomenological whole-brain model, not a patient-specific forward
model. Amplitudes and energies are in model units; the meaningful quantities are
the **relative** comparisons (suppression vs energy, connectome-aware vs naive,
ours vs the PI baseline). See `results/week12_final_report.md` after running
Week 12 for the assembled results and a fuller discussion.

## Revision note (peer review)

The scripts and results in this repository were regenerated for the revised
manuscript. `REVISION_CHANGES.md` lists every code change and the new analysis
scripts; `RUN_COMMANDS.md` (section 5) lists the commands. Run everything with
single-threaded linear algebra (`OMP_NUM_THREADS=1`) to reproduce the reported
per-trial outcomes exactly.
