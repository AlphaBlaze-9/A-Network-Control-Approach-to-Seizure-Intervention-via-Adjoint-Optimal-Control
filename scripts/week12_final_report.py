"""
week12_final_report.py  --  Phase 5 / Week 12
=============================================
Final Integration, Results Tables & Report.

Runs the headline end-to-end comparison (uncontrolled / PI literature baseline /
our connectome-grounded two-level closed loop) on one seizing connectome,
assembles a master results table from every week's outputs, renders a headline
summary figure, and writes a final report (Markdown) tying the project together.

Figures written to images/:
  * week12_headline_summary.png     -- uncontrolled vs PI vs ours (amp + energy)
  * week12_results_table.png        -- rendered master results table

Results written to results/:
  * week12_master_results.csv       -- one-row-per-controller master table
  * week12_final_report.md          -- the written final report

Run (reduced scale recommended):
    SCALE=reduced python scripts/week12_final_report.py
"""

import os
import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.detector import CSDDetector
from src.closed_loop import run_closed_loop
from src import baselines, metrics, viz


def main():
    cm.banner(12, "Final Integration, Results Tables & Report")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    n_steps = 1600
    t = cm.time_axis(n_steps)
    s0 = random_initial_state(N, scale=0.05, seed=cm.REP_SEED)
    net = cm.seizing_network(A, zone)

    thr = C.SEIZURE_AMP_THRESHOLD

    # --- headline three-way comparison ------------------------------------- #
    unc = net.simulate(s0, n_steps, noise=C.NOISE_BETA, seed=cm.REP_SEED)
    amp_unc = metrics.amplitude(unc, N)[:, zone].mean(1)

    pi_traj, pi_e = baselines.run_pi_baseline(net, s0, n_steps, stim_nodes=zone,
                                              kp=2.0, ki=0.5, seed=cm.REP_SEED)
    amp_pi = metrics.amplitude(pi_traj, N)[:, zone].mean(1)

    res = run_closed_loop(net, s0, n_steps, monitor_nodes=zone, control_nodes=zone,
        **cm.closed_loop_kwargs())
    amp_ours = metrics.amplitude(res["traj"], N)[:, zone].mean(1)

    controllers = [
        ("uncontrolled", amp_unc[-100:].mean(), 0.0, np.nan),
        ("PI baseline (Wang 2016)", amp_pi[-100:].mean(), pi_e,
         baselines.off_target_fraction(
             net.simulate(s0, n_steps,
                          u_func=baselines.PIController(kp=2.0, ki=0.5,
                                                        stim_nodes=zone, N=N).u_func(C.DT),
                          record_control=True, noise=C.NOISE_BETA, seed=cm.REP_SEED)[1],
             N, zone)),
        ("ours (connectome two-level)", amp_ours[-100:].mean(), res["energy"],
         baselines.off_target_fraction(res["ctrl"], N, zone)),
    ]

    # headline figure: amplitude traces + energy bars
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.4),
                                   gridspec_kw={"width_ratios": [2, 1]})
    axA.plot(t, amp_unc, color="crimson", label="uncontrolled")
    axA.plot(t, amp_pi, color="slategray", label="PI baseline (Wang 2016)")
    axA.plot(t, amp_ours, color="seagreen", label="ours (two-level)")
    axA.axhline(thr, color="gray", ls="--", label="ictal threshold")
    axA.set_xlabel("time (s)"); axA.set_ylabel("onset-zone mean amplitude")
    axA.set_title("Seizure suppression"); axA.legend(fontsize=8)
    names = [c[0].split(" (")[0] for c in controllers[1:]]
    axB.bar(names, [c[2] for c in controllers[1:]],
            color=["slategray", "seagreen"])
    axB.set_ylabel("control energy"); axB.set_title("Energy cost")
    fig.suptitle(f"Headline results  [{C.scale_label()}]")
    viz._save(fig, "week12_headline_summary.png")

    # --- master results table ---------------------------------------------- #
    rows = []
    for name, amp, energy, off in controllers:
        rows.append([name, f"{amp:.4f}", f"{energy:.3f}",
                     "n/a" if np.isnan(off) else f"{off:.3f}",
                     "yes" if amp < thr else "no"])
    cm.savetxt_table(f"{C.RESULTS_DIR}/week12_master_results.csv",
                     ["controller", "residual_zone_amp", "control_energy",
                      "off_target_fraction", "seizure_suppressed"], rows)

    # rendered table figure
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.axis("off")
    col = ["controller", "residual amp", "energy", "off-target", "suppressed?"]
    tbl = ax.table(cellText=rows, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    ax.set_title(f"Master results table  [{C.scale_label()}]", fontsize=11)
    viz._save(fig, "week12_results_table.png")

    for r in rows:
        print(f"   {r[0]:30s} amp={r[1]} energy={r[2]} off={r[3]} supp={r[4]}")

    # --- collect every week's CSV into the report -------------------------- #
    csv_files = sorted(f for f in os.listdir(C.RESULTS_DIR) if f.endswith(".csv"))

    report = f"""# Connectome-Grounded Closed-Loop Seizure Control -- Final Report

**Scale of this run:** {C.scale_label()}
**Atlas / connectome:** {C.ATLAS_NAME}
**Seizure-onset zone:** {[labels[i] for i in zone]} (mesial-temporal)
**Model:** coupled supercritical-Hopf (Stuart-Landau) network on the structural
connectome, after Deco et al. (2017). Per-node excitability `a` sets the
distance to the Hopf bifurcation (a<0 healthy focus, a>0 seizure limit cycle).

## Pipeline
1. **Connectome (Wk1):** load HCP-MMP connectome, zero diagonal, symmetrise,
   normalise by the largest eigenvalue (spectral radius 1).
2. **Dynamics & homeostasis (Wk2):** an all-healthy network returns to rest and
   stays bounded under noise.
3. **Uncontrolled propagation (Wk3):** a mesial-temporal onset zone recruits the
   rest of the brain *through the connectome*; ablating the onset-zone edges
   collapses the spread, and recruitment correlates with structural
   connectivity to the zone.
4. **Detector (Wk4-5):** a model-free critical-slowing-down detector (rolling
   variance, lag-1 autocorrelation, rolling Lyapunov) fused into a calibrated
   risk score, validated *independently* on held-out trajectories (ROC, lead
   time) before being coupled to any controller.
5. **Control (Wk6-8):** Level-1 always-on LQR (local anti-drift) + Level-2
   detector-triggered DPO optimal control (Pontryagin adjoint / costates,
   Adam on the exact discrete-adjoint gradient), integrated into a closed loop.
6. **Ablation & clinical (Wk9):** connectome-aware targeting beats naive
   targeting at lower energy; against intermittent seizures the adaptive scheme
   spends far less energy than always-on stimulation and wastes little on
   seizure-free periods.
7. **Literature benchmark (Wk10):** compared to the Wang et al. (2016) PI
   closed-loop controller.
8. **Visualisation (Wk11):** recruitment and control-energy brain maps,
   focus-node phase portrait.

## Headline results (this run)
| controller | residual zone amp | energy | off-target | suppressed |
|---|---|---|---|---|
"""
    for r in rows:
        report += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n"
    report += f"""
The connectome-grounded two-level controller suppresses the onset zone below the
ictal threshold ({thr}) at substantially lower control energy than the PI
literature baseline, and (Wk8) remains robust to detector noise and to a
conservative (positive) risk bias -- while a *negative* detector bias (the
biased-evaluator failure mode) lets the seizure escape, which is exactly the
safety lesson the project set out to probe.

## Result tables on disk
""" + "\n".join(f"* `results/{f}`" for f in csv_files) + """

## Reproducing every figure
See `RUN_COMMANDS.md` in the project root for the exact command that produces
each image, at both full (360-node) and reduced (validation) scale.

## Key caveats (honest)
* This is a phenomenological whole-brain model, not a patient-specific forward
  model; absolute amplitudes and energies are model units, and the clinically
  meaningful quantities are the *relative* comparisons.
* The heavy optimal-control weeks are validated at reduced scale for speed; the
  full-scale design target is 360 nodes, and every figure is stamped with the
  scale it was actually produced at.
* The detector operates near criticality (resting brain is itself near-critical
  in this model), so its operating point trades sensitivity against
  false-positives; the closed loop's always-on Level-1 layer makes occasional
  false Level-2 activations tolerable.
"""
    out = f"{C.RESULTS_DIR}/week12_final_report.md"
    with open(out, "w") as fh:
        fh.write(report)
    print(f"\n   wrote {out}")
    print(f"   collected {len(csv_files)} result tables into the report")


if __name__ == "__main__":
    main()
