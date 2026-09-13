"""
week10_literature_benchmark.py  --  Phase 4 / Week 10
=====================================================
Literature Benchmark: PI Controller (Wang et al., 2016).

Benchmarks the project's connectome-grounded two-level closed loop against the
chosen literature standard -- the closed-loop proportional-integral (PI)
controller of Wang, Niebur, Hu & Li (2016), "Suppressing epileptic activity in
a neural mass model using a closed-loop proportional-integral controller"
(Scientific Reports 6:27344). Both controllers act on the same seizing
connectome and onset zone; we compare seizure suppression, control energy, and
off-target energy (energy spent outside the onset zone -- a precision measure).

Figures written to images/:
  * week10_pi_vs_ours_timeseries.png   -- onset-zone amplitude, PI vs ours
  * week10_pi_gain_sweep.png           -- PI suppression/energy vs gains
  * week10_benchmark_bars.png          -- suppression / energy / off-target bars

Results written to results/:
  * week10_benchmark_summary.csv

Run (reduced scale recommended):
    SCALE=reduced python scripts/week10_literature_benchmark.py
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.detector import CSDDetector
from src.closed_loop import run_closed_loop
from src import baselines, metrics, viz


def main():
    cm.banner(10, "Literature Benchmark: PI Controller (Wang et al., 2016)")
    print(f"   baseline: {C.LITERATURE_BASELINE}")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    n_steps = 1600
    t = cm.time_axis(n_steps)
    s0 = random_initial_state(N, scale=0.05, seed=cm.REP_SEED)
    net = cm.seizing_network(A, zone)

    # --- uncontrolled reference -------------------------------------------- #
    unc = net.simulate(s0, n_steps, noise=C.NOISE_BETA, seed=cm.REP_SEED)
    amp_unc = metrics.amplitude(unc, N)[:, zone].mean(1)

    # --- PI literature baseline -------------------------------------------- #
    pi_traj, pi_energy = baselines.run_pi_baseline(
        net, s0, n_steps, stim_nodes=zone, kp=2.0, ki=0.5, seed=cm.REP_SEED)
    amp_pi = metrics.amplitude(pi_traj, N)[:, zone].mean(1)
    # rebuild PI control to measure off-target energy
    pi = baselines.PIController(kp=2.0, ki=0.5, stim_nodes=zone, N=N)
    _, pi_ctrl = net.simulate(s0, n_steps, u_func=pi.u_func(C.DT),
                              record_control=True, noise=C.NOISE_BETA, seed=cm.REP_SEED)
    pi_offtarget = baselines.off_target_fraction(pi_ctrl, N, zone)

    # --- our two-level closed loop ----------------------------------------- #
    res = run_closed_loop(net, s0, n_steps, monitor_nodes=zone, control_nodes=zone,
        **cm.closed_loop_kwargs())
    amp_ours = metrics.amplitude(res["traj"], N)[:, zone].mean(1)
    ours_offtarget = baselines.off_target_fraction(res["ctrl"], N, zone)

    viz.line_plot(t, {"uncontrolled": amp_unc,
                      "PI baseline (Wang 2016)": amp_pi,
                      "ours (connectome two-level)": amp_ours},
                  "Literature benchmark: PI vs connectome-grounded closed loop",
                  "onset-zone mean amplitude", "week10_pi_vs_ours_timeseries.png",
                  hline=C.SEIZURE_AMP_THRESHOLD)
    print(f"   PI   : zone amp={amp_pi[-100:].mean():.4f}  energy={pi_energy:.3f}"
          f"  off-target={pi_offtarget:.3f}")
    print(f"   ours : zone amp={amp_ours[-100:].mean():.4f}  energy={res['energy']:.3f}"
          f"  off-target={ours_offtarget:.3f}")

    # --- PI gain sweep (its sensitivity) ----------------------------------- #
    kps = [0.5, 1.0, 2.0, 4.0, 8.0]
    supp, ener = [], []
    for kp in kps:
        tr, en = baselines.run_pi_baseline(net, s0, n_steps, stim_nodes=zone,
                                           kp=kp, ki=0.25 * kp, seed=cm.REP_SEED)
        supp.append(metrics.amplitude(tr, N)[-100:, zone].mean())
        ener.append(en)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.plot(kps, supp, "o-", color="crimson", label="residual amplitude")
    ax1.axhline(C.SEIZURE_AMP_THRESHOLD, color="gray", ls="--")
    ax1.set_xlabel("PI proportional gain kp"); ax1.set_ylabel(
        "residual zone amplitude", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax2 = ax1.twinx()
    ax2.plot(kps, ener, "s--", color="navy", label="energy")
    ax2.set_ylabel("control energy", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")
    ax1.set_title(f"PI baseline gain sensitivity\n[{C.scale_label()}]")
    viz._save(fig, "week10_pi_gain_sweep.png")

    # --- benchmark bars ---------------------------------------------------- #
    labels_b = ["PI (Wang 2016)", "ours"]
    sup_b = [amp_pi[-100:].mean(), amp_ours[-100:].mean()]
    en_b = [pi_energy, res["energy"]]
    off_b = [pi_offtarget, ours_offtarget]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, vals, ttl in zip(ax, [sup_b, en_b, off_b],
                            ["residual zone amplitude", "control energy",
                             "off-target energy fraction"]):
        a.bar(labels_b, vals, color=["slategray", "seagreen"])
        a.set_title(ttl, fontsize=10)
    ax[0].axhline(C.SEIZURE_AMP_THRESHOLD, color="crimson", ls="--", lw=0.8)
    fig.suptitle(f"PI literature baseline vs ours  [{C.scale_label()}]")
    viz._save(fig, "week10_benchmark_bars.png")

    cm.savetxt_table(f"{C.RESULTS_DIR}/week10_benchmark_summary.csv",
                     ["controller", "residual_zone_amp", "control_energy",
                      "off_target_fraction", "seizure_suppressed"],
                     [["PI (Wang 2016)", f"{amp_pi[-100:].mean():.4f}",
                       f"{pi_energy:.3f}", f"{pi_offtarget:.3f}",
                       "yes" if amp_pi[-100:].mean() < C.SEIZURE_AMP_THRESHOLD else "no"],
                      ["ours (connectome two-level)", f"{amp_ours[-100:].mean():.4f}",
                       f"{res['energy']:.3f}", f"{ours_offtarget:.3f}",
                       "yes" if amp_ours[-100:].mean() < C.SEIZURE_AMP_THRESHOLD else "no"]])


if __name__ == "__main__":
    main()
