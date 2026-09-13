"""
week11_visualisation.py  --  Phase 5 / Week 11
==============================================
Visualisation & Brain Rendering.
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.detector import CSDDetector
from src.closed_loop import run_closed_loop
from src import metrics, viz

def main():
    cm.banner(11, "Visualisation & Brain Rendering")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    n_steps = 1600
    s0 = random_initial_state(N, scale=0.05, seed=cm.REP_SEED)
    net = cm.seizing_network(A, zone)

    # --- uncontrolled recruitment map -------------------------------------- #
    unc = net.simulate(s0, n_steps, noise=C.NOISE_BETA, seed=cm.REP_SEED)
    final_amp = metrics.amplitude(unc, N)[-200:].mean(0)
    viz.plot_brain_values(A, final_amp,
                          "", # Title omitted
                          "week11_brain_recruitment_map.png", focus=focus,
                          cmap="inferno")

    # --- closed-loop control-energy map ------------------------------------ #
    res = run_closed_loop(net, s0, n_steps, monitor_nodes=zone, control_nodes=zone,
        **cm.closed_loop_kwargs())
    ctrl = res["ctrl"]
    node_energy = (ctrl[:, :N] ** 2 + ctrl[:, N:] ** 2).sum(0) * C.DT
    viz.plot_brain_values(A, node_energy,
                          "", # Title omitted
                          "week11_brain_control_map.png", focus=focus,
                          cmap="viridis")
    print(f"   total control energy {node_energy.sum():.3f}; "
          f"top-3 actuated regions: "
          f"{[labels[i] for i in np.argsort(node_energy)[::-1][:3]]}")

    # --- focus-node phase portrait ----------------------------------------- #
    viz.plot_phase_space(unc, res["traj"], N, focus,
                         "week11_focus_phase_portrait.png", "") # Title omitted

    # --- control energy vs structural connectivity to the onset zone ------- #
    conn_to_zone = A[zone].sum(0)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(conn_to_zone, node_energy, s=18, alpha=0.6, color="darkcyan")
    for z in zone:
        ax.scatter(conn_to_zone[z], node_energy[z], s=80, color="crimson",
                   edgecolors="k", zorder=5)
    ax.set_xlabel("structural connectivity to onset zone")
    ax.set_ylabel("control energy at region")
    # BUG FIX B3/B4: Removed ax.set_title(...)
    viz._save(fig, "week11_connectivity_vs_control.png")

if __name__ == "__main__":
    main()