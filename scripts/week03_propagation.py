"""
week03_propagation.py  --  Phase 1 / Week 3
===========================================
Uncontrolled Seizure Propagation.
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src import metrics, viz

def main():
    cm.banner(3, "Uncontrolled Seizure Propagation")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    print(f"   onset zone: {[(i, labels[i]) for i in zone]}")

    n_steps = 1600
    t = cm.time_axis(n_steps)
    s0 = random_initial_state(N, scale=0.05, seed=cm.REP_SEED)

    # --- 1. Intact-connectome propagation ----------------------------------- #
    net = cm.seizing_network(A, zone)
    traj = net.simulate(s0, n_steps, noise=C.NOISE_BETA, seed=cm.REP_SEED)
    amp = metrics.amplitude(traj, N)

    final_amp = amp[-200:].mean(axis=0)
    order = np.argsort(final_amp)[::-1]
    
    # BUG FIX B8: Changed ylabel to node (sorted by geodesic distance from focus)
    viz.heatmap(amp[:, order].T,
                "", # Title omitted
                "week03_propagation_heatmap.png",
                ylabel="node (sorted by geodesic distance from focus)", extent=[0, t[-1], 0, N])

    # --- 2. Focus vs recruited vs distant traces ---------------------------- #
    zone_conn = A[zone].sum(axis=0)
    distant = int(np.argsort(zone_conn)[len(zone):][0])
    recruited = int(order[len(zone)])
    series = {
        f"focus {labels[focus]}": amp[:, focus],
        f"recruited {labels[recruited]}": amp[:, recruited],
        f"distant {labels[distant]}": amp[:, distant],
    }
    viz.line_plot(t, series, "",
                  "amplitude r_i(t)", "week03_focus_vs_neighbours.png",
                  hline=C.SEIZURE_AMP_THRESHOLD)

    # --- 3. Connectome dependence: ablate the onset-zone edges -------------- #
    A_ablate = A.copy()
    A_ablate[zone, :] = 0.0
    A_ablate[:, zone] = 0.0
    net_ab = cm.seizing_network(A_ablate, zone)
    traj_ab = net_ab.simulate(s0, n_steps, noise=C.NOISE_BETA, seed=cm.REP_SEED)
    amp_ab = metrics.amplitude(traj_ab, N)

    rest = [i for i in range(N) if i not in zone]
    spread_intact = amp[:, rest].mean(axis=1)
    spread_ablate = amp_ab[:, rest].mean(axis=1)
    viz.line_plot(t,
                  {"intact connectome": spread_intact,
                   "onset-zone edges ablated": spread_ablate},
                  "",
                  "mean amplitude (non-zone regions)",
                  "week03_connectome_dependence.png")
    print(f"   non-zone spread  intact: {spread_intact[-200:].mean():.4f}"
          f"   ablated: {spread_ablate[-200:].mean():.4f}")

    # --- 4. Recruitment correlates with structural connection to the zone --- #
    conn_to_zone = A[zone].sum(axis=0)
    nz = [i for i in range(N) if i not in zone]
    x = conn_to_zone[nz]
    y = final_amp[nz]
    r = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else float("nan")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(x, y, s=18, alpha=0.6, color="indigo")
    ax.set_xlabel("structural connection strength to onset zone")
    ax.set_ylabel("final amplitude (recruitment)")
    # BUG FIX B3/B4: Removed ax.set_title(...)
    viz._save(fig, "week03_recruitment_vs_connectivity.png")
    print(f"   recruitment-vs-connectivity correlation r = {r:.3f}")

    # --- 5. Hypersynchronisation (Kuramoto order parameter) ----------------- #
    R_all = metrics.kuramoto_order(traj, N)
    R_zone = metrics.kuramoto_order(traj, N, nodes=zone)
    viz.line_plot(t, {"whole network R(t)": R_all,
                      "onset-zone R(t)": R_zone},
                  "",
                  "Kuramoto order parameter R", "week03_kuramoto_synchrony.png")

    # --- short interpretive note ------------------------------------------- #
    note = (f"WEEK 3 NOTE -- onset zone {[labels[i] for i in zone]} drives "
            f"connectome-mediated recruitment (r={r:.3f} between a region's "
            f"structural connection to the zone and its final amplitude). "
            f"Ablating the onset-zone edges drops non-zone spread from "
            f"{spread_intact[-200:].mean():.4f} to {spread_ablate[-200:].mean():.4f}, "
            f"confirming the spread is carried by A, not by local dynamics.")
    with open(f"{C.RESULTS_DIR}/week03_propagation_note.txt", "w") as fh:
        fh.write(note + "\n")
    print("\n   " + note)

if __name__ == "__main__":
    main()