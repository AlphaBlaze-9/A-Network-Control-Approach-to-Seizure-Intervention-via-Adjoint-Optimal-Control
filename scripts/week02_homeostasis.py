"""
week02_homeostasis.py  --  Phase 1 / Week 2
===========================================
Coupled-Oscillator Model & Homeostasis Baseline.

Implements nothing new (the dynamics live in src/hopf_model.py); this script
*verifies* the modelling claim that an all-healthy coupled-Hopf network
maintains homeostasis: starting from a perturbation it decays back toward the
interictal fixed point and stays bounded under physiological noise, with no
chaotic drift. This is the resting baseline every later figure is measured
against.

Figures written to images/:
  * week02_homeostasis_timeseries.png   -- node amplitudes decaying to rest
  * week02_homeostasis_meanfield.png    -- network mean amplitude +/- noise band
  * week02_bifurcation_diagram.png      -- steady-state amplitude vs excitability a

Run:
    python scripts/week02_homeostasis.py
    SCALE=reduced python scripts/week02_homeostasis.py
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import HopfNetwork, healthy_excitability, random_initial_state
from src import metrics, viz


def main():
    cm.banner(2, "Coupled-Oscillator Model & Homeostasis Baseline")
    A, labels, focus = cm.load_network()
    N = A.shape[0]

    # --- 1. Deterministic decay to rest from a perturbed start -------------- #
    net = cm.healthy_network(A)
    s0 = random_initial_state(N, scale=0.4, seed=C.SEED)   # sizeable perturbation
    n_steps = 1200
    traj = net.simulate(s0, n_steps, noise=0.0)
    t = cm.time_axis(n_steps)
    amp = metrics.amplitude(traj, N)

    # plot a representative subset of nodes (focus + a few high-degree regions)
    deg = A.sum(1)
    show = [focus] + list(np.argsort(deg)[::-1][:6])
    series = {f"{labels[i]}": amp[:, i] for i in show}
    viz.line_plot(t, series,
                  "Healthy network returns to rest (deterministic)",
                  "amplitude r_i(t)", "week02_homeostasis_timeseries.png",
                  hline=C.SEIZURE_AMP_THRESHOLD)
    print(f"   max amplitude at end (deterministic): {amp[-50:].max():.4f}")

    # --- 2. Mean-field with noise: bounded homeostasis ---------------------- #
    traj_n = net.simulate(s0, n_steps, noise=C.NOISE_BETA, seed=C.SEED)
    amp_n = metrics.amplitude(traj_n, N)
    mean_amp = amp_n.mean(axis=1)
    std_amp = amp_n.std(axis=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, mean_amp, color="teal", lw=1.4, label="network mean amplitude")
    ax.fill_between(t, mean_amp - std_amp, mean_amp + std_amp, color="teal",
                    alpha=0.2, label="+/- 1 s.d. across regions")
    ax.axhline(C.SEIZURE_AMP_THRESHOLD, color="crimson", ls="--",
               label="ictal threshold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude")
    ax.set_title(f"Homeostasis under noise (beta={C.NOISE_BETA})")
    ax.legend(fontsize=9)
    viz._save(fig, "week02_homeostasis_meanfield.png")
    print(f"   noisy mean amplitude (last 200 steps): {mean_amp[-200:].mean():.4f}"
          f"  (threshold {C.SEIZURE_AMP_THRESHOLD})")

    # --- 3. Single-node bifurcation diagram (the a -> amplitude map) -------- #
    # Sweep the bifurcation parameter for one isolated node to show the
    # supercritical Hopf transition at a = 0 (amplitude ~ sqrt(a) for a>0).
    a_vals = np.linspace(-0.2, 0.8, 41)
    ss_amp = []
    for av in a_vals:
        single = HopfNetwork(np.zeros((1, 1)), a=np.array([av]),
                             omega=np.array([2 * np.pi * C.OMEGA_HZ]))
        tr = single.simulate(np.array([0.1, 0.0]), 2000, noise=0.0)
        ss_amp.append(metrics.amplitude(tr, 1)[-200:].mean())
    ss_amp = np.array(ss_amp)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(a_vals, ss_amp, "o-", ms=3, color="darkorange",
            label="simulated steady-state amplitude")
    pos = a_vals > 0
    ax.plot(a_vals[pos], np.sqrt(a_vals[pos]), "k--", lw=1,
            label=r"theory $\sqrt{a}$ (limit-cycle radius)")
    ax.axvline(0, color="gray", ls=":", label="Hopf bifurcation a=0")
    ax.set_xlabel("excitability a"); ax.set_ylabel("steady-state amplitude")
    ax.set_title("Supercritical Hopf bifurcation (single node)")
    ax.legend(fontsize=9)
    viz._save(fig, "week02_bifurcation_diagram.png")


if __name__ == "__main__":
    main()
