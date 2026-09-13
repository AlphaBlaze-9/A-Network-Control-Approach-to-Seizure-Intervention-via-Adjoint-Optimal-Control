"""
week06_lqr.py  --  Phase 3 / Week 6
===================================
Local Feedback Correction (Level-1, LQR).

The first corrective layer is a per-node linear-quadratic regulator built on
each oscillator's local 2x2 linearisation. It is cheap, always-on, and provably
stabilising near the resting fixed point -- the analogue of a fast reflex that
damps small excursions before they grow. Here we show a single seizure-prone
node being pulled back to rest, the effect of the control-effort weight, and
that the always-on Level-1 layer suppresses a focal node's limit cycle.

Figures written to images/:
  * week06_lqr_node_stabilisation.png   -- amplitude with/without LQR
  * week06_lqr_phase_space.png          -- focus node deflected to fixed point
  * week06_lqr_effort_tradeoff.png      -- settling vs control energy vs r_weight

Run:
    python scripts/week06_lqr.py
    SCALE=reduced python scripts/week06_lqr.py
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import HopfNetwork, random_initial_state
from src.lqr_control import LocalLQR
from src import metrics, viz


def main():
    cm.banner(6, "Local Feedback Correction (Level-1, LQR)")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    n_steps = 1200
    t = cm.time_axis(n_steps)
    s0 = random_initial_state(N, scale=0.1, seed=C.SEED)

    # seizing network (onset zone above the bifurcation)
    net = cm.seizing_network(A, zone)

    # --- 1. Uncontrolled vs LQR-controlled focus amplitude ------------------ #
    unc = net.simulate(s0, n_steps, noise=C.NOISE_BETA, seed=C.SEED)
    lqr = LocalLQR(a=np.full(N, C.A_REST), omega=net.omega,
                   q_weight=1.0, r_weight=1.0)
    ctl = net.simulate(s0, n_steps, u_func=lqr.u_func(gain_scale=1.0),
                       noise=C.NOISE_BETA, seed=C.SEED)
    amp_unc = metrics.amplitude(unc, N)
    amp_ctl = metrics.amplitude(ctl, N)
    viz.line_plot(t,
                  {f"focus {labels[focus]} - uncontrolled": amp_unc[:, focus],
                   f"focus {labels[focus]} - LQR": amp_ctl[:, focus],
                   "zone mean - uncontrolled": amp_unc[:, zone].mean(1),
                   "zone mean - LQR": amp_ctl[:, zone].mean(1)},
                  "Level-1 LQR damps the seizure-prone focus",
                  "amplitude r_i(t)", "week06_lqr_node_stabilisation.png",
                  hline=C.SEIZURE_AMP_THRESHOLD)
    print(f"   focus amplitude  uncontrolled: {amp_unc[-100:, focus].mean():.4f}"
          f"   LQR: {amp_ctl[-100:, focus].mean():.4f}")

    # --- 2. Phase-space deflection of the focus node ------------------------ #
    viz.plot_phase_space(unc, ctl, N, focus, "week06_lqr_phase_space.png",
                         title="Focus node: LQR deflects it off the limit cycle")

    # --- 3. Effort trade-off: sweep the control-penalty weight r ------------ #
    r_weights = [0.1, 0.3, 1.0, 3.0, 10.0]
    settle, energy = [], []
    for rw in r_weights:
        lqr_r = LocalLQR(a=np.full(N, C.A_REST), omega=net.omega,
                         q_weight=1.0, r_weight=rw)
        tr, cc = net.simulate(s0, n_steps, u_func=lqr_r.u_func(1.0),
                              noise=C.NOISE_BETA, seed=C.SEED,
                              record_control=True)
        settle.append(metrics.amplitude(tr, N)[-100:, focus].mean())
        energy.append(float(C.DT * np.sum(cc ** 2)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.plot(r_weights, settle, "o-", color="crimson",
             label="residual focus amplitude")
    ax1.set_xscale("log"); ax1.set_xlabel("control penalty r_weight")
    ax1.set_ylabel("residual focus amplitude", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.axhline(C.SEIZURE_AMP_THRESHOLD, color="gray", ls="--", alpha=0.6)
    ax2 = ax1.twinx()
    ax2.plot(r_weights, energy, "s--", color="navy", label="control energy")
    ax2.set_ylabel("control energy", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")
    ax1.set_title("LQR effort trade-off")
    viz._save(fig, "week06_lqr_effort_tradeoff.png")
    print("   r_weight sweep (residual amp / energy):")
    for rw, sa, en in zip(r_weights, settle, energy):
        print(f"      r={rw:5.1f}  amp={sa:.4f}  energy={en:.3f}")


if __name__ == "__main__":
    main()
