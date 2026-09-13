"""
week07_dpo.py  --  Phase 3 / Week 7
===================================
Optimal Control via the Pontryagin Adjoint (Level-2, "DPO").

The second corrective layer solves a finite-horizon optimal-control problem:
minimise  J = 0.5 * sum_t [ q * ||state||^2 + rho * ||u||^2 ] dt
subject to the Hopf dynamics. We form the control Hamiltonian H = L + lambda^T f,
integrate the costates (lambda, the "conjugate co-pilot") backward via the exact
discrete adjoint recursion, and descend the schedule with Adam. The optimal
control is u* = -lambda / rho (identity actuation). This script verifies the
adjoint gradient against finite differences, shows Adam converging, and shows
the optimal schedule suppressing the focus below the ictal threshold.

Figures written to images/:
  * week07_dpo_convergence.png      -- cost vs Adam iteration
  * week07_dpo_gradient_check.png   -- adjoint vs finite-difference gradient
  * week07_dpo_suppression.png      -- focus amplitude under the optimal schedule
  * week07_dpo_control_schedule.png -- the optimal control signal u*(t)

Run (reduced scale strongly recommended for this heavy week):
    SCALE=reduced python scripts/week07_dpo.py
    python scripts/week07_dpo.py
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.dpo_control import DPOController
from src import metrics, viz


def main():
    cm.banner(7, "Optimal Control via the Pontryagin Adjoint (Level-2)")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    net = cm.seizing_network(A, zone)

    horizon = 200
    dpo = DPOController(net, q=1.0, rho=0.02, control_nodes=zone)
    s0 = random_initial_state(N, scale=0.1, seed=C.SEED)

    # --- 1. Adjoint-gradient correctness check ------------------------------ #
    # Compare the exact discrete-adjoint gradient to a finite-difference probe
    # on a handful of randomly chosen schedule entries.
    U0 = np.zeros((horizon, 2 * N))
    X0 = dpo._forward(s0, U0, C.DT)
    P0 = dpo._backward(X0, C.DT)
    g_adj = dpo.grad(X0, U0, P0, C.DT)
    rng = np.random.default_rng(0)
    probes = [(rng.integers(horizon), focus),
              (rng.integers(horizon), N + focus),
              (rng.integers(horizon), zone[1] if len(zone) > 1 else focus)]
    eps = 1e-6
    fd, ad = [], []
    base = dpo.cost(X0, U0, C.DT)
    for (ti, ni) in probes:
        Up = U0.copy(); Up[ti, ni] += eps
        cp = dpo.cost(dpo._forward(s0, Up, C.DT), Up, C.DT)
        fd.append((cp - base) / eps)
        ad.append(g_adj[ti, ni])
    fd, ad = np.array(fd), np.array(ad)
    rel = np.abs(fd - ad) / (np.abs(fd) + 1e-12)
    print("   adjoint-gradient check (finite-diff vs adjoint):")
    for p, f, a, r in zip(probes, fd, ad, rel):
        print(f"      entry {p}: fd={f:.3e}  adj={a:.3e}  rel.err={r:.2e}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = max(np.abs(fd).max(), np.abs(ad).max()) * 1.2 + 1e-9
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
    ax.scatter(fd, ad, s=60, color="purple", zorder=5)
    ax.set_xlabel("finite-difference gradient"); ax.set_ylabel("adjoint gradient")
    ax.set_title(f"Adjoint gradient matches finite differences\n"
                 f"max rel.err = {rel.max():.1e}")
    ax.legend(fontsize=9)
    viz._save(fig, "week07_dpo_gradient_check.png")

    # --- 2. Optimise the schedule (Adam on the adjoint gradient) ------------ #
    U, hist = dpo.optimize(s0, horizon=horizon, n_iters=150, lr=0.05,
                           verbose=True, u_max=C.U_MAX)
    energy = float(C.DT * np.sum(U**2))
    print(f"   total control energy ||u||^2 = {energy:.4f}")
    # custom convergence plot (x-axis = iteration, not time)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(hist, color="darkgreen", lw=1.5)
    ax.set_xlabel("Adam iteration"); ax.set_ylabel("cost J")
    ax.set_title(f"AOC convergence: cost {hist[0]:.3f} -> {hist[-1]:.3f}")
    viz._save(fig, "week07_dpo_convergence.png")
    print(f"   cost {hist[0]:.4f} -> {hist[-1]:.4f}")

    # --- 3. Apply the optimal schedule to the (RK4) plant ------------------- #
    u_opt = dpo.u_func_from_schedule(U)
    unc = net.simulate(s0, horizon, noise=0.0)
    ctl = net.simulate(s0, horizon, u_func=u_opt, noise=0.0)
    t = cm.time_axis(horizon)
    amp_unc = metrics.amplitude(unc, N)[:, focus]
    amp_ctl = metrics.amplitude(ctl, N)[:, focus]
    viz.line_plot(t, {"uncontrolled": amp_unc, "AOC (closed-loop)": amp_ctl},
                  "Optimal schedule suppresses the focus",
                  "focus amplitude r(t)", "week07_dpo_suppression.png",
                  hline=C.SEIZURE_AMP_THRESHOLD)
    print(f"   focus amplitude  uncontrolled: {amp_unc[-40:].mean():.4f}"
          f"   DPO: {amp_ctl[-40:].mean():.4f}")

    # --- 4. The optimal control schedule itself ----------------------------- #
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(t[:-1], U[:, focus], color="navy", label=f"u_x at focus {labels[focus]}")
    ax.plot(t[:-1], U[:, N + focus], color="orange", label="u_y at focus")
    ax.set_xlabel("time (s)"); ax.set_ylabel("optimal control u*(t)")
    ax.set_title("Optimal control schedule (u* = -lambda/rho)")
    ax.legend(fontsize=9)
    viz._save(fig, "week07_dpo_control_schedule.png")


if __name__ == "__main__":
    main()
