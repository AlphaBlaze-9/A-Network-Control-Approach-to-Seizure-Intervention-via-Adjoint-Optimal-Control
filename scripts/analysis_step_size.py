"""
analysis_step_size.py  --  Reviewer response, Issue #5 (SDE integration order)
==============================================================================
Demonstrates that the production integrator -- RK4 on the drift with an additive
Euler-Maruyama Wiener increment (a drift/diffusion split) -- is *converged* at
the production step ``config.DT`` and carries the strong order expected for
ADDITIVE noise.

Why this, and not "switch to Milstein":
    The plant noise is additive (state-independent diffusion of amplitude
    NOISE_BETA). For additive noise the Milstein correction term vanishes
    (it is (1/2) b b' (dW^2 - dt) and b' = 0), so Milstein is byte-identical to
    Euler-Maruyama and would change nothing. The rigorous thing to show is the
    empirical strong-convergence order of the scheme we actually use, which is
    what this script measures.

Method (shared Brownian path / Brownian-tree refinement):
    1. Draw a fine Wiener path at DT_CONV_FINEST.
    2. The reference solution integrates that fine path.
    3. Each coarser step dt = m * finest is integrated using coarse increments
       that are EXACT SUMS of the corresponding fine increments, so every
       resolution rides the same Brownian path (this is what makes a *strong*
       error meaningful).
    4. Strong error(dt) = mean over paths of || X_dt(T) - X_ref(T) ||.
    5. Fit the slope on log-log axes -> empirical strong order.

Figures written to images/:
    * analysis_step_size_convergence.png
Results written to results/:
    * analysis_step_size_convergence.csv

Run (reduced scale is plenty and fast):
    SCALE=reduced python scripts/analysis_step_size.py
"""

import numpy as np

import _common as cm
from src import config as C


def main():
    cm.banner(0, "Issue #5: SDE strong-convergence study (RK4 drift + additive EM)")
    A, labels, focus = cm.load_network()
    zone = cm.onset_zone(A, labels, focus)
    net = cm.seizing_network(A, zone)        # representative (seizing) dynamics
    N = net.N

    finest = float(C.DT_CONV_FINEST)
    T = float(C.DT_CONV_T)
    n_fine = int(round(T / finest))
    levels = sorted(set(C.DT_CONV_LEVELS))
    for dt in levels:
        m = dt / finest
        if abs(m - round(m)) > 1e-9:
            raise SystemExit(f"dt={dt} is not an integer multiple of the finest "
                             f"step {finest}; adjust DT_CONV_* in config.py")
    noise = C.NOISE_BETA
    paths = int(C.DT_CONV_PATHS)

    from src.hopf_model import random_initial_state
    s0 = random_initial_state(N, scale=0.05, seed=C.SEED)

    print(f"   finest={finest}  T={T}s  n_fine={n_fine}  paths={paths}  "
          f"noise={noise}  [{C.scale_label()}]")
    if noise <= 0:
        print("   [!] NOISE_BETA == 0: this measures deterministic RK4 error only.")

    errs = np.zeros(len(levels))
    for p in range(paths):
        rng = np.random.default_rng(20000 + p)
        dW_fine = np.sqrt(finest) * rng.standard_normal((n_fine, 2 * N))
        # reference solution on the fine path
        s = s0.copy()
        for k in range(n_fine):
            s = net.step_with_dW(s, finest, dW_fine[k], noise=noise)
        ref = s
        # each coarser level on the SAME path (summed increments)
        for li, dt in enumerate(levels):
            m = int(round(dt / finest))
            nc = n_fine // m
            s = s0.copy()
            for j in range(nc):
                dW = dW_fine[j * m:(j + 1) * m].sum(axis=0)   # exact coarse increment
                s = net.step_with_dW(s, dt, dW, noise=noise)
            errs[li] += np.linalg.norm(s - ref)
    errs /= paths

    lv = np.array(levels, dtype=float)
    # least-squares slope on log-log = empirical strong-convergence order
    M = np.vstack([np.log(lv), np.ones_like(lv)]).T
    slope, intercept = np.linalg.lstsq(M, np.log(errs), rcond=None)[0]

    rows = []
    for dt, e in zip(levels, errs):
        rows.append([f"{dt:.5f}", f"{e:.6e}"])
        print(f"   dt={dt:.5f}   strong_err={e:.3e}")
    rows.append(["fitted_strong_order", f"{slope:.4f}"])
    print(f"   >> empirical strong-convergence order = {slope:.3f}")
    prod_err = float(errs[levels.index(C.DT)]) if C.DT in levels else float("nan")
    print(f"   >> strong error at production DT={C.DT}: {prod_err:.3e} "
          f"(limit-cycle radius ~ sqrt(a_seizure) = {np.sqrt(C.A_SEIZURE):.3f})")

    cm.savetxt_table(f"{C.RESULTS_DIR}/analysis_step_size_convergence.csv",
                     ["dt_or_metric", "strong_error_or_value"], rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.loglog(lv, errs, "o-", color="darkgreen", label="measured strong error")
    # order-1 reference line anchored at the production step
    ref_line = errs[len(levels) // 2] * (lv / lv[len(levels) // 2]) ** 1.0
    ax.loglog(lv, ref_line, "k--", lw=0.9, label="slope = 1 (reference)")
    if C.DT in levels:
        ax.axvline(C.DT, color="crimson", ls=":", lw=0.9,
                   label=f"production DT = {C.DT}")
    ax.set_xlabel("integration step dt (s)")
    ax.set_ylabel(r"strong error  $E\,\|X_{dt}(T)-X_{ref}(T)\|$")
    ax.set_title(f"SDE strong convergence (order={slope:.2f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{C.IMAGE_DIR}/analysis_step_size_convergence.png", dpi=200)
    print(f"   saved {C.IMAGE_DIR}/analysis_step_size_convergence.png")


if __name__ == "__main__":
    main()
