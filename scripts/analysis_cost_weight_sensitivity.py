"""
analysis_cost_weight_sensitivity.py  --  Reviewer response (alpha / R2 sensitivity)
====================================================================================
Reviewer 2 (minor): the Level-2 cost weights (alpha = q, R2 = rho) were chosen by
inspection; their robustness should be tested.

Only the ratio alpha/R2 matters for the optimum (the cost is homogeneous in the
two weights), so we hold alpha = 1 and sweep R2 = rho over two decades around the
value used in the paper (0.01). For every rho the FULL closed loop is run with
the standard Monte-Carlo protocol (randomised initial condition + plant noise,
U_MAX saturation) and we report the suppression rate (Wilson 95% CI), the control
energy (mean +/- 95% CI) and the number of Level-2 interventions.

Run one rho per process (parallel-friendly), then combine:
    SCALE=reduced python scripts/analysis_cost_weight_sensitivity.py --rho 0.001
    ...
    SCALE=reduced python scripts/analysis_cost_weight_sensitivity.py --combine

Outputs
    results/analysis_rho_<rho>_per_trial.csv     (one per rho)
    results/analysis_cost_weight_sensitivity.csv (combined)
    images/analysis_cost_weight_sensitivity.png
"""
import argparse
import csv
import os
import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.closed_loop import run_closed_loop
from src import metrics, baselines

RHO_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]
Z = 1.959963984540054


def _wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run_rho(rho, n_trials):
    out = f"{C.RESULTS_DIR}/analysis_rho_{rho}_per_trial.csv"
    if os.path.exists(out) and sum(1 for _ in open(out, encoding="utf-8")) >= n_trials + 1:
        print(f"   rho={rho}: {out} already complete -- skipping (delete it to re-run)")
        return
    A, labels, focus = cm.load_network()
    zone = cm.onset_zone(A, labels, focus)
    N = A.shape[0]
    rows = []
    for i in range(n_trials):
        seed = C.MC_SEED0 + i
        net = cm.seizing_network(A, zone)
        s0 = random_initial_state(N, scale=0.05, seed=seed)
        res = run_closed_loop(net, s0, cm.CL_N_STEPS, monitor_nodes=zone, control_nodes=zone,
                              **cm.closed_loop_kwargs(dpo_rho=rho, seed=seed))
        amp = metrics.amplitude(res["traj"], N)
        final = float(amp[-100:, zone].mean())
        ok = final < C.SEIZURE_AMP_THRESHOLD
        offt = baselines.off_target_fraction(res["ctrl"], N, zone)
        rows.append([rho, i, seed, "yes" if ok else "no", f"{res['energy']:.6f}", f"{offt:.6f}",
                     f"{final:.6f}", len(res["onsets"])])
        print(f"   rho={rho:<6} trial {i:3d} seed={seed} suppressed={'yes' if ok else 'no':3s} "
              f"energy={res['energy']:.4f} final={final:.4f} n_int={len(res['onsets'])}", flush=True)
    cm.savetxt_table(f"{C.RESULTS_DIR}/analysis_rho_{rho}_per_trial.csv",
                     ["rho", "trial", "seed", "suppressed", "energy", "offtarget", "final", "n_interventions"], rows)


def combine():
    out = []
    pts = []
    for rho in RHO_GRID:
        p = f"{C.RESULTS_DIR}/analysis_rho_{rho}_per_trial.csv"
        if not os.path.exists(p):
            print(f"   [!] missing {p}"); continue
        R = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
        n = len(R); k = sum(r["suppressed"] == "yes" for r in R)
        lo, hi = _wilson(k, n)
        e = np.array([float(r["energy"]) for r in R]); ni = np.array([float(r["n_interventions"]) for r in R])
        e_ci = Z * e.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        out.append([rho, 1.0 / rho, n, k, f"{k/n:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{e.mean():.4f}", f"{e_ci:.4f}",
                    f"{ni.mean():.2f}"])
        pts.append((rho, k / n, lo, hi, e.mean(), e_ci))
        print(f"   rho={rho:<6} alpha/R2={1/rho:<7.0f} success={k}/{n} CI[{lo:.2f},{hi:.2f}] energy={e.mean():.3f}+/-{e_ci:.3f} n_int={ni.mean():.2f}")
    cm.savetxt_table(f"{C.RESULTS_DIR}/analysis_cost_weight_sensitivity.csv",
                     ["rho_R2", "alpha_over_R2", "n_trials", "n_success", "success_rate", "ci95_low", "ci95_high",
                      "energy_mean", "energy_ci95_halfwidth", "mean_n_interventions"], out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [p[0] for p in pts]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.errorbar(x, [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                 fmt="o-", capsize=5, color="teal", label="suppression rate (Wilson 95% CI)")
    ax1.set_xscale("log"); ax1.set_xlabel(r"control-energy weight $R_2$ ($\alpha = 1$)")
    ax1.set_ylabel("suppression success rate", color="teal"); ax1.set_ylim(-0.05, 1.08)
    ax1.axvline(0.01, color="crimson", ls=":", lw=1, label="value used in paper")
    ax2 = ax1.twinx()
    ax2.errorbar(x, [p[4] for p in pts], yerr=[p[5] for p in pts], fmt="s--", capsize=4, color="navy",
                 label="control energy (mean +/- 95% CI)")
    ax2.set_ylabel("total control energy", color="navy")
    l1, n1 = ax1.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(C.IMAGE_DIR, "analysis_cost_weight_sensitivity.png"), dpi=150, bbox_inches="tight")
    print("   saved images/analysis_cost_weight_sensitivity.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    cm.banner(0, "Level-2 cost-weight (alpha/R2) sensitivity")
    if a.combine:
        combine(); return
    for rho in (a.rho or RHO_GRID):
        run_rho(rho, a.n)


if __name__ == "__main__":
    main()
