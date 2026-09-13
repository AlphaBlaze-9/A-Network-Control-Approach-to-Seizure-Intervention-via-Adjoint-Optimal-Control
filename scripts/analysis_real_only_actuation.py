"""
analysis_real_only_actuation.py  --  Reviewer response (single-electrode actuation)
===================================================================================
Reviewer 2 (major 4): roughly half of the model's control energy sits in the
imaginary component of u_j, which a scalar DBS electrode cannot deliver.

This script re-runs the closed loop with the actuation matrix restricted to the
real component of z at every actuated node (B_j = [1, 0]^T), for BOTH control
levels: the Level-1 LQR is re-designed as a single-input regulator and the
Level-2 adjoint optimiser's control mask is zeroed on the imaginary half. Every
unit of reported control energy is then physically deliverable by a scalar
electrode. Everything else (seeds, plant noise, U_MAX, detector, horizon,
refractory) is the standard Monte-Carlo protocol, so the result is directly
comparable to Table 2's "Proposed (closed-loop)" column.

Run (split the 100 trials across processes if you like):
    SCALE=reduced python scripts/analysis_real_only_actuation.py --start 0  --count 50
    SCALE=reduced python scripts/analysis_real_only_actuation.py --start 50 --count 50
    SCALE=reduced python scripts/analysis_real_only_actuation.py --combine

Outputs
    results/analysis_realonly_<start>_<count>.csv
    results/analysis_real_only_actuation.csv     (combined summary)
    results/mc_realonly_per_trial.csv            (combined per-trial)
    images/analysis_real_only_actuation.png
"""
import argparse
import csv
import glob
import os
import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.closed_loop import run_closed_loop
from src import metrics, baselines

Z = 1.959963984540054


def _wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run(start, count):
    A, labels, focus = cm.load_network()
    zone = cm.onset_zone(A, labels, focus)
    N = A.shape[0]
    rows = []
    for i in range(start, start + count):
        seed = C.MC_SEED0 + i
        net = cm.seizing_network(A, zone)
        s0 = random_initial_state(N, scale=0.05, seed=seed)
        res = run_closed_loop(net, s0, cm.CL_N_STEPS, monitor_nodes=zone, control_nodes=zone,
                              real_only=True, **cm.closed_loop_kwargs(seed=seed))
        amp = metrics.amplitude(res["traj"], N)
        final = float(amp[-100:, zone].mean())
        ok = final < C.SEIZURE_AMP_THRESHOLD
        ctrl = res["ctrl"]
        tot = float((ctrl ** 2).sum())
        real_frac = float((ctrl[:, :N] ** 2).sum() / tot) if tot > 0 else 1.0
        offt = baselines.off_target_fraction(ctrl, N, zone)
        rows.append([i, seed, "yes" if ok else "no", f"{res['energy']:.6f}", f"{offt:.6f}", f"{final:.6f}",
                     len(res["onsets"]), f"{real_frac:.6f}"])
        print(f"   [real-only] trial {i:3d} seed={seed} suppressed={'yes' if ok else 'no':3s} "
              f"energy={res['energy']:.4f} final={final:.4f} n_int={len(res['onsets'])} real_frac={real_frac:.3f}",
              flush=True)
    cm.savetxt_table(f"{C.RESULTS_DIR}/analysis_realonly_{start}_{count}.csv",
                     ["trial", "seed", "suppressed", "energy", "offtarget", "final", "n_interventions", "real_frac"], rows)


def combine():
    rows = []
    for p in sorted(glob.glob(f"{C.RESULTS_DIR}/analysis_realonly_*_*.csv")):
        rows += list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    rows = {int(r["trial"]): r for r in rows}
    rows = [rows[k] for k in sorted(rows)]
    cm.savetxt_table(f"{C.RESULTS_DIR}/mc_realonly_per_trial.csv",
                     ["trial", "seed", "suppressed", "energy", "offtarget", "final", "n_interventions", "real_frac"],
                     [[r[k] for k in ("trial", "seed", "suppressed", "energy", "offtarget", "final", "n_interventions", "real_frac")] for r in rows])
    n = len(rows); k = sum(r["suppressed"] == "yes" for r in rows)
    lo, hi = _wilson(k, n)
    e = np.array([float(r["energy"]) for r in rows]); o = np.array([float(r["offtarget"]) for r in rows])
    ni = np.array([float(r["n_interventions"]) for r in rows])
    e_ci = Z * e.std(ddof=1) / np.sqrt(n); o_ci = Z * o.std(ddof=1) / np.sqrt(n)
    # reference: standard (two-component) proposed controller, if available
    ref = f"{C.RESULTS_DIR}/mc_proposed_per_trial.csv"
    ref_line = ["Proposed, two-component actuation (reference)", "", "", "", "", "", "", "", ""]
    if os.path.exists(ref):
        R = list(csv.DictReader(open(ref, newline="", encoding="utf-8")))
        nr = len(R); kr = sum(r["suppressed"] == "yes" for r in R); lr, hr = _wilson(kr, nr)
        er = np.array([float(r["energy"]) for r in R]); orr = np.array([float(r["offtarget"]) for r in R])
        ref_line = ["Proposed, two-component actuation (reference)", nr, kr, f"{kr/nr:.4f}", f"{lr:.4f}", f"{hr:.4f}",
                    f"{er.mean():.4f}", f"{Z*er.std(ddof=1)/np.sqrt(nr):.4f}", f"{orr.mean():.4f}"]
    out = [["Proposed, real-only (single-electrode) actuation", n, k, f"{k/n:.4f}", f"{lo:.4f}", f"{hi:.4f}",
            f"{e.mean():.4f}", f"{e_ci:.4f}", f"{o.mean():.4f}"], ref_line]
    cm.savetxt_table(f"{C.RESULTS_DIR}/analysis_real_only_actuation.csv",
                     ["condition", "n_trials", "n_success", "success_rate", "ci95_low", "ci95_high",
                      "energy_mean", "energy_ci95_halfwidth", "offtarget_mean"], out)
    print(f"   real-only: success={k}/{n}={k/n:.0%} CI[{lo:.3f},{hi:.3f}] energy={e.mean():.4f}+/-{e_ci:.4f} "
          f"offtarget={o.mean():.4f} n_int={ni.mean():.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    names = ["two-component\n(x and y)", "real-only\n(x, single electrode)"]
    if os.path.exists(ref):
        rates = [kr / nr, k / n]; errs = [[kr / nr - lr, k / n - lo], [hr - kr / nr, hi - k / n]]
        en = [er.mean(), e.mean()]; ene = [Z * er.std(ddof=1) / np.sqrt(nr), e_ci]
    else:
        names = names[1:]; rates = [k / n]; errs = [[k / n - lo], [hi - k / n]]; en = [e.mean()]; ene = [e_ci]
    axes[0].bar(names, rates, yerr=errs, capsize=6, color=["teal", "darkorange"][-len(names):])
    axes[0].set_ylim(0, 1.08); axes[0].set_ylabel("suppression success rate (Wilson 95% CI)")
    axes[1].bar(names, en, yerr=ene, capsize=6, color=["slateblue", "darkorange"][-len(names):])
    axes[1].set_ylabel("total control energy (mean +/- 95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.IMAGE_DIR, "analysis_real_only_actuation.png"), dpi=150, bbox_inches="tight")
    print("   saved images/analysis_real_only_actuation.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=C.N_SEEDS)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    cm.banner(0, "Single-electrode (real-only) actuation, Monte Carlo")
    if a.combine:
        combine(); return
    run(a.start, a.count)


if __name__ == "__main__":
    main()
