"""
analysis_robustness_mc.py  --  Reviewer response (detector-error robustness, Monte Carlo)
=========================================================================================
The original week08 robustness probe was a SINGLE representative trial per condition
(and its figure caption in the manuscript wrongly described it as a Monte-Carlo
success-rate plot -- Reviewer 3, comment 9). This script runs each detector-error
condition as a proper Monte-Carlo experiment under the standard closed-loop protocol
and reports the suppression success rate with a Wilson 95% CI, so the figure now
shows what the caption says.

Conditions (risk-score corruption injected at the detector output):
    clean          noise SD 0     bias  0
    noise0.15      noise SD 0.15  bias  0
    bias+0.20      noise SD 0     bias +0.20   (conservative: over-reports risk)
    bias-0.05      noise SD 0     bias -0.05
    bias-0.10      noise SD 0     bias -0.10
    bias-0.20      noise SD 0     bias -0.20   (under-reports risk)

Run one condition per process (parallel-friendly), then combine:
    SCALE=reduced python scripts/analysis_robustness_mc.py --condition clean
    ...
    SCALE=reduced python scripts/analysis_robustness_mc.py --combine

Outputs
    results/analysis_robustness_<condition>_per_trial.csv
    results/analysis_robustness_mc.csv
    images/analysis_robustness_mc.png
"""
import argparse
import csv
import os
import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.closed_loop import run_closed_loop
from src import metrics

CONDITIONS = {
    "clean":     dict(detector_noise=0.0,  detector_bias=0.0),
    "noise0.15": dict(detector_noise=0.15, detector_bias=0.0),
    "bias+0.20": dict(detector_noise=0.0,  detector_bias=+0.20),
    "bias-0.05": dict(detector_noise=0.0,  detector_bias=-0.05),
    "bias-0.10": dict(detector_noise=0.0,  detector_bias=-0.10),
    "bias-0.20": dict(detector_noise=0.0,  detector_bias=-0.20),
}
Z = 1.959963984540054


def _wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run_condition(name, n_trials):
    kw = CONDITIONS[name]
    A, labels, focus = cm.load_network()
    zone = cm.onset_zone(A, labels, focus)
    N = A.shape[0]
    rows = []
    for i in range(n_trials):
        seed = C.MC_SEED0 + i
        net = cm.seizing_network(A, zone)
        s0 = random_initial_state(N, scale=0.05, seed=seed)
        res = run_closed_loop(net, s0, cm.CL_N_STEPS, monitor_nodes=zone, control_nodes=zone,
                              **cm.closed_loop_kwargs(seed=seed, **kw))
        amp = metrics.amplitude(res["traj"], N)
        final = float(amp[-100:, zone].mean())
        ok = final < C.SEIZURE_AMP_THRESHOLD
        rows.append([name, i, seed, "yes" if ok else "no", f"{res['energy']:.6f}", f"{final:.6f}", len(res["onsets"])])
        print(f"   [{name:10s}] trial {i:3d} seed={seed} suppressed={'yes' if ok else 'no':3s} "
              f"energy={res['energy']:.4f} final={final:.4f} n_int={len(res['onsets'])}", flush=True)
    cm.savetxt_table(f"{C.RESULTS_DIR}/analysis_robustness_{name}_per_trial.csv",
                     ["condition", "trial", "seed", "suppressed", "energy", "final_zone_amp", "n_interventions"], rows)


def combine():
    out, pts = [], []
    for name, kw in CONDITIONS.items():
        p = f"{C.RESULTS_DIR}/analysis_robustness_{name}_per_trial.csv"
        if not os.path.exists(p):
            print(f"   [!] missing {p}"); continue
        R = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
        n = len(R); k = sum(r["suppressed"] == "yes" for r in R)
        lo, hi = _wilson(k, n)
        e = np.array([float(r["energy"]) for r in R]); ni = np.array([float(r["n_interventions"]) for r in R])
        e_ci = Z * e.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        out.append([name, kw["detector_noise"], kw["detector_bias"], n, k, f"{k/n:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                    f"{e.mean():.4f}", f"{e_ci:.4f}", f"{ni.mean():.2f}"])
        pts.append((name, k / n, lo, hi))
        print(f"   {name:10s} success={k}/{n}={k/n:.0%} CI[{lo:.2f},{hi:.2f}] energy={e.mean():.3f}+/-{e_ci:.3f} n_int={ni.mean():.2f}")
    cm.savetxt_table(f"{C.RESULTS_DIR}/analysis_robustness_mc.csv",
                     ["condition", "risk_noise_sd", "risk_bias", "n_trials", "n_success", "success_rate",
                      "ci95_low", "ci95_high", "energy_mean", "energy_ci95_halfwidth", "mean_n_interventions"], out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    names = [p[0] for p in pts]; rates = [p[1] for p in pts]
    err = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
    colors = ["teal" if not n.startswith("bias-") else "indianred" for n in names]
    ax.bar(names, rates, yerr=err, capsize=6, color=colors)
    ax.set_ylim(0, 1.08); ax.set_ylabel("suppression success rate (Wilson 95% CI)")
    ax.set_xlabel("injected risk-score corruption (noise SD / additive bias)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(C.IMAGE_DIR, "analysis_robustness_mc.png"), dpi=150, bbox_inches="tight")
    print("   saved images/analysis_robustness_mc.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", nargs="*", default=None, choices=list(CONDITIONS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    cm.banner(0, "Detector-error robustness (Monte Carlo)")
    if a.combine:
        combine(); return
    for name in (a.condition or list(CONDITIONS)):
        run_condition(name, a.n)


if __name__ == "__main__":
    main()
