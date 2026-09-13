"""
scripts/analysis_complex_energy.py  --  real vs imaginary control-energy split
==============================================================================
Reads the per-trial real-energy fraction recorded by monte_carlo_baselines.py
(column ``real_frac`` of results/mc_proposed_per_trial.csv, standard Monte-Carlo
protocol: plant noise + U_MAX saturation) and, if present, the single-electrode
run (results/mc_realonly_per_trial.csv), and reports the mean share of control
energy carried by the real (physically deliverable) component of u_j.

Outputs
    results/analysis_complex_energy.csv
    images/analysis_complex_energy.png
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as cm
from src import config as C

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 10})


def _load(name):
    p = os.path.join(C.RESULTS_DIR, name)
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p, newline="", encoding="utf-8")))


def main():
    cm.banner(0, "Complex Energy Fraction (Real vs Imaginary)")
    R = _load("mc_proposed_per_trial.csv")
    if R is None:
        raise SystemExit("run monte_carlo_baselines.py --conditions proposed first")
    rf = np.array([float(r["real_frac"]) for r in R])
    e = np.array([float(r["energy"]) for r in R])
    real_frac = float(rf.mean()); imag_frac = 1 - real_frac
    print(f"  trials: {len(R)}   mean total energy {e.mean():.4f}")
    print(f"  Real component fraction : {real_frac:.2%} (SD {rf.std(ddof=1):.2%})")
    print(f"  Imag component fraction : {imag_frac:.2%}")
    rows = [["n_trials", len(R)], ["mean_total_energy", f"{e.mean():.4f}"],
            ["real_fraction_mean", f"{real_frac:.4f}"], ["real_fraction_sd", f"{rf.std(ddof=1):.4f}"],
            ["imag_fraction_mean", f"{imag_frac:.4f}"]]
    RO = _load("mc_realonly_per_trial.csv")
    if RO:
        rfo = np.array([float(r["real_frac"]) for r in RO]); eo = np.array([float(r["energy"]) for r in RO])
        rows += [["realonly_n_trials", len(RO)], ["realonly_mean_total_energy", f"{eo.mean():.4f}"],
                 ["realonly_real_fraction_mean", f"{rfo.mean():.4f}"]]
        print(f"  single-electrode run: real fraction {rfo.mean():.2%}, mean energy {eo.mean():.4f}")
    cm.savetxt_table(os.path.join(C.RESULTS_DIR, "analysis_complex_energy.csv"), ["quantity", "value"], rows)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([real_frac, imag_frac],
           labels=['Real component\n(physically realizable)', 'Imaginary component\n(no scalar-electrode counterpart)'],
           autopct='%1.2f%%', colors=['#1f77b4', '#d62728'], startangle=90,
           wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax.axis('equal')
    out = os.path.join(C.IMAGE_DIR, "analysis_complex_energy.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {out}")


if __name__ == "__main__":
    main()
