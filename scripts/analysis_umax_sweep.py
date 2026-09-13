"""
analysis_umax_sweep.py  --  Reviewer-response addition (Actuator Saturation)
============================================================================
Reviewer #3: "an optimal schedule that minimises energy but requires
instantaneous infinite current is clinically useless -- impose a bound on u."

This script imposes a hard saturation bound on the Level-2 (DPO) control
schedule and sweeps it from generous down to tight, reporting for each bound:
  * whether the seizure is still suppressed (final onset-zone amplitude below
    the seizure threshold);
  * the residual onset-zone amplitude;
  * the total control energy;
  * the number of interventions.

USE IT TO CHOOSE config.U_MAX: pick the smallest bound that still suppresses
with a little margin, set U_MAX in src/config.py, then regenerate the DPO and
closed-loop figures (week07 / week08) so the reported result is the saturated
one. The "unbounded" row is the original behaviour, for reference.

Run:
    SCALE=reduced python scripts/analysis_umax_sweep.py
    python scripts/analysis_umax_sweep.py

Outputs:
    results/umax_sweep.csv
    images/umax_sweep.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.detector import CSDDetector
from src.closed_loop import run_closed_loop
from src import metrics


# Bounds to test. None == unconstrained (reference). Edit this list freely.
U_MAX_GRID = [None, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]


def main():
    print("=" * 64)
    print(f"  ACTUATOR-BOUND (u_max) SWEEP   [{C.scale_label()}]")
    print("=" * 64)

    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)

    n_steps = 1600
    s0 = random_initial_state(N, scale=0.05, seed=cm.REP_SEED)

    rows = []
    finite_bounds, energies, resid_amps, suppressed_flags = [], [], [], []

    for umax in U_MAX_GRID:
        net = cm.seizing_network(A, zone)
        res = run_closed_loop(
            net, s0, n_steps, monitor_nodes=zone,
            detector=CSDDetector(window=100), risk_threshold=0.5,
            dpo_horizon=200, dpo_iters=50, dpo_rho=0.01, dpo_q=1.0,
            l1_gain=0.25, control_nodes=zone, refractory=20,
            decision_stride=20, plant_noise=C.NOISE_BETA,
            dpo_u_max=umax, seed=cm.REP_SEED, verbose=False)

        a_ctl = metrics.amplitude(res["traj"], N)
        fa = float(a_ctl[-100:, zone].mean())
        ok = bool(fa < C.SEIZURE_AMP_THRESHOLD)
        label = "unbounded" if umax is None else f"{umax:g}"
        rows.append([label, f"{fa:.4f}", f"{res['energy']:.4f}",
                     len(res["onsets"]), "yes" if ok else "no"])
        print(f"  u_max={label:9s}  zone_amp={fa:.4f}  "
              f"energy={res['energy']:.3f}  suppressed={'yes' if ok else 'no'}")

        if umax is not None:
            finite_bounds.append(umax)
            energies.append(res["energy"])
            resid_amps.append(fa)
            suppressed_flags.append(ok)

    cm.savetxt_table(
        os.path.join(C.RESULTS_DIR, "umax_sweep.csv"),
        ["u_max", "final_zone_amp", "control_energy",
         "n_interventions", "suppressed"], rows)

    # ---- figure: residual amplitude (left axis) & energy (right axis) --- #
    order = np.argsort(finite_bounds)
    b = np.asarray(finite_bounds)[order]
    ra = np.asarray(resid_amps)[order]
    en = np.asarray(energies)[order]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(b, ra, "o-", color="darkorange", label="residual onset-zone amplitude")
    ax1.axhline(C.SEIZURE_AMP_THRESHOLD, color="k", ls=":",
                label=f"seizure threshold ({C.SEIZURE_AMP_THRESHOLD})")
    ax1.set_xlabel("actuator bound  u_max")
    ax1.set_ylabel("residual onset-zone amplitude")
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.plot(b, en, "s--", color="slateblue", label="control energy")
    ax2.set_ylabel("total control energy")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="best")
    ax1.set_title(f"Suppression vs actuator saturation bound\n[{C.scale_label()}]")
    fig.tight_layout()
    out = os.path.join(C.IMAGE_DIR, "umax_sweep.png")
    fig.savefig(out, dpi=200)
    print(f"  saved {out}")

    # ---- recommendation ------------------------------------------------- #
    ok_bounds = [bb for bb, f in zip(finite_bounds, suppressed_flags) if f]
    if ok_bounds:
        print(f"\n  Smallest bound that still suppressed: u_max = {min(ok_bounds):g}")
        print("  -> consider setting config.U_MAX to this (with a little margin).")
    else:
        print("\n  No finite bound in the grid suppressed; widen U_MAX_GRID upward.")


if __name__ == "__main__":
    main()
