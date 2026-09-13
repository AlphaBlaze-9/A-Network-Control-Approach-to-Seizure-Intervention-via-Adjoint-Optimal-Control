"""
analysis_dbs_amplitude_sweep.py  --  Reviewer response, Issue #21
=================================================================
The open-loop DBS surrogate uses a single radial-push amplitude (0.06). Reviewer
#21 notes that the energy comparison could be an artefact of that arbitrary
value. This script re-runs the SAME open-loop surrogate (the constant radial
inward push on the onset zone from week09_ablation.py / monte_carlo_baselines.py)
across a grid of amplitudes and reports, for each: whether it still suppresses
the (intermittent) seizures, and its total control energy. That shows whether
the proposed method's energy comparison holds across a range of amplitudes
rather than at one hand-picked point.

For context it overlays the proposed method's energy (read from
results/monte_carlo_baselines_summary.csv if present, else from
results/mc_proposed_per_trial.csv) so the comparison is on one axis.

Figures written to images/:
    * analysis_dbs_amplitude_sweep.png
Results written to results/:
    * analysis_dbs_amplitude_sweep.csv

Run (reduced scale recommended):
    SCALE=reduced python scripts/analysis_dbs_amplitude_sweep.py
"""

import csv
import os
import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state


def _proposed_energy_reference():
    """Best-effort: proposed-method mean control energy from existing MC output."""
    summ = os.path.join(C.RESULTS_DIR, "monte_carlo_baselines_summary.csv")
    if os.path.exists(summ):
        with open(summ, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("condition", "").lower().startswith("proposed"):
                    try:
                        return float(r["energy_mean"])
                    except (KeyError, ValueError):
                        pass
    per = os.path.join(C.RESULTS_DIR, "mc_proposed_per_trial.csv")
    if os.path.exists(per):
        with open(per, newline="", encoding="utf-8") as f:
            e = [float(r["energy"]) for r in csv.DictReader(f)]
        if e:
            return float(np.mean(e))
    return None


def main():
    cm.banner(0, "Issue #21: open-loop DBS amplitude sweep")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = np.asarray(cm.onset_zone(A, labels, focus))
    net = cm.seizing_network(A, zone)
    s0 = random_initial_state(N, scale=0.05, seed=C.SEED)

    mask = np.zeros(N)
    mask[zone] = 1.0

    # Intermittent-seizure scenario, identical to week09 / monte_carlo_baselines
    ep_steps = 2400
    ictal_windows = [(500, 850), (1500, 1950)]
    te = cm.time_axis(ep_steps)

    def a_at(k):
        a = np.full(N, C.A_REST)
        if any(lo <= k < hi for lo, hi in ictal_windows):
            a[zone] = C.A_SEIZURE
        return a

    ictal_mask = np.zeros(ep_steps, dtype=bool)
    for lo, hi in ictal_windows:
        ictal_mask[lo:hi] = True

    def run_amp(amp):
        rng = np.random.default_rng(C.SEED)
        s = s0.copy()
        e_tot = 0.0
        amps = np.empty(ep_steps)
        for k in range(ep_steps):
            net.a = a_at(k)
            x, y = s[:N], s[N:]
            r = np.sqrt(x * x + y * y) + 1e-9
            u = -amp * np.concatenate([mask * x / r, mask * y / r])
            e_tot += C.DT * float(np.sum(u ** 2))
            s = net.rk4_step(s, C.DT, u=u, noise=C.NOISE_BETA, rng=rng)
            amps[k] = np.sqrt(s[zone] ** 2 + s[zone + N] ** 2).mean()
        ictal_amp = float(amps[ictal_mask].mean())
        return ictal_amp, e_tot

    grid = list(C.DBS_AMP_GRID)
    rows = []
    ener, supp = [], []
    for amp in grid:
        ictal_amp, e_tot = run_amp(amp)
        ok = ictal_amp < C.SEIZURE_AMP_THRESHOLD
        ener.append(e_tot); supp.append(ictal_amp)
        flag = "  <-- value used in paper" if abs(amp - 0.06) < 1e-9 else ""
        print(f"   [amp={amp:.2f}] ictal amp={ictal_amp:.4f}  energy={e_tot:.3f}  "
              f"suppressed={'yes' if ok else 'no'}{flag}")
        rows.append([f"{amp:.3f}", f"{ictal_amp:.4f}", f"{e_tot:.4f}",
                     "yes" if ok else "no"])
    net.a = a_at(0)

    prop_e = _proposed_energy_reference()
    cm.savetxt_table(
        f"{C.RESULTS_DIR}/analysis_dbs_amplitude_sweep.csv",
        ["openloop_amplitude", "ictal_zone_amp", "total_energy",
         "seizure_suppressed"], rows)
    if prop_e is not None:
        print(f"   proposed-method energy reference = {prop_e:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(grid, ener, "s-", color="navy", label="open-loop energy")
    if prop_e is not None:
        ax1.axhline(prop_e, color="seagreen", ls="--",
                    label=f"proposed energy ({prop_e:.3f})")
    ax1.set_xlabel("open-loop DBS amplitude")
    ax1.set_ylabel("total control energy", color="navy")
    ax1.tick_params(axis="y", labelcolor="navy")
    ax2 = ax1.twinx()
    ax2.plot(grid, supp, "o:", color="crimson", label="residual ictal amplitude")
    ax2.axhline(C.SEIZURE_AMP_THRESHOLD, color="gray", ls=":",
                label="ictal threshold")
    ax2.set_ylabel("residual ictal amplitude", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="best")
    ax1.set_title(f"Open-loop DBS amplitude sweep  [{C.scale_label()}]")
    fig.tight_layout()
    fig.savefig(f"{C.IMAGE_DIR}/analysis_dbs_amplitude_sweep.png", dpi=200)
    print(f"   saved {C.IMAGE_DIR}/analysis_dbs_amplitude_sweep.png")


if __name__ == "__main__":
    main()
