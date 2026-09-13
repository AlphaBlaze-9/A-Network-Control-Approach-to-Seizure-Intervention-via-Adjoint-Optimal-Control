"""
monte_carlo_validation.py  --  Reviewer-response addition (Statistical Rigor)
=============================================================================
Replaces the single-seed (N=1) headline numbers with a proper Monte-Carlo
estimate. Runs the full closed-loop detect-then-intervene system over
``config.N_SEEDS`` independent trials, each with a different random initial
condition AND a different plant-noise realisation, then reports:

  * suppression success rate as a proportion with a 95% Wilson confidence
    interval (binomial), instead of "9/10";
  * total control energy as mean +/- 95% CI;
  * final onset-zone amplitude as mean +/- 95% CI.

This directly answers the reviewer's "you cannot prove robustness with N=1 /
the controller may have gotten lucky on one noise path" objection.

It uses the SAME closed-loop configuration as week08_closed_loop.py so the
numbers are comparable, and it applies config.U_MAX (the saturation bound) so
that -- once you have set U_MAX from the sweep -- the statistics describe the
clinically-meaningful saturated controller.

Run (start small to estimate wall-clock, then full):
    SCALE=reduced N_SEEDS=20  python scripts/monte_carlo_validation.py
    SCALE=reduced              python scripts/monte_carlo_validation.py
    python scripts/monte_carlo_validation.py        # full 360-node, N_SEEDS trials

Outputs:
    results/monte_carlo_per_trial.csv   -- one row per trial
    results/monte_carlo_summary.csv     -- the headline statistics + CIs
    images/monte_carlo_success_energy.png
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


def _wilson_ci(k, n):
    """95% Wilson interval for a binomial proportion (no SciPy dependency)."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    # allow a quick override of the trial count from the environment
    n_trials = int(os.environ.get("N_SEEDS", C.N_SEEDS))

    print("=" * 64)
    print(f"  MONTE-CARLO VALIDATION  ({n_trials} trials, {C.scale_label()})")
    print(f"  actuator bound U_MAX = {C.U_MAX}")
    print("=" * 64)

    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    print(f"  onset zone: {[labels[i] for i in zone]}")

    n_steps = 1600
    rows = []
    successes, energies, final_amps = [], [], []

    for i in range(n_trials):
        seed = C.MC_SEED0 + i
        s0 = random_initial_state(N, scale=0.05, seed=seed)
        net = cm.seizing_network(A, zone)

        res = run_closed_loop(net, s0, n_steps, monitor_nodes=zone, control_nodes=zone,
                              verbose=False, **cm.closed_loop_kwargs(seed=seed))

        a_ctl = metrics.amplitude(res["traj"], N)
        fa = float(a_ctl[-100:, zone].mean())
        suppressed = bool(fa < C.SEIZURE_AMP_THRESHOLD)

        successes.append(suppressed)
        energies.append(float(res["energy"]))
        final_amps.append(fa)
        rows.append([i, seed, "yes" if suppressed else "no",
                     f"{fa:.4f}", f"{res['energy']:.4f}",
                     len(res["onsets"])])
        print(f"  trial {i:3d}  seed={seed:5d}  "
              f"suppressed={'yes' if suppressed else 'no':3s}  "
              f"zone_amp={fa:.4f}  energy={res['energy']:.3f}")

    # ---- aggregate statistics ------------------------------------------- #
    n = len(successes)
    k = int(np.sum(successes))
    rate = k / n if n else float("nan")
    lo, hi = _wilson_ci(k, n)

    e = np.asarray(energies, dtype=float)
    a = np.asarray(final_amps, dtype=float)
    e_ci = 1.959963984540054 * e.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    a_ci = 1.959963984540054 * a.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")

    print("-" * 64)
    print(f"  SUCCESS: {k}/{n} = {rate:.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  ENERGY : mean {e.mean():.4f}  +/- {e_ci:.4f} (95% CI)")
    print(f"  ZONE AMP: mean {a.mean():.4f}  +/- {a_ci:.4f} (95% CI)")
    print("-" * 64)

    # ---- write CSVs ----------------------------------------------------- #
    cm.savetxt_table(
        os.path.join(C.RESULTS_DIR, "monte_carlo_per_trial.csv"),
        ["trial", "seed", "suppressed", "final_zone_amp",
         "control_energy", "n_interventions"], rows)

    cm.savetxt_table(
        os.path.join(C.RESULTS_DIR, "monte_carlo_summary.csv"),
        ["metric", "value"],
        [["n_trials", n],
         ["successes", k],
         ["success_rate", f"{rate:.4f}"],
         ["success_ci95_low", f"{lo:.4f}"],
         ["success_ci95_high", f"{hi:.4f}"],
         ["energy_mean", f"{e.mean():.4f}"],
         ["energy_ci95_halfwidth", f"{e_ci:.4f}"],
         ["zone_amp_mean", f"{a.mean():.4f}"],
         ["zone_amp_ci95_halfwidth", f"{a_ci:.4f}"],
         ["u_max", str(C.U_MAX)],
         ["scale", C.scale_label()]])

    # ---- figure --------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar([0], [rate], yerr=[[rate - lo], [hi - rate]],
                capsize=8, color="teal", width=0.5)
    axes[0].set_xticks([0]); axes[0].set_xticklabels([f"{k}/{n} trials"])
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("suppression success rate")
    axes[0].set_title(f"Success {rate:.1%}  (95% CI [{lo:.1%}, {hi:.1%}])")

    axes[1].hist(e, bins=min(20, max(5, n // 5)), color="slateblue", alpha=0.85)
    axes[1].axvline(e.mean(), color="crimson", ls="--",
                    label=f"mean {e.mean():.3f} +/- {e_ci:.3f}")
    axes[1].set_xlabel("total control energy"); axes[1].set_ylabel("count")
    axes[1].set_title("Control-energy distribution across trials")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"Monte-Carlo validation  [{C.scale_label()}]  U_MAX={C.U_MAX}")
    fig.tight_layout()
    out = os.path.join(C.IMAGE_DIR, "monte_carlo_success_energy.png")
    fig.savefig(out, dpi=200)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
