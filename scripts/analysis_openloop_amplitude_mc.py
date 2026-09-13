"""
analysis_openloop_amplitude_mc.py  --  Reviewer response (open-loop amplitude)
==============================================================================
Reviewer 3 (comment 3): the open-loop amplitude (0.06) was hand-picked; "if the
open-loop amplitude changes, the results may change substantially."

The open-loop surrogate is a constant radial inward push of amplitude a on the
onset zone, so its total energy is exactly a^2 * |zone| * T -- it scales with the
square of the amplitude and is not a tuned optimum. This script sweeps the
amplitude over the same grid as analysis_dbs_amplitude_sweep.py, but as a
Monte-Carlo experiment (randomised initial condition + plant noise per trial)
and in BOTH scenarios used in the paper:

  * "constant"     -- onset zone seizure-prone for the whole 80-s run (1600 steps);
                      the scenario used for the proposed method and the PI baseline
                      in Table 2, suppression judged on the final 100 steps.
  * "intermittent" -- 120-s run (2400 steps) with two ictal windows
                      (25-42.5 s, 75-97.5 s); suppression judged inside the windows.

Outputs
    results/analysis_openloop_amplitude_mc.csv
    images/analysis_openloop_amplitude_mc.png
Run:
    SCALE=reduced python scripts/analysis_openloop_amplitude_mc.py [--n 20]
"""
import argparse
import os
import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state

Z = 1.959963984540054


def _wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run_trial(A, zone, seed, amp, scenario):
    net = cm.seizing_network(A, zone)
    N = net.N
    s0 = random_initial_state(N, scale=0.05, seed=seed)
    rng = np.random.default_rng(seed)
    mask = np.zeros(N); mask[np.asarray(zone)] = 1.0
    zone_arr = np.asarray(zone)
    if scenario == "constant":
        n_steps = cm.CL_N_STEPS
        windows = [(0, n_steps)]
    else:
        n_steps = 2400
        windows = [(500, 850), (1500, 1950)]

    def a_at(k):
        a = np.full(N, C.A_REST)
        if any(lo <= k < hi for lo, hi in windows):
            a[zone_arr] = C.A_SEIZURE
        return a

    s = s0.copy(); e_tot = 0.0; amps = np.empty(n_steps)
    for k in range(n_steps):
        net.a = a_at(k)
        x, y = s[:N], s[N:]
        r = np.sqrt(x * x + y * y) + 1e-9
        u = -amp * np.concatenate([mask * x / r, mask * y / r])
        e_tot += C.DT * float(np.sum(u ** 2))
        s = net.rk4_step(s, C.DT, u=u, noise=C.NOISE_BETA, rng=rng)
        amps[k] = np.sqrt(s[zone_arr] ** 2 + s[zone_arr + N] ** 2).mean()
    if scenario == "constant":
        judged = float(amps[-100:].mean())
    else:
        m = np.zeros(n_steps, dtype=bool)
        for lo, hi in windows:
            m[lo:hi] = True
        judged = float(amps[m].mean())
    return judged < C.SEIZURE_AMP_THRESHOLD, e_tot, judged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--amps", type=float, nargs="*", default=list(C.DBS_AMP_GRID))
    a = ap.parse_args()
    cm.banner(0, "Open-loop DBS surrogate: amplitude sweep, Monte Carlo, both scenarios")
    A, labels, focus = cm.load_network()
    zone = cm.onset_zone(A, labels, focus)
    rows, pts = [], {}
    for scenario in ("constant", "intermittent"):
        for amp in a.amps:
            ok, en, jg = [], [], []
            for i in range(a.n):
                seed = C.MC_SEED0 + i
                s_ok, s_e, s_j = run_trial(A, zone, seed, amp, scenario)
                ok.append(s_ok); en.append(s_e); jg.append(s_j)
            k = int(np.sum(ok)); n = len(ok); lo, hi = _wilson(k, n)
            en = np.array(en); jg = np.array(jg)
            e_ci = Z * en.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            print(f"   [{scenario:12s} amp={amp:.2f}] success={k}/{n} CI[{lo:.2f},{hi:.2f}] "
                  f"energy={en.mean():.4f} (+/-{e_ci:.4f}) judged amp={jg.mean():.4f}", flush=True)
            rows.append([scenario, f"{amp:.3f}", n, k, f"{k/n:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                         f"{en.mean():.4f}", f"{e_ci:.4f}", f"{jg.mean():.4f}"])
            pts.setdefault(scenario, []).append((amp, k / n, lo, hi, en.mean(), jg.mean()))
    cm.savetxt_table(f"{C.RESULTS_DIR}/analysis_openloop_amplitude_mc.csv",
                     ["scenario", "openloop_amplitude", "n_trials", "n_success", "success_rate", "ci95_low", "ci95_high",
                      "energy_mean", "energy_ci95_halfwidth", "judged_zone_amp_mean"], rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, scenario in zip(axes, ("constant", "intermittent")):
        P = pts[scenario]
        x = [p[0] for p in P]
        ax.errorbar(x, [p[1] for p in P], yerr=[[p[1] - p[2] for p in P], [p[3] - p[1] for p in P]],
                    fmt="o-", capsize=5, color="teal", label="suppression rate (Wilson 95% CI)")
        ax.set_ylim(-0.05, 1.08); ax.set_xlabel("open-loop push amplitude"); ax.set_ylabel("suppression success rate", color="teal")
        ax2 = ax.twinx()
        ax2.plot(x, [p[4] for p in P], "s--", color="navy", label="total control energy")
        ax2.set_ylabel("total control energy", color="navy")
        ax.set_title(f"{scenario} scenario")
        l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, n1 + n2, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(C.IMAGE_DIR, "analysis_openloop_amplitude_mc.png"), dpi=150, bbox_inches="tight")
    print("   saved images/analysis_openloop_amplitude_mc.png")


if __name__ == "__main__":
    main()
