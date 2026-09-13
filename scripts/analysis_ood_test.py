"""
analysis_ood_test.py  --  Out-of-distribution generalisation of the closed loop
===============================================================================
The detector, controller and evaluation were all built at one operating point
(G=2.5, omega=0.10 Hz, a_seizure=0.6). This script re-runs the FULL closed loop
on networks whose plant parameters are shifted away from those defaults, one
axis at a time, and reports the suppression success rate (Wilson 95% CI).

Axes swept (held-out values around the in-distribution default):
    G_COUPLING   : config.OOD_G_GRID
    OMEGA_HZ     : config.OOD_OMEGA_GRID
    A_SEIZURE    : config.OOD_ASEIZ_GRID

For each (axis, value) we run config.OOD_N_SEEDS trials (seeds MC_SEED0..),
each with randomized init + plant noise, using the SAME closed-loop settings
as the Monte-Carlo benchmark (``_common.closed_loop_kwargs``), so only the
plant changes.

Run one axis per process (parallel-friendly), then combine:
    SCALE=reduced python scripts/analysis_ood_test.py --axis G_COUPLING
    SCALE=reduced python scripts/analysis_ood_test.py --axis OMEGA_HZ
    SCALE=reduced python scripts/analysis_ood_test.py --axis A_SEIZURE
    # or one grid value per process:
    SCALE=reduced python scripts/analysis_ood_test.py --axis A_SEIZURE --values 0.75
    SCALE=reduced python scripts/analysis_ood_test.py --combine

Results written to results/:
    * analysis_ood_<axis>.csv        (per axis)
    * analysis_ood_success.csv       (combined)
Figures written to images/:
    * analysis_ood_success.png
"""
import argparse
import csv
import os
import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import HopfNetwork, zone_excitability, random_initial_state
from src.closed_loop import run_closed_loop
from src import metrics


Z = 1.959963984540054
AXES = {
    "G_COUPLING": (C.OOD_G_GRID, lambda v: (v, C.OMEGA_HZ, C.A_SEIZURE)),
    "OMEGA_HZ": (C.OOD_OMEGA_GRID, lambda v: (C.G_COUPLING, v, C.A_SEIZURE)),
    "A_SEIZURE": (C.OOD_ASEIZ_GRID, lambda v: (C.G_COUPLING, C.OMEGA_HZ, v)),
}
HEADER = ["axis", "value", "n_trials", "n_success", "success_rate", "ci95_low", "ci95_high",
          "is_in_distribution", "energy_mean", "mean_n_interventions"]


def _wilson_ci(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = (Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _build_net(A, zone, G, omega_hz, a_seizure):
    N = A.shape[0]
    a = zone_excitability(N, zone, a_seizure=a_seizure, a_rest=C.A_REST)
    omega = np.full(N, 2 * np.pi * omega_hz)
    return HopfNetwork(A, a=a, omega=omega, G=G)


def _run_point(A, zone, G, omega_hz, a_seizure, n_seeds, n_steps=cm.CL_N_STEPS):
    k = 0; energies = []; n_int = []
    for i in range(n_seeds):
        seed = C.MC_SEED0 + i
        net = _build_net(A, zone, G, omega_hz, a_seizure)
        N = net.N
        s0 = random_initial_state(N, scale=0.05, seed=seed)
        res = run_closed_loop(net, s0, n_steps, monitor_nodes=zone, control_nodes=zone,
                              **cm.closed_loop_kwargs(seed=seed))
        amp = metrics.amplitude(res["traj"], N)
        final = float(amp[-100:, zone].mean())
        if final < C.SEIZURE_AMP_THRESHOLD:
            k += 1
        energies.append(res["energy"]); n_int.append(len(res["onsets"]))
    return k, float(np.mean(energies)), float(np.mean(n_int))


def _merge_axis(axis):
    """Assemble results/analysis_ood_<axis>.csv from the per-value CSVs."""
    grid, _ = AXES[axis]
    rows = []
    for v in grid:
        pv = f"{C.RESULTS_DIR}/analysis_ood_{axis}_{v}.csv"
        if os.path.exists(pv):
            rows += [[r[h] for h in HEADER] for r in csv.DictReader(open(pv, newline="", encoding="utf-8"))]
    if rows:
        cm.savetxt_table(f"{C.RESULTS_DIR}/analysis_ood_{axis}.csv", HEADER, rows)


def run_axis(axis, n, values=None):
    """Run the grid points of one axis (optionally a subset via ``values``).
    Each point is written to its own CSV (skipped if it already exists), so
    points can run in parallel processes; the axis CSV is then re-assembled."""
    A, labels, focus = cm.load_network()
    zone = cm.onset_zone(A, labels, focus)
    grid, unpack = AXES[axis]
    todo = grid if values is None else [v for v in grid if any(abs(v - x) < 1e-9 for x in values)]
    for v in todo:
        pv = f"{C.RESULTS_DIR}/analysis_ood_{axis}_{v}.csv"
        if os.path.exists(pv):
            print(f"   [{axis:10s}={v:<6}] per-value CSV exists -- skipping (delete it to re-run)")
            continue
        G, om, asz = unpack(v)
        k, e_mean, ni = _run_point(A, zone, G, om, asz, n)
        lo, hi = _wilson_ci(k, n)
        is_default = (abs(G - C.G_COUPLING) < 1e-9 and abs(om - C.OMEGA_HZ) < 1e-9 and abs(asz - C.A_SEIZURE) < 1e-9)
        print(f"   [{axis:10s}={v:<6}] success={k}/{n}={k/n:.1%} CI[{lo:.1%},{hi:.1%}] energy={e_mean:.3f} "
              f"n_int={ni:.2f}{'  <-- in-distribution' if is_default else ''}", flush=True)
        cm.savetxt_table(pv, HEADER, [[axis, f"{v}", n, k, f"{k/n:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                                       "yes" if is_default else "no", f"{e_mean:.4f}", f"{ni:.2f}"]])
    _merge_axis(axis)


def combine():
    rows = []
    for axis in AXES:
        _merge_axis(axis)
        p = f"{C.RESULTS_DIR}/analysis_ood_{axis}.csv"
        if not os.path.exists(p):
            print(f"   [!] missing {p}"); continue
        rows += [[r[h] for h in HEADER] for r in csv.DictReader(open(p, newline="", encoding="utf-8"))]
    cm.savetxt_table(f"{C.RESULTS_DIR}/analysis_ood_success.csv", HEADER, rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    defaults = {"G_COUPLING": C.G_COUPLING, "OMEGA_HZ": C.OMEGA_HZ, "A_SEIZURE": C.A_SEIZURE}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, axis in zip(axs, AXES):
        R = [r for r in rows if r[0] == axis]
        if not R:
            continue
        xs = np.array([float(r[1]) for r in R]); rates = np.array([float(r[4]) for r in R])
        lo = np.array([float(r[5]) for r in R]); hi = np.array([float(r[6]) for r in R])
        ax.errorbar(xs, rates, yerr=[rates - lo, hi - rates], fmt="o-", capsize=5, color="teal")
        ax.axvline(defaults[axis], color="crimson", ls=":", lw=1, label="in-distribution")
        ax.axhline(1.0, color="gray", ls="--", lw=0.7)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel({"G_COUPLING": "global coupling G", "OMEGA_HZ": "intrinsic frequency (Hz)",
                       "A_SEIZURE": "focus excitability r_focus"}[axis])
        ax.set_ylabel("suppression success rate")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{C.IMAGE_DIR}/analysis_ood_success.png", dpi=200, bbox_inches="tight")
    print(f"   saved {C.IMAGE_DIR}/analysis_ood_success.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", nargs="*", default=None, choices=list(AXES))
    ap.add_argument("--values", type=float, nargs="*", default=None,
                    help="subset of grid values to run (one process per value is fine)")
    ap.add_argument("--n", type=int, default=int(C.OOD_N_SEEDS))
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    cm.banner(0, "Out-of-distribution sweep over plant parameters")
    if a.combine:
        combine(); return
    print(f"   in-distribution default: G={C.G_COUPLING}, omega={C.OMEGA_HZ}Hz, a_seizure={C.A_SEIZURE}   |   "
          f"{a.n} seeds/point   [{C.scale_label()}]")
    for axis in (a.axis or list(AXES)):
        run_axis(axis, a.n, a.values)


if __name__ == "__main__":
    main()
