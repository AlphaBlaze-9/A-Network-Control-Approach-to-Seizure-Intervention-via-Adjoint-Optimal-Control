"""
analysis_scale_timing.py  --  Reviewer-response addition (60 vs 360 scaling)
============================================================================
Reviewer #5: "the scalability of DPO over the full 360-node matrix is unproven;
the adjoint sweep might be too slow for an online clinical device."

This times the actual cost of the Level-2 adjoint optimisation and one full
closed-loop run at whatever scale is active, so you can report concrete
wall-clock numbers (and the offline-vs-real-time distinction) instead of
hand-waving. Run it once at each scale and quote both:

    SCALE=reduced python scripts/analysis_scale_timing.py     # 60-node
    SCALE=full    python scripts/analysis_scale_timing.py     # 360-node

It uses config.U_MAX, so the timing reflects the saturated controller you
actually report.

Outputs:
    results/scale_timing.csv   (appended one row per run)
    (prints timings to stdout)
"""

import csv
import os
import time

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.detector import CSDDetector
from src.dpo_control import DPOController
from src.closed_loop import run_closed_loop


def main():
    print("=" * 64)
    print(f"  SCALE / TIMING  [{C.scale_label()}]")
    print("=" * 64)

    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    s0 = random_initial_state(N, scale=0.05, seed=C.SEED)
    net = cm.seizing_network(A, zone)

    # --- 1. time a single Level-2 DPO solve (one intervention) ----------- #
    dpo = DPOController(net, q=1.0, rho=0.01, control_nodes=zone)
    t0 = time.perf_counter()
    U, hist = dpo.optimize(s0, horizon=200, dt=C.DT, n_iters=50, lr=0.05,
                           u_max=C.U_MAX)
    t_dpo = time.perf_counter() - t0
    print(f"  N = {N} nodes")
    print(f"  one DPO solve (horizon=200, iters=50): {t_dpo:.3f} s")

    # --- 2. time a full closed-loop run ---------------------------------- #
    n_steps = 1600
    t0 = time.perf_counter()
    res = run_closed_loop(
        net, s0, n_steps, monitor_nodes=zone,
        detector=CSDDetector(window=100), risk_threshold=0.5,
        dpo_horizon=200, dpo_iters=50, dpo_rho=0.01, dpo_q=1.0,
        l1_gain=0.25, control_nodes=zone, refractory=20,
        decision_stride=20, plant_noise=C.NOISE_BETA,
        dpo_u_max=C.U_MAX, seed=C.SEED, verbose=False)
    t_loop = time.perf_counter() - t0
    n_int = len(res["onsets"])
    print(f"  full closed-loop run ({n_steps} steps, {n_int} interventions): "
          f"{t_loop:.3f} s")
    sim_seconds = n_steps * C.DT
    print(f"  -> {t_loop / sim_seconds:.4f} s of wall-clock per second of "
          f"simulated brain time")
    print("     (offline design cost; not a real-time guarantee)")

    # --- append to CSV --------------------------------------------------- #
    path = os.path.join(C.RESULTS_DIR, "scale_timing.csv")
    header = ["scale", "N_nodes", "dpo_solve_s", "closed_loop_s",
              "n_interventions", "wallclock_per_sim_s", "u_max"]
    row = [C.scale_label(), N, f"{t_dpo:.3f}", f"{t_loop:.3f}", n_int,
           f"{t_loop / sim_seconds:.4f}", str(C.U_MAX)]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"  appended timing row to {path}")


if __name__ == "__main__":
    main()
