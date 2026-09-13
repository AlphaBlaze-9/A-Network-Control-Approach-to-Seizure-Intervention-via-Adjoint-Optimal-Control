"""
scripts/analysis_duty_cycle_energy.py
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as cm
from src import config as C

def _read_e_int_from_summary():
    path = os.path.join(C.RESULTS_DIR, "week08_closed_loop_summary.csv")
    if not os.path.exists(path): return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("condition", "").strip() == "clean":
                    energy = float(row["control_energy"])
                    n_int = max(int(row["n_interventions"]), 1)
                    return energy / n_int
    except Exception: return None
    return None

def _read_fpr_from_testfold():
    path = os.path.join(C.RESULTS_DIR, "week05_detector_testfold.csv")
    if not os.path.exists(path): return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("fold") == "test":
                    return float(row["FPR"])
    except Exception: return None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-int", type=float, default=None)
    ap.add_argument("--n-seizures", type=float, default=8.0)
    ap.add_argument("--n-control", type=int, default=None)
    ap.add_argument("--openloop-amp", type=float, default=0.06)
    ap.add_argument("--max-false", type=float, default=1000.0)
    args = ap.parse_args()

    e_int = args.e_int if args.e_int is not None else _read_e_int_from_summary()
    if e_int is None: e_int = 0.099

    fpr = _read_fpr_from_testfold()
    if fpr is None: 
        fpr = 0.05
        print("  [!] Could not read FPR from week05 testfold; assuming 0.05.")

    n_control = args.n_control
    if n_control is None:
        try:
            A, labels, focus = cm.load_network()
            n_control = len(cm.onset_zone(A, labels, focus))
        except Exception:
            n_control = 6

    T_day = 86400.0
    e_day_open = n_control * (args.openloop_amp ** 2) * T_day

    # FIX: Calculate exact false triggers per day based on debounce rate.
    # The detector is evaluated on 120-second (2400 step * 0.05s) blocks in week05.
    blocks_per_day = T_day / 120.0
    actual_false_per_day = fpr * blocks_per_day

    n_false = np.linspace(0.0, args.max_false, 400)
    e_day_closed = (args.n_seizures + n_false) * e_int

    f_star = e_day_open / e_int - args.n_seizures

    print("=" * 64)
    print("  24-HOUR DUTY-CYCLE ENERGY")
    print("=" * 64)
    print(f"  E_intervention (Level-2)          : {e_int:.4f}")
    print(f"  genuine seizures/day              : {args.n_seizures:g}")
    print(f"  actuated nodes (n_control)        : {n_control}")
    print(f"  open-loop daily energy            : {e_day_open:.2f}")
    print(f"  -> CROSSOVER false activations/day: {f_star:.1f}")
    print(f"  -> ACTUAL false activations/day   : {actual_false_per_day:.1f} (based on FPR={fpr:.3f})")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_false, e_day_closed, color="seagreen", label="closed-loop daily energy")
    ax.axhline(e_day_open, color="crimson", ls="--", label="continuous open-loop DBS")
    
    if 0 < f_star < args.max_false:
        ax.axvline(f_star, color="gray", ls=":", label=f"crossover @ {f_star:.0f}/day")
    
    if actual_false_per_day < args.max_false:
        ax.axvline(actual_false_per_day, color="blue", ls="-.", label=f"actual @ {actual_false_per_day:.0f}/day")

    ax.set_xlabel("false activations per day")
    ax.set_ylabel("aggregate daily control energy")
    ax.set_title("Closed-loop vs open-loop daily energy\n" f"(E_intervention={e_int:.3f}, FPR={fpr:.3f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(C.IMAGE_DIR, "duty_cycle_energy.png")
    fig.savefig(out, dpi=200)
    print(f"  saved {out}")

if __name__ == "__main__":
    main()