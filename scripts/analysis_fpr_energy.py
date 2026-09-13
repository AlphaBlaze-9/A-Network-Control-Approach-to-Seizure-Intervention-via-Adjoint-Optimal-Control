"""
analysis_fpr_energy.py  --  FPR-aware daily energy accounting (Section 5.3)
===========================================================================
Puts the detector's measured false-positive rate into the daily-energy argument
explicitly and reports whether the intermittent (closed-loop) scheme is still
cheaper than continuous open-loop stimulation.

Model (all energies in the paper's dimensionless control-energy units; time in
model seconds):
    E_episode        = mean over Monte-Carlo trials of (trial energy / number of
                       Level-2 interventions) -- the cost of one intervention
                       episode INCLUDING the always-on Level-1 share
    FPR              = test-fold per-record false-positive rate of the detector
                       (fraction of 120-s genuine-negative records with >= 1
                       sustained alarm), read from week05_detector_testfold.csv
    records/day      = 86400 / 120 = 720
    false acts/day   = FPR * (records/day - genuine events/day)
    closed-loop/day  = (genuine events + false acts) * E_episode
    open-loop/day    = amp^2 * |zone| * 86400        (constant radial push, amp 0.06)
    break-even       = open-loop/day / E_episode - genuine events

Outputs
    results/analysis_fpr_energy.csv
    results/analysis_fpr_energy.md
"""
import csv
import os
import numpy as np

import _common as cm
from src import config as C

OPENLOOP_AMP = 0.06


def _rows(name):
    p = os.path.join(C.RESULTS_DIR, name)
    return list(csv.DictReader(open(p, newline="", encoding="utf-8"))) if os.path.exists(p) else None


def main():
    cm.banner(0, "FPR-aware daily energy accounting")
    A, labels, focus = cm.load_network()
    zone = cm.onset_zone(A, labels, focus)
    R = _rows("mc_proposed_per_trial.csv")
    per_ep = np.array([float(r["energy"]) / max(int(float(r["n_interventions"])), 1) for r in R])
    e_episode = float(per_ep.mean())
    e_trial = float(np.mean([float(r["energy"]) for r in R]))
    n_int = float(np.mean([float(r["n_interventions"]) for r in R]))
    fpr = C.DETECTOR_FPR
    if fpr is None:
        T = _rows("week05_detector_testfold.csv")
        fpr = float([r for r in T if r["fold"] == "test"][0]["FPR"])
    events = C.ASSUMED_SEIZURES_DAY
    records = C.DETECTOR_EVALS_DAY
    false_acts = fpr * max(records - events, 0)
    acts = events + false_acts
    closed_daily = acts * e_episode
    open_daily = OPENLOOP_AMP ** 2 * len(zone) * 86400.0
    open_per_80s = OPENLOOP_AMP ** 2 * len(zone) * cm.CL_N_STEPS * C.DT
    breakeven = open_daily / e_episode - events

    print(f"   E per intervention episode (incl. Level-1)  = {e_episode:.4f}  (trial mean {e_trial:.4f} over {n_int:.2f} episodes)")
    print(f"   detector FPR (per 120-s record, test fold)  = {fpr:.3f}")
    print(f"   genuine events/day (ASSUMPTION)              = {events}")
    print(f"   detector records/day                         = {records}")
    print(f"   expected false activations/day               = {false_acts:.1f}")
    print(f"   closed-loop activations/day                  = {acts:.1f}")
    print(f"   closed-loop daily energy                     = {closed_daily:.1f}")
    print(f"   open-loop energy per 80-s run                = {open_per_80s:.4f}")
    print(f"   open-loop (continuous) daily energy          = {open_daily:.1f}")
    print(f"   break-even false activations/day             = {breakeven:.0f}")
    print(f"   >> closed loop is {'CHEAPER' if closed_daily < open_daily else 'NOT cheaper'} "
          f"by a factor of {open_daily / closed_daily:.1f}")

    cm.savetxt_table(
        f"{C.RESULTS_DIR}/analysis_fpr_energy.csv", ["quantity", "value"],
        [["energy_per_episode_proposed", f"{e_episode:.4f}"],
         ["energy_per_trial_proposed", f"{e_trial:.4f}"],
         ["mean_interventions_per_trial", f"{n_int:.2f}"],
         ["detector_fpr_per_record", f"{fpr:.3f}"],
         ["record_length_s", f"{C.DETECTOR_RECORD_S:.0f}"],
         ["assumed_events_per_day", f"{events}"],
         ["detector_records_per_day", f"{records}"],
         ["expected_false_activations_per_day", f"{false_acts:.1f}"],
         ["closedloop_activations_per_day", f"{acts:.1f}"],
         ["closedloop_daily_energy", f"{closed_daily:.1f}"],
         ["openloop_amplitude", f"{OPENLOOP_AMP}"],
         ["openloop_energy_per_80s", f"{open_per_80s:.4f}"],
         ["openloop_daily_energy", f"{open_daily:.1f}"],
         ["breakeven_false_activations_per_day", f"{breakeven:.0f}"],
         ["energy_ratio_open_over_closed", f"{open_daily / closed_daily:.2f}"]])
    para = (f"With the measured per-record false-positive rate of {fpr:.2f} (fraction of 120-s "
            f"seizure-free records producing at least one alarm), a day of {records} such records "
            f"yields about {false_acts:.0f} spurious Level-2 episodes in addition to the {events} assumed "
            f"genuine events, i.e. about {acts:.0f} episodes/day at {e_episode:.3f} energy units each "
            f"({closed_daily:.0f} units/day), versus {open_daily:.0f} units/day for a continuous open-loop "
            f"push of amplitude {OPENLOOP_AMP} on {len(zone)} nodes. The crossover is {breakeven:.0f} false "
            f"episodes/day.")
    with open(f"{C.RESULTS_DIR}/analysis_fpr_energy.md", "w", encoding="utf-8") as f:
        f.write("# Section 5.3 -- FPR-aware energy accounting\n\n" + para + "\n")
    print("   " + para)


if __name__ == "__main__":
    main()
