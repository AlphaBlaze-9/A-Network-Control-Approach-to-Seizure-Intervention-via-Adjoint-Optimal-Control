"""
week08_closed_loop.py  --  Phase 3 / Week 8
===========================================
Closed-Loop System Integration.

Ties everything together: the always-on Level-1 LQR, the CSD detector watching
the network's mean amplitude, and the Level-2 DPO optimal controller that is
triggered (receding-horizon / MPC style) when risk crosses threshold. We
compare the controlled trajectory to the uncontrolled seizure, mark the
detector's intervention onsets, and -- crucially -- probe robustness by feeding
the detector a biased/noisy risk signal to check the controller still drives
the network off the unsafe set.

Figures written to images/:
  * week08_closed_loop_timeseries.png   -- uncontrolled vs closed-loop amplitude
  * week08_closed_loop_risk.png         -- risk signal with intervention onsets
  * week08_closed_loop_raster.png       -- node x time amplitude, controlled
  * week08_detector_robustness.png      -- suppression under detector bias/noise

Results written to results/:
  * week08_closed_loop_summary.csv

Run (reduced scale recommended; full scale is slow):
    SCALE=reduced python scripts/week08_closed_loop.py
    python scripts/week08_closed_loop.py
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.detector import CSDDetector
from src.closed_loop import run_closed_loop
from src import metrics, viz


def main():
    cm.banner(8, "Closed-Loop System Integration")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    print(f"   onset zone: {[labels[i] for i in zone]}")

    n_steps = 1600
    t = cm.time_axis(n_steps)
    s0 = random_initial_state(N, scale=0.05, seed=cm.REP_SEED)
    net = cm.seizing_network(A, zone)

    # --- 1. Uncontrolled reference seizure ---------------------------------- #
    unc = net.simulate(s0, n_steps, noise=C.NOISE_BETA, seed=cm.REP_SEED)
    amp_unc = metrics.amplitude(unc, N)

    # --- 2. Closed loop (detector-triggered two-level control) -------------- #
    res = run_closed_loop(net, s0, n_steps, monitor_nodes=zone, control_nodes=zone,
                          verbose=True, **cm.closed_loop_kwargs())
    amp_ctl = metrics.amplitude(res["traj"], N)
    onsets_s = [o * C.DT for o in res["onsets"]]

    viz.line_plot(t,
                  {"zone mean - uncontrolled": amp_unc[:, zone].mean(1),
                   "zone mean - closed loop": amp_ctl[:, zone].mean(1),
                   "network mean - uncontrolled": amp_unc.mean(1),
                   "network mean - closed loop": amp_ctl.mean(1)},
                  "Closed-loop control vs uncontrolled seizure",
                  "amplitude r(t)", "week08_closed_loop_timeseries.png",
                  vlines=onsets_s, hline=C.SEIZURE_AMP_THRESHOLD)
    print(f"   zone amplitude  uncontrolled: {amp_unc[-100:, zone].mean():.4f}"
          f"   closed-loop: {amp_ctl[-100:, zone].mean():.4f}")
    print(f"   interventions at steps {res['onsets']}  total energy {res['energy']:.3f}")

    # --- 3. Risk signal with intervention markers --------------------------- #
    viz.line_plot(t[:-1], {"detector risk": res["risk"]},
                  "Detector risk and Level-2 interventions",
                  "risk", "week08_closed_loop_risk.png",
                  vlines=onsets_s, hline=0.5)

    # --- 4. Controlled raster (nodes sorted by uncontrolled recruitment) ---- #
    order = np.argsort(amp_unc[-200:].mean(0))[::-1]
    viz.heatmap(amp_ctl[:, order].T,
                "Closed-loop controlled activity (nodes sorted by recruitment)",
                "week08_closed_loop_raster.png",
                ylabel="region (sorted)", extent=[0, t[-1], 0, N])

    # --- 5. Detector-robustness sweep: biased / noisy risk ------------------ #
    # The Week-5 safeguard in action: even when the detector's risk read-out is
    # corrupted, does the closed loop still drive the network off the unsafe set?
    conditions = {
        "clean": dict(detector_noise=0.0, detector_bias=0.0),
        "noisy risk": dict(detector_noise=0.15, detector_bias=0.0),
        "negative bias": dict(detector_noise=0.0, detector_bias=-0.2),
        "positive bias": dict(detector_noise=0.0, detector_bias=+0.2),
    }
    rows = []
    final_amps = {}
    for name, kw in conditions.items():
        r = run_closed_loop(net, s0, n_steps, monitor_nodes=zone, control_nodes=zone,
                            **cm.closed_loop_kwargs(**kw))
        a_ctl = metrics.amplitude(r["traj"], N)
        fa = a_ctl[-100:, zone].mean()
        final_amps[name] = a_ctl[:, zone].mean(1)
        rows.append([name, f"{fa:.4f}", len(r["onsets"]), f"{r['energy']:.3f}",
                     "yes" if fa < C.SEIZURE_AMP_THRESHOLD else "no"])
        print(f"   [{name:14s}] zone amp={fa:.4f}  onsets={len(r['onsets'])}"
              f"  energy={r['energy']:.3f}")

    viz.line_plot(t, {f"{k}": v for k, v in final_amps.items()},
                  "Closed-loop robustness to detector bias / noise",
                  "onset-zone mean amplitude",
                  "week08_detector_robustness.png",
                  hline=C.SEIZURE_AMP_THRESHOLD)

    cm.savetxt_table(f"{C.RESULTS_DIR}/week08_closed_loop_summary.csv",
                     ["condition", "final_zone_amp", "n_interventions",
                      "control_energy", "seizure_suppressed"], rows)


if __name__ == "__main__":
    main()
