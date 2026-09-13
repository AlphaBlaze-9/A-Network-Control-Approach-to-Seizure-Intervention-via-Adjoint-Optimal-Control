"""
analysis_refractory.py  --  Reviewer response, Issue #20 (refractory period)
============================================================================
Reviewer #20 asks: after a Level-2 schedule ends, a fixed 20-step refractory
period blocks re-solving -- so what happens if a SECOND seizure onset begins
inside that window? This script answers it directly and honestly.

What it isolates:
    The concern is about the trigger/refractory *scheduler*, not about which
    optimiser computed the burst. So we reproduce the project's exact
    scheduling logic -- always-on Level-1 LQR (the real src.lqr_control.LocalLQR)
    plus a detector-triggered Level-2 burst that plays open-loop for
    DPO_HORIZON steps, followed by a REFRACTORY-step lockout before another
    solve is allowed -- and we inject a second ictal onset at a controlled time.
    The Level-2 burst here is the same triggered radial-inward push used as the
    adaptive clinical controller in week09_ablation.py (ctl_adaptive), so the
    scheduling we test is the project's own; only the optimiser-internals (which
    do not affect refractory timing) are abstracted.

Three scenarios (same two-onset disease course, different 2nd-onset timing):
    A. 2nd onset DURING the active Level-2 schedule  -> covered by the ongoing
       open-loop burst (+ L1). Expected: suppressed.
    B. 2nd onset INSIDE the refractory lockout (schedule ended, no re-solve yet)
       -> only Level-1 is acting. This is the exact failure mode #20 names;
       we report truthfully whether L1 alone holds the network.
    C. 2nd onset AFTER refractory ends -> a fresh Level-2 solve fires normally.

Figures written to images/:
    * analysis_refractory_timeseries.png
Results written to results/:
    * analysis_refractory_summary.csv

Run (reduced scale recommended):
    SCALE=reduced python scripts/analysis_refractory.py
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.lqr_control import LocalLQR


def main():
    cm.banner(0, "Issue #20: second onset inside the refractory window")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = np.asarray(cm.onset_zone(A, labels, focus))
    net = cm.seizing_network(A, zone)          # focus/zone seizure-capable
    s0 = random_initial_state(N, scale=0.05, seed=C.SEED)

    lqr = LocalLQR(a=np.full(N, C.A_REST), omega=net.omega)
    mask = np.zeros(N)
    mask[zone] = 1.0

    horizon = C.DPO_HORIZON
    refractory = C.REFRACTORY
    trigger_frac = 0.5                          # escalation trigger (as in week09)
    burst_amp = 0.15                            # triggered radial-push amplitude
    l1_gain = 0.25

    ep_steps = 1600
    te = cm.time_axis(ep_steps)

    first_onset = (300, 650)                    # first ictal window (steps)
    # Place the SECOND onset's start in each regime relative to the schedule the
    # first onset triggers. The first trigger fires shortly after step 300; the
    # schedule then occupies ~[trig, trig+horizon] and the refractory lockout is
    # ~[trig+horizon, trig+horizon+refractory]. We approximate trig ~ 330.
    trig_guess = 330
    second_starts = {
        "A: during active schedule": trig_guess + horizon // 2,
        "B: inside refractory lockout": trig_guess + horizon + refractory // 2,
        "C: after refractory ends": trig_guess + horizon + refractory + 60,
    }

    def excitability_at(k, second_start):
        a = np.full(N, C.A_REST)
        in_first = first_onset[0] <= k < first_onset[1]
        in_second = second_start <= k < second_start + 350
        if in_first or in_second:
            a[zone] = C.A_SEIZURE
        return a

    def run(second_start):
        """Explicit two-level loop with the real trigger/refractory scheduler."""
        rng = np.random.default_rng(C.SEED)
        s = s0.copy()
        amp_tr = np.empty(ep_steps + 1)
        amp_tr[0] = np.sqrt(s[zone] ** 2 + s[zone + N] ** 2).mean()
        schedule_active_until = -1
        refractory_until = -1
        solves = []
        blocked = []                            # steps where a 2nd onset was blocked from re-solving
        for k in range(ep_steps):
            net.a = excitability_at(k, second_start)
            zone_amp = np.sqrt(s[zone] ** 2 + s[zone + N] ** 2).mean()

            # Level 1 is ALWAYS on
            u = l1_gain * lqr.control(s)

            escalating = zone_amp > trigger_frac * C.SEIZURE_AMP_THRESHOLD
            if k < schedule_active_until:
                # an open-loop Level-2 burst is still playing -> apply it (+L1)
                x, y = s[:N], s[N:]
                r = np.sqrt(x * x + y * y) + 1e-9
                u = u - burst_amp * np.concatenate([mask * x / r, mask * y / r])
            elif escalating and k >= refractory_until:
                # fresh Level-2 solve: start a new open-loop schedule
                schedule_active_until = k + horizon
                refractory_until = schedule_active_until + refractory
                solves.append(k)
                x, y = s[:N], s[N:]
                r = np.sqrt(x * x + y * y) + 1e-9
                u = u - burst_amp * np.concatenate([mask * x / r, mask * y / r])
            elif escalating and k < refractory_until:
                # escalating but LOCKED OUT by refractory and no active schedule
                blocked.append(k)               # <-- the #20 failure mode, if any

            s = net.rk4_step(s, C.DT, u=u, noise=C.NOISE_BETA, rng=rng)
            amp_tr[k + 1] = np.sqrt(s[zone] ** 2 + s[zone + N] ** 2).mean()

        # judge suppression DURING the second onset window
        lo, hi = second_start, min(second_start + 350, ep_steps)
        second_amp = float(amp_tr[lo:hi].mean())
        suppressed = second_amp < C.SEIZURE_AMP_THRESHOLD
        return amp_tr, solves, blocked, second_amp, suppressed

    net.a = excitability_at(0, 10**9)           # restore healthy baseline
    traces = {}
    rows = []
    for name, ss in second_starts.items():
        amp_tr, solves, blocked, second_amp, suppressed = run(ss)
        traces[name] = amp_tr
        print(f"   [{name:30s}] 2nd-onset start={ss:4d}  solves@{solves}  "
              f"blocked-while-escalating={len(blocked)}  "
              f"2nd-window amp={second_amp:.4f}  "
              f"suppressed={'yes' if suppressed else 'no'}")
        rows.append([name, ss, len(solves), len(blocked),
                     f"{second_amp:.4f}", "yes" if suppressed else "no"])

    cm.savetxt_table(
        f"{C.RESULTS_DIR}/analysis_refractory_summary.csv",
        ["scenario", "second_onset_start_step", "n_level2_solves",
         "steps_blocked_while_escalating", "second_window_mean_amp",
         "second_onset_suppressed"], rows)

    from src import viz
    viz.line_plot(
        te, traces,
        f"Refractory probe: 2nd onset timing vs suppression "
        f"(horizon={horizon}, refractory={refractory})",
        "onset-zone mean amplitude", "analysis_refractory_timeseries.png",
        hline=C.SEIZURE_AMP_THRESHOLD,
        vlines=[s * C.DT for s in second_starts.values()])
    print(f"   saved {C.IMAGE_DIR}/analysis_refractory_timeseries.png")


if __name__ == "__main__":
    main()
