"""
week09_ablation.py  --  Phase 4 / Week 9
========================================
Ablation Studies & Clinical Comparison.

Two questions:
  (1) Does connectome grounding matter? We compare the closed-loop controller
      when it actuates the *connectome-identified onset zone* against actuating
      a same-sized set of random (connectome-naive) nodes, and against a single
      isolated focus. If the structure matters, targeting the structurally
      central onset zone should suppress the seizure at far lower energy.
  (2) How does the adaptive closed loop compare to conventional always-on
      open-loop stimulation (the clinical DBS analogue) on suppression vs energy?

Figures written to images/:
  * week09_targeting_ablation.png       -- zone amplitude under each targeting
  * week09_targeting_energy.png         -- suppression vs energy per strategy
  * week09_clinical_vs_closedloop.png   -- always-on DBS vs adaptive closed loop

Results written to results/:
  * week09_ablation_summary.csv

Run (reduced scale recommended):
    SCALE=reduced python scripts/week09_ablation.py
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.detector import CSDDetector
from src.closed_loop import run_closed_loop
from src import baselines, metrics, viz


def _run_cl(net, s0, n_steps, zone, control_nodes, monitor_nodes):
    """Helper: run the closed loop with the tuned Week-8 parameters."""
    return run_closed_loop(net, s0, n_steps, monitor_nodes=monitor_nodes,
                           control_nodes=control_nodes, **cm.closed_loop_kwargs())


def main():
    cm.banner(9, "Ablation Studies & Clinical Comparison")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    n_steps = 1600
    t = cm.time_axis(n_steps)
    s0 = random_initial_state(N, scale=0.05, seed=cm.REP_SEED)
    net = cm.seizing_network(A, zone)

    rng = np.random.default_rng(cm.REP_SEED)
    nonzone = [i for i in range(N) if i not in zone]
    random_target = sorted(rng.choice(nonzone, size=len(zone), replace=False).tolist())

    # --- 1. Targeting ablation --------------------------------------------- #
    strategies = {
        "onset zone (connectome)": zone,
        "random nodes (naive)": random_target,
        "single focus only": [focus],
    }
    amp_traces = {}
    rows = []
    for name, ctrl_nodes in strategies.items():
        res = _run_cl(net, s0, n_steps, zone, ctrl_nodes, monitor_nodes=zone)
        amp = metrics.amplitude(res["traj"], N)
        amp_traces[name] = amp[:, zone].mean(1)
        final = amp[-100:, zone].mean()
        rows.append([name, f"{final:.4f}", len(res["onsets"]),
                     f"{res['energy']:.3f}",
                     "yes" if final < C.SEIZURE_AMP_THRESHOLD else "no"])
        print(f"   [{name:26s}] zone amp={final:.4f}  energy={res['energy']:.3f}")

    viz.line_plot(t, amp_traces,
                  "Connectome-aware vs naive control targeting",
                  "onset-zone mean amplitude", "week09_targeting_ablation.png",
                  hline=C.SEIZURE_AMP_THRESHOLD)

    # suppression vs energy scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = ["seagreen", "indianred", "slategray"]
    for (name, _), c in zip(strategies.items(), colors):
        row = next(r for r in rows if r[0] == name)
        ax.scatter(float(row[3]), float(row[1]), s=140, color=c, label=name,
                   edgecolors="k", zorder=5)
    ax.axhline(C.SEIZURE_AMP_THRESHOLD, color="gray", ls="--",
               label="ictal threshold")
    ax.set_xlabel("control energy"); ax.set_ylabel("residual onset-zone amplitude")
    ax.set_title("Suppression vs energy by targeting strategy")
    ax.legend(fontsize=8)
    viz._save(fig, "week09_targeting_energy.png")

    # --- 2. Clinical always-on open-loop vs adaptive, INTERMITTENT seizures - #
    # The fair clinical contrast needs an *intermittent* disease course: the
    # onset zone is healthy most of the time and only crosses into the ictal
    # regime during bounded windows. Always-on stimulation then pays an energy
    # (and off-state stimulation) penalty during the long seizure-free periods,
    # while an adaptive scheme only spends energy during events. We integrate a
    # time-varying excitability schedule directly here.
    ep_steps = 2400
    te = cm.time_axis(ep_steps)
    ictal_windows = [(500, 850), (1500, 1950)]      # two seizures

    def a_at(k):
        """Time-varying excitability: zone is ictal only inside the windows."""
        a = np.full(N, C.A_REST)
        if any(lo <= k < hi for lo, hi in ictal_windows):
            a[np.asarray(zone)] = C.A_SEIZURE
        return a

    def episodic_run(controller):
        """RK4 integrate the episodic scenario. ``controller(k, s)->u`` returns
        the stacked control (or None). Returns (zone_amp_trace, energy,
        energy_seizure_free)."""
        rng = np.random.default_rng(cm.REP_SEED)
        s = s0.copy()
        amp_tr = np.empty(ep_steps + 1)
        amp_tr[0] = np.sqrt(s[zone] ** 2 + s[np.asarray(zone) + N] ** 2).mean()
        e_tot = e_free = 0.0
        for k in range(ep_steps):
            net.a = a_at(k)
            u = controller(k, s)
            if u is not None:
                step_e = C.DT * float(np.sum(u ** 2))
                e_tot += step_e
                if not any(lo <= k < hi for lo, hi in ictal_windows):
                    e_free += step_e
            s = net.rk4_step(s, C.DT, u=u, noise=C.NOISE_BETA, rng=rng)
            amp_tr[k + 1] = np.sqrt(s[zone] ** 2 + s[np.asarray(zone) + N] ** 2).mean()
        return amp_tr, e_tot, e_free

    from src.lqr_control import LocalLQR
    lqr = LocalLQR(a=np.full(N, C.A_REST), omega=net.omega)
    mask = np.zeros(N); mask[np.asarray(zone)] = 1.0

    def ctl_none(k, s):
        return None

    def ctl_alwayson(k, s):                       # constant open-loop DBS
        x, y = s[:N], s[N:]
        r = np.sqrt(x * x + y * y) + 1e-9
        amp = 0.06
        return -amp * np.concatenate([mask * x / r, mask * y / r])

    def ctl_adaptive(k, s):                        # gentle L1 + state-triggered burst
        u = 0.25 * lqr.control(s)
        zone_amp = np.sqrt(s[zone] ** 2 + s[np.asarray(zone) + N] ** 2).mean()
        if zone_amp > 0.5 * C.SEIZURE_AMP_THRESHOLD:   # only act when escalating
            x, y = s[:N], s[N:]
            r = np.sqrt(x * x + y * y) + 1e-9
            u = u - 0.15 * np.concatenate([mask * x / r, mask * y / r])
        return u

    none_amp, _, _ = episodic_run(ctl_none)
    ao_amp, ao_e, ao_free = episodic_run(ctl_alwayson)
    ad_amp, ad_e, ad_free = episodic_run(ctl_adaptive)
    net.a = a_at(0)                                # restore (healthy) for safety

    # Suppression is judged DURING the ictal windows (where it matters), not at
    # the post-window tail.
    ictal_mask = np.zeros(ep_steps + 1, dtype=bool)
    for lo, hi in ictal_windows:
        ictal_mask[lo:hi] = True

    def ictal_mean(trace):
        return float(trace[ictal_mask].mean())

    viz.line_plot(te, {"uncontrolled": none_amp,
                       "always-on open-loop": ao_amp,
                       "adaptive (gentle L1 + triggered)": ad_amp},
                  "Intermittent seizures: adaptive vs always-on stimulation",
                  "onset-zone mean amplitude", "week09_clinical_vs_closedloop.png",
                  vlines=[w[0] * C.DT for w in ictal_windows]
                         + [w[1] * C.DT for w in ictal_windows],
                  hline=C.SEIZURE_AMP_THRESHOLD)
    print("   intermittent-seizure clinical comparison (amplitude DURING windows):")
    print(f"      uncontrolled : ictal amp={ictal_mean(none_amp):.4f}")
    print(f"      always-on : ictal amp={ictal_mean(ao_amp):.4f}  total energy={ao_e:.3f}  "
          f"seizure-FREE energy={ao_free:.3f} ({100*ao_free/max(ao_e,1e-9):.0f}% wasted)")
    print(f"      adaptive  : ictal amp={ictal_mean(ad_amp):.4f}  total energy={ad_e:.3f}  "
          f"seizure-FREE energy={ad_free:.3f} ({100*ad_free/max(ad_e,1e-9):.0f}% wasted)")

    ol_rows = [
        ["uncontrolled (intermittent)", f"{ictal_mean(none_amp):.4f}", "0.000",
         "0.000", "no"],
        ["always-on open-loop", f"{ictal_mean(ao_amp):.4f}", f"{ao_e:.3f}",
         f"{ao_free:.3f}", "yes" if ictal_mean(ao_amp) < C.SEIZURE_AMP_THRESHOLD else "no"],
        ["adaptive (L1+triggered)", f"{ictal_mean(ad_amp):.4f}", f"{ad_e:.3f}",
         f"{ad_free:.3f}", "yes" if ictal_mean(ad_amp) < C.SEIZURE_AMP_THRESHOLD else "no"],
    ]

    cm.savetxt_table(f"{C.RESULTS_DIR}/week09_ablation_summary.csv",
                     ["strategy", "ictal_zone_amp", "total_energy",
                      "seizure_free_energy", "seizure_suppressed"],
                     rows + [["--- intermittent clinical comparison ---",
                              "ictal amp", "total energy", "free-time energy", ""]]
                     + ol_rows)


if __name__ == "__main__":
    main()
