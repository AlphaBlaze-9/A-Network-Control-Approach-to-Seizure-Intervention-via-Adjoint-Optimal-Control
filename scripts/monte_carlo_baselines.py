"""
monte_carlo_baselines.py  --  Unified Monte-Carlo benchmark (proposed + 3 baselines)
====================================================================================
Runs the PROPOSED method and all three Table-2 baselines under the IDENTICAL
Monte-Carlo protocol -- same N_SEEDS trials (config.N_SEEDS, seeds
MC_SEED0..MC_SEED0+N-1), same randomized initial conditions, same plant noise
(config.NOISE_BETA), the same 80-s (1600-step) continuously seizure-prone
scenario, and the same U_MAX actuator saturation wherever the Level-2 optimiser
is used -- so every entry in Table 2 is apples-to-apples. The closed-loop
settings come from ``_common.closed_loop_kwargs`` (shared with every
representative-trial figure script).

Conditions (map to Table 2 columns):
  proposed  -- our two-level closed loop, control_nodes = onset zone
  ablation  -- "connectome ablation (naive)": identical closed loop, but
               control_nodes = a random same-size set of non-zone nodes
               (re-drawn per trial from that trial's seed)
  openloop  -- "open-loop DBS surrogate": always-on radial inward push
               (amplitude OPENLOOP_AMP = 0.06) on the onset zone, on the
               SAME 80-s continuously seizure-prone scenario as the other
               conditions (the intermittent two-window scenario is analysed
               separately in analysis_openloop_amplitude_mc.py)
  pi        -- literature PI controller (Wang et al. 2016), per
               baselines.PIController

For each condition, per trial we record: suppressed (bool), control energy,
off-target stimulation fraction, residual onset-zone amplitude (final 100
steps), the number of Level-2 interventions (closed-loop conditions only) and
the fraction of control energy carried by the real (x) component of the
actuation signal. Off-target is 0.0 by construction for openloop/pi (their
actuation masks are restricted to the onset zone).

Run (one condition per terminal, in parallel):
    SCALE=reduced python scripts/monte_carlo_baselines.py --conditions proposed
    SCALE=reduced python scripts/monte_carlo_baselines.py --conditions ablation
    SCALE=reduced python scripts/monte_carlo_baselines.py --conditions openloop
    SCALE=reduced python scripts/monte_carlo_baselines.py --conditions pi
Then combine:
    SCALE=reduced python scripts/monte_carlo_baselines.py --combine

Outputs:
    results/mc_<condition>_per_trial.csv
    results/monte_carlo_baselines_summary.csv   (written by --combine)
    results/monte_carlo_summary.csv             (headline, proposed column; --combine)
    images/monte_carlo_baselines_bars.png       (written by --combine)
"""

import argparse
import csv
import os

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src.closed_loop import run_closed_loop
from src import baselines, metrics

OPENLOOP_AMP = 0.06
Z = 1.959963984540054  # 95% normal quantile


def _wilson_ci(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = (Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _mean_ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return float(x.mean()) if n else float("nan"), float("nan")
    return float(x.mean()), float(Z * x.std(ddof=1) / np.sqrt(n))


def _real_frac(ctrl, N):
    tot = float((ctrl ** 2).sum())
    return float((ctrl[:, :N] ** 2).sum() / tot) if tot > 0 else float("nan")


# --------------------------------------------------------------------------- #
#  Per-trial runners -- one full simulation each, same protocol throughout
# --------------------------------------------------------------------------- #
def run_proposed_trial(A, zone, seed, n_steps=cm.CL_N_STEPS):
    net = cm.seizing_network(A, zone)
    N = net.N
    s0 = random_initial_state(N, scale=0.05, seed=seed)
    res = run_closed_loop(net, s0, n_steps, monitor_nodes=zone, control_nodes=zone,
                          **cm.closed_loop_kwargs(seed=seed))
    amp = metrics.amplitude(res["traj"], N)
    final = float(amp[-100:, zone].mean())
    suppressed = final < C.SEIZURE_AMP_THRESHOLD
    offt = baselines.off_target_fraction(res["ctrl"], N, zone)
    return suppressed, float(res["energy"]), float(offt), final, len(res["onsets"]), _real_frac(res["ctrl"], N)


def run_ablation_trial(A, zone, seed, n_steps=cm.CL_N_STEPS):
    """Connectome-ablation (naive) baseline: identical closed loop, but the
    actuated set is a random same-size set of non-zone nodes, re-drawn from
    this trial's own seed."""
    net = cm.seizing_network(A, zone)
    N = net.N
    s0 = random_initial_state(N, scale=0.05, seed=seed)
    rng = np.random.default_rng(seed)
    nonzone = [i for i in range(N) if i not in zone]
    random_target = sorted(rng.choice(nonzone, size=len(zone), replace=False).tolist())
    res = run_closed_loop(net, s0, n_steps, monitor_nodes=zone, control_nodes=random_target,
                          **cm.closed_loop_kwargs(seed=seed))
    amp = metrics.amplitude(res["traj"], N)
    final = float(amp[-100:, zone].mean())
    suppressed = final < C.SEIZURE_AMP_THRESHOLD
    offt = baselines.off_target_fraction(res["ctrl"], N, zone)
    return suppressed, float(res["energy"]), float(offt), final, len(res["onsets"]), _real_frac(res["ctrl"], N)


def run_openloop_trial(A, zone, seed, n_steps=cm.CL_N_STEPS, amp=OPENLOOP_AMP):
    """Open-loop DBS surrogate: constant radial inward push of amplitude ``amp``
    on the onset zone for the whole run (always on), on the same continuously
    seizure-prone 80-s scenario as the other conditions. Its energy is
    deterministic (amp^2 * |zone| * T). Off-target = 0 by construction."""
    net = cm.seizing_network(A, zone)
    N = net.N
    s0 = random_initial_state(N, scale=0.05, seed=seed)
    mask = np.zeros(N); mask[np.asarray(zone)] = 1.0

    def u_func(t, s):
        x, y = s[:N], s[N:]
        r = np.sqrt(x * x + y * y) + 1e-9
        return -amp * np.concatenate([mask * x / r, mask * y / r])

    traj, ctrl = net.simulate(s0, n_steps, u_func=u_func, record_control=True,
                              noise=C.NOISE_BETA, seed=seed)
    energy = float(C.DT * np.sum(ctrl ** 2))
    ampl = metrics.amplitude(traj, N)
    final = float(ampl[-100:, zone].mean())
    suppressed = final < C.SEIZURE_AMP_THRESHOLD
    return suppressed, energy, 0.0, final, 0, _real_frac(ctrl, N)


def run_pi_trial(A, zone, seed, n_steps=cm.CL_N_STEPS):
    """Literature PI baseline (Wang et al. 2016). Off-target = 0.0 by
    construction (PIController's mask only covers the onset zone)."""
    net = cm.seizing_network(A, zone)
    N = net.N
    s0 = random_initial_state(N, scale=0.05, seed=seed)
    pi = baselines.PIController(kp=2.0, ki=0.5, stim_nodes=zone, N=N)
    traj, ctrl = net.simulate(s0, n_steps, u_func=pi.u_func(C.DT), record_control=True,
                              noise=C.NOISE_BETA, seed=seed)
    energy = float(C.DT * np.sum(ctrl ** 2))
    amp = metrics.amplitude(traj, N)
    final = float(amp[-100:, zone].mean())
    suppressed = final < C.SEIZURE_AMP_THRESHOLD
    return suppressed, energy, 0.0, final, 0, _real_frac(ctrl, N)


RUNNERS = {
    "proposed": run_proposed_trial,
    "ablation": run_ablation_trial,
    "openloop": run_openloop_trial,
    "pi": run_pi_trial,
}

DISPLAY_NAME = {
    "proposed": "Proposed (closed-loop)",
    "ablation": "Connectome ablation (naive)",
    "openloop": "Open-loop DBS surrogate",
    "pi": "PI (Wang 2016)",
}

COLS = ["trial", "seed", "suppressed", "energy", "offtarget", "final", "n_interventions", "real_frac"]


def _run_condition(cond, n_trials, A, zone):
    fn = RUNNERS[cond]
    rows = []
    print(f"=== condition: {cond}  ({DISPLAY_NAME[cond]}, {n_trials} trials) ===")
    for i in range(n_trials):
        seed = C.MC_SEED0 + i
        suppressed, energy, offt, final, n_int, rf = fn(A, zone, seed)
        rows.append([i, seed, "yes" if suppressed else "no",
                     f"{energy:.6f}", f"{offt:.6f}", f"{final:.6f}", n_int, f"{rf:.6f}"])
        print(f"  [{cond:8s}] trial {i:3d} seed={seed:5d} "
              f"suppressed={'yes' if suppressed else 'no':3s} "
              f"energy={energy:.4f} offtarget={offt:.4f} final={final:.4f} n_int={n_int} real_frac={rf:.3f}",
              flush=True)
    path = os.path.join(C.RESULTS_DIR, f"mc_{cond}_per_trial.csv")
    cm.savetxt_table(path, COLS, rows)
    print(f"  wrote {path}\n")


def _combine():
    print("=" * 64)
    print(f"  COMBINING MONTE-CARLO BASELINES  [{C.scale_label()}, U_MAX={C.U_MAX}]")
    print("=" * 64)
    summary_rows = []
    bar_data = {}
    for cond in ("proposed", "ablation", "openloop", "pi"):
        path = os.path.join(C.RESULTS_DIR, f"mc_{cond}_per_trial.csv")
        if not os.path.exists(path):
            print(f"  [!] {path} not found -- skipping {cond}")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        n = len(reader)
        k = sum(1 for r in reader if r["suppressed"].strip() == "yes")
        lo, hi = _wilson_ci(k, n)
        e = [float(r["energy"]) for r in reader]
        o = [float(r["offtarget"]) for r in reader]
        fz = [float(r["final"]) for r in reader]
        ni = [float(r.get("n_interventions", 0) or 0) for r in reader]
        rf = [float(r["real_frac"]) for r in reader if r.get("real_frac") not in (None, "", "nan")]
        e_mean, e_ci = _mean_ci(e)
        o_mean, o_ci = _mean_ci(o)
        f_mean, f_ci = _mean_ci(fz)
        rate = k / n if n else float("nan")
        rf_mean = float(np.mean(rf)) if rf else float("nan")
        print(f"  {DISPLAY_NAME[cond]:30s} n={n:4d}  success={k}/{n}={rate:.1%}  CI[{lo:.1%},{hi:.1%}]  "
              f"energy={e_mean:.4f}+/-{e_ci:.4f} (median {np.median(e):.4f})  offtarget={o_mean:.4f}+/-{o_ci:.4f}  "
              f"final={f_mean:.4f}  n_int={np.mean(ni):.2f}  real_frac={rf_mean:.3f}")
        summary_rows.append([
            DISPLAY_NAME[cond], n, k, f"{rate:.4f}", f"{lo:.4f}", f"{hi:.4f}",
            f"{e_mean:.4f}", f"{e_ci:.4f}", f"{np.median(e):.4f}", f"{o_mean:.4f}", f"{o_ci:.4f}",
            f"{f_mean:.4f}", f"{f_ci:.4f}", f"{np.mean(ni):.2f}", f"{rf_mean:.4f}"])
        bar_data[cond] = (rate, lo, hi, e_mean, e_ci, o_mean, o_ci)

    if not summary_rows:
        print("  Nothing to combine yet.")
        return

    cm.savetxt_table(
        os.path.join(C.RESULTS_DIR, "monte_carlo_baselines_summary.csv"),
        ["condition", "n_trials", "n_success", "success_rate",
         "success_ci95_low", "success_ci95_high",
         "energy_mean", "energy_ci95_halfwidth", "energy_median",
         "offtarget_mean", "offtarget_ci95_halfwidth",
         "final_zone_amp_mean", "final_zone_amp_ci95_halfwidth",
         "mean_n_interventions", "real_energy_fraction_mean"],
        summary_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [c for c in ("proposed", "ablation", "openloop", "pi") if c in bar_data]
    names = [DISPLAY_NAME[c] for c in conds]
    rates = [bar_data[c][0] for c in conds]
    rate_err = [[bar_data[c][0] - bar_data[c][1] for c in conds],
                [bar_data[c][2] - bar_data[c][0] for c in conds]]
    energies = [bar_data[c][3] for c in conds]
    e_err = [bar_data[c][4] for c in conds]
    offt = [bar_data[c][5] for c in conds]
    o_err = [bar_data[c][6] for c in conds]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].bar(names, rates, yerr=rate_err, capsize=6, color="teal")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("suppression success rate (Wilson 95% CI)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, energies, yerr=e_err, capsize=6, color="slateblue")
    axes[1].set_ylabel("control energy (mean +/- 95% CI)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[2].bar(names, offt, yerr=o_err, capsize=6, color="darkorange")
    axes[2].set_ylabel("off-target stimulation fraction (mean +/- 95% CI)")
    axes[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out = os.path.join(C.IMAGE_DIR, "monte_carlo_baselines_bars.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")

    # keep results/monte_carlo_summary.csv (headline) in sync with the proposed column
    if "proposed" in bar_data:
        with open(os.path.join(C.RESULTS_DIR, "mc_proposed_per_trial.csv"), newline="", encoding="utf-8") as f:
            R = list(csv.DictReader(f))
        n = len(R); k = sum(r["suppressed"] == "yes" for r in R); lo, hi = _wilson_ci(k, n)
        e = np.array([float(r["energy"]) for r in R]); a = np.array([float(r["final"]) for r in R])
        cm.savetxt_table(
            os.path.join(C.RESULTS_DIR, "monte_carlo_summary.csv"), ["metric", "value"],
            [["n_trials", n], ["successes", k], ["success_rate", f"{k/n:.4f}"],
             ["success_ci95_low", f"{lo:.4f}"], ["success_ci95_high", f"{hi:.4f}"],
             ["energy_mean", f"{e.mean():.4f}"], ["energy_ci95_halfwidth", f"{Z*e.std(ddof=1)/np.sqrt(n):.4f}"],
             ["zone_amp_mean", f"{a.mean():.4f}"], ["zone_amp_ci95_halfwidth", f"{Z*a.std(ddof=1)/np.sqrt(n):.4f}"],
             ["u_max", str(C.U_MAX)], ["scale", C.scale_label()]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="proposed,ablation,openloop,pi",
                    help="comma-separated subset of: proposed,ablation,openloop,pi")
    ap.add_argument("--combine", action="store_true",
                    help="combine existing per-trial CSVs into the summary + figure")
    args = ap.parse_args()

    if args.combine:
        _combine()
        return

    n_trials = int(os.environ.get("N_SEEDS", C.N_SEEDS))
    A, labels, focus = cm.load_network()
    zone = cm.onset_zone(A, labels, focus)
    print(f"  onset zone: {[labels[i] for i in zone]}")
    print(f"  N_SEEDS={n_trials}  U_MAX={C.U_MAX}  [{C.scale_label()}]\n")

    for cond in args.conditions.split(","):
        cond = cond.strip()
        if cond not in RUNNERS:
            raise SystemExit(f"Unknown condition '{cond}'. Choose from: {', '.join(RUNNERS)}")
        _run_condition(cond, n_trials, A, zone)


if __name__ == "__main__":
    main()
