"""
analysis_surrogates.py  --  Connectome dependence: degree-preserving surrogates
==============================================================================
Implements the Maslov-Sneppen degree-preserving rewiring test described in the
manuscript (Section 3.4) and the recruitment statistics quoted in Section 4.2,
so that every number in that section is produced by a script in this repository.

For the uncontrolled seizing network (representative trial, seed = cm.REP_SEED):
  * recruited node fraction  -- fraction of nodes whose amplitude stays above the
                                 ictal threshold for >= 60% of the last 200 steps
  * time-to-recruitment      -- per recruited non-zone node, first time its
                                 amplitude exceeds the ictal threshold (median, s)
  * Kuramoto order parameter -- phase-synchrony R(t), whole network and onset zone
                                 (the proper synchrony measure; the amplitude
                                 criterion above is a recruitment measure)
  * onset-zone edge ablation -- non-zone mean amplitude intact vs ablated
  * surrogate test           -- 20 degree-preserving surrogates (double-edge swaps
                                 carrying edge weights, 10 x |E| accepted swaps,
                                 re-normalised to spectral radius 1); one-sample
                                 t-test of the surrogate ensemble against the true
                                 connectome's non-zone amplitude.

Outputs
    results/analysis_surrogates.csv
    images/analysis_surrogates.png
Run:
    SCALE=reduced python scripts/analysis_surrogates.py
"""
import os
import numpy as np
from scipy import stats

import _common as cm
from src import config as C
from src.hopf_model import random_initial_state
from src import metrics

N_SURR = 20
SWAPS_PER_EDGE = 10


def maslov_sneppen(A, rng, swaps_per_edge=SWAPS_PER_EDGE):
    """Degree-preserving rewiring of a weighted undirected graph.

    Each accepted double-edge swap (a-b, c-d) -> (a-d, c-b) moves the two edge
    weights with the edges, so the binary degree sequence AND the multiset of
    edge weights are preserved exactly; the specific wiring is randomised.
    """
    N = A.shape[0]
    iu = np.triu_indices(N, 1)
    mask = A[iu] > 0
    edges = [[int(i), int(j), float(w)] for i, j, w in zip(iu[0][mask], iu[1][mask], A[iu][mask])]
    present = {(e[0], e[1]) for e in edges}
    target = swaps_per_edge * len(edges)
    accepted = tries = 0
    while accepted < target and tries < 50 * target:
        tries += 1
        i, j = rng.integers(len(edges), size=2)
        if i == j:
            continue
        a, b, w1 = edges[i]; c, d, w2 = edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1 = (min(a, d), max(a, d)); e2 = (min(c, b), max(c, b))
        if e1 in present or e2 in present:
            continue
        present.discard((a, b)); present.discard((c, d))
        present.add(e1); present.add(e2)
        edges[i] = [e1[0], e1[1], w1]; edges[j] = [e2[0], e2[1], w2]
        accepted += 1
    S = np.zeros_like(A)
    for a, b, w in edges:
        S[a, b] = S[b, a] = w
    r = np.max(np.abs(np.linalg.eigvalsh(S)))
    return S / r if r > 0 else S, accepted


def _run(A, zone, s0, seed, n_steps):
    net = cm.seizing_network(A, zone)
    traj = net.simulate(s0, n_steps, noise=C.NOISE_BETA, seed=seed)
    return metrics.amplitude(traj, net.N), traj


def main():
    cm.banner(0, "Connectome dependence: recruitment statistics + degree-preserving surrogates")
    A, labels, focus = cm.load_network()
    N = A.shape[0]
    zone = cm.onset_zone(A, labels, focus)
    rest = np.array([i for i in range(N) if i not in zone])
    n_steps = cm.CL_N_STEPS
    seed = cm.REP_SEED
    s0 = random_initial_state(N, scale=0.05, seed=seed)
    thr = C.SEIZURE_AMP_THRESHOLD
    t = cm.time_axis(n_steps)

    # ---- intact connectome ------------------------------------------------- #
    amp, traj = _run(A, zone, s0, seed, n_steps)
    ictal = metrics.sustained_ictal_mask(traj, N, thr, tail=200)
    recruited_frac = float(ictal.mean())
    recruited_nonzone = [i for i in rest if ictal[i]]
    ttr = []
    for i in recruited_nonzone:
        above = np.where(amp[:, i] > thr)[0]
        if len(above):
            ttr.append(t[above[0]])
    ttr_median = float(np.median(ttr)) if ttr else float("nan")
    R_all = metrics.kuramoto_order(traj, N)
    R_zone = metrics.kuramoto_order(traj, N, nodes=zone)
    nonzone_intact = float(amp[-200:][:, rest].mean())
    print(f"   recruited node fraction (sustained > {thr}): {recruited_frac:.3f} ({int(ictal.sum())}/{N})")
    print(f"   non-zone nodes recruited: {len(recruited_nonzone)}; median time-to-recruitment {ttr_median:.2f} s")
    print(f"   Kuramoto R (last 200 steps): whole network {R_all[-200:].mean():.3f}, onset zone {R_zone[-200:].mean():.3f}")
    print(f"   non-zone mean amplitude (last 200 steps), intact: {nonzone_intact:.4f}")

    # ---- onset-zone edge ablation ------------------------------------------- #
    A_ab = A.copy(); A_ab[zone, :] = 0.0; A_ab[:, zone] = 0.0
    amp_ab, _ = _run(A_ab, zone, s0, seed, n_steps)
    nonzone_ablated = float(amp_ab[-200:][:, rest].mean())
    red = 100 * (1 - nonzone_ablated / nonzone_intact)
    print(f"   non-zone mean amplitude, onset-zone edges ablated: {nonzone_ablated:.4f}  ({red:.1f}% reduction)")

    # ---- degree-preserving surrogates -------------------------------------- #
    rng = np.random.default_rng(C.SEED)
    sur_vals, sur_rec, acc = [], [], []
    for k in range(N_SURR):
        S, a = maslov_sneppen(A, rng)
        amp_s, traj_s = _run(S, zone, s0, seed, n_steps)
        sur_vals.append(float(amp_s[-200:][:, rest].mean()))
        sur_rec.append(float(metrics.sustained_ictal_mask(traj_s, N, thr, tail=200).mean()))
        acc.append(a)
        print(f"      surrogate {k:2d}: accepted swaps={a}  non-zone amp={sur_vals[-1]:.4f}  recruited={sur_rec[-1]:.3f}")
    sur_vals = np.array(sur_vals)
    tt = stats.ttest_1samp(sur_vals, nonzone_intact)
    print(f"   surrogates: mean +/- SD = {sur_vals.mean():.4f} +/- {sur_vals.std(ddof=1):.4f}; "
          f"true connectome = {nonzone_intact:.4f}; one-sample t({N_SURR-1}) = {tt.statistic:.2f}, p = {tt.pvalue:.2e}")
    print(f"   surrogate recruited fraction mean = {np.mean(sur_rec):.3f} (true {recruited_frac:.3f})")

    cm.savetxt_table(
        f"{C.RESULTS_DIR}/analysis_surrogates.csv",
        ["quantity", "value"],
        [["seed", seed], ["n_steps", n_steps],
         ["recruited_node_fraction_true", f"{recruited_frac:.4f}"],
         ["n_recruited_nodes_true", int(ictal.sum())],
         ["median_time_to_recruitment_s_nonzone", f"{ttr_median:.3f}"],
         ["kuramoto_R_whole_last200", f"{R_all[-200:].mean():.4f}"],
         ["kuramoto_R_zone_last200", f"{R_zone[-200:].mean():.4f}"],
         ["nonzone_amp_intact", f"{nonzone_intact:.4f}"],
         ["nonzone_amp_ablated", f"{nonzone_ablated:.4f}"],
         ["ablation_reduction_percent", f"{red:.1f}"],
         ["n_surrogates", N_SURR], ["swaps_per_edge", SWAPS_PER_EDGE],
         ["surrogate_nonzone_amp_mean", f"{sur_vals.mean():.4f}"],
         ["surrogate_nonzone_amp_sd", f"{sur_vals.std(ddof=1):.4f}"],
         ["surrogate_recruited_fraction_mean", f"{np.mean(sur_rec):.4f}"],
         ["t_statistic", f"{tt.statistic:.3f}"], ["df", N_SURR - 1], ["p_value", f"{tt.pvalue:.3e}"],
         ["true_minus_surrogate_mean", f"{nonzone_intact - sur_vals.mean():.4f}"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(sur_vals, bins=10, color="slategray", alpha=0.85, label="degree-preserving surrogates (n=20)")
    axes[0].axvline(nonzone_intact, color="crimson", ls="--", lw=1.6, label="true connectome")
    axes[0].axvline(nonzone_ablated, color="darkorange", ls=":", lw=1.6, label="onset-zone edges ablated")
    axes[0].set_xlabel("mean non-zone amplitude (last 200 steps)"); axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, R_all, color="tab:blue", lw=1.2, label="whole network R(t)")
    axes[1].plot(t, R_zone, color="tab:orange", lw=1.2, label="onset-zone R(t)")
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("Kuramoto order parameter R"); axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.IMAGE_DIR, "analysis_surrogates.png"), dpi=150, bbox_inches="tight")
    print("   saved images/analysis_surrogates.png")


if __name__ == "__main__":
    main()
