"""
analysis_energy_significance.py  --  Reviewer response, Issue #23
=================================================================
Reviewer #23 notes the proposed vs PI control-energy comparison reports means
with overlapping CIs but no significance test. This script loads the per-trial
energy arrays produced by monte_carlo_baselines.py and runs the appropriate
tests:

  * Wilcoxon signed-rank (PAIRED): the two conditions share the same trial seeds
    (MC_SEED0 + i), so they are naturally paired trial-by-trial. This is the
    primary test.
  * Mann-Whitney U (unpaired): reported alongside as a distribution-free
    cross-check that does not assume pairing.

It also reports the rank-biserial effect size and the median difference, and
appends everything to a summary CSV you can cite in Section 4.7 / Table 3.

Inputs (must exist -- produce them first with monte_carlo_baselines.py):
    results/mc_proposed_per_trial.csv
    results/mc_pi_per_trial.csv

Results written to results/:
    * analysis_energy_significance.csv

Run:
    python scripts/analysis_energy_significance.py
    # optionally compare other conditions:
    python scripts/analysis_energy_significance.py --a proposed --b openloop
"""

import argparse
import csv
import os
import numpy as np

import _common as cm
from src import config as C


def _load_per_trial(cond):
    path = os.path.join(C.RESULTS_DIR, f"mc_{cond}_per_trial.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n   -> run: SCALE=reduced python "
            f"scripts/monte_carlo_baselines.py --conditions {cond}")
    seed_to_energy = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seed_to_energy[int(r["seed"])] = float(r["energy"])
    return seed_to_energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="proposed")
    ap.add_argument("--b", default="pi")
    args = ap.parse_args()
    cm.banner(0, f"Issue #23: energy significance test ({args.a} vs {args.b})")

    A = _load_per_trial(args.a)
    B = _load_per_trial(args.b)
    shared = sorted(set(A) & set(B))
    if len(shared) < 5:
        raise SystemExit(f"only {len(shared)} shared seeds between "
                         f"{args.a} and {args.b}; need the same MC protocol.")
    ea = np.array([A[s] for s in shared])
    eb = np.array([B[s] for s in shared])
    n = len(shared)

    from scipy import stats
    # Primary: paired Wilcoxon signed-rank
    try:
        w_stat, w_p = stats.wilcoxon(ea, eb)
    except ValueError as e:                       # e.g. all differences zero
        w_stat, w_p = float("nan"), float("nan")
        print(f"   [wilcoxon] {e}")
    # Cross-check: Mann-Whitney U (unpaired)
    u_stat, u_p = stats.mannwhitneyu(ea, eb, alternative="two-sided")
    # rank-biserial effect size from U
    rbc = 1.0 - 2.0 * u_stat / (len(ea) * len(eb))

    md = float(np.median(ea - eb))
    print(f"   n paired trials           : {n}")
    print(f"   median energy {args.a:9s}: {np.median(ea):.4f}  "
          f"(mean {ea.mean():.4f} +/- {1.96*ea.std(ddof=1)/np.sqrt(n):.4f})")
    print(f"   median energy {args.b:9s}: {np.median(eb):.4f}  "
          f"(mean {eb.mean():.4f} +/- {1.96*eb.std(ddof=1)/np.sqrt(n):.4f})")
    print(f"   median paired difference  : {md:+.4f}")
    print(f"   Wilcoxon signed-rank (paired)  W={w_stat:.1f}  p={w_p:.4g}")
    print(f"   Mann-Whitney U (unpaired)      U={u_stat:.1f}  p={u_p:.4g}  "
          f"rank-biserial={rbc:+.3f}")
    verdict = ("NO significant difference (p >= 0.05)" if (not np.isfinite(w_p) or w_p >= 0.05)
               else "significant difference (p < 0.05)")
    print(f"   >> {verdict}")

    # merge into the summary CSV (one row per comparison; re-running a
    # comparison replaces its row, other comparisons are kept)
    header = ["comparison", "n_paired", "median_a", "median_b", "median_diff",
              "wilcoxon_W", "wilcoxon_p", "mannwhitney_U", "mannwhitney_p",
              "rank_biserial", "verdict_alpha0.05"]
    new_row = [f"{args.a}_vs_{args.b}", n, f"{np.median(ea):.4f}",
               f"{np.median(eb):.4f}", f"{md:.4f}",
               f"{w_stat:.2f}", f"{w_p:.4g}", f"{u_stat:.2f}", f"{u_p:.4g}",
               f"{rbc:.3f}", verdict]
    out = f"{C.RESULTS_DIR}/analysis_energy_significance.csv"
    existing = []
    if os.path.exists(out):
        with open(out, newline="", encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r["comparison"] != new_row[0]]
    rows_out = [[r.get(h, "") for h in header] for r in existing] + [new_row]
    cm.savetxt_table(out, header, rows_out)


if __name__ == "__main__":
    main()
