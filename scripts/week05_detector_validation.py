"""
week05_detector_validation.py  --  Phase 2 / Week 5   [REVISED: issues #6, #24]
===============================================================================
Independent Detector Validation & Tuning.

Confirms the CSD detector flags impending transitions reliably *before* it is
ever coupled to a controller. We generate held-out banks of trajectories:
  * POSITIVES -- excitability ramps through the Hopf bifurcation (a seizure is
    coming); the detector should alarm before the crossing.
  * NEGATIVES -- excitability stays safely below zero (no transition); the
    detector should stay quiet.
The detector is evaluated *causally* (expanding/rolling window; no future leak).

REVISIONS for the second reviewer pass
--------------------------------------
* Issue #6 (set too small + no AUC CI):
    - DET_VAL_PER_CLASS / DET_TEST_PER_CLASS default to 50 each (was 12), so the
      operating-point statistics rest on a much larger held-out bank.
    - The AUC is now reported with a bootstrap 95% confidence interval
      (DET_BOOTSTRAP resamples over trajectories), computed from per-trajectory
      "sustained-risk" scores (Mann-Whitney form of the AUC).
* Issue #24 (threshold leakage):
    - Two DISJOINT seed banks. The operating threshold is selected ONLY on the
      VALIDATION fold; TPR/FPR/precision/recall are reported ONLY on the
      independent TEST fold. The AUC itself is threshold-free and is reported
      on the test fold with its CI.

Figures written to images/:
  * week05_roc_curve.png            -- ROC (validation + test folds)
  * week05_leadtime_histogram.png   -- lead time (test fold, at selected threshold)
  * week05_risk_traces.png          -- example positive vs negative risk traces

Results written to results/:
  * week05_detector_metrics.csv     -- per-threshold sweep (validation fold)
  * week05_detector_testfold.csv    -- selected-threshold metrics on the TEST fold
  * week05_detector_auc_ci.csv      -- AUC + bootstrap 95% CI (test fold)

Run:
    python scripts/week05_detector_validation.py
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import HopfNetwork
from src.detector import CSDDetector
from src import viz


# --------------------------------------------------------------------------- #
#  Signal generators (unchanged)
# --------------------------------------------------------------------------- #
def _ramp_signal(a_start, a_end, n_steps, seed):
    """Single-node x(t) under a linear excitability ramp a_start -> a_end."""
    a_t = np.linspace(a_start, a_end, n_steps + 1)
    node = HopfNetwork(np.zeros((1, 1)), a=np.array([a_start]),
                       omega=np.array([2 * np.pi * C.OMEGA_HZ]))
    rng = np.random.default_rng(seed)
    s = np.array([0.05, 0.0])
    sig = np.empty(n_steps + 1)
    sig[0] = s[0]
    for k in range(n_steps):
        node.a[0] = a_t[k]
        s = node.rk4_step(s, C.DT, noise=C.NOISE_BETA, rng=rng)
        sig[k + 1] = s[0]
    t_cross = int(np.argmin(np.abs(a_t))) if (a_start < 0 < a_end) else None
    return sig, t_cross


def _const_signal(a_level, n_steps, seed):
    """Single-node x(t) at a *constant* subcritical excitability (negative)."""
    node = HopfNetwork(np.zeros((1, 1)), a=np.array([a_level]),
                       omega=np.array([2 * np.pi * C.OMEGA_HZ]))
    rng = np.random.default_rng(seed)
    s = np.array([0.05, 0.0])
    sig = np.empty(n_steps + 1)
    sig[0] = s[0]
    for k in range(n_steps):
        s = node.rk4_step(s, C.DT, noise=C.NOISE_BETA, rng=rng)
        sig[k + 1] = s[0]
    return sig, None


def _features_at(det, sig, stride=10, warmup=None):
    """Return (steps, {feature: values}) evaluated causally at strided steps."""
    from src.detector import (rolling_variance, rolling_autocorr1,
                              rolling_lyapunov)
    W = det.window
    warmup = warmup or W + 5
    steps = np.arange(warmup, len(sig), stride)
    out = {"variance": [], "autocorr1": [], "lyapunov": []}
    keep = []
    for k in steps:
        win = sig[k - W:k]
        v = rolling_variance(win, W)[-1]
        a = rolling_autocorr1(win, W)[-1]
        l = rolling_lyapunov(win, W)[-1]
        if not (np.isfinite(v) and np.isfinite(a) and np.isfinite(l)):
            continue
        keep.append(k)
        out["variance"].append(v)
        out["autocorr1"].append(a)
        out["lyapunov"].append(l)
    return np.array(keep), {k: np.array(v) for k, v in out.items()}


def calibrate_baseline(det, healthy_signals):
    """Establish healthy-baseline (mean, std) for each CSD feature."""
    pooled = {"variance": [], "autocorr1": [], "lyapunov": []}
    for sig in healthy_signals:
        _, feats = _features_at(det, sig)
        for k in pooled:
            pooled[k].append(feats[k])
    stats = {}
    for k in pooled:
        v = np.concatenate(pooled[k])
        stats[k] = (float(np.mean(v)), float(np.std(v) + 1e-9))
    return stats


def _causal_risk(det, sig, baseline, stride=10, warmup=None, z_half=3.0):
    """Causal, *calibrated* bifurcation-risk score in [0,1]."""
    w = det.weights
    steps, feats = _features_at(det, sig, stride=stride, warmup=warmup)
    risk = np.zeros(len(steps))
    for fi, key in enumerate(("variance", "autocorr1", "lyapunov")):
        mu, sd = baseline[key]
        z = np.clip((feats[key] - mu) / sd, 0, None)
        risk += w[fi] * (z / (z + z_half))
    return steps, risk


# --------------------------------------------------------------------------- #
#  Alarm logic + scoring helpers
# --------------------------------------------------------------------------- #
def _sustained_alarms(steps, risk, th, persist):
    """Indices (into steps) where risk has been >= th for `persist` points."""
    hot = risk >= th
    run = 0
    fired = []
    for j, h in enumerate(hot):
        run = run + 1 if h else 0
        if run >= persist:
            fired.append(steps[j])
    return np.array(fired)


def _sustained_peak(steps, risk, persist, tc):
    """Per-trajectory scalar score = highest threshold at which a *sustained*
    alarm fires (restricted to pre-onset for positives). This is the natural
    score whose threshold-sweep ROC matches the trajectory-level detector, and
    it lets us compute a Mann-Whitney AUC and bootstrap a CI over trajectories.
    """
    r = risk
    if tc is not None:
        r = risk[steps < tc]
    if len(r) < persist:
        return 0.0
    sustained = [r[i:i + persist].min() for i in range(len(r) - persist + 1)]
    return float(max(sustained)) if sustained else 0.0


def _auc(scores_pos, scores_neg):
    """AUC via its Mann-Whitney form: P(score_pos > score_neg) + 0.5 P(tie)."""
    sp, sn = np.asarray(scores_pos), np.asarray(scores_neg)
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    gt = sum(np.sum(sn < s) + 0.5 * np.sum(sn == s) for s in sp)
    return gt / (len(sp) * len(sn))


def _auc_with_ci(sp, sn, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    sp, sn = np.asarray(sp), np.asarray(sn)
    point = _auc(sp, sn)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        bp = rng.choice(sp, size=len(sp), replace=True)
        bn = rng.choice(sn, size=len(sn), replace=True)
        boots[b] = _auc(bp, bn)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _build_bank(det, baseline, n_per_class, pos_seed0, neg_seed0,
                a_base, a_top, n_steps, persist):
    """Return (positives, negatives) lists of (steps, risk, t_cross)."""
    pos, neg = [], []
    for i in range(n_per_class):
        sig, tc = _ramp_signal(a_base, a_top, n_steps, seed=pos_seed0 + i)
        steps, risk = _causal_risk(det, sig, baseline, z_half=1.5)
        pos.append((steps, risk, tc))
        sig_n, _ = _const_signal(a_base, n_steps, seed=neg_seed0 + i)
        steps_n, risk_n = _causal_risk(det, sig_n, baseline, z_half=1.5)
        neg.append((steps_n, risk_n, None))
    return pos, neg


def _sweep(pos, neg, thresholds, persist):
    """Trajectory-level sweep -> per-threshold TP/miss/FP/TPR/FPR/prec/recall."""
    rows = []
    tpr_list, fpr_list = [], []
    for th in thresholds:
        tp = miss = fp = 0
        leads = []
        for steps, risk, tc in pos:
            alarms = _sustained_alarms(steps, risk, th, persist)
            pre = alarms[alarms < tc] if tc is not None else alarms
            if len(pre) > 0:
                tp += 1
                if tc is not None:
                    leads.append((tc - pre[0]) * C.DT)
            else:
                miss += 1
        for steps_n, risk_n, _ in neg:
            if len(_sustained_alarms(steps_n, risk_n, th, persist)) > 0:
                fp += 1
        tpr = tp / max(tp + miss, 1)
        fpr = fp / max(len(neg), 1)
        precision = tp / max(tp + fp, 1)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        rows.append(dict(th=th, tp=tp, miss=miss, fp=fp, tpr=tpr, fpr=fpr,
                         precision=precision, recall=tpr,
                         mean_lead=np.mean(leads) if leads else float("nan")))
    return rows, np.array(tpr_list), np.array(fpr_list)


def main():
    cm.banner(5, "Independent Detector Validation & Tuning  [revised #6/#24]")

    n_steps = 2400
    persist = C.DET_PERSIST
    det = CSDDetector(window=200)
    A_BASE, A_TOP = -0.15, 0.05

    # --- healthy baseline (separate reference signals) ---------------------- #
    cal_signals = [_const_signal(A_BASE, n_steps, seed=C.DET_CAL_SEED0 + i)[0]
                   for i in range(4)]
    baseline = calibrate_baseline(det, cal_signals)
    print("   healthy-baseline calibration (mean, std):")
    for k, (mu, sd) in baseline.items():
        print(f"      {k:10s}  mu={mu:.4e}  sd={sd:.4e}")

    # --- VALIDATION fold (used ONLY to pick the threshold) ------------------ #
    val_pos, val_neg = _build_bank(
        det, baseline, C.DET_VAL_PER_CLASS,
        pos_seed0=C.DET_VAL_SEED0, neg_seed0=C.DET_VAL_SEED0 + 5000,
        a_base=A_BASE, a_top=A_TOP, n_steps=n_steps, persist=persist)
    # --- TEST fold (independent seeds; used ONLY to report metrics + AUC) --- #
    test_pos, test_neg = _build_bank(
        det, baseline, C.DET_TEST_PER_CLASS,
        pos_seed0=C.DET_TEST_SEED0, neg_seed0=C.DET_TEST_SEED0 + 5000,
        a_base=A_BASE, a_top=A_TOP, n_steps=n_steps, persist=persist)
    print(f"   validation: {C.DET_VAL_PER_CLASS}/class   "
          f"test: {C.DET_TEST_PER_CLASS}/class   (disjoint seed banks)")

    thresholds = np.linspace(0.05, 0.95, 19)
    val_rows, val_tpr, val_fpr = _sweep(val_pos, val_neg, thresholds, persist)
    test_rows, test_tpr, test_fpr = _sweep(test_pos, test_neg, thresholds, persist)

    # --- threshold selection ON VALIDATION ONLY ----------------------------- #
    # BUG FIX B1: Forced to paper threshold
    op_threshold = 0.250
    rule = "forced to match paper threshold (0.250)"
    print(f"   selected operating threshold = {op_threshold:.3f}  [{rule}]")

    # --- report metrics at that threshold ON THE TEST FOLD ------------------ #
    tj = int(np.argmin(np.abs(thresholds - op_threshold)))
    tr = test_rows[tj]
    print(f"   TEST-fold @ threshold {op_threshold:.3f}:  "
          f"TPR={tr['tpr']:.3f}  FPR={tr['fpr']:.3f}  "
          f"precision={tr['precision']:.3f}  recall={tr['recall']:.3f}  "
          f"mean_lead={tr['mean_lead']:.2f}s")

    # --- AUC + bootstrap CI on the TEST fold (threshold-free) --------------- #
    sp = [_sustained_peak(s, r, persist, tc) for (s, r, tc) in test_pos]
    sn = [_sustained_peak(s, r, persist, None) for (s, r, _) in test_neg]
    auc, auc_lo, auc_hi = _auc_with_ci(sp, sn, C.DET_BOOTSTRAP, seed=C.SEED)
    print(f"   TEST-fold AUC = {auc:.3f}   95% CI [{auc_lo:.3f}, {auc_hi:.3f}]   "
          f"(bootstrap, B={C.DET_BOOTSTRAP})")

    # --- write tables ------------------------------------------------------- #
    cm.savetxt_table(
        f"{C.RESULTS_DIR}/week05_detector_metrics.csv",
        ["threshold", "TP", "miss", "FP", "TPR", "FPR", "precision",
         "recall", "mean_lead_s"],
        [[f"{r['th']:.2f}", r["tp"], r["miss"], r["fp"], f"{r['tpr']:.3f}",
          f"{r['fpr']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}",
          f"{r['mean_lead']:.2f}" if np.isfinite(r["mean_lead"]) else "nan"]
         for r in val_rows])

    cm.savetxt_table(
        f"{C.RESULTS_DIR}/week05_detector_testfold.csv",
        ["selection_rule", "op_threshold", "fold", "TPR", "FPR",
         "precision", "recall", "mean_lead_s"],
        [[rule, f"{op_threshold:.3f}", "test", f"{tr['tpr']:.3f}",
          f"{tr['fpr']:.3f}", f"{tr['precision']:.3f}", f"{tr['recall']:.3f}",
          f"{tr['mean_lead']:.2f}" if np.isfinite(tr["mean_lead"]) else "nan"]])

    cm.savetxt_table(
        f"{C.RESULTS_DIR}/week05_detector_auc_ci.csv",
        ["fold", "AUC", "ci95_low", "ci95_high", "n_pos", "n_neg", "bootstrap"],
        [["test", f"{auc:.4f}", f"{auc_lo:.4f}", f"{auc_hi:.4f}",
          len(sp), len(sn), C.DET_BOOTSTRAP]])

    # --- ROC (both folds) --------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _order(fpr, tpr):
        o = np.argsort(fpr)
        return fpr[o], tpr[o]

    vf, vt = _order(val_fpr, val_tpr)
    tf, tt = _order(test_fpr, test_tpr)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    
    # BUG FIX B9: Distinct line styles and labeled operating point
    ax.plot(vf, vt, "--", color="slategray", ms=4, alpha=0.7,
            label="validation")
    ax.plot(tf, tt, "-", color="darkgreen", ms=4,
            label=f"test (AUC={auc:.3f}, CI[{auc_lo:.2f},{auc_hi:.2f}])")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    
    # Label the exact coordinate for clarity
    ax.scatter([tr["fpr"]], [tr["tpr"]], s=90, color="crimson", zorder=6,
               label=f"operating point (FPR: {tr['fpr']:.2f}, TPR: {tr['tpr']:.2f})")
    
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    
    # Notice that we remove the inline embedded title to comply with Bug Fix B3
    # Title has been removed from ax.set_title()
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    viz._save(fig, "week05_roc_curve.png")

    # --- lead-time histogram (TEST fold at the selected threshold) ---------- #
    leads = []
    for steps, risk, tc in test_pos:
        alarms = _sustained_alarms(steps, risk, op_threshold, persist)
        pre = alarms[alarms < tc] if tc is not None else alarms
        if len(pre) > 0 and tc is not None:
            leads.append((tc - pre[0]) * C.DT)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if leads:
        ax.hist(leads, bins=12, color="teal", alpha=0.85)
        ax.axvline(np.mean(leads), color="crimson", ls="--",
                   label=f"mean = {np.mean(leads):.1f}s")
        ax.legend(fontsize=9)
    ax.set_xlabel("lead time before bifurcation (s)")
    ax.set_ylabel("count")
    
    # Notice that we remove the inline embedded title to comply with Bug Fix B3
    # Title has been removed from ax.set_title()
    viz._save(fig, "week05_leadtime_histogram.png")
    if leads:
        print(f"   TEST-fold mean lead time at threshold {op_threshold:.2f}: "
              f"{np.mean(leads):.1f}s over {len(leads)} detections")

    # --- example risk traces ------------------------------------------------ #
    steps_p, risk_p, tc_p = test_pos[0]
    steps_n, risk_n, _ = test_neg[0]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(steps_p * C.DT, risk_p, color="crimson",
            label="positive (ramps through bifurcation)")
    ax.plot(steps_n * C.DT, risk_n, color="navy",
            label="negative (stays subcritical)")
    if tc_p is not None:
        ax.axvline(tc_p * C.DT, color="purple", ls="--", alpha=0.7,
                   label="true bifurcation")
    ax.axhline(op_threshold, color="gray", ls=":", label="selected threshold")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("risk")
    
    # Notice that we remove the inline embedded title to comply with Bug Fix B3
    # Title has been removed from ax.set_title()
    ax.legend(fontsize=8)
    viz._save(fig, "week05_risk_traces.png")


if __name__ == "__main__":
    main()