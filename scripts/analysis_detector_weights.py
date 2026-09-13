"""
analysis_detector_weights.py  --  Reviewer response (detector feature-weight ablation)
======================================================================================
Reviewer 2 (minor): "The detector combines its features with an equal, untested
weighting. A brief ablation of these weights would strengthen the result."

This script re-uses the EXACT validation machinery of week05_detector_validation.py
(same calibration signals, same disjoint validation / test seed banks, same causal
strided feature evaluation, same sustained-alarm rule) but computes the three raw
CSD features ONCE per trajectory and then fuses them under a grid of weightings.
For every weighting it reports, on the independent TEST fold:
  * AUC with a bootstrap 95% CI (threshold-free, Mann-Whitney form),
  * TPR / FPR / precision / mean lead time at the threshold selected on the
    VALIDATION fold by the paper's rule (highest threshold at full recall),
  * the same operating-point metrics at the fixed paper threshold 0.250.

Outputs
    results/analysis_detector_weights.csv
    images/analysis_detector_weights.png

Run:
    python scripts/analysis_detector_weights.py
"""
import os
import numpy as np

import _common as cm
from src import config as C
from src.detector import CSDDetector
import week05_detector_validation as w5

WEIGHT_SETS = [
    ("var+AC1+Lyap (0.4,0.4,0.2) [used]", (0.4, 0.4, 0.2)),
    ("var+AC1 equal (0.5,0.5,0)", (0.5, 0.5, 0.0)),
    ("var only (1,0,0)", (1.0, 0.0, 0.0)),
    ("AC1 only (0,1,0)", (0.0, 1.0, 0.0)),
    ("Lyap only (0,0,1)", (0.0, 0.0, 1.0)),
    ("var-heavy (0.6,0.2,0.2)", (0.6, 0.2, 0.2)),
    ("AC1-heavy (0.2,0.6,0.2)", (0.2, 0.6, 0.2)),
    ("Lyap-heavy (0.25,0.25,0.5)", (0.25, 0.25, 0.5)),
    ("var+AC1 (0.7,0.3,0)", (0.7, 0.3, 0.0)),
    ("var+AC1 (0.3,0.7,0)", (0.3, 0.7, 0.0)),
]
KEYS = ("variance", "autocorr1", "lyapunov")
Z_HALF = 1.5          # identical to week05 (_build_bank passes z_half=1.5)


def _bank_features(det, n_per_class, pos_seed0, neg_seed0, a_base, a_top, n_steps):
    pos, neg = [], []
    for i in range(n_per_class):
        sig, tc = w5._ramp_signal(a_base, a_top, n_steps, seed=pos_seed0 + i)
        steps, feats = w5._features_at(det, sig)
        pos.append((steps, feats, tc))
        sig_n, _ = w5._const_signal(a_base, n_steps, seed=neg_seed0 + i)
        steps_n, feats_n = w5._features_at(det, sig_n)
        neg.append((steps_n, feats_n, None))
    return pos, neg


def _fuse(feats, baseline, weights):
    risk = None
    for fi, key in enumerate(KEYS):
        mu, sd = baseline[key]
        z = np.clip((feats[key] - mu) / sd, 0, None)
        term = weights[fi] * (z / (z + Z_HALF))
        risk = term if risk is None else risk + term
    return risk


def _to_bank(bank, baseline, weights):
    return [(steps, _fuse(feats, baseline, weights), tc) for steps, feats, tc in bank]


def main():
    cm.banner(0, "Detector feature-weight ablation (validation -> test)")
    n_steps = 2400
    persist = C.DET_PERSIST
    det = CSDDetector(window=200)
    A_BASE, A_TOP = -0.15, 0.05

    cal_signals = [w5._const_signal(A_BASE, n_steps, seed=C.DET_CAL_SEED0 + i)[0] for i in range(4)]
    baseline = w5.calibrate_baseline(det, cal_signals)

    print("   computing raw features once per trajectory (validation + test banks) ...")
    val_pos, val_neg = _bank_features(det, C.DET_VAL_PER_CLASS, C.DET_VAL_SEED0, C.DET_VAL_SEED0 + 5000,
                                      A_BASE, A_TOP, n_steps)
    test_pos, test_neg = _bank_features(det, C.DET_TEST_PER_CLASS, C.DET_TEST_SEED0, C.DET_TEST_SEED0 + 5000,
                                        A_BASE, A_TOP, n_steps)
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for name, w in WEIGHT_SETS:
        w = np.asarray(w, float); w = w / w.sum()
        vp, vn = _to_bank(val_pos, baseline, w), _to_bank(val_neg, baseline, w)
        tp_, tn_ = _to_bank(test_pos, baseline, w), _to_bank(test_neg, baseline, w)
        val_rows, _, _ = w5._sweep(vp, vn, thresholds, persist)
        test_rows, _, _ = w5._sweep(tp_, tn_, thresholds, persist)
        # paper's selection rule on the VALIDATION fold: highest threshold with full recall
        full = [r for r in val_rows if r["tpr"] >= 1.0 - 1e-12]
        th_sel = max(r["th"] for r in full) if full else thresholds[0]
        tj = int(np.argmin(np.abs(thresholds - th_sel)))
        tr = test_rows[tj]
        t25 = test_rows[int(np.argmin(np.abs(thresholds - 0.25)))]
        sp = [w5._sustained_peak(s, r, persist, tc) for (s, r, tc) in tp_]
        sn = [w5._sustained_peak(s, r, persist, None) for (s, r, _) in tn_]
        auc, lo, hi = w5._auc_with_ci(sp, sn, C.DET_BOOTSTRAP, seed=C.SEED)
        print(f"   {name:36s} AUC={auc:.3f} [{lo:.3f},{hi:.3f}]  th_val={th_sel:.2f} -> "
              f"test TPR={tr['tpr']:.2f} FPR={tr['fpr']:.2f} lead={tr['mean_lead']:.1f}s | "
              f"@0.25: TPR={t25['tpr']:.2f} FPR={t25['fpr']:.2f}")
        rows.append([name, f"{w[0]:.2f}", f"{w[1]:.2f}", f"{w[2]:.2f}",
                     f"{auc:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{th_sel:.3f}",
                     f"{tr['tpr']:.3f}", f"{tr['fpr']:.3f}", f"{tr['precision']:.3f}",
                     f"{tr['mean_lead']:.2f}" if np.isfinite(tr['mean_lead']) else "nan",
                     f"{t25['tpr']:.3f}", f"{t25['fpr']:.3f}", f"{t25['precision']:.3f}",
                     f"{t25['mean_lead']:.2f}" if np.isfinite(t25['mean_lead']) else "nan"])

    cm.savetxt_table(
        f"{C.RESULTS_DIR}/analysis_detector_weights.csv",
        ["weighting", "w_var", "w_ac1", "w_lyap", "test_AUC", "auc_ci95_low", "auc_ci95_high",
         "threshold_selected_on_validation", "test_TPR_at_selected", "test_FPR_at_selected",
         "test_precision_at_selected", "test_lead_s_at_selected",
         "test_TPR_at_0.25", "test_FPR_at_0.25", "test_precision_at_0.25", "test_lead_s_at_0.25"],
        rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r[0] for r in rows]
    auc = np.array([float(r[4]) for r in rows]); lo = np.array([float(r[5]) for r in rows]); hi = np.array([float(r[6]) for r in rows])
    fpr = np.array([float(r[9]) for r in rows]); tpr = np.array([float(r[8]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    y = np.arange(len(names))
    axes[0].barh(y, auc, xerr=[auc - lo, hi - auc], capsize=4, color="teal")
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8); axes[0].invert_yaxis()
    axes[0].set_xlabel("test-fold AUC (95% bootstrap CI)"); axes[0].set_xlim(0.5, 1.0)
    axes[1].barh(y - 0.2, tpr, height=0.4, color="seagreen", label="TPR (test)")
    axes[1].barh(y + 0.2, fpr, height=0.4, color="indianred", label="FPR (test)")
    axes[1].set_yticks(y); axes[1].set_yticklabels([""] * len(names)); axes[1].invert_yaxis()
    axes[1].set_xlabel("operating point selected on validation fold (full recall)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(C.IMAGE_DIR, "analysis_detector_weights.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"   saved {out}")


if __name__ == "__main__":
    main()
