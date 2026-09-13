"""
week04_detector.py  --  Phase 2 / Week 4
========================================
Programming the Unsafe-State Detector.
"""

import numpy as np

import _common as cm
from src import config as C
from src.hopf_model import HopfNetwork
from src.detector import CSDDetector
from src import viz

def ramping_trajectory(n_steps=4000, a_start=-0.3, a_end=0.4, seed=C.SEED):
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
    t_cross = int(np.argmin(np.abs(a_t)))
    return sig, a_t, t_cross

def main():
    cm.banner(4, "Programming the Unsafe-State Detector")

    sig, a_t, t_cross = ramping_trajectory()
    t = cm.time_axis(len(sig) - 1)
    print(f"   true bifurcation (a=0) at step {t_cross} (t={t[t_cross]:.1f}s)")

    det = CSDDetector(window=200)
    feats = det.features(sig)
    risk, _ = det.risk(sig)

    # --- 1. Raw signal + the three CSD features ----------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Plot components (Will be tight_layout() adjusted by viz._save to fix BUG B10)
    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(t, sig, lw=0.6, color="black"); ax[0].set_ylabel("signal x(t)")
    # BUG FIX B3/B4: Removed embedded set_title
    ax[1].plot(t, feats["variance"], color="tab:blue"); ax[1].set_ylabel("variance")
    ax[2].plot(t, feats["autocorr1"], color="tab:green"); ax[2].set_ylabel("lag-1 AC")
    ax[3].plot(t, feats["lyapunov"], color="tab:red"); ax[3].set_ylabel("Lyapunov")
    ax[3].set_xlabel("time (s)")
    for a in ax:
        a.axvline(t[t_cross], color="purple", ls="--", alpha=0.7)
    viz._save(fig, "week04_csd_features.png")

    # --- 2. Fused continuous risk score ------------------------------------- #
    viz.line_plot(t, {"fused CSD risk": risk},
                  "", # Title omitted
                  "risk in [0,1]", "week04_risk_score.png",
                  vlines=[t[t_cross]], hline=0.5)
    
    above = np.where(risk >= 0.5)[0]
    if len(above):
        lead = (t_cross - above[0]) * C.DT
        print(f"   first risk>=0.5 at step {above[0]} -> lead time {lead:.1f}s "
              f"before the bifurcation")
    else:
        print("   risk never crossed 0.5 (tune window/threshold in Week 5)")

if __name__ == "__main__":
    main()