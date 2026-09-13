"""
week01_connectome.py  --  Phase 1 / Week 1
==========================================
Structural Network Acquisition & Scale Strategy.
"""

import numpy as np

import _common as cm
from src import config as C
from src.connectome import (load_raw_connectome, clean_and_normalise,
                            degree_vector)
from src import viz

def main():
    cm.banner(1, "Structural Network Acquisition & Scale Strategy")

    # --- 1. Load the raw connectome and clean/normalise it ------------------ #
    raw, labels = load_raw_connectome()
    A_full = clean_and_normalise(raw)
    np.save(C.A_MATRIX_NPY, A_full.astype(np.float32))
    print(f"Loaded {raw.shape[0]}-node {C.ATLAS_NAME} connectome.")
    print(f"   density            : {np.count_nonzero(A_full)/A_full.size:6.3f}")
    print(f"   spectral radius (A): {np.max(np.abs(np.linalg.eigvalsh(A_full))):6.3f}")
    print(f"   focus region       : idx {C.FOCUS_NODE} = {labels[C.FOCUS_NODE]}")
    print(f"   cached normalised A -> {C.A_MATRIX_NPY}")

    A, lab, focus = cm.load_network()
    N = A.shape[0]

    # --- 2. Connectivity-matrix heatmap ------------------------------------- #
    # BUG FIX B6: changed cbar_label to "normalized connectivity weight"
    viz.heatmap(np.log1p(A), 
                    "", # Title omitted
                    "week01_connectome_matrix.png",
                    xlabel="region", ylabel="region", focus=focus, 
                    cbar_label="normalized connectivity weight")

    # --- 3. Weighted-degree distribution ------------------------------------ #
    deg = degree_vector(A)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(deg, bins=40, color="slateblue", alpha=0.85)
    ax.axvline(deg[focus], color="crimson", ls="--",
               label=f"focus {lab[focus]} (deg={deg[focus]:.2f})")
    ax.set_xlabel("weighted degree"); ax.set_ylabel("number of regions")
    ax.legend(fontsize=9)
    # BUG FIX B3/B4: Removed embedded set_title
    viz._save(fig, "week01_degree_distribution.png")

    # --- 4. Eigenvalue spectrum before vs after normalisation --------------- #
    cleaned_unnorm = raw.astype(float).copy()
    np.fill_diagonal(cleaned_unnorm, 0.0)
    cleaned_unnorm = 0.5 * (cleaned_unnorm + cleaned_unnorm.T)
    ev_before = np.linalg.eigvalsh(cleaned_unnorm)
    ev_after = np.linalg.eigvalsh(clean_and_normalise(raw))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(np.sort(ev_before)[::-1], ".", ms=3, color="gray")
    ax[0].set_xlabel("index"); ax[0].set_ylabel("eigenvalue")
    
    ax[1].plot(np.sort(ev_after)[::-1], ".", ms=3, color="teal")
    ax[1].axhline(1.0, color="crimson", ls="--", lw=0.8, label="spectral radius = 1")
    ax[1].axhline(-1.0, color="crimson", ls="--", lw=0.8)
    ax[1].set_xlabel("index")
    ax[1].legend(fontsize=9)
    # BUG FIX B3/B4: Removed embedded set_title and suptitle
    viz._save(fig, "week01_eigenvalue_spectrum.png")

    # --- 5. Written scale plan + literature baseline (Week-1 deliverable) --- #
    plan = f"""WEEK 1 DELIVERABLE -- SCALE PLAN & LITERATURE BASELINE
======================================================
Atlas / parcellation : {C.ATLAS_NAME}
Design target        : {C.N_NODES_FULL} nodes (full Glasser cortical parcellation)
Reduced dev scale    : {C.REDUCED_SIZE} nodes (dense sub-network around the focus)
Normalisation        : diagonal zeroed, matrix symmetrised, divided by the
                       largest-magnitude eigenvalue so spectral radius = 1.0
                       (matches the user's original week1.py recipe).
Seizure-onset focus  : idx {C.FOCUS_NODE} = {C.FOCUS_NAME}
                       mesial-temporal zone = {C.MESIAL_TEMPORAL}

Scale strategy (honest):
  * All heavy optimal-control development (Weeks 6-10) is validated first on
    the {C.REDUCED_SIZE}-node sub-network for speed, then re-run at the full
    {C.N_NODES_FULL}-node design target at the documented scale-up checkpoint.
  * EVERY figure is stamped with the scale it was actually produced at via
    config.scale_label(); a reduced-scale result is never presented as a
    360-node result.
  * Switch scales with the SCALE environment variable: `SCALE=reduced ...`.

Chosen Week-10 literature baseline:
  {C.LITERATURE_BASELINE}
"""
    out = f"{C.RESULTS_DIR}/week01_scale_plan.txt"
    with open(out, "w") as fh:
        fh.write(plan)
    print("\n" + plan)
    print(f"   wrote {out}")

if __name__ == "__main__":
    main()