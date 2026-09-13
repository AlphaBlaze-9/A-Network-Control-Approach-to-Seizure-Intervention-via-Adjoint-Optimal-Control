"""
viz.py  --  Phase 5 / Week 11: Visualisation helpers
====================================================
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# BUG FIX B7: Standardize axis font sizes across all figures
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 10})

from . import config

def _save(fig, name):
    path = os.path.join(config.IMAGE_DIR, name)
    fig.tight_layout() # BUG FIX B10: Ensures all subplots fit in exported image
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {path}")
    return path

def node_coordinates_2d(A):
    # Offline fallback 2D coordinates
    try:
        import networkx as nx
        thr = np.percentile(A[A > 0], 80) if np.any(A > 0) else 0
        G = nx.from_numpy_array((A > thr) * A)
        pos = nx.spring_layout(G, seed=config.SEED, k=1.5 / np.sqrt(len(A)))
        coords = np.array([pos[i] for i in range(len(A))])
    except Exception:
        deg = A.sum(1)
        L = np.diag(deg) - A
        w, v = np.linalg.eigh(L)
        coords = v[:, 1:3]
    return coords, "connectome graph layout"

def plot_brain_values(A, values, title, fname, focus=None, cmap="magma"):
    # Title argument is deliberately ignored (BUG FIX B3/B4)
    v = np.asarray(values, dtype=float)
    vmax = np.percentile(v, 99) if np.any(v) else 1.0
    
    # BUG FIX B15: Attempt to render a proper brain using nilearn if available
    try:
        from nilearn import plotting
        import networkx as nx
        
        # Generate anatomical-style 3D proxy coordinates 
        thr = np.percentile(A[A > 0], 80) if np.any(A > 0) else 0
        G = nx.from_numpy_array((A > thr) * A)
        pos = nx.spring_layout(G, dim=3, seed=config.SEED)
        coords_3d = np.array([pos[i] for i in range(len(A))]) * 50 # scale for brain mapping
        
        fig = plt.figure(figsize=(8, 6))
        plotting.plot_markers(
            v, coords_3d, node_size=20 + 240 * (v / (vmax + 1e-12)).clip(0, 1),
            node_cmap=cmap, node_vmin=0, node_vmax=vmax,
            display_mode='ortho', figure=fig
        )
        return _save(fig, fname)
    except Exception:
        pass
    
    # Fallback to 2D scatter plot if nilearn fails
    coords, src = node_coordinates_2d(A)
    sizes = 20 + 240 * (v / (vmax + 1e-12)).clip(0, 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=v, s=sizes, cmap=cmap,
                    vmin=0, vmax=vmax, edgecolors="k", linewidths=0.3)
    if focus is not None:
        ax.scatter(coords[focus, 0], coords[focus, 1], s=420, facecolors="none",
                   edgecolors="cyan", linewidths=2.2, label="seizure focus")
        ax.legend(loc="upper right")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="value")
    
    ax.set_xticks([]); ax.set_yticks([])
    return _save(fig, fname)

def plot_phase_space(traj_unc, traj_ctl, N, focus, fname, title=None):
    # Title argument is deliberately ignored (BUG FIX B3/B4)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot(traj_unc[:, focus], traj_unc[:, N + focus], lw=1.0, alpha=0.8,
            color="crimson", label="uncontrolled (-> limit cycle)")
    ax.plot(traj_ctl[:, focus], traj_ctl[:, N + focus], lw=1.0, alpha=0.9,
            color="royalblue", label="controlled (-> fixed point)")
    ax.scatter([0], [0], color="k", zorder=5, label="healthy fixed point")
    th = np.linspace(0, 2 * np.pi, 200)
    rr = np.sqrt(max(config.A_SEIZURE, 1e-6))
    ax.plot(rr * np.cos(th), rr * np.sin(th), "k--", lw=0.8, alpha=0.5,
            label="seizure limit cycle (r=sqrt(a))")
    ax.set_xlabel("x (Re z)"); ax.set_ylabel("y (Im z)")
    ax.legend(loc="upper left"); ax.set_aspect("equal")
    return _save(fig, fname)

def line_plot(t, series: dict, title, ylabel, fname, vlines=None, hline=None):
    # Title argument is deliberately ignored (BUG FIX B3/B4)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for label, y in series.items():
        ax.plot(t, y, lw=1.3, label=label)
    if vlines:
        for x in vlines:
            ax.axvline(x, color="green", ls=":", alpha=0.7)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", alpha=0.7, label="threshold")
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel)
    ax.legend()
    return _save(fig, fname)

def heatmap(M, title, fname, xlabel="time (s)", ylabel="node", cmap="viridis",
            extent=None, focus=None, cbar_label="amplitude"):
    # Title argument is deliberately ignored (BUG FIX B3/B4)
    # BUG FIX B19: Explicit 'viridis' fallback cmap, customizable cbar_label
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(M, aspect="auto", cmap=cmap, origin="lower", extent=extent)
    fig.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
    if focus is not None:
        ax.axhline(focus, color="cyan", ls="--", lw=1.0, alpha=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return _save(fig, fname)