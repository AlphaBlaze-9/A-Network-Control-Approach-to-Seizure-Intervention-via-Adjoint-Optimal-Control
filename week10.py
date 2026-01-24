#!/usr/bin/env python3
"""
Week 10: Advanced Visualization (3D brain mesh + videos)
FIXED: Corrected color mapping (Blue=Healthy) and sustained seizure rendering.
"""

import argparse
import os
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore

# Nilearn is used to fetch + load cortical surface meshes.
try:
    from nilearn import datasets
    from nilearn import surface
    _HAVE_NILEARN = True
except Exception:
    _HAVE_NILEARN = False


# ------------------------------
# Utilities
# ------------------------------

def _compute_amplitude(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return np.asarray(x, dtype=float)
    if x.ndim == 3:
        return np.linalg.norm(x, axis=-1)
    raise ValueError(f"Unsupported x shape {x.shape}; expected (T,N) or (T,N,d).")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _maybe_get(data: dict, key: str, default=None):
    return data[key] if key in data else default


# ------------------------------
# Surface mesh handling
# ------------------------------

def _load_fsaverage5_surfaces(kind: str = "infl") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not _HAVE_NILEARN:
        raise RuntimeError("nilearn is not available")

    fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    if kind == "infl":
        left_path, right_path = fs["infl_left"], fs["infl_right"]
    elif kind == "pial":
        left_path, right_path = fs["pial_left"], fs["pial_right"]
    else:
        raise ValueError("kind must be 'infl' or 'pial'")

    vL, fL = surface.load_surf_mesh(left_path)
    vR, fR = surface.load_surf_mesh(right_path)
    return np.asarray(vL), np.asarray(fL), np.asarray(vR), np.asarray(fR)


def _sphere_mesh(subdiv: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0, 2*np.pi, subdiv, endpoint=False)
    v = np.linspace(0, np.pi, subdiv)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    faces = []
    n_u = subdiv
    n_v = subdiv
    def idx(iu, iv):
        return iv*n_u + iu

    for iv in range(n_v - 1):
        for iu in range(n_u):
            iu2 = (iu + 1) % n_u
            a = idx(iu, iv)
            b = idx(iu2, iv)
            c = idx(iu, iv+1)
            d = idx(iu2, iv+1)
            faces.append([a, c, b])
            faces.append([b, c, d])
    return verts, np.asarray(faces, dtype=int)


def _nearest_vertices(points: np.ndarray, verts: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    verts = np.asarray(verts, dtype=float)
    out = np.empty_like(points)
    chunk = 512
    for i in range(0, points.shape[0], chunk):
        P = points[i:i+chunk]
        d2 = np.sum((P[:, None, :] - verts[None, :, :])**2, axis=-1)
        j = np.argmin(d2, axis=1)
        out[i:i+chunk] = verts[j]
    return out


# ------------------------------
# Rendering
# ------------------------------

def _poly3d_from_faces(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return verts[faces]


def _setup_3d_ax(fig, elev=15, azim=110):
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 0.75])
    return ax


def _add_brain_mesh(ax, vL, fL, vR, fR, alpha=0.12):
    polyL = Poly3DCollection(_poly3d_from_faces(vL, fL), linewidths=0.0, alpha=alpha)
    polyR = Poly3DCollection(_poly3d_from_faces(vR, fR), linewidths=0.0, alpha=alpha)
    polyL.set_facecolor((0.6, 0.6, 0.6, alpha))
    polyR.set_facecolor((0.6, 0.6, 0.6, alpha))
    ax.add_collection3d(polyL)
    ax.add_collection3d(polyR)


def _set_limits_from_mesh(ax, vL, vR, pad=5.0):
    V = np.vstack([vL, vR])
    mins = V.min(axis=0) - pad
    maxs = V.max(axis=0) + pad
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])


def _frame_scatter(
    ax,
    node_xyz: np.ndarray,
    amp_t: np.ndarray,
    threshold: float,
    mode: str,
    u_t: Optional[np.ndarray] = None,
    stim_nodes: Optional[np.ndarray] = None,
    title: Optional[str] = None,
):
    """
    Draw scatter for one frame.
    mode:
      - "uncontrolled": red seizure nodes, light gray others
      - "controlled": cool/blue for healthy nodes, red for seizure nodes
    """
    N = amp_t.shape[0]
    seizure = amp_t >= threshold

    # Marker sizes
    base = 16.0
    sizes = np.full(N, base, dtype=float)
    if u_t is not None:
        u_abs = np.abs(u_t).astype(float)
        # Normalize robustly: scale sizes based on control magnitude
        denom = np.percentile(u_abs, 95) if np.any(u_abs > 0) else 1.0
        denom = max(denom, 1e-9)
        # Dynamic sizing: large control = large nodes
        sizes += 140.0 * np.clip(u_abs / denom, 0, 1)

    # Colors
    if mode == "uncontrolled":
        colors = np.zeros((N, 4), dtype=float)
        colors[:] = (0.75, 0.75, 0.75, 0.95)  # non-seizure
        colors[seizure] = (1.0, 0.15, 0.15, 0.98)  # seizure red
    elif mode == "controlled":
        colors = np.zeros((N, 4), dtype=float)
        # Health score: 1 means very healthy (Blue/Cyan), 0 means at threshold
        health = np.clip(1.0 - (amp_t / max(threshold, 1e-9)), 0.0, 1.0)
        
        # FIX: Flip the color map input. 
        # cm.cool(0.0) is Cyan (Healthy). cm.cool(1.0) is Magenta (Borderline).
        # We want Healthy -> Cyan, so we map health=1.0 to index 0.0
        rgba = cm.cool(1.0 - health) 
        
        colors[:] = rgba
        # If any node exceeds threshold, show it in red (Failure/Spike)
        colors[seizure] = (1.0, 0.15, 0.15, 0.98)
    else:
        raise ValueError("mode must be 'uncontrolled' or 'controlled'")

    ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], s=sizes, c=colors, depthshade=False)

    # Optional: highlight stimulatable nodes with an outline (ring)
    if stim_nodes is not None and len(stim_nodes) > 0:
        stim_xyz = node_xyz[stim_nodes]
        ax.scatter(
            stim_xyz[:, 0], stim_xyz[:, 1], stim_xyz[:, 2],
            s=np.maximum(sizes[stim_nodes], base) + 30.0,
            facecolors="none",
            edgecolors=(0.0, 0.0, 0.0, 0.55),
            linewidths=0.8,
            depthshade=False,
        )

    if title:
        ax.set_title(title, fontsize=11)


def _render_video(
    out_path: Path,
    amp: np.ndarray,
    node_xyz: np.ndarray,
    threshold: float,
    mesh: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dt: float,
    mode: str,
    u_node: Optional[np.ndarray] = None,
    stim_nodes: Optional[np.ndarray] = None,
    fps: int = 24,
    frame_stride: int = 2,
    elev: float = 15,
    azim: float = 110,
    dpi: int = 180,
    max_frames: Optional[int] = 900,
):
    vL, fL, vR, fR = mesh
    T, N = amp.shape

    frame_ids = list(range(0, T, frame_stride))
    if max_frames is not None and len(frame_ids) > max_frames:
        frame_ids = frame_ids[:max_frames]

    _ensure_dir(out_path.parent)

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    try:
        for k, t in enumerate(frame_ids):
            fig = plt.figure(figsize=(7.2, 7.2))
            ax = _setup_3d_ax(fig, elev=elev, azim=azim)
            _add_brain_mesh(ax, vL, fL, vR, fR, alpha=0.12)
            _set_limits_from_mesh(ax, vL, vR, pad=6.0)

            time_s = t * dt
            title = f"{mode.title()}  |  t = {time_s:.2f}s"
            _frame_scatter(
                ax=ax,
                node_xyz=node_xyz,
                amp_t=amp[t],
                threshold=threshold,
                mode=mode,
                u_t=(u_node[t] if u_node is not None else None),
                stim_nodes=stim_nodes,
                title=title,
            )

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(img)
            plt.close(fig)
    finally:
        writer.close()


def _save_key_figures(
    out_dir: Path,
    amp_un: np.ndarray,
    amp_cl: np.ndarray,
    node_xyz: np.ndarray,
    threshold: float,
    mesh: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dt: float,
    u_node: Optional[np.ndarray],
    stim_nodes: Optional[np.ndarray],
    times: Tuple[float, ...] = (0.0, 2.5, 5.0, 7.5),
    dpi: int = 300,
):
    vL, fL, vR, fR = mesh
    T = amp_un.shape[0]

    for mode, amp, suffix in [
        ("uncontrolled", amp_un, "uncontrolled"),
        ("controlled", amp_cl, "controlled"),
    ]:
        for tt in times:
            t_idx = int(round(tt / max(dt, 1e-9)))
            t_idx = max(0, min(T - 1, t_idx))
            fig = plt.figure(figsize=(8.0, 8.0))
            ax = _setup_3d_ax(fig, elev=15, azim=110)
            _add_brain_mesh(ax, vL, fL, vR, fR, alpha=0.12)
            _set_limits_from_mesh(ax, vL, vR, pad=6.0)

            title = f"{mode.title()}  |  t = {t_idx*dt:.2f}s"
            _frame_scatter(
                ax=ax,
                node_xyz=node_xyz,
                amp_t=amp[t_idx],
                threshold=threshold,
                mode=mode,
                u_t=(u_node[t_idx] if (u_node is not None and mode == "controlled") else None),
                stim_nodes=stim_nodes,
                title=title,
            )
            out_path = out_dir / f"fig_{suffix}_t{t_idx:05d}.png"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Week 10: Advanced Visualization")
    ap.add_argument("--npz", type=str, required=True, help="Path to .npz containing simulation outputs")
    ap.add_argument("--out", type=str, default="week10_viz_out", help="Output directory")
    ap.add_argument("--threshold", type=float, default=None, help="Seizure amplitude threshold (overrides file)")
    ap.add_argument("--dt", type=float, default=None, help="Time step in seconds (overrides file)")
    ap.add_argument("--fps", type=int, default=24, help="Video frames per second")
    ap.add_argument("--stride", type=int, default=2, help="Render every k-th simulation step")
    ap.add_argument("--mesh", type=str, default="infl", choices=["infl", "pial"], help="Surface mesh type")
    ap.add_argument("--no_nilearn_download", action="store_true",
                    help="Do not attempt nilearn dataset download; fallback mesh will be used.")
    args = ap.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    _ensure_dir(out_dir)

    data = _load_npz(npz_path)

    # Required arrays
    x_un = _maybe_get(data, "x_uncontrolled")
    x_cl = _maybe_get(data, "x_controlled")
    node_coords = _maybe_get(data, "node_coords")

    if x_un is None or x_cl is None or node_coords is None:
        missing = [k for k in ["x_uncontrolled", "x_controlled", "node_coords"] if k not in data]
        raise SystemExit(f"Missing required arrays in NPZ: {missing}")

    amp_un = _compute_amplitude(np.asarray(x_un))
    amp_cl = _compute_amplitude(np.asarray(x_cl))

    # dt + threshold
    dt = float(args.dt if args.dt is not None else _maybe_get(data, "dt", 0.01))
    threshold = float(args.threshold if args.threshold is not None else _maybe_get(data, "threshold", np.percentile(amp_un, 90)))

    # Control input (optional)
    u_ctrl = _maybe_get(data, "u_controlled", None)
    B = _maybe_get(data, "B", None)
    u_node = None
    if u_ctrl is not None:
        u_ctrl = np.asarray(u_ctrl)
        if u_ctrl.shape[1] == amp_cl.shape[1]:
            u_node = u_ctrl
        else:
            if B is None:
                raise SystemExit("u_controlled appears to be actuator-space (T,M) but B (N,M) is missing.")
            B = np.asarray(B)
            u_node = (B @ u_ctrl.T).T

    # Stim nodes (optional)
    stim_nodes = _maybe_get(data, "stim_nodes", None)
    if stim_nodes is None and B is not None:
        row_norm = np.linalg.norm(np.asarray(B), axis=1)
        stim_nodes = np.where(row_norm > 1e-12)[0]
    if stim_nodes is not None:
        stim_nodes = np.asarray(stim_nodes, dtype=int)

    # Load mesh
    use_nilearn = _HAVE_NILEARN and (not args.no_nilearn_download)
    if use_nilearn:
        try:
            vL, fL, vR, fR = _load_fsaverage5_surfaces(kind=args.mesh)
        except Exception as e:
            print(f"[WARN] Could not load fsaverage surfaces via nilearn ({e}). Falling back to sphere mesh.")
            use_nilearn = False

    if not use_nilearn:
        verts, faces = _sphere_mesh(subdiv=50)
        vL = verts[verts[:, 0] <= 0]
        vR = verts[verts[:, 0] >= 0]
        left_mask = verts[:, 0] <= 0
        right_mask = verts[:, 0] >= 0
        left_idx = np.where(left_mask)[0]
        right_idx = np.where(right_mask)[0]
        left_map = -np.ones(verts.shape[0], dtype=int); left_map[left_idx] = np.arange(left_idx.size)
        right_map = -np.ones(verts.shape[0], dtype=int); right_map[right_idx] = np.arange(right_idx.size)
        fL = []
        fR = []
        for tri in faces:
            if left_mask[tri].all():
                fL.append([left_map[i] for i in tri])
            if right_mask[tri].all():
                fR.append([right_map[i] for i in tri])
        fL = np.asarray(fL, dtype=int) if len(fL) else np.zeros((0, 3), dtype=int)
        fR = np.asarray(fR, dtype=int) if len(fR) else np.zeros((0, 3), dtype=int)

    mesh = (vL, fL, vR, fR)
    all_verts = np.vstack([vL, vR])
    node_xyz = _nearest_vertices(np.asarray(node_coords), all_verts)

    # Outputs
    vid_un = out_dir / "video_uncontrolled_seizure_spread.mp4"
    vid_cl = out_dir / "video_controlled_suppression.mp4"

    print(f"[INFO] Writing videos to:\n  {vid_un}\n  {vid_cl}")
    _render_video(
        out_path=vid_un,
        amp=amp_un,
        node_xyz=node_xyz,
        threshold=threshold,
        mesh=mesh,
        dt=dt,
        mode="uncontrolled",
        u_node=None,
        stim_nodes=stim_nodes,
        fps=args.fps,
        frame_stride=args.stride,
        dpi=180,
    )
    _render_video(
        out_path=vid_cl,
        amp=amp_cl,
        node_xyz=node_xyz,
        threshold=threshold,
        mesh=mesh,
        dt=dt,
        mode="controlled",
        u_node=u_node,
        stim_nodes=stim_nodes,
        fps=args.fps,
        frame_stride=args.stride,
        dpi=180,
    )

    # High-res figures
    fig_dir = out_dir / "figures"
    _ensure_dir(fig_dir)
    _save_key_figures(
        out_dir=fig_dir,
        amp_un=amp_un,
        amp_cl=amp_cl,
        node_xyz=node_xyz,
        threshold=threshold,
        mesh=mesh,
        dt=dt,
        u_node=u_node,
        stim_nodes=stim_nodes,
        times=(0.0, 2.5, 5.0, 7.5),
        dpi=320,
    )
    print(f"[DONE] Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()