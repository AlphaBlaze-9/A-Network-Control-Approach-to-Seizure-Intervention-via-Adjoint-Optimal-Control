#!/usr/bin/env python3
"""
Week 11: Quantitative Deliverable Compilation

What this script produces (research-ready artifacts):
1) Core dataset package:
   - adjacency_matrix.npy (A)
   - excitability_weights.csv (I_ext / "excitability")
   - actuator_matrix.npy (B)
   - node_coords.npy (for Week 10/figures)
   - week11_core_data.npz (all of the above + metadata)

2) Optimization results ("which nodes were stimulated"):
   - stimulation_ranking.csv (per-node energy / magnitude statistics)
   - stimulation_summary.json (key scalar metrics + top-K nodes)

3) Baseline comparison metric (Week 8-style):
   - performance_comparison.csv
   - week11_results_section.md (drop-in write-up section)

Designed to work with your existing pipeline:
- Uses results.npz produced by week7.py (saved for Week 10). See week7 handoff save block.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ControlStats:
    node: int
    energy: float
    mean_abs_u: float
    peak_abs_u: float
    active_fraction: float


@dataclass(frozen=True)
class PerformanceMetrics:
    n_nodes: int
    n_steps: int
    dt: float
    threshold: float
    energy_cbf: float
    baseline_k: int
    baseline_mag: float
    baseline_nodes: List[int]
    energy_baseline: float
    efficiency_ratio: float
    percent_energy_saved: float
    overlap_cbf_topk_with_baseline: int


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _maybe_get(d: Dict[str, np.ndarray], key: str, default=None):
    return d[key] if key in d else default


def _write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _rank_control_nodes(
    u_node: np.ndarray,
    threshold_u: Optional[float] = None,
) -> List[ControlStats]:
    """Compute per-node stimulation statistics from node-level u(t)."""
    if u_node.ndim != 2:
        raise ValueError(f"u_node must be (T,N). Got {u_node.shape}")

    abs_u = np.abs(u_node)
    if threshold_u is None:
        p95 = float(np.percentile(abs_u, 95)) if abs_u.size else 0.0
        threshold_u = 0.05 * p95

    energy = np.sum(u_node ** 2, axis=0)
    mean_abs = np.mean(abs_u, axis=0)
    peak_abs = np.max(abs_u, axis=0)
    active_frac = np.mean(abs_u >= threshold_u, axis=0)

    stats = [
        ControlStats(
            node=i,
            energy=float(energy[i]),
            mean_abs_u=float(mean_abs[i]),
            peak_abs_u=float(peak_abs[i]),
            active_fraction=float(active_frac[i]),
        )
        for i in range(u_node.shape[1])
    ]
    stats.sort(key=lambda s: s.energy, reverse=True)
    return stats


def _infer_u_node(data: Dict[str, np.ndarray], shape_txn: Tuple[int, int]) -> Optional[np.ndarray]:
    """Return node-level u(t) as (T,N). Maps actuator-space u via B if needed."""
    u_ctrl = _maybe_get(data, "u_controlled", None)
    if u_ctrl is None:
        return None

    u_ctrl = np.asarray(u_ctrl)
    T, N = shape_txn

    if u_ctrl.ndim != 2:
        raise ValueError(f"u_controlled must be 2D. Got {u_ctrl.shape}")

    if u_ctrl.shape == (T, N):
        return u_ctrl

    B = _maybe_get(data, "B", None)
    if B is None:
        raise ValueError("u_controlled is not (T,N); cannot map without B (N,M).")

    B = np.asarray(B)
    if B.ndim != 2 or B.shape[0] != N:
        raise ValueError(f"B must be (N,M) with N={N}. Got {B.shape}")

    if u_ctrl.shape[0] != T:
        raise ValueError(f"u_controlled has wrong time length. Expected T={T}, got {u_ctrl.shape[0]}")
    return (B @ u_ctrl.T).T


def _reconstruct_A_and_excitability(n_nodes: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Rebuild A and I_ext deterministically by instantiating week7. (results.npz does not store them.)"""
    try:
        from week7 import ClosedLoopSimulation  # type: ignore
    except Exception as e:
        raise RuntimeError("Could not import week7.py. Put week11.py in the same folder.") from e

    sim = ClosedLoopSimulation(num_nodes=int(n_nodes), seed=int(seed))
    A = np.asarray(sim.adj_matrix, dtype=float)
    I_ext = np.asarray(sim.I_ext, dtype=float).reshape(-1)

    if A.shape != (n_nodes, n_nodes):
        raise ValueError(f"Adjacency A has wrong shape {A.shape} (expected {(n_nodes, n_nodes)})")
    if I_ext.shape != (n_nodes,):
        raise ValueError(f"I_ext has wrong shape {I_ext.shape} (expected {(n_nodes,)})")
    return A, I_ext


def _gramian_hubs(n_nodes: int, top_k: int = 3, seed: int = 42) -> List[int]:
    """Compute baseline hub nodes using Week 8 Gramian ranking."""
    try:
        from week8 import Week8Analysis  # type: ignore
    except Exception as e:
        raise RuntimeError("Could not import week8.py. Put week11.py in the same folder.") from e

    analysis = Week8Analysis(num_nodes=int(n_nodes), seed=int(seed))
    Wc = analysis.compute_gramian()
    top_nodes, _ = analysis.identify_hubs(Wc, top_k=int(top_k))
    return [int(x) for x in np.asarray(top_nodes).reshape(-1).tolist()]


def _baseline_energy(steps: int, baseline_nodes: List[int], baseline_mag: float) -> float:
    k = len(baseline_nodes)
    return float(steps * (k * (baseline_mag ** 2)))


def _write_results_section_md(
    path: Path,
    metrics: PerformanceMetrics,
    top_stats: List[ControlStats],
    focus_node: int = 0,
) -> None:
    _ensure_dir(path.parent)
    lines: List[str] = []
    lines.append("# Optimization Results (Week 11)\n\n")

    lines.append("## Core data package\n")
    lines.append(
        "We package the structural adjacency matrix $A$, node excitability weights ($I_{ext}$), actuator matrix $B$, "
        "node coordinates, and simulation metadata into a single reproducible repository artifact "
        "(`week11_core_data.npz`) plus CSV exports.\n\n"
    )

    lines.append("## Optimization results: stimulation targets selected by the CBF-QP\n")
    lines.append(
        "At each time step, the controller solves a minimum-energy quadratic program to keep voltages below the "
        "safety threshold. The resulting control signal $u(t)$ is analyzed per node using an energy proxy "
        "$E_i = \\sum_t u_i(t)^2$.\n\n"
    )

    lines.append(f"- Nodes: **{metrics.n_nodes}**\n")
    lines.append(
        f"- Steps: **{metrics.n_steps}** (dt = **{metrics.dt} s**, total T ≈ **{metrics.n_steps*metrics.dt:.2f} s**)\n"
    )
    lines.append(f"- Safety threshold: **{metrics.threshold:.3f}**\n")
    lines.append(f"- Focus node (seizure onset in this simulation): **{focus_node}**\n\n")

    lines.append("### Top stimulated nodes (ranked by $\\sum_t u_i(t)^2$)\n")
    lines.append("| Rank | Node | Energy $E_i$ | Mean $|u_i|$ | Peak $|u_i|$ | Active fraction |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|\n")
    for r, s in enumerate(top_stats, start=1):
        lines.append(
            f"| {r} | {s.node} | {s.energy:.3f} | {s.mean_abs_u:.4f} | {s.peak_abs_u:.4f} | {s.active_fraction:.3f} |\n"
        )

    lines.append("\n## Baseline comparison (Controllability Gramian hubs)\n")
    lines.append(
        "We compute a standard linear controllability Gramian ranking (Week 8 baseline) and compare it to the "
        "closed-loop CBF controller from Weeks 5–7.\n\n"
    )
    lines.append(f"- Baseline hub nodes (top {metrics.baseline_k}): **{metrics.baseline_nodes}**\n")
    lines.append(f"- Energy (CBF): **{metrics.energy_cbf:.2f}**\n")
    lines.append(
        f"- Energy (Baseline constant suppression): **{metrics.energy_baseline:.2f}** (mag={metrics.baseline_mag})\n"
    )
    lines.append(f"- Efficiency ratio (Baseline / CBF): **{metrics.efficiency_ratio:.2f}×**\n")
    lines.append(f"- Percent energy saved vs baseline: **{metrics.percent_energy_saved:.1f}%**\n")
    lines.append(
        f"- Overlap between CBF top-K stimulated nodes and baseline hubs: **{metrics.overlap_cbf_topk_with_baseline}**\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Week 11: Quantitative Deliverable Compilation")
    ap.add_argument("--npz", type=str, default="results.npz", help="Path to results.npz (from Week 7/10 pipeline)")
    ap.add_argument("--out", type=str, default="week11_out", help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Seed used in week7 network construction")
    ap.add_argument("--topk", type=int, default=10, help="Top-K stimulated nodes to report")
    ap.add_argument("--baseline_k", type=int, default=3, help="Top-K baseline hub nodes (Gramian)")
    ap.add_argument("--baseline_mag", type=float, default=2.0, help="Constant suppression magnitude (baseline)")
    args = ap.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    _ensure_dir(out_dir)

    if not npz_path.exists():
        raise SystemExit(
            f"Could not find NPZ: {npz_path}\n"
            "Tip: run week7.py to generate results.npz, then run:\n"
            "  python week11.py --npz results.npz --out week11_out"
        )

    data = _load_npz(npz_path)
    x_un = _maybe_get(data, "x_uncontrolled")
    x_cl = _maybe_get(data, "x_controlled")
    node_coords = _maybe_get(data, "node_coords")
    B = _maybe_get(data, "B")
    dt = float(_maybe_get(data, "dt", 0.01))
    threshold = float(_maybe_get(data, "threshold", 1.0))

    if x_un is None or x_cl is None:
        missing = [k for k in ["x_uncontrolled", "x_controlled"] if k not in data]
        raise SystemExit(f"Missing required arrays in NPZ: {missing}")

    x_un = np.asarray(x_un, dtype=float)
    x_cl = np.asarray(x_cl, dtype=float)
    if x_un.ndim != 2 or x_cl.ndim != 2:
        raise SystemExit("Expected x_uncontrolled and x_controlled to have shape (T,N).")

    if x_un.shape != x_cl.shape:
        raise SystemExit(f"x_uncontrolled shape {x_un.shape} does not match x_controlled shape {x_cl.shape}.")

    steps, n_nodes = x_un.shape

    # Reconstruct A and excitability (I_ext)
    A, I_ext = _reconstruct_A_and_excitability(n_nodes=n_nodes, seed=args.seed)

    # Node-level u(t)
    u_node = _infer_u_node(data, shape_txn=(steps, n_nodes))
    if u_node is None:
        raise SystemExit("results.npz is missing u_controlled; cannot compile stimulation targets.")

    u_node = np.asarray(u_node, dtype=float)
    if u_node.shape != (steps, n_nodes):
        raise SystemExit(f"u_node shape {u_node.shape} does not match expected {(steps, n_nodes)}.")

    # Rank control
    stats = _rank_control_nodes(u_node=u_node)
    topk = max(1, int(args.topk))
    top_stats = stats[:topk]

    # Baseline hubs + energy metrics
    baseline_nodes = _gramian_hubs(n_nodes=n_nodes, top_k=int(args.baseline_k), seed=args.seed)
    energy_cbf = float(np.sum(np.linalg.norm(u_node, axis=1) ** 2))
    energy_base = _baseline_energy(steps=steps, baseline_nodes=baseline_nodes, baseline_mag=float(args.baseline_mag))

    ratio = energy_base / energy_cbf if energy_cbf > 1e-12 else float("inf")
    pct_saved = (1.0 - (energy_cbf / energy_base)) * 100.0 if energy_base > 1e-12 else float("nan")
    overlap = len(set([s.node for s in top_stats]).intersection(set(baseline_nodes)))

    metrics = PerformanceMetrics(
        n_nodes=n_nodes,
        n_steps=steps,
        dt=dt,
        threshold=threshold,
        energy_cbf=energy_cbf,
        baseline_k=len(baseline_nodes),
        baseline_mag=float(args.baseline_mag),
        baseline_nodes=baseline_nodes,
        energy_baseline=energy_base,
        efficiency_ratio=float(ratio),
        percent_energy_saved=float(pct_saved),
        overlap_cbf_topk_with_baseline=int(overlap),
    )

    # -----------------------------
    # Write outputs
    # -----------------------------
    dataset_dir = out_dir / "core_dataset"
    _ensure_dir(dataset_dir)

    np.save(dataset_dir / "adjacency_matrix.npy", A)
    np.save(dataset_dir / "actuator_matrix.npy", np.asarray(B) if B is not None else np.eye(n_nodes, dtype=float))
    if node_coords is not None:
        np.save(dataset_dir / "node_coords.npy", np.asarray(node_coords))

    # Excitability CSV
    exc_rows: List[List[object]] = []
    focus_node = 0
    for i in range(n_nodes):
        exc_rows.append([i, _safe_float(I_ext[i]), 1 if i == focus_node else 0])
    _write_csv(dataset_dir / "excitability_weights.csv", ["node", "I_ext", "is_focus"], exc_rows)

    # Pack dataset
    core_npz = dataset_dir / "week11_core_data.npz"
    np.savez(
        core_npz,
        adjacency_matrix=A,
        excitability_weights=I_ext,
        B=(np.asarray(B) if B is not None else np.eye(n_nodes, dtype=float)),
        node_coords=(np.asarray(node_coords) if node_coords is not None else None),
        dt=dt,
        threshold=threshold,
        n_nodes=n_nodes,
        n_steps=steps,
        source_results_npz=str(npz_path),
        seed=int(args.seed),
    )

    # Optimization results
    results_dir = out_dir / "optimization_results"
    _ensure_dir(results_dir)

    rank_rows = [
        [s.node, f"{s.energy:.6f}", f"{s.mean_abs_u:.8f}", f"{s.peak_abs_u:.8f}", f"{s.active_fraction:.6f}"]
        for s in stats
    ]
    _write_csv(
        results_dir / "stimulation_ranking.csv",
        ["node", "energy_sum_u2", "mean_abs_u", "peak_abs_u", "active_fraction"],
        rank_rows,
    )

    with open(results_dir / "stimulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "topk": topk,
                "top_nodes_by_energy": [asdict(s) for s in top_stats],
                "energy_cbf_total": energy_cbf,
                "active_definition": "active := |u| >= 0.05 * p95(|u|)",
            },
            f,
            indent=2,
        )

    # Baseline comparison table
    _write_csv(
        out_dir / "performance_comparison.csv",
        [
            "n_nodes",
            "n_steps",
            "dt",
            "threshold",
            "baseline_k",
            "baseline_mag",
            "baseline_nodes",
            "energy_cbf",
            "energy_baseline",
            "efficiency_ratio_baseline_over_cbf",
            "percent_energy_saved_vs_baseline",
            "overlap_cbf_topk_with_baseline",
        ],
        [[
            metrics.n_nodes,
            metrics.n_steps,
            f"{metrics.dt:.6f}",
            f"{metrics.threshold:.6f}",
            metrics.baseline_k,
            f"{metrics.baseline_mag:.3f}",
            json.dumps(metrics.baseline_nodes),
            f"{metrics.energy_cbf:.6f}",
            f"{metrics.energy_baseline:.6f}",
            f"{metrics.efficiency_ratio:.6f}",
            f"{metrics.percent_energy_saved:.3f}",
            metrics.overlap_cbf_topk_with_baseline,
        ]],
    )

    # Write-up section
    _write_results_section_md(out_dir / "week11_results_section.md", metrics=metrics, top_stats=top_stats, focus_node=focus_node)

    # Console summary
    print("==========================================================")
    print("WEEK 11 DELIVERABLES WRITTEN")
    print("==========================================================")
    print(f"Core dataset:           {dataset_dir}")
    print(f"Optimization results:   {results_dir}")
    print(f"Write-up section:       {out_dir / 'week11_results_section.md'}")
    print("")
    print("Top stimulated nodes (by energy):")
    for r, s in enumerate(top_stats, start=1):
        print(f"  {r:02d}. node={s.node:3d}  E={s.energy:10.3f}  mean|u|={s.mean_abs_u:8.4f}  peak|u|={s.peak_abs_u:8.4f}")
    print("")
    print(f"Baseline hubs (Gramian top-{metrics.baseline_k}): {metrics.baseline_nodes}")
    print(f"Energy CBF:      {metrics.energy_cbf:.2f}")
    print(f"Energy baseline: {metrics.energy_baseline:.2f}")
    print(f"Efficiency ratio (baseline/cbf): {metrics.efficiency_ratio:.2f}x")
    print(f"Percent energy saved vs baseline: {metrics.percent_energy_saved:.1f}%")
    print("==========================================================")


if __name__ == "__main__":
    main()
