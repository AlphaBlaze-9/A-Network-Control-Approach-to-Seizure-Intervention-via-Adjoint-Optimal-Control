"""
_common.py  --  Shared helpers for the Week 1-12 driver scripts
================================================================
Small conveniences imported by every ``weekNN_*.py`` script so the individual
scripts stay focused on the *analysis* for that week rather than boilerplate.

Nothing here changes the science: it only assembles the already-verified
building blocks in ``src/`` (connectome, Hopf model, controllers, detector) in
the standard way the project uses them, and defines the seizure-onset zone
consistently across all figures.
"""

from __future__ import annotations
import os
import sys
import numpy as np

# Make ``src`` importable when a script is run directly (python scripts/weekNN.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C                                    # noqa: E402
from src.connectome import get_network                         # noqa: E402
from src.hopf_model import (HopfNetwork, healthy_excitability,  # noqa: E402
                            focal_excitability, random_initial_state)


# --------------------------------------------------------------------------- #
#  Network / onset-zone assembly
# --------------------------------------------------------------------------- #
def load_network(scale: str | None = None):
    """Return (A, labels, focus_idx) for the active (or requested) scale.

    Thin wrapper over ``connectome.get_network`` so every script picks up the
    SCALE environment variable identically.
    """
    return get_network(scale)


def onset_zone(A, labels, focus, zone_size: int = 4):
    """Return the list of node indices that form the seizure-onset zone.

    Temporal-lobe epilepsy onsets in a mesial-temporal *zone*, not a single
    point. At full Glasser scale we use the anatomically named mesial-temporal
    regions (entorhinal cortex, presubiculum, hippocampus, perirhinal cortex)
    from ``config.MESIAL_TEMPORAL``. At reduced scale (where those original
    indices no longer exist) we fall back to the focus plus its most strongly
    connected neighbours -- i.e. the regions the connectome would actually
    recruit first. Either way the zone is a small, contiguous limbic patch.
    """
    N = A.shape[0]
    anat = [i for i in C.MESIAL_TEMPORAL if i < N]
    if len(anat) >= zone_size and focus in anat:
        return sorted(anat[:zone_size])
    # connectivity-based fallback (reduced scale)
    strength = A[focus].copy()
    strength[focus] = -np.inf
    neigh = list(np.argsort(strength)[::-1][: zone_size - 1])
    return sorted([focus] + [int(j) for j in neigh])


def seizing_network(A, focus_zone, omega_hz: float = C.OMEGA_HZ):
    """Build a HopfNetwork whose onset zone is seizure-prone (a>0), rest healthy.

    This is the standard 'patient' network used from Week 3 onward: a single
    limbic onset zone is pushed past the Hopf bifurcation while every other
    region sits at the healthy resting value, so any activity elsewhere must
    have arrived through the structural connectome.
    """
    N = A.shape[0]
    a = healthy_excitability(N)
    a[np.asarray(focus_zone)] = C.A_SEIZURE
    omega = np.full(N, 2 * np.pi * omega_hz)
    return HopfNetwork(A, a=a, omega=omega)


def healthy_network(A, omega_hz: float = C.OMEGA_HZ):
    """Build an all-healthy resting HopfNetwork (every a_i < 0)."""
    N = A.shape[0]
    omega = np.full(N, 2 * np.pi * omega_hz)
    return HopfNetwork(A, a=healthy_excitability(N), omega=omega)


# --------------------------------------------------------------------------- #
#  Closed-loop protocol shared by every representative-trial script AND the
#  Monte-Carlo scripts, so a "representative trial" is literally trial 0 of
#  the Monte-Carlo ensemble (same seed, same plant noise, same actuator bound).
# --------------------------------------------------------------------------- #
REP_SEED = C.MC_SEED0            # representative trial == Monte-Carlo trial 0
CL_N_STEPS = 1600                # closed-loop / propagation run length (80 s)


def closed_loop_kwargs(**overrides):
    """Standard keyword arguments for ``src.closed_loop.run_closed_loop``.

    These are exactly the settings used by monte_carlo_validation.py and
    monte_carlo_baselines.py. Pass overrides (e.g. ``control_nodes=...``,
    ``detector_bias=...``) to vary one factor at a time.
    """
    from src.detector import CSDDetector
    kw = dict(detector=CSDDetector(window=100), risk_threshold=0.5,
              dpo_horizon=200, dpo_iters=50, dpo_rho=0.01, dpo_q=1.0,
              l1_gain=0.25, refractory=20, decision_stride=20,
              plant_noise=C.NOISE_BETA, dpo_u_max=C.U_MAX, seed=REP_SEED)
    kw.update(overrides)
    return kw


# --------------------------------------------------------------------------- #
#  Misc
# --------------------------------------------------------------------------- #
def time_axis(n_steps, dt: float = C.DT):
    """Return a (n_steps+1,) physical-time vector for a trajectory."""
    return np.arange(n_steps + 1) * dt


def banner(week: int, title: str):
    """Print a consistent header so console logs are easy to scan."""
    print("=" * 74)
    print(f"  WEEK {week:02d}  |  {title}")
    print(f"  {C.scale_label()}")
    print("=" * 74)


def savetxt_table(path, header, rows, fmt="%s"):
    """Write a small CSV results table (used by the benchmarking weeks)."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"   wrote {path}")
