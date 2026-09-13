"""
metrics.py  --  Seizure-state quantification and network synchrony
==================================================================
Turns raw (x, y) Hopf trajectories into the scalar quantities the rest of the
pipeline reasons about: per-node oscillation amplitude, the Kuramoto order
parameter (global phase synchrony), and a binary "ictal" classification based
on a sustained-amplitude threshold (the Week-3 numerical seizure definition).
"""

from __future__ import annotations
import numpy as np


def amplitude(traj, N):
    """Per-node oscillation amplitude r_i(t) = sqrt(x_i^2 + y_i^2).

    Returns an (T, N) array. Amplitude is the natural seizure read-out for a
    Hopf model: it is ~0 at the stable focus (interictal) and ~sqrt(a) on the
    limit cycle (ictal).
    """
    x, y = traj[:, :N], traj[:, N:]
    return np.sqrt(x * x + y * y)


def phase(traj, N):
    """Per-node phase theta_i(t) = atan2(y_i, x_i), in radians."""
    return np.arctan2(traj[:, N:], traj[:, :N])


def kuramoto_order(traj, N, nodes=None):
    """Global Kuramoto order parameter R(t) in [0, 1].

    R = | (1/M) sum_k exp(i theta_k) |. R ~ 0 means incoherent (healthy
    desynchronised rest); R -> 1 means hypersynchronisation, the hallmark of a
    generalised seizure. ``nodes`` optionally restricts to a sub-set.
    """
    th = phase(traj, N)
    if nodes is not None:
        th = th[:, nodes]
    return np.abs(np.mean(np.exp(1j * th), axis=1))


def sustained_ictal_mask(traj, N, threshold, min_fraction=0.6, tail=None):
    """Classify each node as ictal if its amplitude stays above ``threshold``.

    A node counts as seizing if its amplitude exceeds ``threshold`` for at least
    ``min_fraction`` of the analysed window (the ``tail`` last samples, or the
    whole trajectory). Requiring *sustained* supra-threshold amplitude rejects
    transient noise spikes, which is what makes this a defensible seizure state
    rather than an instantaneous trip.

    Returns a boolean (N,) array.
    """
    amp = amplitude(traj, N)
    if tail is not None:
        amp = amp[-tail:]
    frac_above = (amp > threshold).mean(axis=0)
    return frac_above >= min_fraction


def seizure_load(traj, N, threshold, tail=None):
    """Fraction of nodes that are ictal (a scalar seizure-severity index)."""
    return sustained_ictal_mask(traj, N, threshold, tail=tail).mean()
