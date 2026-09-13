"""
lqr_control.py  --  Phase 3 / Week 6: Level-1 Local Stabilisation (LQR)
======================================================================
Level 1 of the two-level corrector. Each Hopf oscillator is linearised about
its interictal (resting) fixed point at the origin, and a Linear-Quadratic
Regulator is designed on that *local linearisation*. The LQR acts node-locally
(no system-wide override): its job is to keep each region from drifting away
from the healthy fixed point so that open loops do not slowly wander.

Linearisation of a single Hopf node about (x, y) = (0, 0):

    d/dt [x, y]^T = M [x, y]^T,   M = [[a, -w], [w, a]]

with additive control matrix B = I_2. The continuous-time algebraic Riccati
equation (CARE) for (M, B, Q, R) yields the local feedback gain K, and the
control law is u_i = -K (state_i - fixed_point_i).
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_continuous_are

from . import config


class LocalLQR:
    """Per-node LQR built on each oscillator's local linearisation.

    Parameters
    ----------
    a, omega : (N,) per-node bifurcation parameters and frequencies
    q_weight : float  -- state penalty (Q = q_weight * I_2)
    r_weight : float  -- control penalty (R = r_weight * I_2)
    """

    def __init__(self, a, omega, q_weight=1.0, r_weight=1.0, real_only=False):
        """``real_only=True`` designs the regulator for a single-input actuator
        that can only drive the real component x of z (B = [1, 0]^T), i.e. the
        physically realisable single-electrode case. (M, B) stays controllable
        for omega != 0 because the rotation couples x and y, so the CARE is
        solvable. Default False == original 2-input behaviour, byte-for-byte."""
        self.a = np.asarray(a, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.N = len(self.a)
        self.real_only = bool(real_only)
        self.K = np.zeros((self.N, 2, 2))            # per-node 2x2 gains
        if self.real_only:
            B = np.array([[1.0], [0.0]])
            R = r_weight * np.eye(1)
        else:
            B = np.eye(2)
            R = r_weight * np.eye(2)
        Q = q_weight * np.eye(2)
        for i in range(self.N):
                    M = np.array([[self.a[i], -self.omega[i]],
                                [self.omega[i], self.a[i]]])
                    P = solve_continuous_are(M, B, Q, R)
                    
                    # BUG FIX B21: Riccati residual check
                    residual = M.T @ P + P @ M - P @ B @ np.linalg.inv(R) @ B.T @ P + Q
                    assert np.max(np.abs(residual)) < 1e-8, f"CARE solver failed to converge at node {i}"
                    
                    Ki = np.linalg.solve(R, B.T @ P)   # K = R^{-1} B^T P  (2x2 or 1x2)
                    if self.real_only:
                        self.K[i, 0, :] = Ki[0]         # only the x-row is actuated; u^y == 0
                    else:
                        self.K[i] = Ki

    def control(self, state, fixed_point=None):
        """Return the stacked Level-1 control u = [u^x, u^y] for a full state.

        Each node's deviation from its fixed point (origin by default) is fed
        through that node's local gain.
        """
        N = self.N
        x, y = state[:N], state[N:]
        if fixed_point is not None:
            x = x - fixed_point[:N]
            y = y - fixed_point[N:]
        ux = -(self.K[:, 0, 0] * x + self.K[:, 0, 1] * y)
        uy = -(self.K[:, 1, 0] * x + self.K[:, 1, 1] * y)
        return np.concatenate([ux, uy])

    def u_func(self, gain_scale=1.0, fixed_point=None):
        """Build a control callable u(t, state) for HopfNetwork.simulate.

        ``gain_scale`` lets the closed loop run Level-1 at a gentle, always-on
        gain (small) and rely on Level-2 for the heavy lifting during events.
        """
        def _u(t, state):
            return gain_scale * self.control(state, fixed_point)
        return _u
