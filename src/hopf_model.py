"""
hopf_model.py  --  Phase 1 / Weeks 2-3: Coupled Supercritical-Hopf Dynamics
===========================================================================
Implements the whole-brain Hopf (Stuart-Landau) model of Deco et al. (2017):
each Glasser region is a normal-form supercritical Hopf oscillator, coupled
through the structural connectome A via diffusive coupling.

Per node i, with state (x_i, y_i) = Re/Im of the complex amplitude z_i:

    r2_i = x_i^2 + y_i^2
    dx_i = (a_i - r2_i) x_i - w_i y_i + G * sum_j A_ij (x_j - x_i) + u_i^x
    dy_i = (a_i - r2_i) y_i + w_i x_i + G * sum_j A_ij (y_j - y_i) + u_i^y

The bifurcation parameter a_i is the project's "excitability score":
    a_i < 0  -> stable focus  (damped / healthy / interictal)
    a_i > 0  -> limit cycle   (sustained oscillation / seizure), radius sqrt(a_i)
The supercritical Hopf bifurcation occurs at a_i = 0.

Control enters additively as u = (u^x, u^y); with this additive actuation the
control matrix is the identity, which keeps the optimal-control algebra in
Week 7 clean (u* = -B^T lambda / rho = -lambda / rho).

Stochastic integration (reviewer note, Issue #5)
------------------------------------------------
The plant noise is ADDITIVE (state-independent diffusion: a constant amplitude
`noise` times a Wiener increment). For additive-noise SDEs the Milstein scheme
is *identical* to Euler-Maruyama, because the Milstein correction term is
(1/2) b b' (dW^2 - dt) and the derivative b' of a constant diffusion is zero.
Euler-Maruyama on additive noise therefore already has strong order 1.0 (the
order-1/2 penalty only applies to multiplicative noise). We integrate the
drift with RK4 and append the exact Gaussian Wiener increment (a Lie-Trotter
drift/diffusion split); ``scripts/analysis_step_size.py`` empirically measures
the strong-convergence order of this exact scheme and shows it is converged at
the production step DT, which is the rigorous substitute for "switch to
Milstein" (which would be a no-op here).
"""

from __future__ import annotations
import numpy as np
import warnings

from . import config


class HopfNetwork:
    """Coupled supercritical-Hopf network on a structural connectome.

    Parameters
    ----------
    A : (N, N) ndarray   -- normalised structural adjacency (the connectome)
    a : (N,) ndarray     -- per-node bifurcation parameters ("excitability")
    omega : (N,) ndarray -- per-node intrinsic angular frequencies
    G : float            -- global coupling factor
    """

    def __init__(self, A, a=None, omega=None, G=config.G_COUPLING):
        self.A = np.asarray(A, dtype=float)
        self.N = self.A.shape[0]
        self.deg = self.A.sum(axis=1)                       # weighted degree
        self.G = float(G)
        # Default: healthy resting brain (all a_i slightly negative).
        self.a = (np.full(self.N, config.A_REST) if a is None
                  else np.asarray(a, dtype=float).copy())
        # Default: common intrinsic frequency.
        w = 2 * np.pi * config.OMEGA_HZ
        self.omega = (np.full(self.N, w) if omega is None
                      else np.asarray(omega, dtype=float).copy())

    # ------------------------------------------------------------------ #
    #  Vector field
    # ------------------------------------------------------------------ #
    def f(self, state, u=None):
        """Right-hand side dot(state) = f(state, u) of the coupled network.

        ``state`` is the stacked vector [x (N), y (N)]; ``u`` is the stacked
        control [u^x (N), u^y (N)] or None. Returns d(state)/dt of the same shape.
        """
        x, y = state[: self.N], state[self.N:]
        r2 = x * x + y * y
        # Diffusive coupling  G * (A x - deg * x)  ==  G * sum_j A_ij (x_j - x_i)
        cx = self.G * (self.A @ x - self.deg * x)
        cy = self.G * (self.A @ y - self.deg * y)
        dx = (self.a - r2) * x - self.omega * y + cx
        dy = (self.a - r2) * y + self.omega * x + cy
        if u is not None:
            dx = dx + u[: self.N]
            dy = dy + u[self.N:]
        return np.concatenate([dx, dy])

    def jacobian_T_times(self, state, lam):
        """Compute J(state)^T @ lam analytically (needed for the costate ODE).

        Forming the full 2N x 2N Jacobian is wasteful at N=360; because A is
        symmetric the transpose-times-vector has a clean closed form derived
        from the partial derivatives of f. This is the exact gradient operator
        the Pontryagin costates ride on in Week 7.
        """
        x, y = state[: self.N], state[self.N:]
        r2 = x * x + y * y
        lx, ly = lam[: self.N], lam[self.N:]
        diag_x = self.a - r2 - 2 * x * x - self.G * self.deg
        diag_y = self.a - r2 - 2 * y * y - self.G * self.deg
        cross = -2 * x * y
        # (J^T lam) for the x-block and y-block (A symmetric => A^T = A)
        gx = diag_x * lx + self.G * (self.A @ lx) + (cross + self.omega) * ly
        gy = (cross - self.omega) * lx + diag_y * ly + self.G * (self.A @ ly)
        return np.concatenate([gx, gy])

    # ------------------------------------------------------------------ #
    #  Integrators
    # ------------------------------------------------------------------ #
    def _rk4_drift(self, state, dt, u=None):
        """One classical 4th-order Runge-Kutta step on the DRIFT only."""
        k1 = self.f(state, u)
        k2 = self.f(state + 0.5 * dt * k1, u)
        k3 = self.f(state + 0.5 * dt * k2, u)
        k4 = self.f(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step_with_dW(self, state, dt, dW, u=None, noise=0.0):
        """Deterministic RK4 drift step plus a SUPPLIED Wiener increment ``dW``.

        ``dW`` must be a sample of the Brownian increment over the step (i.e.
        distributed N(0, dt) per component). Exposing the increment lets the
        step-size convergence study (Issue #5) drive coarse and fine
        integrations with the *same* underlying Brownian path. ``rk4_step``
        below draws ``dW`` internally and calls this, so the production
        behaviour is byte-for-byte the original scheme.
        """
        new = self._rk4_drift(state, dt, u)
        if noise > 0.0:
            new = new + noise * np.asarray(dW)
        return new

    def rk4_step(self, state, dt, u=None, noise=0.0, rng=None):
        """One classical 4th-order Runge-Kutta step (optional additive noise).

        Noise is added Euler-Maruyama style after the deterministic RK4 update.
        Because the diffusion is additive (state-independent), this exact scheme
        carries strong order 1.0 -- see the module docstring and Issue #5.
        """
        # BUG FIX B13: Warn if dt is too large for the operator-splitting
        if dt > 0.025:
            warnings.warn("dt > 0.025 degrades strong convergence of the operator-splitting scheme. Proceed with caution.")
            
        if noise > 0.0:
            rng = rng or np.random
            dW = np.sqrt(dt) * rng.standard_normal(state.shape)
        else:
            dW = 0.0
        return self.step_with_dW(state, dt, dW, u=u, noise=noise)

    def simulate(self, state0, n_steps, dt=config.DT, u_func=None,
                 noise=0.0, seed=None, record_control=False):
        """Integrate the network for ``n_steps`` and return the trajectory.

        Parameters
        ----------
        state0 : (2N,) initial condition [x0, y0]
        u_func : callable(t_index, state) -> (2N,) control, or None
        Returns
        -------
        traj : (n_steps+1, 2N) ndarray of states
        (optionally) ctrl : (n_steps, 2N) applied controls
        """
        rng = np.random.default_rng(seed)
        traj = np.empty((n_steps + 1, 2 * self.N))
        traj[0] = state0
        ctrl = np.zeros((n_steps, 2 * self.N)) if record_control else None
        s = state0.copy()
        for t in range(n_steps):
            u = None if u_func is None else u_func(t, s)
            if record_control and u is not None:
                ctrl[t] = u
            s = self.rk4_step(s, dt, u=u, noise=noise, rng=rng)
            traj[t + 1] = s
        return (traj, ctrl) if record_control else traj


# ---------------------------------------------------------------------- #
#  Convenience constructors
# ---------------------------------------------------------------------- #
def healthy_excitability(N):
    """All-negative excitability vector: a uniformly healthy resting network."""
    return np.full(N, config.A_REST)


def focal_excitability(N, focus, a_seizure=config.A_SEIZURE, a_rest=config.A_REST):
    """Excitability vector with a single seizure-prone focus node (a_focus > 0)."""
    a = np.full(N, a_rest)
    a[focus] = a_seizure
    return a


def zone_excitability(N, zone, a_seizure=config.A_SEIZURE, a_rest=config.A_REST):
    """Excitability vector with a whole onset ZONE driven seizure-prone.

    Mirrors what ``_common.seizing_network`` does (the onset zone, not just the
    single focus, is elevated). Provided here so the out-of-distribution and
    amplitude-sweep analyses can build networks at non-default parameters
    without depending on the cached default constants.
    """
    a = np.full(N, a_rest)
    a[np.asarray(zone)] = a_seizure
    return a


def random_initial_state(N, scale=0.1, seed=config.SEED):
    """Small random initial condition near the origin (interictal rest)."""
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal(2 * N)