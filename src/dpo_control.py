"""
dpo_control.py  --  Phase 3 / Week 7: Optimal-Control Hamiltonian & DPO
======================================================================
Level 2 of the corrector: a connectome-wide optimal controller built on the
*Pontryagin optimal-control Hamiltonian* (NOT a conservative physical
Hamiltonian -- the brain dynamics stay the coupled-Hopf ODEs f(x, u), which we
need so the limit cycles still exist).

    H(x, u, lambda) = L(x, u) + lambda^T f(x, u)

with running cost

    L(x, u) = 0.5 [ q * sum_i (x_i^2 + y_i^2)  +  rho * ||u||^2 ]
                    (  proximity-to-bifurcation  )   ( control energy )

The amplitude term q*sum r_i^2 penalises proximity to the seizure boundary
(under-treatment / "hitting the crowd"); the rho*||u||^2 term penalises control
energy (over-treatment / "flipping the car"). The costates lambda are the
"conjugate co-pilot": they carry the exact gradient of the cost back through
the dynamics.

Differentiable Policy Optimization (DPO) here means: the control schedule u(t)
is the policy, and we descend the cost using the *exact adjoint gradient*

    dJ/du_t = dt ( rho * u_t + B^T p_{t+1} )

obtained from the discrete costate recursion (predictor forward, corrector
backward, adjusted in lockstep). The implementation uses an explicit-Euler
discretisation so the discrete adjoint is exact -- the cost is guaranteed to
decrease for a small enough step, which is what the convergence figure shows.

[Reviewer #3 -- actuator saturation] ``optimize`` now accepts ``u_max``: an
optional hard bound that projects the control schedule onto [-u_max, u_max]
after every gradient step, so the returned schedule respects a finite actuator
limit instead of being unconstrained. ``u_max=None`` reproduces the original
unconstrained behaviour exactly.
"""

from __future__ import annotations
import numpy as np

from . import config


class DPOController:
    """Differentiable Policy Optimization controller (Pontryagin / adjoint).

    Parameters
    ----------
    net : HopfNetwork    -- the plant whose f, J^T are reused for the adjoint
    q   : float          -- amplitude (proximity-to-bifurcation) penalty
    rho : float          -- control-energy penalty
    control_nodes : array or None -- node indices allowed to actuate (default all)
    """

    def __init__(self, net, q=1.0, rho=0.05, control_nodes=None, real_only=False):
        """``real_only=True`` restricts the actuation matrix B to the real
        component x of each actuated node (single-electrode case): the y-half
        of the mask is zeroed, so the adjoint gradient and the returned
        schedule carry no imaginary-component control. Default False == original."""
        self.net = net
        self.N = net.N
        self.q = float(q)
        self.rho = float(rho)
        self.real_only = bool(real_only)
        if control_nodes is None:
            self.mask = np.ones(2 * self.N)
        else:
            m = np.zeros(self.N)
            m[np.asarray(control_nodes)] = 1.0
            self.mask = np.concatenate([m, m])        # actuate x and y of those nodes
        if self.real_only:
            self.mask[self.N:] = 0.0                  # B restricted to Re(z)

    # ------------------------------------------------------------------ #
    #  Forward predictor and backward corrector (the predictor-corrector pair)
    # ------------------------------------------------------------------ #
    def _forward(self, x0, U, dt):
        """Explicit-Euler forward roll-out of the state under control schedule U.

        Returns the state trajectory X of shape (T+1, 2N).
        """
        T = U.shape[0]
        X = np.empty((T + 1, 2 * self.N))
        X[0] = x0
        for t in range(T):
            X[t + 1] = X[t] + dt * self.net.f(X[t], U[t] * self.mask)
        return X

    def _backward(self, X, dt):
        """Backward costate recursion (the exact discrete adjoint of Euler).

            p_t = p_{t+1} + dt [ (df/dx)^T p_{t+1} + q * x_t ],   p_T = 0

        Reuses the network's analytic J^T-times-vector operator. Returns P of
        shape (T+1, 2N); P[t+1] is the costate paired with control U[t].
        """
        T = X.shape[0] - 1
        P = np.zeros((T + 1, 2 * self.N))
        for t in range(T - 1, -1, -1):
            JT_p = self.net.jacobian_T_times(X[t], P[t + 1])
            dLdx = self.q * X[t]                       # d/dx of 0.5 q ||state||^2
            P[t] = P[t + 1] + dt * (JT_p + dLdx)
        return P

    def cost(self, X, U, dt):
        """Discrete total cost J = sum_t dt * L(x_t, u_t)."""
        amp = self.q * np.sum(X[:-1] ** 2, axis=1)
        ctrl = self.rho * np.sum((U * self.mask) ** 2, axis=1)
        return float(0.5 * dt * np.sum(amp + ctrl))

    def grad(self, X, U, P, dt):
        """Exact adjoint gradient dJ/dU_t = dt (rho U_t + B^T p_{t+1})."""
        return dt * (self.rho * U + P[1:]) * self.mask

    # ------------------------------------------------------------------ #
    #  Differentiable Policy Optimization
    # ------------------------------------------------------------------ #
    def optimize(self, x0, horizon, dt=config.DT, n_iters=150, lr=0.05,
                 U_init=None, verbose=False, clip=None, u_max=None):
        """Optimise the open-loop control schedule over ``horizon`` steps.

        Policy = the control schedule U(t). We descend the cost with the exact
        adjoint (costate) gradient, using Adam so the step adapts to the costate
        magnitude (which can be large for long horizons over an unstable focus).
        Returns (U, history) where ``history`` is the per-iteration cost.

        Parameters
        ----------
        clip : float or None
            Optional gradient-norm clip (numerical stabiliser; unchanged).
        u_max : float or None
            Optional actuator-saturation bound. When set, the control schedule
            is projected onto [-u_max, u_max] after every Adam step, so the
            optimiser returns a schedule that respects a finite stimulation
            limit (Reviewer #3). ``None`` == unconstrained (original behaviour).
        """
        U = (np.zeros((horizon, 2 * self.N)) if U_init is None
             else U_init.copy())
        # Respect the saturation bound on any provided warm-start schedule too.
        if u_max is not None:
            U = np.clip(U, -float(u_max), float(u_max))
        # Adam moments
        m = np.zeros_like(U)
        v = np.zeros_like(U)
        b1, b2, eps = 0.9, 0.999, 1e-8
        history = []
        for it in range(1, n_iters + 1):
            X = self._forward(x0, U, dt)               # predictor
            P = self._backward(X, dt)                  # corrector (costates)
            g = self.grad(X, U, P, dt)
            if clip is not None:
                gn = np.linalg.norm(g)
                if gn > clip:
                    g = g * (clip / gn)
            history.append(self.cost(X, U, dt))
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            mhat = m / (1 - b1 ** it)
            vhat = v / (1 - b2 ** it)
            U = U - lr * mhat / (np.sqrt(vhat) + eps)
            if u_max is not None:                      # actuator saturation
                U = np.clip(U, -float(u_max), float(u_max))
            if verbose and it % 25 == 0:
                print(f"   DPO iter {it:3d}  cost={history[-1]:.5f}  "
                      f"|grad|={np.linalg.norm(g):.3e}")
        return U, np.array(history)

    def u_func_from_schedule(self, U, dt=config.DT):
        """Wrap a precomputed schedule U as a u(t, state) callable for the plant.

        Used by the closed loop: the optimal schedule is computed once per
        receding-horizon window and replayed by the (RK4) plant integrator.
        """
        T = U.shape[0]

        def _u(t, state):
            return (U[t] * self.mask) if t < T else np.zeros(2 * self.N)
        return _u
