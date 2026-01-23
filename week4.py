# week4.py

"""
Week 4: Defining the Control Barrier Function (CBF) (Aligned to Week 3 FHN)

Safe Set (from Week 3 seizure threshold on v):
- Week 3 seizure criterion: max(v_i) > v_th  (one-sided threshold)
- Define Safe Set for constrained nodes: v_i <= v_th
- Barrier per node i:
    h_i(x) = v_th - v_i
  so h_i(x) > 0 is safe, h_i(x) -> 0 at the seizure boundary, h_i(x) < 0 is unsafe.

Dynamics (control-affine, consistent with FHN + stimulation current):
- State x = [v; w] in R^(2N), with v,w in R^N.
- Drift f(x) is the FHN network from Week 3:
    dv = v - v^3/3 - w + I_ext + coupling*(A v - deg*v)
    dw = (v + a - b*w)/tau
- Control enters as an additive current into dv:
    dv += B u
  => x_dot = f(x) + g(x)u, where:
    g(x) = [[B],
            [0]]

CBF derivatives and constraint:
- ∇h_i(x) is constant: ∂h_i/∂v_i = -1, all other partials = 0.
- Lie derivatives:
    L_f h(x) = (∂h/∂x) f(x)
    L_g h(x) = (∂h/∂x) g(x)
- CBF inequality (elementwise for all constrained nodes):
    L_f h(x) + L_g h(x) u + α(h(x)) >= 0
  with α(h) = k h (elementwise).
- QP form:
    (L_g h(x)) u >= -L_f h(x) - α(h(x))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple
import numpy as np

Array = np.ndarray


# -----------------------------
# Methods: State Helpers
# -----------------------------
def as_column(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError(f"Expected (n,) or (n,1). Got {x.shape}.")


def split_vw(x: Array, N: int) -> Tuple[Array, Array]:
    x = as_column(x)
    if x.shape[0] != 2 * N:
        raise ValueError(f"State must have length 2N={2*N}. Got {x.shape[0]}.")
    v = x[:N, :]
    w = x[N:, :]
    return v, w


# -----------------------------
# Methods: Barrier (Safe Set)
# -----------------------------
@dataclass(frozen=True)
class VoltageThresholdBarrier:
    """
    Barrier aligned to Week 3 seizure threshold on v:

    For each constrained node i:
        h_i(x) = v_th - v_i

    Safe set: h_i(x) >= 0  <=>  v_i <= v_th

    Gradient derivation:
        h_i depends only on v_i linearly
        ∂h_i/∂v_i = -1
        ∂h_i/∂(all other states) = 0
    """

    n_nodes: int
    v_threshold: float
    constrained_nodes: Optional[Iterable[int]] = None  # indices in [0, n_nodes-1]

    def __post_init__(self) -> None:
        if self.n_nodes <= 0:
            raise ValueError("n_nodes must be positive.")
        if self.v_threshold <= 0:
            raise ValueError("v_threshold must be positive.")
        if self.constrained_nodes is not None:
            for i in self.constrained_nodes:
                if i < 0 or i >= self.n_nodes:
                    raise ValueError("constrained_nodes contains an out-of-range node index.")

    @property
    def state_dim(self) -> int:
        return 2 * self.n_nodes

    def _node_list(self) -> Array:
        if self.constrained_nodes is None:
            return np.arange(self.n_nodes, dtype=int)
        return np.array(list(self.constrained_nodes), dtype=int)

    def h(self, x: Array) -> Array:
        """
        Returns h(x) shape (k,1), one per constrained node.
        """
        x = as_column(x)
        N = self.n_nodes
        v, _ = split_vw(x, N)

        nodes = self._node_list()
        h_vec = (float(self.v_threshold) - v[nodes, 0]).reshape(-1, 1)
        return h_vec

    def grad(self, x: Array) -> Array:
        """
        Returns stacked gradients ∂h/∂x with shape (k, 2N).
        Each row j corresponds to node i=constrained_nodes[j].

        For node i: gradient has -1 at the v_i coordinate, zeros elsewhere.
        """
        _ = x  # gradient does not depend on x for this linear barrier
        nodes = self._node_list()
        k = nodes.size
        n = self.state_dim

        gradH = np.zeros((k, n), dtype=float)
        gradH[np.arange(k), nodes] = -1.0  # v_i positions are in the first N entries
        return gradH


# -----------------------------------------
# Methods: Control-Affine Dynamics (Week 3)
# -----------------------------------------
@dataclass(frozen=True)
class ControlAffineSystem:
    """
    Control-affine dynamics:
        x_dot = f(x) + g(x) u

    Shapes:
        x in R^n       -> (n,1)
        u in R^m       -> (m,1)
        f(x) in R^n    -> (n,1)
        g(x) in R^(n,m)
    """

    f: Callable[[Array], Array]
    g: Callable[[Array], Array]

    def eval(self, x: Array) -> Tuple[Array, Array]:
        x = as_column(x)
        fx = as_column(self.f(x))
        gx = np.asarray(self.g(x), dtype=float)

        n = x.shape[0]
        if fx.shape != (n, 1):
            raise ValueError(f"f(x) must return shape ({n},1). Got {fx.shape}.")
        if gx.ndim != 2 or gx.shape[0] != n:
            raise ValueError(f"g(x) must return shape ({n},m). Got {gx.shape}.")
        return fx, gx


def make_fhn_control_affine(
    adj: Array,
    coupling: float,
    currents: Array,
    a: float,
    b: float,
    tau: float,
    B: Array,
) -> ControlAffineSystem:
    """
    Builds x_dot = f(x) + g(x)u for the Week 3 FHN network with control in dv.

    - adj: (N,N) adjacency matrix
    - currents: (N,) external drive vector I_ext
    - B: (N,m) input matrix mapping u to dv (stimulation current)
    """
    adj = np.asarray(adj, dtype=float)
    N = adj.shape[0]
    if adj.shape != (N, N):
        raise ValueError("adj must be square (N,N).")

    currents = np.asarray(currents, dtype=float).reshape(-1)
    if currents.shape[0] != N:
        raise ValueError("currents must have length N.")

    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[0] != N:
        raise ValueError("B must have shape (N,m).")
    m = B.shape[1]

    deg = np.sum(adj, axis=1).reshape(-1, 1)  # (N,1)

    def f(x: Array) -> Array:
        # x = [v; w]
        v, w = split_vw(x, N)

        # Diffusive coupling: (A v) - deg*v
        neighbor_diffs = (adj @ v) - (deg * v)
        coupling_current = float(coupling) * neighbor_diffs

        # Week 3 FHN drift
        dv = v - (v**3) / 3.0 - w + currents.reshape(-1, 1) + coupling_current
        dw = (v + float(a) - float(b) * w) / float(tau)

        return np.vstack([dv, dw])

    def g(x: Array) -> Array:
        # Control enters only in dv: dv += B u, dw unchanged
        _ = x
        top = B                  # (N,m)
        bottom = np.zeros((N, m)) # (N,m)
        return np.vstack([top, bottom])  # (2N,m)

    return ControlAffineSystem(f=f, g=g)


# -----------------------------
# Methods: CBF (Lie Derivatives)
# -----------------------------
@dataclass(frozen=True)
class CBF:
    """
    Vector CBF for k constrained nodes.

    With h(x) in R^k and x_dot = f(x) + g(x)u:
        d/dt h(x) = (∂h/∂x) f(x) + (∂h/∂x) g(x) u
                 = L_f h(x) + L_g h(x) u

    CBF condition (elementwise):
        L_f h(x) + L_g h(x) u + α(h(x)) >= 0
    """

    barrier: VoltageThresholdBarrier
    system: ControlAffineSystem
    alpha: Callable[[Array], Array]  # (k,1) -> (k,1)

    def lie_derivatives(self, x: Array) -> Tuple[Array, Array, Array, Array]:
        """
        Returns:
          h(x)      : (k,1)
          ∂h/∂x     : (k,2N)
          L_f h(x)  : (k,1)
          L_g h(x)  : (k,m)
        """
        x = as_column(x)

        h_val = self.barrier.h(x)        # (k,1)
        gradH = self.barrier.grad(x)     # (k,2N)
        f_x, g_x = self.system.eval(x)   # (2N,1), (2N,m)

        Lf = gradH @ f_x                 # (k,1)
        Lg = gradH @ g_x                 # (k,m)

        return h_val, gradH, Lf, Lg

    def safety_inequality_qp_form(self, x: Array) -> Tuple[Array, Array]:
        """
        QP-ready inequality:
            (L_g h(x)) u >= -L_f h(x) - α(h(x))

        Returns:
            A = L_g h(x)                 shape (k,m)
            b = -L_f h(x) - α(h(x))      shape (k,1)
        """
        h_val, _, Lf, Lg = self.lie_derivatives(x)
        b = -Lf - self.alpha(h_val)
        return np.asarray(Lg, dtype=float), np.asarray(b, dtype=float)


# -----------------------------
# Methods: Minimal Sanity Test
# -----------------------------
if __name__ == "__main__":
    # Small test matching your Week 3 setup style (but without plotting)
    N = 5
    focus_node = 0

    # Example adjacency (replace with your Week 1/3 adj_matrix)
    rng = np.random.default_rng(42)
    adj = rng.random((N, N))
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0.0)

    # Week 3 parameters
    a = 0.7
    b = 0.8
    tau = 12.5
    coupling_strength = 0.5

    # Week 3 currents
    I_base = 0.35
    I_focus = 1.0
    I_ext = np.full(N, I_base, dtype=float)
    I_ext[focus_node] = I_focus

    # Control input matrix B (Week 6 will refine this)
    # Example: stimulate only two nodes (columns are control channels)
    stim_nodes = [0, 1]
    m = len(stim_nodes)
    B = np.zeros((N, m), dtype=float)
    for j, node in enumerate(stim_nodes):
        B[node, j] = 1.0

    # Week 3 seizure threshold on v
    v_th = 1.0

    # Constrain all nodes except the focus (common seizure-propagation framing)
    constrained_nodes = [i for i in range(N) if i != focus_node]

    barrier = VoltageThresholdBarrier(
        n_nodes=N,
        v_threshold=v_th,
        constrained_nodes=constrained_nodes,
    )

    system = make_fhn_control_affine(
        adj=adj,
        coupling=coupling_strength,
        currents=I_ext,
        a=a,
        b=b,
        tau=tau,
        B=B,
    )

    k_alpha = 2.0
    alpha = lambda h: k_alpha * h

    cbf = CBF(barrier=barrier, system=system, alpha=alpha)

    # Example state: x = [v; w]
    v = np.zeros(N)
    w = np.zeros(N)

    v[focus_node] = 1.2        # focus above threshold (unconstrained)
    v[1] = 0.8                 # constrained node near threshold
    x = np.concatenate([v, w])

    h_val, gradH, Lf, Lg = cbf.lie_derivatives(x)
    A_qp, b_qp = cbf.safety_inequality_qp_form(x)

    print("h(x) =\n", h_val)
    print("∂h/∂x =\n", gradH)
    print("L_f h(x) =\n", Lf)
    print("L_g h(x) =\n", Lg)
    print("QP form: A u >= b")
    print("A =\n", A_qp)
    print("b =\n", b_qp)
