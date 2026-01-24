import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cvxpy as cp
    _HAVE_CVXPY = True
except Exception:
    _HAVE_CVXPY = False


@dataclass(frozen=True)
class QPResult:
    u: np.ndarray
    solve_time_ms: float
    status: str


def solve_min_norm_qp(
    A: np.ndarray,
    b: np.ndarray,
    u_max: Optional[float] = None,
    solver: str = "OSQP",
    eps_abs: float = 1e-3,
    eps_rel: float = 1e-3,
) -> QPResult:
    """
    Week 5 deliverable (generic): solve

        minimize    0.5 * ||u||^2
        subject to  A u >= b

    where:
      - A: (k,m)
      - b: (k,) or (k,1)
      - u: (m,)

    Optional:
      - u_max: if provided, adds box constraint -u_max <= u <= u_max
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D (k,m). Got {A.shape}.")
    k, m = A.shape
    if b.shape[0] != k:
        raise ValueError(f"b must have length k={k}. Got {b.shape}.")

    if not _HAVE_CVXPY:
        # Safe fallback: return zeros if cvxpy isn't installed (keeps pipeline runnable).
        return QPResult(u=np.zeros(m, dtype=float), solve_time_ms=0.0, status="NO_CVXPY_FALLBACK")

    u = cp.Variable(m)
    objective = cp.Minimize(0.5 * cp.sum_squares(u))
    constraints = [A @ u >= b]
    if u_max is not None:
        constraints += [u <= float(u_max), u >= -float(u_max)]

    prob = cp.Problem(objective, constraints)

    start = time.time()
    try:
        prob.solve(solver=getattr(cp, solver), eps_abs=eps_abs, eps_rel=eps_rel)
    except Exception:
        # Try a more general solver if available
        try:
            prob.solve()
        except Exception:
            return QPResult(u=np.zeros(m, dtype=float), solve_time_ms=(time.time() - start) * 1000.0, status="SOLVE_FAILED")

    solve_time_ms = (time.time() - start) * 1000.0
    u_val = u.value if u.value is not None else np.zeros(m, dtype=float)
    return QPResult(u=np.asarray(u_val, dtype=float).reshape(-1), solve_time_ms=solve_time_ms, status=str(prob.status))


# Convenience wrapper matching your Week 4 CBF API
def solve_cbf_qp(A_qp: np.ndarray, b_qp: np.ndarray, **kwargs) -> Tuple[np.ndarray, float, str]:
    res = solve_min_norm_qp(A_qp, b_qp, **kwargs)
    return res.u, res.solve_time_ms, res.status


if __name__ == "__main__":
    # Quick sanity test
    A = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    b = np.array([0.5, -0.2])
    u, t_ms, status = solve_cbf_qp(A, b, u_max=2.0)
    print("u =", u)
    print("time_ms =", t_ms)
    print("status =", status)
