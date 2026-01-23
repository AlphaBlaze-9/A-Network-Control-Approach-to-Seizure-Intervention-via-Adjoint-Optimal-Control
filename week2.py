# ===========================
# WEEK 2: NODE DYNAMICS & EXCITABILITY IMPLEMENTATION
# ===========================

import numpy as np

# --- Excitability Map ---
def create_excitability_map(N, seizure_foci_indices):
    a_vec = np.full(N, -0.5)
    for idx in seizure_foci_indices:
        a_vec[idx] = 0.5
    return a_vec

# --- Nonlinear Node Dynamics (Supercritical Hopf Oscillator) ---
def hopf_network_dynamics(state, t, A, a_vec, G, omega):
    N = len(a_vec)
    x = state[:N]
    y = state[N:]
    r2 = x**2 + y**2
    dxdt = (a_vec - r2) * x - omega * y
    dydt = (a_vec - r2) * y + omega * x
    coupling_x = G * (A @ x - np.sum(A, axis=1) * x)
    coupling_y = G * (A @ y - np.sum(A, axis=1) * y)
    return np.concatenate([dxdt + coupling_x, dydt + coupling_y])
