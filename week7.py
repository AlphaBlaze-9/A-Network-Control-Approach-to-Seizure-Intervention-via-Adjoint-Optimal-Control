import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as optimize

from week4 import VoltageThresholdBarrier, make_fhn_control_affine, CBF
from week6 import ActuatorConfig


class ClosedLoopSimulation:
    def __init__(self, num_nodes=50, T=400.0, dt=0.05, alpha=1.0, seed=42):
        self.num_nodes = int(num_nodes)
        self.T = float(T)
        self.dt = float(dt)
        self.steps = int(self.T / self.dt)
        self.time_points = np.arange(self.steps) * self.dt
        self.alpha_decay = float(alpha)
        self.seed = int(seed)

        # --- 1. Network Setup (Matching Week 3) ---
        self.G = nx.watts_strogatz_graph(n=self.num_nodes, k=6, p=0.1, seed=self.seed)
        self.adj_matrix = nx.to_numpy_array(self.G)

        # Focus Node (Seizure Onset Zone)
        self.focus_node_index = 0

        # --- 2. FHN Parameters ---
        # Adjusted to ensure sustained oscillations (Limit Cycle) for the seizure
        self.a = 0.7
        self.b = 0.8
        self.tau = 12.5
        self.coupling_strength = 0.5

        # External Drive
        # FIX: Increased I_focus to 2.0 to ensure the seizure doesn't die out spontaneously.
        self.I_base = 0.35
        self.I_focus = 2.0
        self.I_ext = np.full(self.num_nodes, self.I_base, dtype=float)
        self.I_ext[self.focus_node_index] = self.I_focus

        # --- 3. Actuator Configuration (Week 6) ---
        self.actuator_config = ActuatorConfig(num_nodes=self.num_nodes)
        self.B = self.actuator_config.get_B_matrix(mode='identity')
        self.num_actuators = self.B.shape[1]

        # --- 4. CBF / Safety Setup ---
        self.seizure_threshold = 1.0  # v_th

    def solve_qp(self, A_cbf, b_cbf):
        """
        Solve: min 0.5||u||^2  s.t. A u >= b
        Uses scipy SLSQP.
        """
        A_cbf = np.asarray(A_cbf, dtype=float)
        b_cbf = np.asarray(b_cbf, dtype=float).reshape(-1)

        def objective(u):
            return 0.5 * np.sum(u**2)

        def constraint_func(u):
            return A_cbf @ u - b_cbf

        # Start guess at 0
        u0 = np.zeros(self.num_actuators, dtype=float)
        cons = {'type': 'ineq', 'fun': constraint_func}

        # Optimization to find minimal control u
        res = optimize.minimize(objective, u0, method='SLSQP', constraints=cons, tol=1e-3)

        if res.success and res.x is not None:
            return res.x
        return np.zeros(self.num_actuators, dtype=float)

    def run(self, controlled=True):
        print(f"Starting Simulation (Controlled={controlled})...")

        rng = np.random.default_rng(self.seed)
        # Start near rest to let the seizure develop naturally
        current_state = rng.random((2, self.num_nodes)) * 0.1

        state_history = np.zeros((self.steps, self.num_nodes), dtype=float)
        control_history = np.zeros((self.steps, self.num_actuators), dtype=float)

        barrier = VoltageThresholdBarrier(
            n_nodes=self.num_nodes,
            v_threshold=self.seizure_threshold,
            constrained_nodes=None  # constrain all nodes
        )

        system_wrapper = make_fhn_control_affine(
            adj=self.adj_matrix,
            coupling=self.coupling_strength,
            currents=self.I_ext,
            a=self.a,
            b=self.b,
            tau=self.tau,
            B=self.B
        )

        alpha_func = lambda h: self.alpha_decay * h
        cbf = CBF(barrier=barrier, system=system_wrapper, alpha=alpha_func)

        x = current_state.reshape(-1, 1)  # (2N, 1)

        for i in range(self.steps):
            t = i * self.dt

            # Store v
            v = x[:self.num_nodes].flatten()
            state_history[i, :] = v

            # Compute control
            u_opt = np.zeros(self.num_actuators, dtype=float)
            if controlled:
                A_qp, b_qp = cbf.safety_inequality_qp_form(x)
                u_opt = self.solve_qp(A_qp, b_qp)

            control_history[i, :] = u_opt

            # Dynamics for RK4 with constant u during dt
            def dynamics_func(y):
                y = y.reshape(-1, 1)
                f_val, g_val = system_wrapper.eval(y)
                dx = f_val + g_val @ u_opt.reshape(-1, 1)
                return dx.flatten()

            k1 = dynamics_func(x)
            k2 = dynamics_func(x + 0.5 * self.dt * k1.reshape(-1, 1))
            k3 = dynamics_func(x + 0.5 * self.dt * k2.reshape(-1, 1))
            k4 = dynamics_func(x + self.dt * k3.reshape(-1, 1))

            x_next = x.flatten() + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x = x_next.reshape(-1, 1)

        return state_history, control_history

    def make_node_coords(self):
        """
        Provides an (N,3) coordinate set for Week 10 visualization.
        """
        pos = nx.spring_layout(self.G, dim=3, seed=self.seed)
        coords = np.array([pos[i] for i in range(self.num_nodes)], dtype=float)
        # Scale up a bit to spread points
        coords *= 60.0
        return coords


if __name__ == "__main__":
    os.makedirs("Images", exist_ok=True)

    # 1. Setup
    # FIX: Increased alpha to 20.0 to handle the stronger seizure drive
    sim = ClosedLoopSimulation(num_nodes=50, alpha=20.0)

    # 2. Run Uncontrolled (Baseline)
    print(">>> Running Baseline Simulation (Uncontrolled)...")
    v_uncontrolled, _ = sim.run(controlled=False)

    # 3. Run Controlled (CBF)
    print(">>> Running CBF Simulation (Controlled)...")
    v_controlled, u_controlled = sim.run(controlled=True)

    # --- Visualization ---
    time_points = sim.time_points
    focus_idx = sim.focus_node_index

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax = axes[0]
    ax.plot(time_points, v_uncontrolled[:, focus_idx], 'r--', label='Uncontrolled (Focus)', alpha=0.7)
    ax.plot(time_points, v_controlled[:, focus_idx], 'g-', label='Controlled (Focus)', linewidth=2)
    ax.axhline(sim.seizure_threshold, color='k', linestyle=':', label='Safety Threshold')
    ax.set_ylabel("Voltage (v)")
    ax.set_title("Focus Node Dynamics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    max_v_un = np.max(v_uncontrolled, axis=1)
    max_v_con = np.max(v_controlled, axis=1)
    ax.plot(time_points, max_v_un, 'r--', label='Max Voltage (Uncontrolled)', alpha=0.5)
    ax.plot(time_points, max_v_con, 'g-', label='Max Voltage (Controlled)')
    ax.axhline(sim.seizure_threshold, color='k', linestyle=':', label='Safety Threshold')
    ax.set_ylabel("Max Voltage (v)")
    ax.set_title("Network Maximum Activity (Global Safety)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(time_points, u_controlled[:, focus_idx], 'b-', label='Input to Focus Node')
    total_energy_t = np.sum(u_controlled**2, axis=1)
    ax.plot(time_points, total_energy_t, 'k:', alpha=0.35, label='Total Input Energy (per-step)')
    ax.set_ylabel("Control Input (u)")
    ax.set_xlabel("Time (ms)")
    ax.set_title("Control Inputs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    image_path = "Images/week7_comparison.png"
    plt.savefig(image_path, dpi=200)
    print(f"Saved comparison plot to '{image_path}'")
    plt.show()

    # --- Numerical Verification ---
    max_un = np.max(v_uncontrolled)
    max_con = np.max(v_controlled)
    print(f"Max Voltage (Uncontrolled): {max_un:.4f}")
    print(f"Max Voltage (Controlled):   {max_con:.4f}")
    print(f"Seizure Threshold:          {sim.seizure_threshold}")

    if max_con < sim.seizure_threshold + 0.1:
        print("✅ SUCCESS: Seizure suppressed.")
    else:
        print("❌ FAILURE: Seizure NOT fully suppressed.")

    # --- Week 10 handoff: save NPZ for visualization ---
    node_coords = sim.make_node_coords()
    stim_nodes = np.where(np.linalg.norm(sim.B, axis=1) > 1e-12)[0]

    np.savez(
        "results.npz",
        x_uncontrolled=v_uncontrolled,   # (T,N)
        x_controlled=v_controlled,       # (T,N)
        u_controlled=u_controlled,       # (T,N)
        node_coords=node_coords,         # (N,3)
        B=sim.B,                         # (N,N)
        dt=sim.dt,
        threshold=sim.seizure_threshold,
        stim_nodes=stim_nodes,
    )
    print("✅ Saved Week 10 input file: results.npz")