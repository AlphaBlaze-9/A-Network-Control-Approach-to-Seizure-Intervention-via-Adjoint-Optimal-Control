
import os
import numpy as np
import scipy.linalg

from week7 import ClosedLoopSimulation


class Week8Analysis(ClosedLoopSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def linearize_dynamics(self, v_fixed=None):
        """
        Linearizes the FHN network dynamics around a fixed point.
        Returns A_sys (2N x 2N).
        """
        N = self.num_nodes
        if v_fixed is None:
            v_fixed = np.zeros(N, dtype=float)

        degrees = np.sum(self.adj_matrix, axis=1)
        L_diff = self.adj_matrix - np.diag(degrees)

        df_v_dv = np.diag(1.0 - v_fixed**2) + self.coupling_strength * L_diff
        df_v_dw = -np.eye(N)
        df_w_dv = (1.0 / self.tau) * np.eye(N)
        df_w_dw = (-self.b / self.tau) * np.eye(N)

        A_sys = np.block([
            [df_v_dv, df_v_dw],
            [df_w_dv, df_w_dw]
        ])
        return A_sys

    def compute_gramian(self):
        print("Linearizing dynamics...")
        A_sys = self.linearize_dynamics(v_fixed=np.zeros(self.num_nodes, dtype=float))

        B_full = np.vstack([self.B, np.zeros_like(self.B)])
        Q = -B_full @ B_full.T

        print("Solving Lyapunov equation for Gramian (this may take a moment)...")
        evals = np.linalg.eigvals(A_sys)
        max_real = np.max(evals.real)
        if max_real >= 0:
            print(f"⚠️ Warning: System may be unstable (max real eig={max_real:.4f}). Gramian may be unreliable.")

        Wc = scipy.linalg.solve_continuous_lyapunov(A_sys, Q)
        return Wc

    def identify_hubs(self, Wc, top_k=3):
        N = self.num_nodes
        node_scores = np.array([Wc[i, i] + Wc[N+i, N+i] for i in range(N)], dtype=float)
        sorted_indices = np.argsort(node_scores)[::-1]
        top_indices = sorted_indices[:top_k]
        top_score = float(np.mean(node_scores[top_indices]))
        return top_indices, top_score

    def run_baseline_suppression(self, target_nodes, suppression_magnitude=2.0):
        print(f"Running Baseline Simulation: Constant Suppression on nodes {target_nodes}...")

        os.makedirs("Images", exist_ok=True)

        rng = np.random.default_rng(42)
        current_state = rng.random((2, self.num_nodes)) * 0.1
        state_history = np.zeros((self.steps, self.num_nodes), dtype=float)

        from week4 import make_fhn_control_affine
        system_wrapper = make_fhn_control_affine(
            adj=self.adj_matrix,
            coupling=self.coupling_strength,
            currents=self.I_ext,
            a=self.a,
            b=self.b,
            tau=self.tau,
            B=self.B
        )

        u_baseline = np.zeros(self.num_actuators, dtype=float)
        for node in target_nodes:
            if 0 <= int(node) < self.num_actuators:
                u_baseline[int(node)] = -float(suppression_magnitude)

        x = current_state.reshape(-1, 1)

        for i in range(self.steps):
            state_history[i, :] = x[:self.num_nodes].flatten()

            def dynamics_func(y):
                y = y.reshape(-1, 1)
                f, g = system_wrapper.eval(y)
                return (f + g @ u_baseline.reshape(-1, 1)).flatten()

            k1 = dynamics_func(x)
            k2 = dynamics_func(x + 0.5 * self.dt * k1.reshape(-1, 1))
            k3 = dynamics_func(x + 0.5 * self.dt * k2.reshape(-1, 1))
            k4 = dynamics_func(x + self.dt * k3.reshape(-1, 1))

            x_next = x.flatten() + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x = x_next.reshape(-1, 1)

        return state_history, u_baseline


if __name__ == "__main__":
    analysis = Week8Analysis(num_nodes=50)

    print("\n==========================================================")
    print("        WEEK 8: CONTROL EFFICIENCY COMPARISON")
    print("==========================================================")

    Wc = analysis.compute_gramian()
    print("   > Controllability Gramian Computed.")

    top_nodes, hub_score = analysis.identify_hubs(Wc, top_k=3)
    print(f"   > Identified Hub Nodes (Top 3): {top_nodes}")
    print(f"   > Hub Controllability Score: {hub_score:.2f}")

    v_cbf, u_cbf = analysis.run(controlled=True)
    energy_cbf = float(np.sum(np.linalg.norm(u_cbf, axis=1)**2))
    max_v_cbf = float(np.max(v_cbf))
    cbf_success = max_v_cbf < analysis.seizure_threshold + 0.1
    cbf_success_str = "YES" if cbf_success else "NO"

    print("   > Method 1: Control Barrier Function (CBF)")
    print(f"     - Seizure Suppressed: {cbf_success_str}")
    print(f"     - Total Energy (||u||^2): {energy_cbf:.1f} units")

    baseline_mag = 2.0
    v_base, u_vec_base = analysis.run_baseline_suppression(top_nodes, suppression_magnitude=baseline_mag)
    energy_base = float(analysis.steps * np.sum(u_vec_base**2))
    max_v_base = float(np.max(v_base))
    base_success = max_v_base < analysis.seizure_threshold + 0.1
    base_success_str = "YES" if base_success else "NO"

    print("\n   > Method 2: Baseline (Constant Hub Suppression)")
    print(f"     - Seizure Suppressed: {base_success_str}")
    print(f"     - Total Energy (||u||^2): {energy_base:.1f} units")

    ratio = energy_base / energy_cbf if energy_cbf > 0 else float('inf')

    print("\n3. FINAL COMPARISON")
    print("----------------------------------------------------------")
    print("   Metric                  | CBF (Yours)   | Baseline (Standard)")
    print("   ----------------------- | ------------- | -------------------")
    print(f"   Suppression Success     |     {cbf_success_str.ljust(4)}      |       {base_success_str.ljust(4)}")
    print(f"   Energy Cost             |   {energy_cbf:8.1f}    |     {energy_base:8.1f}")
    print("   ----------------------- | ------------- | -------------------")

    if energy_cbf > 0:
        print(f"\n   >>> CONCLUSION: CBF is {ratio:.1f}x more efficient than Baseline.")
    else:
        print("\n   >>> CONCLUSION: CBF used 0 energy (Perfect efficiency).")

    with open("week8_results.txt", "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("        WEEK 8: CONTROL EFFICIENCY COMPARISON\n")
        f.write("==========================================================\n\n")
        f.write(f"Hub Nodes (Top 3): {top_nodes}\n")
        f.write(f"Hub Controllability Score: {hub_score:.2f}\n\n")
        f.write(f"CBF Suppressed: {cbf_success_str}\n")
        f.write(f"CBF Energy: {energy_cbf:.1f}\n\n")
        f.write(f"Baseline Suppressed: {base_success_str}\n")
        f.write(f"Baseline Energy: {energy_base:.1f}\n\n")
        if energy_cbf > 0:
            f.write(f"Conclusion: CBF is {ratio:.1f}x more efficient.\n")
        else:
            f.write("Conclusion: CBF used 0 energy.\n")

    print("\n✅ Saved results to 'week8_results.txt'")
