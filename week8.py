
import numpy as np
import scipy.linalg
from week7 import ClosedLoopSimulation

class Week8Analysis(ClosedLoopSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def linearize_dynamics(self, v_fixed=None, w_fixed=None):
        """
        Linearizes the FHN network dynamics around a fixed point.
        Returns the Jacobian matrix A_sys (2N x 2N).
        State vector x = [v1...vN, w1...wN].
        """
        N = self.num_nodes
        
        # If no fixed point provided, assume the healthy rest state approx
        # For FHN with I=0, fixed point is near (0,0) or determined by nullclines.
        # Let's approximate around v=0, w=0 for simplicity of "structural" controllability,
        # OR use the 'healthy' I_base fixed point.
        # Let's use v=0 (near rest) to evaluate structural controllability.
        if v_fixed is None:
            v_fixed = np.zeros(N)
        
        # Jacobian Blocks:
        # dv/dt = v - v^3/3 - w + C(L v) ...
        # df_v / dv = I - diag(v^2) + C * L_diffusive
        # df_v / dw = -I
        # df_w / dv = (1/tau) * I
        # df_w / dw = (-b/tau) * I
        
        # 1. Laplacian (Diffusive Coupling)
        # Week 3 Coupling: neighbor_diffs = (adj @ v) - (v * degree)
        degrees = np.sum(self.adj_matrix, axis=1)
        # Laplacian L_diff = A - D
        L_diff = self.adj_matrix - np.diag(degrees)
        
        # Block 11: df_v / dv
        # diag(1 - v^2) + C * L_diff
        df_v_dv = np.diag(1.0 - v_fixed**2) + self.coupling_strength * L_diff
        
        # Block 12: df_v / dw
        df_v_dw = -np.eye(N)
        
        # Block 21: df_w / dv
        df_w_dv = (1.0 / self.tau) * np.eye(N)
        
        # Block 22: df_w / dw
        df_w_dw = (-self.b / self.tau) * np.eye(N)
        
        # Assembly A_sys
        # [[df_v_dv, df_v_dw],
        #  [df_w_dv, df_w_dw]]
        A_sys = np.block([
            [df_v_dv, df_v_dw],
            [df_w_dv, df_w_dw]
        ])
        
        return A_sys

    def compute_gramian(self):
        """
        Computes the infinite-horizon Controllability Gramian Wc.
        A_sys Wc + Wc A_sys^T = -BB^T
        """
        print("Linearizing dynamics...")
        A_sys = self.linearize_dynamics(v_fixed=np.zeros(self.num_nodes)) # Linearize at rest
        
        # Construct B matrix for the full state [v; w]
        # B applies to v dynamics only.
        # B_full = [B; 0]
        B_full = np.vstack([self.B, np.zeros_like(self.B)])
        
        # Solve Continuous Lyapunov Equation
        # A X + X A^T = Q
        # Here Q = -BB^T
        Q = -B_full @ B_full.T
        
        print("Solving Lyapunov equation for Gramian (this may take a moment)...")
        # Note: A_sys must be stable (eigenvalues < 0) for infinite horizon.
        # Check stability
        evals = np.linalg.eigvals(A_sys)
        max_real = np.max(evals.real)
        if max_real >= 0:
            print(f"⚠️ Warning: System is unstable (max eig={max_real:.4f}). Gramian might not be positive definite or solver might fail.")
            # Shift A slightly to make it stable for structural analysis?
            # Or just proceed and see.
            # Usually for FHN at rest it IS stable.
        
        Wc = scipy.linalg.solve_continuous_lyapunov(A_sys, Q)
        return Wc

    def identify_hubs(self, Wc, top_k=3):
        """
        Rank nodes based on trace of Gramian (average controllability).
        """
        N = self.num_nodes
        # Wc is 2N x 2N. Diagonal elements reflect energy to move each state.
        # Sum diagonal blocks for each node i: Wc[i,i] (voltage) + Wc[N+i, N+i] (recovery)
        node_scores = []
        for i in range(N):
            score = Wc[i, i] + Wc[N+i, N+i]
            node_scores.append(score)
            
        node_scores = np.array(node_scores)
        sorted_indices = np.argsort(node_scores)[::-1] # Descending
        
        top_indices = sorted_indices[:top_k]
        top_score = np.mean(node_scores[top_indices])
        
        return top_indices, top_score

    def run_baseline_suppression(self, target_nodes, suppression_magnitude=2.0):
        """
        Applies a constant negative current to target nodes.
        """
        print(f"Running Baseline Simulation: Constant Suppression on nodes {target_nodes}...")
        
        # Setup similar to ClosedLoopSimulation.run()
        np.random.seed(42)
        current_state = np.random.rand(2, self.num_nodes) * 0.1
        state_history = np.zeros((self.steps, self.num_nodes))
        
        # System wrapper for dynamics
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
        
        # Constant Control Vector u
        # We need to map target_nodes to actuator indices.
        # Assuming B is Identity for now (from week7), so actuator index = node index.
        u_baseline = np.zeros(self.num_actuators)
        for node in target_nodes:
            # Check if this node has an actuator
            # In Identity B, column j stimulates node j.
            # So u[node] = -magnitude
            u_baseline[node] = -suppression_magnitude

        x = current_state.reshape(-1, 1)

        for i in range(self.steps):
            t = i * self.dt
            # Store v
            state_history[i, :] = x[:self.num_nodes].flatten()
            
            # Step Dynamics with constant u
            f_val, g_val = system_wrapper.eval(x)
            dx = f_val + g_val @ u_baseline.reshape(-1, 1)
            
            # Euler step (simpler, or copy RK4)
            # Let's use simple Euler for baseline to save code space if acceptable, 
            # or better yet, reuse the RK4 logic properly.
            
            def dynamics_func(y, t_): # Consistent RK4
                y = y.reshape(-1, 1)
                f, g = system_wrapper.eval(y)
                return (f + g @ u_baseline.reshape(-1, 1)).flatten()

            # RK4
            k1 = dynamics_func(x, t)
            k2 = dynamics_func(x + 0.5 * self.dt * k1.reshape(-1, 1), t + 0.5 * self.dt)
            k3 = dynamics_func(x + 0.5 * self.dt * k2.reshape(-1, 1), t + 0.5 * self.dt)
            k4 = dynamics_func(x + self.dt * k3.reshape(-1, 1), t + self.dt)
            
            x_next = x.flatten() + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x = x_next.reshape(-1, 1)
            
        return state_history, u_baseline

if __name__ == "__main__":
    analysis = Week8Analysis(num_nodes=50)
    
    # 1. Controllability Analysis
    print("\n==========================================================")
    print("        WEEK 8: CONTROL EFFICIENCY COMPARISON")
    print("==========================================================")
    print("\n1. Controllability Analysis (Standard Control Theory)")
    print("----------------------------------------------------------")
    
    Wc = analysis.compute_gramian()
    print("   > Controllability Gramian Computed.")
    
    top_nodes, hub_score = analysis.identify_hubs(Wc, top_k=3)
    print(f"   > Identified Hub Nodes (Top 3): {top_nodes}")
    print(f"   > Hub Controllability Score: {hub_score:.2f}")

    # 2. Simulation Results
    print("\n2. Simulation Results")
    print("----------------------------------------------------------")
    
    # --- Method 1: CBF (Week 7) ---
    # We run the CBF one
    v_cbf, u_cbf = analysis.run(controlled=True)
    
    # Calculate Energy
    energy_cbf = np.sum(np.linalg.norm(u_cbf, axis=1)**2)
    # Success?
    max_v_cbf = np.max(v_cbf)
    cbf_success = max_v_cbf < analysis.seizure_threshold + 0.1
    cbf_success_str = "YES" if cbf_success else "NO"
    
    print("   > Method 1: Control Barrier Function (CBF)")
    print(f"     - Seizure Suppressed: {cbf_success_str}")
    print(f"     - Total Energy (||u||^2): {energy_cbf:.1f} units")
    
    # --- Method 2: Baseline (Constant Hub Suppression) ---
    # Heuristic magnitude: needs to be enough to suppress.
    # Try 2.0 or 5.0? Let's try matching the max u from CBF?
    # Or just a standard strong suppression.
    baseline_mag = 2.0 
    v_base, u_vec_base = analysis.run_baseline_suppression(top_nodes, suppression_magnitude=baseline_mag)
    
    # Calc Energy for baseline (u is constant vector over all steps)
    # Total Energy = steps * ||u_vec||^2
    energy_base = analysis.steps * np.sum(u_vec_base**2)
    
    # Success?
    max_v_base = np.max(v_base)
    base_success = max_v_base < analysis.seizure_threshold + 0.1
    base_success_str = "YES" if base_success else "NO"

    print("\n   > Method 2: Baseline (Constant Hub Suppression)")
    print(f"     - Seizure Suppressed: {base_success_str}")
    print(f"     - Total Energy (||u||^2): {energy_base:.1f} units")

    # 3. Final Comparison
    print("\n3. FINAL COMPARISON")
    print("----------------------------------------------------------")
    print("   Metric                  | CBF (Yours)   | Baseline (Standard)")
    print("   ----------------------- | ------------- | -------------------")
    print(f"   Suppression Success     |     {cbf_success_str.ljust(4)}      |       {base_success_str.ljust(4)}")
    print(f"   Energy Cost             |   {energy_cbf:8.1f}    |     {energy_base:8.1f}")
    print("   ----------------------- | ------------- | -------------------")
    
    ratio = energy_base / energy_cbf if energy_cbf > 0 else float('inf')
    
    # Print to console
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
        
    # Save to file
    with open("week8_results.txt", "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("        WEEK 8: CONTROL EFFICIENCY COMPARISON\n")
        f.write("==========================================================\n\n")
        
        f.write("1. Controllability Analysis (Standard Control Theory)\n")
        f.write("----------------------------------------------------------\n")
        f.write(f"   > Identified Hub Nodes (Top 3): {top_nodes}\n")
        f.write(f"   > Hub Controllability Score: {hub_score:.2f}\n\n")
        
        f.write("2. Simulation Results\n")
        f.write("----------------------------------------------------------\n")
        f.write("   > Method 1: Control Barrier Function (CBF)\n")
        f.write(f"     - Seizure Suppressed: {cbf_success_str}\n")
        f.write(f"     - Total Energy (||u||^2): {energy_cbf:.1f} units\n\n")
        
        f.write("   > Method 2: Baseline (Constant Hub Suppression)\n")
        f.write(f"     - Seizure Suppressed: {base_success_str}\n")
        f.write(f"     - Total Energy (||u||^2): {energy_base:.1f} units\n\n")
        
        f.write("3. FINAL COMPARISON\n")
        f.write("----------------------------------------------------------\n")
        f.write("   Metric                  | CBF (Yours)   | Baseline (Standard)\n")
        f.write("   ----------------------- | ------------- | -------------------\n")
        f.write(f"   Suppression Success     |     {cbf_success_str.ljust(4)}      |       {base_success_str.ljust(4)}\n")
        f.write(f"   Energy Cost             |   {energy_cbf:8.1f}    |     {energy_base:8.1f}\n")
        f.write("   ----------------------- | ------------- | -------------------\n\n")
        
        if energy_cbf > 0:
            f.write(f"   >>> CONCLUSION: CBF is {ratio:.1f}x more efficient than Baseline.\n")
        else:
            f.write(f"   >>> CONCLUSION: CBF used 0 energy.\n")
            
    print("\n✅ Saved results to 'week8_results.txt'")

