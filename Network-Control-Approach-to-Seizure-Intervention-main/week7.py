
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as optimize
from week4 import VoltageThresholdBarrier, ControlAffineSystem, make_fhn_control_affine, CBF
from week6 import ActuatorConfig

class ClosedLoopSimulation:
    def __init__(self, num_nodes=50, T=400.0, dt=0.05, alpha=1.0):
        self.num_nodes = num_nodes
        self.T = T
        self.dt = dt
        self.steps = int(T / dt)
        self.time_points = np.linspace(0, T, self.steps)
        self.alpha_decay = alpha

        # --- 1. Network Setup (Matching Week 3) ---
        # Watts-Strogatz small-world graph
        self.G = nx.watts_strogatz_graph(n=num_nodes, k=6, p=0.1, seed=42)
        self.adj_matrix = nx.to_numpy_array(self.G)
        
        # Focus Node (Seizure Onset Zone)
        self.focus_node_index = 0
        
        # --- 2. FHN Parameters (Matching Week 3) ---
        self.a = 0.7
        self.b = 0.8
        self.tau = 12.5
        self.coupling_strength = 0.5
        
        # External Drive
        self.I_base = 0.35
        self.I_focus = 1.0
        self.I_ext = np.full(num_nodes, self.I_base)
        self.I_ext[self.focus_node_index] = self.I_focus
        
        # --- 3. Actuator Configuration ---
        # We'll use a simple "Identity" actuation for now (can stimulate any node)
        # or a specific set. Let's assume we can stimulate all nodes for maximum control authority first.
        # Week 6 ActuatorConfig can be reused.
        self.actuator_config = ActuatorConfig(num_nodes=num_nodes)
        self.B = self.actuator_config.get_B_matrix(mode='identity')
        self.num_actuators = self.B.shape[1]

        # --- 4. CBF / Safety Setup ---
        self.seizure_threshold = 1.0 # v_th
        # Constrain all nodes? Or just healthy ones? 
        # Typically we want to prevent high amplitude everywhere or specifically in healthy tissue.
        # Let's constrain ALL nodes for now to strictly suppress seizure.
        pass
        
    def solve_qp(self, A_cbf, b_cbf):
        """
        Solve Min ||u||^2 s.t. A_cbf @ u >= b_cbf
        
        Using scipy.optimize.minimize
        Objective: f(u) = sum(u^2)
        Constraint: A_cbf @ u - b_cbf >= 0
        """
        def objective(u):
            return 0.5 * np.sum(u**2)
            
        def constraint_func(u):
            # A u >= b  <=>  A u - b >= 0
            return A_cbf @ u - b_cbf.flatten()
            
        # Initial guess (zero control)
        u0 = np.zeros(self.num_actuators)
        
        # Constraints definition for scipy
        # 'type': 'ineq' means constraint_func(u) >= 0
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        # Bounds? Infinite for now, or could limit amplitude
        # bounds = [(-5.0, 5.0) for _ in range(self.num_actuators)]
        
        # Solve
        # SLSQP is good for smooth nonlinear/linear constraints
        res = optimize.minimize(objective, u0, method='SLSQP', constraints=cons, tol=1e-3)
        
        if res.success:
            return res.x
        else:
            # Fallback or warning if solver fails
            # print(f"QP Solver failed: {res.message}")
            return np.zeros(self.num_actuators)

    def run(self, controlled=True):
        print(f"Starting Simulation (Controlled={controlled})...")
        
        # Initial State (Random perturbations around rest)
        np.random.seed(42) # Ensure reproducible comparison
        current_state = np.random.rand(2, self.num_nodes) * 0.1
        
        # History Setup
        state_history = np.zeros((self.steps, self.num_nodes))
        control_history = np.zeros((self.steps, self.num_actuators))
        
        # CBF Setup (Re-initialized here to be sure)
        barrier = VoltageThresholdBarrier(
            n_nodes=self.num_nodes,
            v_threshold=self.seizure_threshold,
            constrained_nodes=None # All nodes
        )
        
        # System Dynamics Wrapper for weak4.py utils
        system_wrapper = make_fhn_control_affine(
            adj=self.adj_matrix,
            coupling=self.coupling_strength,
            currents=self.I_ext,
            a=self.a,
            b=self.b,
            tau=self.tau,
            B=self.B
        )
        
        # Alpha function for CBF
        alpha_func = lambda h: self.alpha_decay * h
        
        cbf = CBF(barrier=barrier, system=system_wrapper, alpha=alpha_func)
        
        # --- Dynamics Integration (RK4 with Control) ---
        # We need a custom step function because 'u' changes every step based on state
        
        x = current_state.reshape(-1, 1) # (2N, 1) column vector
        
        for i in range(self.steps):
            t = i * self.dt
            
            # 1. Store History
            # x is [v; w], we just want v (first N)
            v = x[:self.num_nodes].flatten()
            state_history[i, :] = v
            
            # 2. Calculate Control (if active)
            u_opt = np.zeros(self.num_actuators)
            if controlled:
                # Get Inequality A u >= b
                # Note: weak4 CBF returns A, b such that A u >= b
                A_qp, b_qp = cbf.safety_inequality_qp_form(x)
                
                # Check if we are already safe? 
                # If constraints are satisfied with u=0, we can skip solver (energy saving),
                # But strict QP formulation will just return u=0 closest to 0.
                u_opt = self.solve_qp(A_qp, b_qp)
            
            control_history[i, :] = u_opt
            
            # 3. Step Dynamics
            # RK4 Integration manually because 'u' is effectively constant for this timestep dt
            # x_dot = f(x) + g(x)u
            # Define dynamics func for this specific u
            def dynamics_func(y, t_):
                y = y.reshape(-1, 1)
                f_val, g_val = system_wrapper.eval(y)
                # x_dot = f + g @ u
                dx = f_val + g_val @ u_opt.reshape(-1, 1)
                return dx.flatten()
            
            # RK4 Step
            k1 = dynamics_func(x, t)
            k2 = dynamics_func(x + 0.5 * self.dt * k1.reshape(-1, 1), t + 0.5 * self.dt)
            k3 = dynamics_func(x + 0.5 * self.dt * k2.reshape(-1, 1), t + 0.5 * self.dt)
            k4 = dynamics_func(x + self.dt * k3.reshape(-1, 1), t + self.dt)
            
            x_next = x.flatten() + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x = x_next.reshape(-1, 1)
            
        return state_history, control_history

if __name__ == "__main__":
    # 1. Setup
    sim = ClosedLoopSimulation(num_nodes=50, alpha=5.0) # Using higher alpha for stronger correction if needed
    
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
    
    # Plot 1: Focus Node Comparison
    ax = axes[0]
    ax.plot(time_points, v_uncontrolled[:, focus_idx], 'r--', label='Uncontrolled (Focus)', alpha=0.7)
    ax.plot(time_points, v_controlled[:, focus_idx], 'g-', label='Controlled (Focus)', linewidth=2)
    ax.axhline(sim.seizure_threshold, color='k', linestyle=':', label='Safety Threshold')
    ax.set_ylabel("Voltage (v)")
    ax.set_title("Focus Node Dynamics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Max Network Activity (Global Safety)
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
    
    # Plot 3: Control Input Energy
    ax = axes[2]
    # Sum of absolute control input at each step or just for focus node
    # Let's plot the control input specifically for the focus node (since B is Identity)
    ax.plot(time_points, u_controlled[:, focus_idx], 'b-', label='Input to Focus Node')
    # Maybe also total energy?
    total_energy = np.sum(u_controlled**2, axis=1)
    ax.plot(time_points, total_energy, 'k:', alpha=0.3, label='Total Input Energy')
    
    ax.set_ylabel("Control Input (u)")
    ax.set_xlabel("Time (ms)")
    ax.set_title("Control Inputs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure BEFORE showing it (otherwise it might be blank)
    image_path = "Images/week7_comparison.png"
    plt.savefig(image_path)
    print(f"Saved comparison plot to '{image_path}'")
    
    # Show plot
    plt.show()

    # --- Numerical Verification ---
    max_un = np.max(v_uncontrolled)
    max_con = np.max(v_controlled)
    print(f"Max Voltage (Uncontrolled): {max_un:.4f}")
    print(f"Max Voltage (Controlled):   {max_con:.4f}")
    print(f"Seizure Threshold:          {sim.seizure_threshold}")
    
    if max_con < sim.seizure_threshold + 0.1: # Allow small tolerance
        print("✅ SUCCESS: Seizure suppressed.")
    else:
        print("❌ FAILURE: Seizure NOT fully suppressed.")
