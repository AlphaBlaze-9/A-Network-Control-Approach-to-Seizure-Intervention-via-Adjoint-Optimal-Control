
import numpy as np
import matplotlib.pyplot as plt
from week7 import ClosedLoopSimulation
from week4 import make_fhn_control_affine, VoltageThresholdBarrier, CBF

class Week9Sensitivity(ClosedLoopSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run_perturbed(self, noise_std=0.0):
        """
        Runs the simulation with a perturbed adjacency matrix.
        A_perturbed = A + N(0, noise_std)
        """
        # Create perturbed adjacency
        noise = np.random.normal(0, noise_std, self.adj_matrix.shape)
        # Keep it symmetric and zero diagonal typically, or just allow general noise?
        # Biological noise might not be symmetric. Let's add general noise but keep positive?
        # Typically structural connectivity is non-negative.
        A_perturbed = self.adj_matrix + noise
        A_perturbed = np.maximum(A_perturbed, 0) # Ensure non-negative? 
        np.fill_diagonal(A_perturbed, 0)
        
        # We need to re-initialize system wrapper with this new A
        # BUT the controller might assume the ORIGINAL A (model mismatch) 
        # OR the controller knows the perturbed A?
        # "Robustness" usually implies Model Mismatch: System has A_plant, Controller has A_model.
        # Let's assume the controller assumes the IDEAL model (self.adj_matrix),
        # but the actual system evolves with A_perturbed.
        
        # 1. Controller Model (Ideal)
        system_model = make_fhn_control_affine(
            adj=self.adj_matrix, # Controller thinks this is the network
            coupling=self.coupling_strength,
            currents=self.I_ext,
            a=self.a,
            b=self.b,
            tau=self.tau,
            B=self.B
        )
        
        # 2. Plant (Real System with Noise)
        system_plant = make_fhn_control_affine(
            adj=A_perturbed, # Actual network
            coupling=self.coupling_strength,
            currents=self.I_ext,
            a=self.a,
            b=self.b,
            tau=self.tau,
            B=self.B
        )
        
        # CBF Setup (using Ideal Model)
        barrier = VoltageThresholdBarrier(self.num_nodes, self.seizure_threshold)
        alpha_func = lambda h: self.alpha_decay * h
        cbf = CBF(barrier, system_model, alpha_func)
        
        # Simulation Loop
        np.random.seed(42)
        x = (np.random.rand(2, self.num_nodes) * 0.1).reshape(-1, 1)
        state_history = np.zeros((self.steps, self.num_nodes))
        energy = 0.0
        
        for i in range(self.steps):
            t = i * self.dt
            state_history[i, :] = x[:self.num_nodes].flatten()
            
            # Controller calculates u based on Model
            A_qp, b_qp = cbf.safety_inequality_qp_form(x)
            u_opt = self.solve_qp(A_qp, b_qp)
            energy += np.sum(u_opt**2)
            
            # Plant evolves with u (RK4 on Plant dynamics)
            def dynamics_func(y, t_):
                y = y.reshape(-1, 1)
                f, g = system_plant.eval(y)
                return (f + g @ u_opt.reshape(-1, 1)).flatten()
            
            # RK4 Step
            k1 = dynamics_func(x, t)
            k2 = dynamics_func(x + 0.5 * self.dt * k1.reshape(-1, 1), t + 0.5 * self.dt)
            k3 = dynamics_func(x + 0.5 * self.dt * k2.reshape(-1, 1), t + 0.5 * self.dt)
            k4 = dynamics_func(x + self.dt * k3.reshape(-1, 1), t + self.dt)
            
            x_next = x.flatten() + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x = x_next.reshape(-1, 1)
            
        return state_history, energy

    def run_sensitivity(self, intensity_factor=1.0):
        """
        Runs simulation with Scaled I_focus (Seizure Intensity).
        """
        # Update currents
        I_current = self.I_ext.copy()
        # Scale the focus drive
        focus_drive = self.I_focus * intensity_factor
        I_current[self.focus_node_index] = focus_drive
        
        # Controller AND Plant see this change (assuming we can measure the state, we react to it)
        # Or does controller assume normal params? 
        # Let's assume controller knows the model parameters adapted to patient state, 
        # or simply that the dynamics 'f(x)' includes I_ext.
        
        system = make_fhn_control_affine(
            adj=self.adj_matrix,
            coupling=self.coupling_strength,
            currents=I_current,
            a=self.a,
            b=self.b,
            tau=self.tau,
            B=self.B
        )
        
        barrier = VoltageThresholdBarrier(self.num_nodes, self.seizure_threshold)
        alpha_func = lambda h: self.alpha_decay * h
        cbf = CBF(barrier, system, alpha_func)
        
        np.random.seed(42)
        x = (np.random.rand(2, self.num_nodes) * 0.1).reshape(-1, 1)
        state_history = np.zeros((self.steps, self.num_nodes))
        energy = 0.0
        
        for i in range(self.steps):
            state_history[i, :] = x[:self.num_nodes].flatten()
            A_qp, b_qp = cbf.safety_inequality_qp_form(x)
            u_opt = self.solve_qp(A_qp, b_qp)
            energy += np.sum(u_opt**2)
            
            def dynamics_func(y, t_):
                y = y.reshape(-1, 1)
                f, g = system.eval(y)
                return (f + g @ u_opt.reshape(-1, 1)).flatten()
            
            k1 = dynamics_func(x, 0)
            k2 = dynamics_func(x + 0.5 * self.dt * k1.reshape(-1, 1), 0)
            k3 = dynamics_func(x + 0.5 * self.dt * k2.reshape(-1, 1), 0)
            k4 = dynamics_func(x + self.dt * k3.reshape(-1, 1), 0)
            x_next = x.flatten() + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x = x_next.reshape(-1, 1)
            
        return state_history, energy

if __name__ == "__main__":
    # Reduce duration for faster sensitivity analysis
    sim = Week9Sensitivity(num_nodes=50, T=100.0)
    
    print("==========================================================")
    print("        WEEK 9: ROBUSTNESS & SENSITIVITY ANALYSIS")
    print("==========================================================")
    print("Note: Reduced simulation duration T=100 for speed.")
    
    # --- 1. Robustness Analysis (Noise) ---
    print("\n1. Running Robustness Analysis (Connectivity Noise)...")
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    max_voltages = []
    
    for sigma in noise_levels:
        print(f"   > Simulating Noise Std = {sigma}...")
        v_hist, _ = sim.run_perturbed(noise_std=sigma)
        max_v = np.max(v_hist)
        max_voltages.append(max_v)
        
    # Plot Robustness
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, max_voltages, 'o-', linewidth=2, color='purple')
    plt.axhline(sim.seizure_threshold, color='red', linestyle='--', label='Seizure Threshold')
    plt.xlabel('Connectivity Noise (Std Dev)')
    plt.ylabel('Max Network Voltage')
    plt.title('Robustness: Controller Performance vs Model Mismatch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Images/week9_robustness.png')
    print("   ✅ Saved 'Images/week9_robustness.png'")
    
    # --- 2. Sensitivity Analysis (Intensity) ---
    print("\n2. Running Sensitivity Analysis (Seizure Intensity)...")
    intensities = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    energies = []
    successes = []
    
    for intensity in intensities:
        print(f"   > Simulating Intensity Factor = {intensity}x...")
        v_hist, erg = sim.run_sensitivity(intensity_factor=intensity)
        energies.append(erg)
        successes.append(np.max(v_hist) < sim.seizure_threshold + 0.1)
        
    # Plot Sensitivity
    plt.figure(figsize=(8, 6))
    plt.plot(intensities, energies, 's-', linewidth=2, color='orange')
    plt.xlabel('Seizure Intensity (Factor of I_focus)')
    plt.ylabel('Total Control Energy (||u||^2)')
    plt.title('Sensitivity: Energy Cost vs Seizure Intensity')
    plt.grid(True, alpha=0.3)
    
    # Mark operational breakdown
    breakdown_idx = next((i for i, s in enumerate(successes) if not s), None)
    if breakdown_idx is not None:
        plt.axvline(intensities[breakdown_idx], color='red', linestyle=':', label='Operational Limit')
        plt.legend()
        
    plt.savefig('Images/week9_sensitivity.png')
    print("   ✅ Saved 'Images/week9_sensitivity.png'")
    
    print("\nAnalysis Complete.")
