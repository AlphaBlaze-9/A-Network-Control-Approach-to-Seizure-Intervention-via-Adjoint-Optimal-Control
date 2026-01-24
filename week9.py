
import os
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
        Plant uses perturbed A; controller uses nominal A (model mismatch).
        """
        noise = np.random.normal(0, noise_std, self.adj_matrix.shape)
        A_perturbed = self.adj_matrix + noise
        A_perturbed = np.maximum(A_perturbed, 0)
        np.fill_diagonal(A_perturbed, 0)

        system_model = make_fhn_control_affine(
            adj=self.adj_matrix,
            coupling=self.coupling_strength,
            currents=self.I_ext,
            a=self.a,
            b=self.b,
            tau=self.tau,
            B=self.B
        )

        system_plant = make_fhn_control_affine(
            adj=A_perturbed,
            coupling=self.coupling_strength,
            currents=self.I_ext,
            a=self.a,
            b=self.b,
            tau=self.tau,
            B=self.B
        )

        barrier = VoltageThresholdBarrier(self.num_nodes, self.seizure_threshold)
        alpha_func = lambda h: self.alpha_decay * h
        cbf = CBF(barrier, system_model, alpha_func)

        rng = np.random.default_rng(42)
        x = (rng.random((2, self.num_nodes)) * 0.1).reshape(-1, 1)
        state_history = np.zeros((self.steps, self.num_nodes), dtype=float)
        energy = 0.0

        for i in range(self.steps):
            state_history[i, :] = x[:self.num_nodes].flatten()

            A_qp, b_qp = cbf.safety_inequality_qp_form(x)
            u_opt = self.solve_qp(A_qp, b_qp)
            energy += float(np.sum(u_opt**2))

            def dynamics_func(y):
                y = y.reshape(-1, 1)
                f, g = system_plant.eval(y)
                return (f + g @ u_opt.reshape(-1, 1)).flatten()

            k1 = dynamics_func(x)
            k2 = dynamics_func(x + 0.5 * self.dt * k1.reshape(-1, 1))
            k3 = dynamics_func(x + 0.5 * self.dt * k2.reshape(-1, 1))
            k4 = dynamics_func(x + self.dt * k3.reshape(-1, 1))

            x_next = x.flatten() + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x = x_next.reshape(-1, 1)

        return state_history, energy

    def run_sensitivity(self, intensity_factor=1.0):
        """
        Runs simulation with scaled focus excitability (seizure intensity).
        """
        I_current = self.I_ext.copy()
        I_current[self.focus_node_index] = self.I_focus * float(intensity_factor)

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

        rng = np.random.default_rng(42)
        x = (rng.random((2, self.num_nodes)) * 0.1).reshape(-1, 1)
        state_history = np.zeros((self.steps, self.num_nodes), dtype=float)
        energy = 0.0

        for i in range(self.steps):
            state_history[i, :] = x[:self.num_nodes].flatten()

            A_qp, b_qp = cbf.safety_inequality_qp_form(x)
            u_opt = self.solve_qp(A_qp, b_qp)
            energy += float(np.sum(u_opt**2))

            def dynamics_func(y):
                y = y.reshape(-1, 1)
                f, g = system.eval(y)
                return (f + g @ u_opt.reshape(-1, 1)).flatten()

            k1 = dynamics_func(x)
            k2 = dynamics_func(x + 0.5 * self.dt * k1.reshape(-1, 1))
            k3 = dynamics_func(x + 0.5 * self.dt * k2.reshape(-1, 1))
            k4 = dynamics_func(x + self.dt * k3.reshape(-1, 1))

            x_next = x.flatten() + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x = x_next.reshape(-1, 1)

        return state_history, energy


if __name__ == "__main__":
    os.makedirs("Images", exist_ok=True)

    sim = Week9Sensitivity(num_nodes=50, T=100.0)

    print("==========================================================")
    print("        WEEK 9: ROBUSTNESS & SENSITIVITY ANALYSIS")
    print("==========================================================")
    print("Note: Reduced simulation duration T=100 for speed.")

    print("\n1. Running Robustness Analysis (Connectivity Noise)...")
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    max_voltages = []

    for sigma in noise_levels:
        print(f"   > Simulating Noise Std = {sigma}...")
        v_hist, _ = sim.run_perturbed(noise_std=sigma)
        max_v = float(np.max(v_hist))
        max_voltages.append(max_v)

    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, max_voltages, 'o-', linewidth=2)
    plt.axhline(sim.seizure_threshold, linestyle='--', label='Seizure Threshold')
    plt.xlabel('Connectivity Noise (Std Dev)')
    plt.ylabel('Max Network Voltage')
    plt.title('Robustness: Controller Performance vs Model Mismatch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Images/week9_robustness.png', dpi=200)
    print("   ✅ Saved 'Images/week9_robustness.png'")

    print("\n2. Running Sensitivity Analysis (Seizure Intensity)...")
    intensities = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    energies = []
    successes = []

    for intensity in intensities:
        print(f"   > Simulating Intensity Factor = {intensity}x...")
        v_hist, erg = sim.run_sensitivity(intensity_factor=intensity)
        energies.append(float(erg))
        successes.append(float(np.max(v_hist)) < sim.seizure_threshold + 0.1)

    plt.figure(figsize=(8, 6))
    plt.plot(intensities, energies, 's-', linewidth=2)
    plt.xlabel('Seizure Intensity (Factor of I_focus)')
    plt.ylabel('Total Control Energy (||u||^2)')
    plt.title('Sensitivity: Energy Cost vs Seizure Intensity')
    plt.grid(True, alpha=0.3)

    breakdown_idx = next((i for i, s in enumerate(successes) if not s), None)
    if breakdown_idx is not None:
        plt.axvline(intensities[breakdown_idx], linestyle=':', label='Operational Limit')
        plt.legend()

    plt.savefig('Images/week9_sensitivity.png', dpi=200)
    print("   ✅ Saved 'Images/week9_sensitivity.png'")

    print("\nAnalysis Complete.")
