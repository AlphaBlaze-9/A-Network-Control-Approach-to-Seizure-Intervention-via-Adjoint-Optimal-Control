import numpy as np
import matplotlib.pyplot as plt
from optimization_engine import compute_glasser_control

# 1. Setup simulation parameters
num_trials = 50
nodes = 360
h_values = np.linspace(0.01, 1.0, num_trials) # From "Near Seizure" to "Very Safe"
energy_results = []
solve_times = []

# 2. Run the Benchmark
print("Starting Verification Test for 360-node Glasser Atlas...")

for h in h_values:
    # Mocking current brain state and dynamics
    f_x = np.random.normal(0, 1, nodes) 
    grad_h = np.random.normal(0, 1, nodes)
    x = np.random.normal(0, 1, nodes)
    
    # Solve for optimal stimulation
    u_opt, t_solve = compute_glasser_control(x, f_x, h, grad_h)
    
    # Calculate Total Energy (L2 Norm)
    energy = np.linalg.norm(u_opt)
    energy_results.append(energy)
    solve_times.append(t_solve)

# 3. Plotting the Results
plt.figure(figsize=(10, 5))

# Plot A: Energy vs Safety
plt.subplot(1, 2, 1)
plt.plot(h_values, energy_results, 'r-', linewidth=2)
plt.title('Energy Usage vs. Brain Safety')
plt.xlabel('Safety Margin (h_val)')
plt.ylabel('Stimulation Energy (||u||)')
plt.grid(True)

# Plot B: Latency Check
plt.subplot(1, 2, 2)
plt.hist(solve_times, bins=10, color='skyblue', edgecolor='black')
plt.title('Solver Latency (360 Nodes)')
plt.xlabel('Solve Time (ms)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('week5_verification.png')
plt.show()

print(f"Average Solve Time: {np.mean(solve_times):.2f} ms")