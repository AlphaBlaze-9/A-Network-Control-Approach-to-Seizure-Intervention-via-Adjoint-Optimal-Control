import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: FIX THE NAME ERROR (Load Week 1 Deliverable) ---
try:
    # This loads the physical 'highway' map you built in Week 1
    normalized_A = np.load('brain_A_matrix.npy')
    print("✅ Success: Week 1 Matrix loaded correctly.")
except FileNotFoundError:
    print("❌ Error: 'brain_A_matrix.npy' not found. Run your Week 1 script first!")
    exit()

# --- ACTION ITEM 1 & 2: Implement Nonlinear Hopf Oscillator Dynamics ---
# This function defines the mathematical rules for how nodes behave
def hopf_network_dynamics(state, t, A, a_vec, G, omega):
    N = len(a_vec)
    x = state[:N]
    y = state[N:]
    
    # Calculate amplitude squared (x^2 + y^2)
    r2 = x**2 + y**2
    
    # Action Item 4: Nonlinear equations where 'a' controls stability
    # Local dynamics: If a < 0, it damps to zero. If a > 0, it oscillates.
    dxdt = (a_vec - r2) * x - omega * y
    dydt = (a_vec - r2) * y + omega * x
    
    # Network coupling: How activity spreads through the A matrix
    # G is global coupling strength
    coupling_x = G * (A @ x - np.sum(A, axis=1) * x)
    coupling_y = G * (A @ y - np.sum(A, axis=1) * y)
    
    return np.concatenate([dxdt + coupling_x, dydt + coupling_y])

# --- ACTION ITEM 3: Map Excitability Scores to Bifurcation Parameter ---
# We create a map where specific nodes are 'sick' (prone to seizures)
def create_excitability_map(N, seizure_foci_indices):
    # Action Item 4: Negative parameters = damping (Healthy state)
    a_parameters = np.full(N, -0.5) 
    
    # Action Item 4: Positive parameters = oscillations (Seizure state)
    for idx in seizure_foci_indices:
        a_parameters[idx] = 0.5 
        
    return a_parameters

# --- FINAL WEEK 2 SETUP ---
N = len(normalized_A)
# We pick node 10 as our primary seizure focus (the 'sick' region)
a_vec = create_excitability_map(N, seizure_foci_indices=[10])

# Save this map as your Week 2 deliverable
np.save('excitability_map_a.npy', a_vec)

print("------------------------------------------------")
print(f"✅ WEEK 2 COMPLETE: Dynamics function defined for {N} nodes.")
print("✅ Deliverable 'excitability_map_a.npy' saved.")
print("------------------------------------------------")

import numpy as np

# 1. Load the binary file
a_map = np.load('excitability_map_a.npy')

# 2. Check the basics
print(f"Total Brain Regions: {len(a_map)}")
print(f"First 5 values: {a_map[:5]}")

# 3. Find your seizure focus (the 'sick' node)
seizure_indices = np.where(a_map > 0)[0]
print(f"Seizure focus found at indices: {seizure_indices}")
print(f"Value at focus: {a_map[seizure_indices[0]]}")

# 4. Count healthy vs sick
healthy_count = np.sum(a_map < 0)
sick_count = np.sum(a_map > 0)
print(f"Brain Summary: {healthy_count} healthy regions, {sick_count} seizure regions.")