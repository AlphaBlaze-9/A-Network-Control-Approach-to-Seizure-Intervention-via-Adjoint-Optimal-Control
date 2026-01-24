import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ===========================
# WEEK 3: SIMULATING UNCONTROLLED SEIZURE PROPAGATION (FHN)
# ===========================

# --- Network Generation and Setup ---
num_nodes = 50
# Watts-Strogatz small-world graph to mimic brain topology
G = nx.watts_strogatz_graph(n=num_nodes, k=6, p=0.1, seed=42)
adj_matrix = nx.to_numpy_array(G)

# Define Focus Node (Seizure Onset Zone)
focus_node_index = 0  # Node 0 is the "Hippocampus"

# --- Model Parameters (FitzHugh-Nagumo) ---
T = 400.0             # Total simulation time (ms, conceptual)
dt = 0.05             # Time step
steps = int(T / dt)
time_points = np.arange(steps) * dt

# FHN Parameters
a = 0.7
b = 0.8
tau = 12.5

# Global Coupling Strength
coupling_strength = 0.5

# External Drive (Excitability)
I_base = 0.35
I_focus = 1.0
I_ext = np.full(num_nodes, I_base)
I_ext[focus_node_index] = I_focus

# Seizure Threshold (Amplitude)
seizure_threshold = 1.0

# --- Dynamics Functions ---
def fhn_dynamics(state, t, adj, coupling, currents):
    # state shape: (2, num_nodes) -> v = state[0], w = state[1]
    v = state[0]
    w = state[1]

    # Diffusive coupling term: sum(A_ij * (v_j - v_i))
    neighbor_diffs = (adj @ v) - (v * np.sum(adj, axis=1))
    coupling_current = coupling * neighbor_diffs

    # Differential Equations
    dv = v - (v**3)/3.0 - w + currents + coupling_current
    dw = (v + a - b * w) / tau

    return np.stack((dv, dw))

# --- RK4 Integration ---
def rk4_step(func, state, t, dt, adj, coupling, currents):
    k1 = func(state, t, adj, coupling, currents)
    k2 = func(state + 0.5 * dt * k1, t + 0.5 * dt, adj, coupling, currents)
    k3 = func(state + 0.5 * dt * k2, t + 0.5 * dt, adj, coupling, currents)
    k4 = func(state + dt * k3, t + dt, adj, coupling, currents)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --- Simulation Loop ---
# Initial State (Random perturbations around rest)
rng = np.random.default_rng(42)
current_state = rng.random((2, num_nodes)) * 0.1

simulation_history = np.zeros((steps, num_nodes))

print(f"Simulating network with {num_nodes} nodes for {T} (dt={dt})...")

for i in range(steps):
    t = i * dt
    current_state = rk4_step(fhn_dynamics, current_state, t, dt, adj_matrix, coupling_strength, I_ext)
    simulation_history[i, :] = current_state[0]  # Store voltage 'v' only

# --- Analysis: Seizure State Identification ---
is_seizing = np.max(simulation_history[int(steps/2):], axis=0) > seizure_threshold
seizing_nodes = np.where(is_seizing)[0]
print(f"Number of nodes recruited into seizure: {len(seizing_nodes)}/{num_nodes}")

# --- Visualization: Time Series ---
plt.figure(figsize=(12, 6))
plt.plot(time_points, simulation_history[:, focus_node_index],
         label=f'Focus Node (Node {focus_node_index})', color='red', linewidth=2)

neighbor_node = list(G.neighbors(focus_node_index))[0]
plt.plot(time_points, simulation_history[:, neighbor_node],
         label=f'Direct Neighbor (Node {neighbor_node})', color='orange', alpha=0.8)

distant_node = num_nodes - 1
plt.plot(time_points, simulation_history[:, distant_node],
         label=f'Distant Node (Node {distant_node})', color='blue', alpha=0.6, linestyle='--')

plt.axhline(y=seizure_threshold, color='k', linestyle=':', label='Seizure Threshold')
plt.title('Seizure Propagation: Time Series Activity')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (v)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Visualization: Spatiotemporal Heatmap (No seaborn dependency) ---
plt.figure(figsize=(14, 8))
plt.imshow(simulation_history.T, aspect="auto", origin="lower", interpolation="nearest")
plt.colorbar(label='Activity Amplitude (v)')
plt.title('Spatiotemporal Spread of Seizure Activity')
plt.xlabel('Time Steps')
plt.ylabel('Node Index')
plt.tight_layout()
plt.show()
