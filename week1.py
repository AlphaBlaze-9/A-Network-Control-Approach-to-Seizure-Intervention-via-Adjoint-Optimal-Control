import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Data ---
file_path = 'brain_connectivity.mat' 

try:
    mat_data = scipy.io.loadmat(file_path)
    adj_matrix = mat_data['connectivity']
    print("✅ Success: Raw matrix loaded!")
    print(f"   Shape: {adj_matrix.shape}")
except FileNotFoundError:
    print("❌ Error: Could not find 'brain_connectivity.mat'. Check your file path.")
    exit()

# --- 2. Clean and Normalize ---
np.fill_diagonal(adj_matrix, 0)

eigenvalues = np.linalg.eigvals(adj_matrix)
max_eigenval = np.max(np.abs(eigenvalues))

print(f"   Original Max Eigenvalue: {max_eigenval:.4f}")

if max_eigenval > 0:
    normalized_A = adj_matrix / max_eigenval
else:
    normalized_A = adj_matrix 

# --- 3. Save Deliverable ---
np.save('brain_A_matrix.npy', normalized_A)

print("------------------------------------------------")
print("✅ DONE: Saved 'brain_A_matrix.npy'")
print("------------------------------------------------")

# --- 4. Visualize ---
plt.imshow(np.log1p(normalized_A), cmap='magma')
plt.title("Final Normalized 'A' Matrix")
plt.colorbar()
plt.show()