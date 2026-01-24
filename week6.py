import os
import numpy as np


class ActuatorConfig:
    def __init__(self, num_nodes=360):
        self.num_nodes = int(num_nodes)

    def get_B_matrix(self, mode='identity', target_indices=None):
        """
        Constructs the control input matrix B.

        Parameters:
        - mode: 'identity' (ideal) or 'clinical' (targeted).
        - target_indices: List of node indices to stimulate (only used for 'clinical').

        Returns:
        - B: (num_nodes, num_actuators) matrix.
        """
        if mode == 'identity':
            print(f"Generating Identity B Matrix (All {self.num_nodes} nodes stimulated)...")
            return np.eye(self.num_nodes)

        if mode == 'clinical':
            if target_indices is None:
                print("No target indices provided. Using placeholder (first 50 nodes)...")
                target_indices = np.arange(min(50, self.num_nodes))

            target_indices = np.asarray(target_indices, dtype=int)
            num_actuators = len(target_indices)
            print(f"Generating Clinical B Matrix ({num_actuators} nodes stimulated)...")

            B = np.zeros((self.num_nodes, num_actuators))

            for j, node_idx in enumerate(target_indices):
                if 0 <= node_idx < self.num_nodes:
                    B[node_idx, j] = 1.0
                else:
                    print(f"Warning: Node index {node_idx} out of bounds.")
            return B

        raise ValueError(f"Unknown mode: {mode}")

    def visualize_matrix(self, B, title="B Matrix", filename=None):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 8))
        plt.imshow(B, aspect='auto', cmap='binary', interpolation='none')
        plt.colorbar(label='Connection (1=Active, 0=Inactive)')
        plt.title(title)
        plt.xlabel("Control Inputs (Actuators)")
        plt.ylabel("Brain Nodes (Regions)")
        plt.tight_layout()

        if filename:
            folder = os.path.dirname(filename)
            if folder:
                os.makedirs(folder, exist_ok=True)
            plt.savefig(filename)
            print(f"✅ Saved visualization to {filename}")
        else:
            plt.show()


if __name__ == "__main__":
    config = ActuatorConfig(num_nodes=360)

    B_ideal = config.get_B_matrix(mode='identity')
    print(f"Identity Matrix shape: {B_ideal.shape}")
    print(f"Rank: {np.linalg.matrix_rank(B_ideal)}")

    print("-" * 30)

    temporal_indices = [10, 11, 12, 13, 14, 50, 51, 52]
    B_clinical = config.get_B_matrix(mode='clinical', target_indices=temporal_indices)
    print(f"Clinical Matrix shape: {B_clinical.shape}")
    print(f"Active nodes: {np.where(np.sum(B_clinical, axis=1) > 0)[0]}")

    print("Saving Clinical Matrix Heatmap...")
    image_path = "Images/week6_clinical_heatmap.png"
    config.visualize_matrix(B_clinical, "Clinical Actuator Placement (Temporal Lobe Subset)", filename=image_path)
