import numpy as np
import matplotlib.pyplot as plt

# 1. Load the binary data you created
a_map = np.load('excitability_map_a.npy')

# 2. Reshape or Plot as a 1D Heatmap
# Since it's 360 nodes, we display it as a long strip
plt.figure(figsize=(15, 2))
plt.imshow(a_map.reshape(1, -1), aspect='auto', cmap='coolwarm')

# 3. Add formatting
plt.colorbar(label='Excitability (a)')
plt.title("Week 2 Deliverable: Brain Excitability Map (The 'a' Vector)")
plt.xlabel("Brain Region Index")
plt.yticks([]) # Hide Y-axis since it's 1D

# 4. Annotate the Seizure Focus
plt.annotate('Seizure Focus (a=0.5)', xy=(10, 0), xytext=(20, -0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig('excitability_map_visual.png', dpi=300)
plt.show()