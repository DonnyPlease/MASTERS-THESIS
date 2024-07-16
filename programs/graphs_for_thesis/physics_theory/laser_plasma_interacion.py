import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')

# Add a gradient to represent the laser pulse
x = np.linspace(0, 2, 500)
y = np.linspace(0, 6, 500)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X - 1)**2 * 20)  # Gaussian pulse

# Plot the laser pulse
ax.imshow(Z, extent=[0, 2, 0, 6], origin='lower', cmap='inferno', alpha=0.8)

# Add plasma representation as a rectangle
plasma = patches.Rectangle((2, 1), 6, 4, linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.5)
ax.add_patch(plasma)

# Add text annotations
ax.text(0.5, 5.5, 'Laser Pulse', fontsize=14, color='white', weight='bold')
ax.text(5, 4.5, 'Plasma', fontsize=14, color='blue', weight='bold')

# Add interaction zone illustration
interaction_zone = patches.Ellipse((3, 3), 2, 4, angle=0, linewidth=1, edgecolor='red', facecolor='none', linestyle='--')
ax.add_patch(interaction_zone)
ax.text(2.7, 0.5, 'Interaction Zone', fontsize=12, color='red', weight='bold')

# Save the image
plt.savefig("laser_plasma_interaction.png", dpi=300)

# Show the plot
plt.show()
