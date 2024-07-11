import os
import sys
sys.path.append('C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/')


import numpy as np
import matplotlib.pyplot as plt

from dataset.Dataset import DatasetUtils
from draw_dataset import draw_dataset

from scipy.interpolate import RBFInterpolator

# Load the data
dataset = DatasetUtils.load_final_dataset('dataset')
data = DatasetUtils.dataset_to_dict(dataset)
data17 = np.array(data['1e17'])
print(data17)
x = data17[:,0]/5
y = data17[:,1]/60
z = data17[:,2]

# Combine x and y into a single array of shape (n_samples, 2)
points = np.column_stack((x, y))

# Fit the RBF interpolator
rbf_interpolator = RBFInterpolator(points, z, kernel='cubic')

# Create a grid for plotting the interpolated surface
x_grid = np.linspace(np.min(x), np.max(x), 100)
x_grid = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 100)
y_grid = np.linspace(np.min(y), np.max(y), 100)
x_grid, y_grid = np.meshgrid(x_grid, y_grid)
grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

# Predict values on the grid
z_grid = rbf_interpolator(grid_points)
z_grid = z_grid.reshape(x_grid.shape)

# Plot the original data points and the interpolated surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='r', label='Original Data')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()


