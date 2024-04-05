from scipy.interpolate import griddata
from fit_tool.Dataset import DatasetUtils
import numpy as np
import matplotlib.pyplot as plt


dataset, _ = DatasetUtils.load_datasets_to_dicts('dataset')

data = DatasetUtils.dataset_to_dict(dataset)
    
# print(data["1e18"])
current = "1e17"

# Extract the data points
x = [item[0] for item in data[current]]
y = [item[1] for item in data[current]]
z = [item[2] for item in data[current]]

# Define the grid for interpolation
# Define the range of x and y
x_min = min(x)
x_max = max(x)
y_min = min(y)
y_max = max(y)

num_points = 30

# Define the grid
x_grid = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), num_points)
y_grid = np.linspace(np.min(y), np.max(y), num_points)
X, Y = np.meshgrid(x_grid, y_grid)

print(X)

# Interpolate z values onto the grid
Z = griddata((x, y), z, (X, Y), method='linear')
print(Z)



plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z, cmap='viridis')
plt.colorbar(label='Z')
plt.xscale('log')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Color Plot of Z values with Logarithmic X Scale (Using griddata)')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')


# # Customize labels and title
# ax.set_xlabel('l')
# ax.set_ylabel('alpha')
# ax.set_zlabel('T')
# ax.set_title('Mesh Plot of temperatures')

# plt.show()

# # Plot the interpolated data
# plt.imshow(Z.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', aspect='auto')
# plt.colorbar()
# plt.xlabel('l')
# plt.xscale('log')
# plt.ylabel('alpha')
# plt.title('Interpolated Data')
# plt.show()

