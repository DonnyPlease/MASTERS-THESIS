from scipy.interpolate import griddata
from fit_tool.Dataset import DatasetUtils
import numpy as np
import matplotlib.pyplot as plt


dataset, _ = DatasetUtils.load_datasets_to_dicts('dataset')

data = DatasetUtils.dataset_to_dict(dataset)
    
for current in ["1e17", "1e18", "1e19"]:
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

    num_points = 20

    # Define the grid
    x_grid = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), num_points)
    y_grid = np.linspace(np.min(y), np.max(y), num_points)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Interpolate z values onto the grid
    Z = griddata((x, y), z, (X, Y), method='linear')

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Z, cmap='viridis')
    plt.colorbar(label='T [keV]')
    plt.xscale('log')
    plt.xlabel('l [nm]')
    plt.ylabel('aplha [deg]')
    plt.title("I = " + current)
    # plt.show()
    
    plt.scatter(x,y,c='black',s=10,marker='o')
    plt.savefig("dataset/I_" + current + ".png")
    