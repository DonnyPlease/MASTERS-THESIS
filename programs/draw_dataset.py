from scipy.interpolate import griddata
from fit_tool.Dataset import DatasetUtils
import numpy as np
import matplotlib.pyplot as plt


dataset, _ = DatasetUtils.load_datasets_to_dicts('dataset')

data = DatasetUtils.dataset_to_dict(dataset)

def plot_maximum_absorption_curve():
    curve_x = np.logspace(-1.13,0.75,30)
    curve_y = np.arcsin(0.68*np.power(curve_x*2*np.pi,np.ones_like(curve_x)*-1/3))*180/np.pi
    plt.plot(curve_x,curve_y, color='#7FFF00',linewidth=2)

def draw(X,Y,Z):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Z, cmap='inferno', shading='auto')
    plt.colorbar(label=r'$T_{hot}$ [keV]')
    plt.xscale('log')
    plt.xlabel(r'$L$ [μm]')
    plt.ylabel(r'$\alpha$ [°]')
    ticks=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5]
    plt.xticks(ticks,ticks)
    plt.show()

def draw_dataset():
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

        num_points = 50
        # Define the grid
        x_grid = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), num_points)
        y_grid = np.linspace(np.min(y), np.max(y), num_points)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate z values onto the grid
        Z = griddata((x, y), z, (X, Y), method='linear')

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(X, Y, Z, cmap='inferno', shading='auto')
        plt.colorbar(label=r'$T_{hot}$ [keV]')
        plt.xscale('log')
        plt.xlabel(r'$L$ [μm]')
        plt.ylabel(r'$\alpha$ [°]')
        ticks=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5]
        plt.xticks(ticks,ticks)
        # plt.show()
        
        if current == "1e17":
            plot_maximum_absorption_curve()
        
        plt.savefig("dataset/I_" + current + ".png")
        plt.scatter(x,y,c='white',s=10,marker='o')
        plt.savefig("dataset/I_" + current + "wp.png")

if __name__ == "__main__": 
    draw_dataset()