import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def integrate_histogram(bins, counts, energy_cut=0):
    """
    Integrate the histogram by summing the counts and multiplying by the bin width.
    """
    starting_index = np.where(bins >= energy_cut)[0][0]
    counts = np.array(counts[starting_index:])
    bins = np.array(bins[starting_index:])
    energies = bins[:-1] + np.diff(bins)/2
    return np.dot(counts, energies)
    
def integrate_only_hot_electrons(histogram):
    # Integrate th histogram analytically N = N0*exp(-E/t_hot)
    
    t_hot = histogram.t_hot
    n0 = histogram.n0_hot
    
    return 
    
 
def create_dict_from_file(filename):
    """
    Create a dictionary from a file. The file should contain the data in the format:
    I,L,alpha,i
    where I,L,alpha are parameters and i is the value of the integral for histogram with those parameters.
    """
    data = {"1e17": [], "1e18": [], "1e19": []} # Initialize the dictionary
    with open(filename, 'r') as file:
        for line in file:
            x, y, z, i = line.split(',')
            data[x].append((float(y), float(z), float(i)))
    
    return data

def draw_integrals(file_name, target_folder, energy_cut):
    data = create_dict_from_file(file_name)
         
    for current in ["1e17", "1e18", "1e19"]:
        # Extract the data points
        x = [item[0] for item in data[current]] # L
        y = [item[1] for item in data[current]] # alpha
        z = [item[2] for item in data[current]] # integral
        
        # Define the grid for interpolation
        num_points = 10

        # Define the grid
        x_grid = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), num_points)
        y_grid = np.linspace(np.min(y), np.max(y), num_points)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate z values onto the grid
        Z = griddata((x, y), z, (X, Y), method='linear')

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(X, Y, Z, cmap="inferno")
        plt.colorbar(label='Total Energy [eV]')
        plt.xscale('log')
        plt.xlabel(r'$l$ [nm]')
        plt.ylabel(r'$\alpha$ '+'[°]')
        plt.title("I = " + current)
        # plt.show()
        
        # plt.scatter(x,y,c='black',s=10,marker='o')
        plt.savefig(target_folder + "I_" + current + "_cut_{}.pdf".format(energy_cut))
        
if __name__ == "__main__":
    ### Test the function integrate_histogram
    bins = np.linspace(0, 200, 101)
    counts = np.random.rand(100)
    print(integrate_histogram(bins, counts, cut=10))