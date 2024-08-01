import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

# Set paths
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)
PATH_TO_MODELS = os.path.join(PATH_TO_PROJECT, 'models', 'models')

# Import custom modules
from models.gp3d_gpy import Gp3dGpy as GP
from models.gp3d_gpy import Gp3dGpy
from transformer import Transformer

# Configure Matplotlib to use LaTeX for rendering text
rc('font', family='serif', serif='Computer Modern')
rc('text', usetex=True)

def load_model(path):
    """Load the GP model from the specified path."""
    return GP.load(path)

def power(x, a, b):
    """Power-law function for curve fitting."""
    return a * x**b

if __name__ == "__main__":
    # Define prediction points and space
    x_pred = np.array([
        [1e17, 1, 21.6],
        [2e17, 1, 21.6],
        [5e17, 1, 21.6],
        [1e18, 1, 21.6],
        [2e18, 1, 21.6],
        [5e18, 1, 21.6],
        [1e19, 1, 21.6],
    ])

    i_samples = np.array([1e17, 2e17, 5e17, 1e18, 2e18, 5e18, 1e19])
    i_samples = np.logspace(17, 19, 20)
    # Load the GP model
    model = load_model(os.path.join(PATH_TO_MODELS, "gp_model.pkl"))

    x_pred_001 = np.array([i_samples, np.ones_like(i_samples) * 0.01, np.ones_like(i_samples) * 21.6]).T
    x_pred_01 = np.array([i_samples, np.ones_like(i_samples) * 0.1, np.ones_like(i_samples) * 21.6]).T
    x_pred_1 = np.array([i_samples, np.ones_like(i_samples) * 1, np.ones_like(i_samples) * 21.6]).T
    x_pred_4 = np.array([i_samples, np.ones_like(i_samples) * 4, np.ones_like(i_samples) * 21.6]).T
    
    list_of_x_pred = [x_pred_001, x_pred_01, x_pred_1, x_pred_4]
    labels = [r"$\lambda = 0.01$", r"$\lambda = 0.1$", r"$\lambda = 1$", r"$\lambda = 4$"]
    colors = ["blue", "green", "orange", "purple"]        
    for i, x_pred in enumerate(list_of_x_pred):
        x_pred_t = model.transformer.transform(x_pred)
        t_pred, _ = model.predict(x_pred, return_std=True)
        x_pred = model.transformer.reverse_transform(x_pred_t)
    

        plt.scatter(x_pred[:, 0], t_pred, color=colors[i], label=labels[i], zorder=7)
        plt.plot(x_pred[:, 0], t_pred, color=colors[i], zorder=7)
    
    
    plt.xscale("log")
    plt.yscale("log")
    plt.yticks([1e1,2e1, 5e1,1e2, 2e2, 5e2, 1e3], ["10","20", "50", "100", "200", "500", "1000"])
    plt.xticks([1e17, 2e17, 5e17, 1e18, 2e18, 5e18, 1e19], 
               [r"$1\times 10^{17}$", r"$2\times10^{17}$", r"$5\times 10^{17}$", 
                r"$1 \times 10^{18}$", r"$2 \times 10^{18}$", r"$5 \times 10^{18}$", r"$1\times 10^{19}$"])
    plt.xlabel(r"$I \, [\mathrm{W \,cm^{-2}}]$")
    plt.ylabel(r"$T_{\mathrm{hot}} \, [\mathrm{keV}]$")
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs_for_thesis/comparison/cui2013_2.pdf")
    plt.show()
