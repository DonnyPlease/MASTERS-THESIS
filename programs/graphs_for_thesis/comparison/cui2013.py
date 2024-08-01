import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rc
from scipy.optimize import curve_fit

# Set paths
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)
PATH_TO_MODELS = os.path.join(PATH_TO_PROJECT, 'models', 'models')

PATH_TO_FIGURES = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/tex/figures/'
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
    i_space = np.logspace(16.9, 19.1, 50)
    x_ss = np.array([[i, 1, 21.6] for i in i_space])

    # Load the GP model
    model = load_model(os.path.join(PATH_TO_MODELS, "gp_model.pkl"))

    # Transform the input data
    x_pred_t = model.transformer.transform(x_pred)
    x_ss_t = model.transformer.transform(x_ss)

    # Make predictions
    t_pred, _ = model.predict(x_pred, return_std=True)
    mean, ss = model.predict(x_ss, return_std=True)
    ss=1.96*ss
    # Reverse transform the data
    x_pred = model.transformer.reverse_transform(x_pred_t)
    x_ss = model.transformer.reverse_transform(x_ss_t)

    # Fit the power-law curve
    popt, pcov = curve_fit(power, x_pred[:, 0], t_pred, p0=[100, 1/3])
    std_popt = np.sqrt(np.diag(pcov))
    print("Fitted parameters:", popt)
    print("Standard deviation of parameters:", std_popt)

    # Generate fitted data
    fitted_x = np.logspace(16.9, 19.1, 20)
    fitted_y = power(fitted_x, popt[0], popt[1])

    # Plotting
    plt.plot(fitted_x, fitted_y, color='black', label=r"Fit $T_{\mathrm{hot}}=a I^b$",zorder=8)
    plt.scatter(x_pred[:, 0], t_pred, color='red', label="GP Prediction", zorder=7)
    plt.fill_between(x_ss[:, 0], mean + ss, mean - ss, color='red', alpha=0.2, label=r"Prediction 95 $\%$ confidence interval")
    plt.xscale("log")
    plt.yscale("log")
    plt.yticks([1e2, 2e2, 5e2], ["100", "200", "500"])
    plt.xticks([1e17, 2e17, 5e17, 1e18, 2e18, 5e18, 1e19], 
               [r"$1\times 10^{17}$", r"$2\times10^{17}$", r"$5\times 10^{17}$", 
                r"$1 \times 10^{18}$", r"$2 \times 10^{18}$", r"$5 \times 10^{18}$", r"$1\times 10^{19}$"])
    plt.xlabel(r"$I \, [\mathrm{W \,cm^{-2}}]$")
    plt.ylabel(r"$T_{\mathrm{hot}} \, [\mathrm{keV}]$")
    plt.grid(True)
    plt.legend()

    # Add fitted parameters to the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    custom_line = Line2D([0], [0], color='white', lw=2, linestyle='--')
    handles.append(custom_line)
    label = r"$a$ = {:.5f} ± {:.5f} keV".format(popt[0], std_popt[0])
    labels.append(label)
    custom_line = Line2D([0], [0], color='white', lw=2, linestyle='--')
    handles.append(custom_line)
    label = r"$b$ = {:.3f} ± {:.3f}".format(popt[1], std_popt[1])
    labels.append(label)
    plt.legend(handles=handles, labels=labels)

    # Set x-axis limits and show the plot
    plt.xlim(10**16.9, 10**19.1)
    plt.ylim(50, 10**3)
    plt.savefig(PATH_TO_FIGURES + "cui_compare1.pdf")
    plt.show()
