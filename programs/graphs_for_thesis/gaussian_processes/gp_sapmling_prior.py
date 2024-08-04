import numpy as np
import matplotlib.pyplot as plt

# Configure Matplotlib to use LaTeX for rendering text
from matplotlib import rc
rc('font', family='serif', serif='Computer Modern')
rc('text', usetex=True)
rc('font', size=18)          # controls default text sizes
rc('axes', titlesize=20)     # fontsize of the axes title
rc('axes', labelsize=16)     # fontsize of the x and y labels
rc('xtick', labelsize=16)    # fontsize of the tick labels
rc('ytick', labelsize=16)    # fontsize of the tick labels
rc('legend', fontsize=16)    # legend fontsize
rc('figure', titlesize=20)   # fontsize of the figure title

def k(a,b):
    return np.exp(-0.5 * np.subtract.outer(a, b)**2)

x = np.linspace(0, 10, 100)
K = k(x, x)
m = np.zeros(100)
samples = np.random.multivariate_normal(m, K, 5)
plt.plot(x, samples.T,zorder=5)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, zorder=0)
plt.show()  