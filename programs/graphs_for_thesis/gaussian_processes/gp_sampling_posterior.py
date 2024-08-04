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

def k(a, b):
    return np.exp(-0.5 * np.subtract.outer(a, b)**2)

def k_matern32(a,b):
    return (1 + np.sqrt(3) * np.abs(np.subtract.outer(a, b))) * np.exp(-np.sqrt(3) * np.abs(np.subtract.outer(a, b)))

def k_matern52(a,b):
    return (1 + np.sqrt(5) * np.abs(np.subtract.outer(a, b)) + 5/3 * np.subtract.outer(a, b)**2) * np.exp(-np.sqrt(5) * np.abs(np.subtract.outer(a, b)))

x = np.linspace(0, 10, 100)
x_train = np.array([1, 3, 5, 7, 9])
y_train = np.array([2, 2, 3, 4, 2])

K = k(x, x)
K_train = k(x_train, x_train) + 1e-6 * np.eye(len(x_train))
K_train_inv = np.linalg.inv(K_train)

K_x = k(x, x_train)
mu_post = K_x @ K_train_inv @ y_train
cov_post = K - K_x @ K_train_inv @ K_x.T

std_dev = np.sqrt(np.diag(cov_post))

samples = np.random.multivariate_normal(mu_post, cov_post, 5)
plt.scatter(x_train, y_train, color='black', zorder=5)
plt.plot(x, samples.T, zorder=1)
plt.fill_between(x, mu_post - std_dev, mu_post + std_dev, color='gray', alpha=0.3, zorder=2)
plt.grid(True, zorder=0)
plt.xlabel("x")
plt.ylabel("y")
plt.show()