import numpy as np
import matplotlib.pyplot as plt

def k(a,b):
    return np.exp(-0.5 * np.subtract.outer(a, b)**2)

x = np.linspace(0, 10, 100)
K = k(x, x)
m = np.zeros(100)
samples = np.random.multivariate_normal(m, K, 5)
plt.plot(x, samples.T)
plt.xlabel("x")
plt.ylabel("y")
plt.show()  