import os
import sys
sys.path.append('C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/')

from dataset.Dataset import DatasetUtils
from draw_dataset import draw_dataset

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt


# Load the data
dataset = DatasetUtils.load_final_dataset('dataset')
data = DatasetUtils.dataset_to_dict(dataset)
data17 = np.array(data['1e17'])
lengths = np.array(data17[:,0])/5
alphas = np.array(data17[:,1])/60
t_hot = np.array(data17[:,2])

# Plot the data
plt.scatter(lengths, alphas, c=t_hot, marker='.')
# plt.colorbar()
plt.xscale('log')

# Fit the model
X = np.array([lengths, alphas]).T
y = t_hot
kernel = RBF(0.01, (1e-3, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
gp.fit(X, y)

# Predict z based on x,y grid
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([X.flatten(), Y.flatten()]).T
Z, _ = gp.predict(Z, return_std=True)
Z = Z.reshape(X.shape)

# Plot the prediction
cax = plt.imshow(Z, extent=(0,1,0,1), origin='lower')
plt.scatter(lengths, alphas, c='k',marker='.')
plt.colorbar(cax)
plt.xscale('log')

plt.show()






