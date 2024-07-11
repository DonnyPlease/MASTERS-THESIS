import os
import sys
sys.path.append('C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/')

from dataset.Dataset import DatasetUtils
from draw_dataset import draw_dataset

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
from draw_dataset import draw
from transformer import Transformer


# Load the data
dataset = DatasetUtils.load_final_dataset('dataset')
data = DatasetUtils.dataset_to_dict(dataset)
data_sliced = np.array(data['1e19'])

transformer = Transformer()
x_temp = np.array([np.ones_like(data_sliced[:,0]),data_sliced[:,0],data_sliced[:,1]]).T
transformer.fit(x_temp)

lengths = transformer.transform_l(data_sliced[:,0])
alphas = transformer.transform_a(data_sliced[:,1])

mean_t_hot = np.mean(data_sliced[:,2])
t_hot = np.array(data_sliced[:,2]) 

# Fit the model
X = np.array([lengths, alphas]).T
y = t_hot
kernel = RBF(1, length_scale_bounds='fixed')
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, normalize_y=True)
gp.fit(X, y)

# Predict z based on x,y grid
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
grid = np.array([X.flatten(), Y.flatten()]).T

t_hot_predicted, ss = gp.predict(grid, return_std=True)

t_hot_predicted = t_hot_predicted.reshape(X.shape)

ss = ss.reshape(X.shape)

x = transformer.reverse_transform_l(x)
y = transformer.reverse_transform_a(y)
draw(x,y,t_hot_predicted, axes=["l","a"])
