import os
import sys
sys.path.append('C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/')


import pykrige
import numpy as np
import matplotlib.pyplot as plt

from dataset.Dataset import DatasetUtils
from draw_dataset import draw

# Load the data
dataset = DatasetUtils.load_final_dataset('dataset')
data = DatasetUtils.dataset_to_dict(dataset)
data17 = np.array(data['1e17'])

lengths = np.array(data17[:,0])/5.0
alphas = np.array(data17[:,1])/60.0
t_hot = np.array(data17[:,2])

# Transform lengths to log scale
lengths_log = np.log10(lengths)

# Transform lengths_log to 0-1 scale
lengths_log_scaled = (lengths_log - np.min(lengths_log))/(np.max(lengths_log) - np.min(lengths_log))


# Plot the data
plt.scatter(lengths_log_scaled, alphas, c=t_hot, marker='.')
plt.colorbar()
plt.show()


UK = pykrige.uk.UniversalKriging(lengths_log_scaled, alphas, t_hot, weight=True, nlags=20, variogram_model='gaussian', verbose=True, enable_plotting=True)
print(UK.variogram_model_parameters)
gridx = np.linspace(0, 1, 100)
gridy = np.linspace(0, 1, 100)
zstar, ss = UK.execute('grid', gridx, gridy)

# Transform gridx with the inverse of the log scale
gridx = gridx*(np.max(lengths_log) - np.min(lengths_log)) + np.min(lengths_log)

print(zstar)

print(gridx.shape)
print(gridy.shape)
print(zstar.shape)
print(zstar)
draw(10**gridx*5.0, gridy*60.0, zstar)
draw(10**gridx*5.0, gridy*60.0, np.sqrt(ss))

