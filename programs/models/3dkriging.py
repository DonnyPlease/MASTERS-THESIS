# 3D kriging model

import os
import sys
sys.path.append('C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/')


import pykrige
import numpy as np
import matplotlib.pyplot as plt

from dataset.Dataset import DatasetUtils
from draw_dataset import draw
from transformer import Transformer
import joblib

TRAIN, TEST = 0, 1


if __name__ == '__main__':
    int_factor = 100
    
    ACTION = TRAIN
    
    # Load the data
    dataset = DatasetUtils.load_final_dataset('dataset')
    data = DatasetUtils.dataset_to_tensors(dataset)
    x, y = data
    x = np.array(x)
    y = np.array(y)

    transformer = Transformer()
    transformer.fit(x)
    x_t = transformer.transform(x, factor_i=int_factor)

    UK = None
    if ACTION == TRAIN:
        # Train the model
        UK = pykrige.uk3d.UniversalKriging3D(x_t[:,0], x_t[:,1], x_t[:,2], y[:,0], nlags=30, verbose=True, enable_plotting=True, weight=True, variogram_model='gaussian')
        joblib.dump(UK, 'models/kriging/3dkriging.pkl')
    elif ACTION == TEST:
        UK = joblib.load('models/kriging/3dkriging.pkl')
    # Create the model
    
    
    # Create the grid
    gridx = np.linspace(0, int_factor, 5)
    gridy = np.linspace(0, 1, 100)
    gridz = np.linspace(0, 1, 100)

    # Calculate the kriging
    z, ss = UK.execute('grid', gridx, gridy, gridz)

    gridx = transformer.reverse_transform_i(gridx, factor=int_factor)
    gridy = transformer.reverse_transform_l(gridy)
    gridz = transformer.reverse_transform_a(gridz)
    # Plot the plot the results as slice at x = 1e17
    draw(gridy, gridz, ss[:,:,0])
