import os
import sys
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)
PATH_TO_MODELS = PATH_TO_PROJECT + 'models/models/'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from dataset.Dataset import DatasetUtils
from draw_dataset import draw, draw_slice
from transformer import Transformer, transform
from prediction_grid import PredictionGrid

from gp3d import Gp3d
from gp3d_gpy import Gp3dGpy

TRAIN, TEST = 0, 1


def load_data():
    dataset = DatasetUtils.load_final_dataset('dataset')
    x, y = DatasetUtils.dataset_to_tensors(dataset)
    x, y = np.array(x), np.array(y)[:,0]    # Only the first output - T_HOT, the second is T_HOT_STDEV
    return x, y

class Options:
    def __init__(self):
        self.action = TEST
        self.int_factor = 10
        self.length_factor = 1
        self.alpha_factor = 1
        

if __name__ == "__main__":
    options = Options()
    x, y = load_data()
    x, y, transformer = transform(x, y, factor_i=options.int_factor)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # Initialize the models
    
    gpGpy = Gp3dGpy()
    
    if options.action == TRAIN:
        gpGpy.train(X_train, y_train)
        gpGpy.save(PATH_TO_MODELS + 'gp3dgpy_model.pkl')
        
    elif options.action == TEST:
        gpGpy.load(PATH_TO_MODELS + 'gp3dgpy_model.pkl')
        
    # Test scores
    gp_r2, gp_rmse = gpGpy.score(X_test, y_test)
    print(f"GP \t\t R2: {gp_r2:.3f} \t RMSE: {gp_rmse:.3f}")
    
    # Residual analysis
    gp_mean_of_residuals = gpGpy.mean_residuals(X_test, y_test)
    print("GP residuals mean: ", gp_mean_of_residuals)
    
    # Predictions
    prediction_grid = PredictionGrid(transformer)
    gp_prediction = gpGpy.predict(X_test)

    