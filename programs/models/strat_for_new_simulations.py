import os
import sys
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)
PATH_TO_MODELS = PATH_TO_PROJECT + 'models/models/'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import json

from dataset.Dataset import DatasetUtils
from draw_dataset import draw, draw_slice
from transformer import Transformer, transform
from models.prediction_grid import PredictionGrid

from models.gp3d_gpy import Gp3dGpy as GP
from models.gp3d_gpy import Gp3dGpy

from models.compare_models import Options, select_model, load_data

TRAIN, TEST = 0, 1


if __name__ == "__main__":
    x, y = load_data()

    model = GP.load(PATH_TO_MODELS + "gp_model.pkl")
    transformer = model.transformer
    prediction_grid = PredictionGrid(transformer)
    
    x_pred = prediction_grid.grid_for_prediction()
    t_pred, t_stdev = model.predict(x_pred, return_std=True)
    
    # Find 50 points with the highest standard deviation
    indices = np.argsort(t_stdev)[-50:]
    x_pred = x_pred[indices]
    t_pred = t_pred[indices]
    t_stdev = t_stdev[indices]
    
    x_pred = transformer.reverse_transform(x_pred)
    
    for i in range(50):
        # print using nice float formatting, names of variables and tabs
        print(f"I = {x_pred[i, 0]:.2e}\tL = {x_pred[i, 1]:.2f}\tA = {x_pred[i, 2]:.2f}\tT = {t_pred[i]:.2f}\tStdev = {t_stdev[i]:.2f}")
    
    