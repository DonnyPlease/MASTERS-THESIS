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
from prediction_grid import PredictionGrid

from gp3d_gpy import Gp3dGpy
from pytorch_first_model import NNModel, Model
from svr import Svr

TRAIN, TEST = 0, 1

MODELS_NAMES = ['nn', 'gp', 'svr']

def select_model(model_name):
    if model_name == 'gp':
        return Gp3dGpy()
    elif model_name == 'svr':
        return Svr()
    elif model_name == 'nn':
        return NNModel()

def load_data():
    dataset = DatasetUtils.load_final_dataset('dataset')
    x, y = DatasetUtils.dataset_to_tensors(dataset)
    x, y = np.array(x), np.array(y)[:,0]    # Only the first output - T_HOT, the second is T_HOT_STDEV
    return x, y

class Options:
    def __init__(self):
        self.action = TEST
        self.int_factor = 1
        self.length_factor = 1
        self.alpha_factor = 1
 
 
def k_fold_for_all_models(x, y):
    x, y, transformer = transform(x, y)
    kf = KFold(n_splits=8, shuffle=True, random_state=40)
    kf.get_n_splits(x)
    stats = {}
    for model_name in MODELS_NAMES:
        stats[model_name] = {"r2": [], "rmse": [], "mean_residuals": [], "params": []}
    
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        print(f"Fold {i+1}")
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        for model_name in MODELS_NAMES:
            model = select_model(model_name)
            model.set_transformer(transformer)
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            r2, rmse = model.scores(y_test, y_pred)
            stats[model_name]["r2"].append(r2)
            stats[model_name]["rmse"].append(rmse)
            mean_of_residuals = model.mean_residuals(y_test, y_pred)
            stats[model_name]["mean_residuals"].append(mean_of_residuals)
            stats[model_name]["params"].append(model.get_params())
        print("\n")
        
    json.dump(stats, open("models/k_fold_stats.json", "w"), indent=4)
        
if __name__ == "__main__":
    options = Options()
    x, y = load_data()
    k_fold_for_all_models(x, y)
    
    # x, y, transformer = transform(x, y)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=40)

    
    # model_name = 'gp' 
    # # Initialize the models
    # model = select_model(model_name)
    # if options.action == TRAIN:
    #     model.set_transformer(transformer)
    #     model.train(X_train, y_train)
    #     model.save(PATH_TO_MODELS + model_name + '_model.pkl')
        
    # elif options.action == TEST:
    #     model = model.load(PATH_TO_MODELS + model_name + '_model.pkl')
    
    # # Predict
    # y_pred = model.predict(X_test)
    
    # # Test scores
    # gp_r2, gp_rmse = model.scores(y_test, y_pred)
    # print(f"GP \t\t R2: {gp_r2:.3f} \t RMSE: {gp_rmse:.3f}")
    
    # # Residual analysis
    # gp_mean_of_residuals = model.mean_residuals(y_test, y_pred)
    # print("GP residuals mean: ", gp_mean_of_residuals)
    

    