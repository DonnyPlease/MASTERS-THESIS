import os
import sys
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

from dataset.Dataset import DatasetUtils
from draw_dataset import draw_dataset

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import joblib
import matplotlib.pyplot as plt
from draw_dataset import draw, draw_slice
from transformer import Transformer, transform
from prediction_grid import PredictionGrid

DATASET_FOLDER = 'dataset'
PATH_TO_MODEL = PATH_TO_PROJECT + 'models/models/gp3d_model.pkl'

TRAIN, TEST = 0, 1

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def non_negative(x):
    return np.array([max(0, t) for t in x])


class Gp3d():
    def __init__(self):
        self.model = None
        self.params = {"C" : 0.15, "l" : 0.5}
        return
    
    def train(self, x, y, C_range=None, l_range=None):
        print("Training the gp3d model...")
        self.best_params = self.optimize_using_crosval(x, y, C_range=C_range, l_range=l_range)
        self.params["C"], self.params["l"] = self.best_params
        
        # Fit the model
        kernel = C(constant_value=self.params["C"], constant_value_bounds="fixed")*RBF(self.params["l"], length_scale_bounds='fixed')
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, normalize_y=True, alpha=1e-8)
        self.model.fit(x, y)
        
        print("GP3d Model trained wtih best parameters: ", self.best_params)
        return
    
    def set_params(self, C, l):
        self.params["C"], self.params["l"] = C, l
        return
    
    def train_with_custom_params(self, x, y, C, l):
        kernel = C(constant_value=C, constant_value_bounds="fixed")*RBF(l, length_scale_bounds='fixed')
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, normalize_y=True, alpha=1e-8)
        self.model.fit(x, y)
        return
    
    def predict(self, x, return_std=False):
        y, ss = self.model.predict(x, return_std=True)
        y = non_negative(y)
        if return_std:
            return y, ss
        return y
    
    def residuals(self, x, y):
        return y - self.model.predict(x)
    
    def mean_residual(self, x, y):
        return np.mean(self.residuals(x, y))
    
    def plot_histogram_of_residuals(self, x, y):
        residuals = self.residuals(x, y)
        plt.hist(residuals, bins=10)
        plt.show()
        return
    
    def save(self, path):
        joblib.dump(self.model, path)
        print("GP3d Model saved")
        return
    
    def load(self, path):
        self.model = joblib.load(path)
        print("GP3d Model loaded")
        return
    
    def optimize_using_crosval(self, X, y, C_range=None, l_range=None):
        params = {'C': np.logspace(-6, 0, 10) if C_range is None else C_range, 
                  'l': np.linspace(0.01, 2, 20) if l_range is None else l_range}

        kf = KFold(n_splits=10, shuffle=True, random_state=41)
        overall_R2_scores, overall_rmse_scores = [], []
        i = 0
        for c in params['C']:
            l_R2_scores, l_rmse_scores = [], []
            for l in params['l']:
                kv_R2_scores, kv_rmse_scores = [], []

                for train_index, val_index in kf.split(X):
                    X_train_fold, X_val_fold = X[train_index], X[val_index]
                    y_train_fold, y_val_fold = y[train_index], y[val_index]

                    kernel = C(constant_value=c, constant_value_bounds="fixed")*RBF(l, length_scale_bounds='fixed')
                    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e1)
                    gp.fit(X_train_fold, y_train_fold)
                    kv_R2_scores.append(gp.score(X_val_fold, y_val_fold))
                    kv_rmse_scores.append(rmse(y_val_fold, gp.predict(X_val_fold)))
                l_R2_scores.append(np.mean(kv_R2_scores))
                l_rmse_scores.append(np.mean(kv_rmse_scores))
            overall_R2_scores.append(l_R2_scores)
            overall_rmse_scores.append(l_rmse_scores)
            print(f"Finished {i+1}/{len(params['C'])}")
            i += 1

        overall_R2_scores = np.array(overall_R2_scores)
        best_R2_C, best_R2_l = np.unravel_index(overall_R2_scores.argmax(), overall_R2_scores.shape)
        print("Best R2 score: ", overall_R2_scores[best_R2_C, best_R2_l], "at positions", best_R2_C, best_R2_l)

        overall_rmse_scores = np.array(overall_rmse_scores)
        best_rmse_C, best_rmse_l = np.unravel_index(overall_rmse_scores.argmin(), overall_rmse_scores.shape)
        print("Best RMSE: ", overall_rmse_scores[best_rmse_C, best_rmse_l], "at positions", best_rmse_C, best_rmse_l)

        return params['C'][best_R2_C], params['l'][best_R2_l]    
    
    def score(self, x, y):
        return self.model.score(x, y) , rmse(y, self.model.predict(x))


if __name__ == '__main__':
    # OPTIONS 
    ACTION = TRAIN
    int_factor = 1 
    
    x, y = DatasetUtils.load_data_for_regression(DATASET_FOLDER)
    x, y = np.array(x), np.array(y)[:,0]    # Only the first output column - T_HOT, the second is T_HOT_STDEV
    X, y, transformer = transform(x, y, factor_i=int_factor)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=41)
   
    # Train or load the model 
    gp = Gp3d()
    if ACTION == TRAIN:
        gp.train(X_train, y_train)
        gp.train_with_custom_params(X_train, y_train, 0.25, 0.5)
        gp.save(PATH_TO_MODEL)
    elif ACTION == TEST:
        gp.load(PATH_TO_MODEL)
    
    # Test scores
    gp_r2, gp_rmse = gp.score(X_test, y_test)
    print(f"GP \t\t R2: {gp_r2:.2f} \t RMSE: {gp_rmse:.2f}")
    
    # Residual analysis
    gp_mean_of_residuals = gp.mean_residual(X_test, y_test)
    print("GP residuals mean: ", gp_mean_of_residuals)
    
    
    # Predict
    prediction_grid = PredictionGrid(transformer=transformer, factor_i=int_factor)
    grid = prediction_grid.grid_for_prediction()
    t_hot_predicted, ss = gp.predict(grid, return_std=True)
    t_hot_predicted = t_hot_predicted.reshape(prediction_grid.X.shape)
    ss = ss.reshape(prediction_grid.X.shape)
    
    # Plot the predictions 
    i_grid, l_grid, a_grid = prediction_grid.grid_for_plotting()
    slices_at = [0, 15, 25, 35, 40, 49]
    for slice_at in slices_at:
        draw_slice(i_grid, l_grid, a_grid, t_hot_predicted, slice_at=slice_at, axes=["l","a"])
        draw_slice(i_grid, l_grid, a_grid, ss, slice_at=slice_at, axes=["l","a"])