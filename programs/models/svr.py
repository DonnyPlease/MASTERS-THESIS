import sys, os
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from sklearn.svm import SVR

from dataset.Dataset import DatasetUtils
from draw_dataset import draw, draw_slice
from transformer import Transformer, transform
from models.prediction_grid import PredictionGrid

DATASET_FOLDER = 'dataset'
PATH_TO_MODEL = PATH_TO_PROJECT + 'models/models/svr_model.pkl'

TRAIN, TEST = 0, 1


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def non_negative(x):
    return np.array([max(0, t) for t in x])

class Svr():
    def __init__(self):
        self.model = None
        self.params = {"C" : 0.15, "l" : 0.5}
        self.transformer = None
        return
    
    def train(self, x, y, C_range=None, gamma_range=None):
        param_grid = {
            'C': [10, 100, 1000, 5000],
            'epsilon': [0.0002,0.001, 0.005,0.01,0.05],
            'gamma': [1.0, 10, 100]  # Adjust gamma for smoothness
        }
        grid_search = GridSearchCV(SVR(epsilon=0.001, kernel="rbf"), param_grid, cv=7, scoring='neg_root_mean_squared_error', verbose=1)
        grid_search.fit(x, y)
        self.model = grid_search.best_estimator_
        print("Best parameters: ", grid_search.best_params_)
        self.best_params = grid_search.best_params_
        return
    
    def set_params(self, C, gamma):
        self.params["C"], self.params["gamma"] = C, gamma
        return
    
    def train_with_custom_params(self, x, y, C, gamma):
        self.model = SVR(C=C, gamma=gamma)
        self.model.fit(x, y)
        return
    
    def predict(self, x):
        y = self.model.predict(x)
        y = non_negative(y)
        return y
    
    def residuals(self, x, y):
        return y - self.model.predict(x)
    
    def scores(self, y_true, y_pred):
        # caclulate the R2 score as 1 - SS_res / SS_tot
        y_pred = y_pred.flatten()
        ss_res = np.sum((y_true - y_pred) ** 2)
        mean_y = np.mean(y_true)
        ss_tot = np.sum((y_true - mean_y) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # calculate the RMSE score
        rmse_score = rmse(y_true, y_pred)
        return r2, rmse_score
    
    def mean_residuals(self, y_true, y_pred):
        return np.mean(y_true - y_pred)
    
    def plot_histogram_of_residuals(self, x, y):
        residuals = self.residuals(x, y)
        plt.hist(residuals, bins=10)
        plt.show()
        return
    
    def save(self, path):
        joblib.dump(self, path)
        return
    
    @staticmethod
    def load(path):
        return joblib.load(path)
    
    def score(self, x, y):
        r2 = self.model.score(x, y)
        rmse = np.sqrt(np.mean((y - self.model.predict(x))**2))
        return r2, rmse
       
    def set_transformer(self, transformer):
        self.transformer = transformer
        return
    
    def get_params(self):
        return self.best_params
     
if __name__ == '__main__':
    # OPTIONS
    int_factor = 1
    ACTION = TEST
    
    # Load the data
    x, y = DatasetUtils.load_data_for_regression(DATASET_FOLDER)
    x, y = np.array(x), np.array(y)[:,0]    # Only the first output column - T_HOT, the second is T_HOT_STDEV
    X, y, transformer = transform(x, y, factor_i=int_factor)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
   
    # Train or load the model 
    svr = Svr()
    if ACTION == TRAIN:
        svr.train(X_train, y_train)
        svr.set_transformer(transformer)
        svr.save(PATH_TO_MODEL)
    elif ACTION == TEST:
        svr = svr.load(PATH_TO_MODEL)
        
    # Test scores
    svr_r2, svr_rmse = svr.score(X_test, y_test)
    print(f"SVR \t\t R2: {svr_r2:.2f} \t RMSE: {svr_rmse:.2f}")
    
    # Residual analysis
    svr_mean_of_residuals = svr.mean_residual(X_test, y_test)
    print("SVR residuals mean: ", svr_mean_of_residuals)
        
    # Predict
    prediction_grid = PredictionGrid(transformer=transformer, factor_i=int_factor,count_i=51)
    grid = prediction_grid.grid_for_prediction()
    t_hot_predicted = svr.predict(grid)
    t_hot_predicted = t_hot_predicted.reshape(prediction_grid.X.shape)
    
    # Plot the predictions 
    i_grid, l_grid, a_grid = prediction_grid.grid_for_plotting()
    slices_at = [0, 25, 50]
    for slice_at in slices_at:
        draw_slice(i_grid, l_grid, a_grid, t_hot_predicted, slice_at=slice_at, axes=["l","a"])