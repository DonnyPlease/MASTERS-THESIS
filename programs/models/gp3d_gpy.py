import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import GPy
import GPyOpt

# Add the project path
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

# Import custom modules
from dataset.Dataset import DatasetUtils
from draw_dataset import draw, draw_slice
from transformer import Transformer, transform
from prediction_grid import PredictionGrid

# Define paths and constants
DATASET_FOLDER = 'dataset'
PATH_TO_MODEL = PATH_TO_PROJECT + 'models/models/gp3dgpy_model.pkl'
TRAIN, TEST = 0, 1

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def non_negative(x):
    return np.array([max(0, t) for t in x])

class Gp3dGpy():
    def __init__(self):
        self.model = None
        self.params = {"C" : 0.15, "l" : 0.5}
        return
    
    def train(self, x, y, C_range=None, l_range=None):
        print("Training the gp3d model...")
        
        # Ensure y is reshaped to (n_samples, 1)
        if y.ndim == 1:
            y = y[:, None]
        
        # Define the kernel
        kernel = GPy.kern.Matern32(input_dim=x.shape[1], variance=self.params["C"], lengthscale=self.params["l"])
        
        # Initialize GP regression model
        self.model = GPy.models.GPRegression(x, y, kernel)
        self.model.optimize(messages=True)
        print(self.model)

        print("GP3d Model trained with best parameters: ", self.model.param_array)
        return
    
    def optimize_model(self, params):
        C = 10 ** params[0, 0]  # Transform to logarithmic scale for variance
        l = params[0, 1]
        kernel = GPy.kern.Matern52(input_dim=self.model.input_dim, variance=C, lengthscale=l)
        self.model.kern = kernel
        self.model.noi
        return -self.model.log_likelihood()  # Minimize negative log-likelihood

    def train_with_custom_params(self, x, y, C, l):
        kernel = GPy.kern.Matern52(input_dim=x.shape[1], variance=C, lengthscale=l)
        self.model = GPy.models.GPRegression(x, y, kernel)
        self.model.optimize()

    def set_params(self, C, l):
        self.params["C"], self.params["l"] = C, l
        return
    
    def predict(self, x, return_std=False):
        y_mean, y_var = self.model.predict(x)
        y_mean = non_negative(y_mean.flatten())
        if return_std:
            return y_mean, np.sqrt(y_var.flatten())
        return y_mean
    
    def residuals(self, x, y):
        y_pred = self.predict(x)
        return y - y_pred
    
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
        params = {'C': np.logspace(-6, 4, 10) if C_range is None else C_range, 
                  'l': np.linspace(0.01, 2, 20) if l_range is None else l_range}

        kf = KFold(n_splits=10, shuffle=True, random_state=41)
        overall_R2_scores, overall_rmse_scores = [], []
        i = 0
        for c in params['C']:
            l_R2_scores, l_rmse_scores = [], []
            for l in params['l']:
                kv_R2_scores, kv_rmse_scores = []

                for train_index, val_index in kf.split(X):
                    X_train_fold, X_val_fold = X[train_index], X[val_index]
                    y_train_fold, y_val_fold = y[train_index], y[val_index]

                    kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=c, lengthscale=l)
                    gp = GPy.models.GPRegression(X_train_fold, y_train_fold[:, None], kernel)
                    gp.optimize(messages=False)
                    kv_R2_scores.append(gp.score(X_val_fold, y_val_fold[:, None]))
                    kv_rmse_scores.append(rmse(y_val_fold, gp.predict(X_val_fold)[0].flatten()))
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
        # predict the values
        y_pred, _ = self.model.predict(x)
        
        # caclulate the R2 score as 1 - SS_res / SS_tot
        y_pred = y_pred.flatten()
        ss_res = np.sum((y - y_pred) ** 2)
        mean_y = np.mean(y)
        ss_tot = np.sum((y - mean_y) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # calculate the RMSE score
        rmse_score = rmse(y, y_pred)
        return r2, rmse_score

if __name__ == '__main__':
    # OPTIONS 
    ACTION = TEST
    int_factor = 1 
    
    x, y = DatasetUtils.load_data_for_regression(DATASET_FOLDER)
    x, y = np.array(x), np.array(y)[:,0]    # Only the first output column - T_HOT, the second is T_HOT_STDEV
    X, y, transformer = transform(x, y, factor_i=int_factor)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=41)
    
    # Train or load the model 
    gp = Gp3dGpy()
    if ACTION == TRAIN:
        gp.train(X_train, y_train)
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
        draw_slice(i_grid, l_grid, a_grid, t_hot_predicted, slice_at=slice_at, axes=["i","l"])
        draw_slice(i_grid, l_grid, a_grid, ss, slice_at=slice_at, axes=["i","l"])
