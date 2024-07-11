import os
import sys
sys.path.append('C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/')

from dataset.Dataset import DatasetUtils
from draw_dataset import draw_dataset

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import matplotlib.pyplot as plt
from draw_dataset import draw
from transformer import Transformer

TRAIN, TEST = 0, 1

def optimize_using_crosval(X, y):
    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    kernel_parameters = np.logspace(-10, -7, 100)
    kf = KFold(n_splits=5, shuffle=True, random_state=41)
    mean_scores = []

    for param in kernel_parameters:
        scores = []
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
            kr = KernelRidge(alpha=param, kernel='rbf')
            kr.fit(X_train_fold, y_train_fold)
            scores.append(kr.score(X_val_fold, y_val_fold))
    
        mean_scores.append(np.mean(scores))
    
    best_param = kernel_parameters[np.argmax(mean_scores)]
    return best_param

if __name__ == '__main__':
    int_factor = 10
    # Load the data
    
    ACTION = TRAIN
    
    # Load the data
    dataset = DatasetUtils.load_final_dataset('dataset')
    x, y = DatasetUtils.dataset_to_tensors(dataset)
    x, y = np.array(x), np.array(y)[:,0]    # Only the first output - T_HOT, the second is T_HOT_STDEV

    transformer = Transformer()
    transformer.fit(x)
    X = transformer.transform(x, factor_i=int_factor)

    mean_t_hot = np.mean(y)
    t_hot = np.array(y) 

    best_param = optimize_using_crosval(X, y)
    print("Best kernel parameter: ", best_param)
    # Fit the model
    kr = KernelRidge(alpha=best_param, kernel="rbf")
    
    kr.fit(X, y)

    # Create the grid for prediction
    i_grid = np.linspace(0, int_factor, 50)
    l_grid  = np.linspace(0, 1, 50)
    a_grid = np.linspace(0, 1, 50)
    X, Y, Z = np.meshgrid(i_grid, l_grid, a_grid)
    grid = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    
    # Predict
    t_hot_predicted = kr.predict(grid)
    t_hot_predicted = t_hot_predicted.reshape(X.shape)
    
    i_grid = transformer.reverse_transform_i(i_grid, factor=int_factor)
    l_grid = transformer.reverse_transform_l(l_grid, factor=1)
    a_grid = transformer.reverse_transform_a(a_grid, factor=1)
    final_values = None
    x_grid = None
    y_grid = None
    
    # OPTIONS
    DRAW_AXES = ["l","a"]
    slice_at = 0
    draw_values = t_hot_predicted
    
    print("Drawing for axes: ", DRAW_AXES)
    if DRAW_AXES == ["l","a"]:
        final_values = draw_values[:,slice_at,:]
        final_values = final_values.T
        x_grid = l_grid
        y_grid = a_grid
        print("\t at I = ", i_grid[slice_at])
    elif DRAW_AXES == ["i","l"]:
        final_values = draw_values[slice_at,:,:]
        final_values = final_values.T
        x_grid = i_grid
        y_grid = l_grid
        print("\t at alpha = ", a_grid[slice_at])
    elif DRAW_AXES == ["i","a"]:
        final_values = draw_values[:,:,slice_at]
        x_grid = i_grid
        y_grid = a_grid
        print("\tat l = ", l_grid[slice_at])
    
    draw(x_grid, y_grid, final_values, axes=DRAW_AXES)
