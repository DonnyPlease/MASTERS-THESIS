import sys, os
IMPORT_PATH = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(IMPORT_PATH)

from fitting.helpful_functions import load_folder_names, load_histogram
from fitting.helpful_functions import create_filenames, create_filename
from optimizer import ParameterOptimizer
from optimizer import scan_fit_double_exp, variances_of_parameters

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize


if __name__ == '__main__':
    folder_names = load_folder_names("old_data/params.txt")
    print(folder_names[310])
    bins, counts = load_histogram("old_data/moved_histograms/trimmed_histograms/"+folder_names[310])
    best_params_from_grid = scan_fit_double_exp(bins, counts)
    optimizer = ParameterOptimizer(bins, counts)
    optimizer.optimize_double_exp(initial_guess=best_params_from_grid)
    optimizer.print_params()
    optimizer.print_temperatures()
    variances = variances_of_parameters(bins, optimizer.params, optimizer.get_variance_estimate())
    
    # Print parameters with variances nicely formatted
    for i in range(len(optimizer.params)):
        print(f"{optimizer.params[i]:.2e} +/- {variances[i]:.2e}")
        
    # Print temperatures with variances nicely formatted with proper error propagation
    print("Temperatures with variances:")
    for i in range(len(optimizer.get_temperatures())):
        temp = optimizer.get_temperatures()[i]
        var = variances[(i)*2+2]*temp**2
        print(f"{temp:.7f} +/- {var:.7f}")
    
    
    