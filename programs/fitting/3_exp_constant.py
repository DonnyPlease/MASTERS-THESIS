import sys, os
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import numpy as np

# CONSTANTS
FOLDER_NAME = "data/"
PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/'+FOLDER_NAME+'spectra/'

FOLDER_SAVE_3EXP = "histograms/"+FOLDER_NAME+"/fitting_results/3exp_plots/"
FOLDER_SAVE_2EXP = "histograms/"+FOLDER_NAME+"/fitting_results/2exp_plots/"
FOLDER_SAVE_2EXP_WC = "histograms/"+FOLDER_NAME+"/fitting_results/2exp_wc_plots/"
FOLDER_SAVE_3EXP_WC = "histograms/"+FOLDER_NAME+"/fitting_results/3exp_wc_plots/"
FOLDER_SAVE_NLSQ = "histograms/"+FOLDER_NAME+"/fitting_results/nlsq_plots/"
FOLDER_SAVE_RESULTS = "histograms/"+FOLDER_NAME+"/fitting_results/fit_params/"

from histograms.HistogramUtils import load_histograms
from fitting.FittingUtils import fit_hot_temperature, plot_fit, reverse_normalize, print_fit_results, save_fit_results, _fit_using_three_exponentials_with_bias, find_new_start_index

if __name__ == "__main__":
    # 1. Load all histograms from the folder
    histograms = load_histograms(PATH_TO_HISTOGRAMS)
    
    # 2. For each histogram, perform the algorithm and save the result to the new folder and to the dataset
    count_failed_fits = 0
    for histogram in histograms:
        
        # 1. NORMALIZE THE HISTOGRAM
        max_hist = np.max(histogram[0])
        histogram = (histogram[0], histogram[1]/max_hist, histogram[2])
        
        start, end = 0, len(histogram[0])-1
        results = _fit_using_three_exponentials_with_bias(histogram)
        if results is None:
            count_failed_fits += 1
            continue
        results["3exp"] = results
        new_start_index = find_new_start_index(histogram, start, end, results["3exp"])
        results["histogram"] = histogram[2] 
        plot_fit(histogram, results, fit_type="3exp", save_path=FOLDER_SAVE_3EXP_WC)
        print(results)
    
    print("Failed fits: ", count_failed_fits)