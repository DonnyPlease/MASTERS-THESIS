import sys, os
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import numpy as np

# CONSTANTS
FOLDER_NAME = "data/"
PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/'+FOLDER_NAME+'spectra/'

FOLDER_SAVE_3EXP = "histograms/"+FOLDER_NAME+"/fitting_results/3exp_plots/"
FOLDER_SAVE_2EXP = "histograms/"+FOLDER_NAME+"/fitting_results/2exp_plots/"
FOLDER_SAVE_NLSQ = "histograms/"+FOLDER_NAME+"/fitting_results/nlsq_plots/"
FOLDER_SAVE_RESULTS = "histograms/"+FOLDER_NAME+"/fitting_results/fit_params/"

from histograms.HistogramUtils import load_histograms
from fitting.FittingUtils import fit_hot_temperature, plot_fit, reverse_normalize, print_fit_results, save_fit_results

if __name__ == "__main__":
    # 1. Load all histograms from the folder
    histograms = load_histograms(PATH_TO_HISTOGRAMS)
    
    # 2. For each histogram, perform the algorithm and save the result to the new folder and to the dataset
    count_failed_fits = 0
    for histogram in histograms:
        max_hist = np.max(histogram[0])
        
        # 1. NORMALIZE THE HISTOGRAM
        histogram = (histogram[0], histogram[1]/max_hist, histogram[2])
        
        # 2. FIT THE HISTOGRAM
        fit_results = fit_hot_temperature(histogram)
        
        # 3. REVERSE THE NORMALIZATION
        histogram, fit_results = reverse_normalize(histogram, fit_results, max_hist)
        fit_results["histogram"] = histogram[2]
        print_fit_results(fit_results)
        save_fit_results(fit_results, FOLDER_SAVE_RESULTS)
        # 3. PLOT THE FITS
        plot_fit(histogram, fit_results, fit_type="3exp", save_path=FOLDER_SAVE_3EXP)
        plot_fit(histogram, fit_results, fit_type="2exp", save_path=FOLDER_SAVE_2EXP)
        plot_fit(histogram, fit_results, fit_type="nlsq", save_path=FOLDER_SAVE_NLSQ)
        
        # 4. SAVE THE FIT RESULTS TO DATASET
        
        print("\n")
    print("Number of failed fits: " + str(count_failed_fits))
    
