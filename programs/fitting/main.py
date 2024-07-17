import sys, os
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import numpy as np

# CONSTANTS
PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/data/spectra/'

from histograms.HistogramUtils import load_histograms
from fitting.FittingUtils import fit_hot_temperature, plot_fit, reverse_normalize, print_fit_results, save_fit_results

if __name__ == "__main__":
    # 1. Load all histograms from the folder
    histograms = load_histograms(PATH_TO_HISTOGRAMS)
    
    # 2. For each histogram, perform the algorithm and save the result to the new folder and to the dataset
    count_failed_fits = 0
    for histogram in histograms:
        max_hist = np.max(histogram[1])
        
        # 1. NORMALIZE THE HISTOGRAM
        histogram = (histogram[0], histogram[1]/max_hist, histogram[2])
        
        # 2. FIT THE HISTOGRAM
        fit_results = fit_hot_temperature(histogram)
        
        # 3. REVERSE THE NORMALIZATION
        histogram, fit_results = reverse_normalize(histogram, fit_results, max_hist)
        fit_results["histogram"] = histogram[2]
        print_fit_results(fit_results)
        save_fit_results(fit_results)
        # 3. PLOT THE FITS
        plot_fit(histogram, fit_results, fit_type="3exp")
        plot_fit(histogram, fit_results, fit_type="2exp")
        plot_fit(histogram, fit_results, fit_type="nlsq")
        
        # 4. SAVE THE FIT RESULTS TO DATASET
        
        print("\n")
    print("Number of failed fits: " + str(count_failed_fits))
    
