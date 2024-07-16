import sys, os
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

# CONSTANTS
PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/data/spectra/'

from histograms.HistogramUtils import load_histograms
from fitting.FittingUtils import fit_hot_temperature

if __name__ == "__main__":
    # 1. Load all histograms from the folder
    histograms = load_histograms(PATH_TO_HISTOGRAMS)
    
    # 2. For each histogram, perform the algorithm and save the result to the new folder and to the dataset
    count_failed_fits = 0
    for histogram in histograms:
        fit_results = fit_hot_temperature(histogram)
        if fit_results is None:
            count_failed_fits += 1
        # Save the result to the new folder
        # Save the result to the dataset
    
    print("Number of failed fits: " + str(count_failed_fits))
    
