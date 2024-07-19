import sys, os
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import numpy as np

from fitting.FittingUtils import load_fit_results, print_fit_results

# CONSTANTS
FOLDER_NAME = "data/"
PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/'+FOLDER_NAME+'spectra/'

FOLDER_SAVE_3EXP = "histograms/"+FOLDER_NAME+"/fitting_results/3exp_plots/"
FOLDER_SAVE_2EXP = "histograms/"+FOLDER_NAME+"/fitting_results/2exp_plots/"
FOLDER_SAVE_NLSQ = "histograms/"+FOLDER_NAME+"/fitting_results/nlsq_plots/"
FOLDER_SAVE_RESULTS = "histograms/"+FOLDER_NAME+"/fitting_results/fit_params/"

if __name__ == "__main__":
    fit_results = []
    for file in os.listdir(FOLDER_SAVE_RESULTS):
        fit_results.append(load_fit_results(FOLDER_SAVE_RESULTS+file))
    
    failed_counts = {"3exp": 0, "2exp": 0, "nlsq": 0}
    failed_fits = []
    for fit in fit_results:
        if fit["3exp"] is  None:
            failed_fits.append(fit["histogram"]["filename"])
            failed_counts["3exp"] += 1
        if fit["2exp"] is  None:
            failed_fits.append(fit["histogram"]["filename"])
            failed_counts["2exp"] += 1
        if fit["nlsq"] is  None:
            failed_fits.append(fit["histogram"]["filename"])
            failed_counts["nlsq"] += 1
    
    print(failed_counts)
    print(failed_fits)