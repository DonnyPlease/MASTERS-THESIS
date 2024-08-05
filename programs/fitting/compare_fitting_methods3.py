import sys, os
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import numpy as np

from fitting.FittingUtils import load_fit_results, print_fit_results

# CONSTANTS
FOLDER_NAME = "data/"
PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/'+FOLDER_NAME+'spectra/'


FOLDER_SAVE_SCANNING = "histograms/"+FOLDER_NAME+"alternative_fitting/results/"
FOLDER_SAVE_3EXP = "histograms/"+FOLDER_NAME+"fitting_results/fit_params/"

if __name__ == "__main__":
    # 1. LOAD ALL FIT_RESULTS FROM THE FOLDER
    fit_results = {}
    for file in os.listdir(FOLDER_SAVE_3EXP):
        results = load_fit_results(FOLDER_SAVE_3EXP+file)
        fit_results[results["histogram"]["filename"]] = {"3exp": results["3exp"]}
        if "true_params" not in results:
            continue
        fit_results[results["histogram"]["filename"]]["true_params"] = results["true_params"]
        
    for file in os.listdir(FOLDER_SAVE_SCANNING):    
        results_scan = load_fit_results(FOLDER_SAVE_SCANNING+file)
        fit_results[results_scan["histogram"]["filename"]]["scan"] = results_scan["best_linear_fit"]
    
    # 2. EXTRACT T_HOT FROM THE FIT_RESULTS AND CALCULATE RELATIVE ERROR
    for folder, results in fit_results.items():

        if "3exp" not in results or "scan" not in results or results["3exp"] is None or results["scan"] is None:
            continue
        relative_error = (results["3exp"]["t_hot"] - results["scan"]["t_hot"])/results["scan"]["t_hot"]
        results["relative_error"] = relative_error
        
        

    # 5. PLOT THE RELATIVE ERRORS AS A FUNCTION OF TRUE_T_HOT
    t_hot_array = []
    for folder, results in fit_results.items():
        
        if "3exp" not in results or "scan" not in results or results["3exp"] is None or results["scan"] is None:
            continue
        
        t_hot_array.append({ "relative_error": results["relative_error"], "scan": results["scan"]["t_hot"]})
    
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # ax.scatter(x, [t_hots["3exp"] for t_hots in t_hot_array], label="3exp")
    ax.scatter([t_hots["scan"] for t_hots in t_hot_array], [t_hots["relative_error"] for t_hots in t_hot_array], s=10, zorder=3, marker="x")
    ax.set_xlabel(r"$T_\mathrm{hot,scan}$ [keV]")
    ax.set_ylabel(r"$RD_\mathrm{3exp,scan}$")
    ax.grid(zorder=0)
    plt.show()
    
