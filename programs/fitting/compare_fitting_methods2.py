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
        if "true_params" not in results:
            continue
        true_t_hot = results["true_params"]["t_hot"]
        
        if "3exp" not in results or "scan" not in results or results["3exp"] is None or results["scan"] is None:
            continue
        relative_error_3exp = (results["3exp"]["t_hot"] - true_t_hot)/true_t_hot
        results["3exp"]["relative_error"] = relative_error_3exp
        
        relative_error_scan = (results["scan"]["t_hot"] - true_t_hot)/true_t_hot
        results["scan"]["relative_error"] = relative_error_scan
        
        
    relative_error_3exp = np.array([results["3exp"]["relative_error"] for results in fit_results.values() if "true_params" in results and results["3exp"] is not None and "relative_error" in results["3exp"]])
    relative_error_scan = np.array([results["scan"]["relative_error"] for results in fit_results.values() if "true_params" in results and results["scan"] is not None and "relative_error" in results["scan"]])
    
    # 3. CALCULATE THE MEAN ABSOLUTE RELATIVE ERROR
    mean_abs_rel_error_3exp = np.mean(sorted(np.abs(relative_error_3exp))[:-40])
    mean_abs_rel_error_scan = np.mean(sorted(np.abs(relative_error_scan))[:-40])
    
    # 4. CALCULATE THE MEAN SQUARED RELATIVE ERROR
    mean_sq_rel_error_3exp = np.mean(sorted(relative_error_3exp**2)[:-40])
    mean_sq_rel_error_scan = np.mean(sorted(relative_error_scan**2)[:-40])
    
    
    
    print("Mean absolute relative error 3exp: " + str(mean_abs_rel_error_3exp))
    print("Mean absolute relative error scan: " + str(mean_abs_rel_error_scan))
    print("Mean squared relative error 3exp: " + str(mean_sq_rel_error_3exp))
    print("Mean squared relative error scan: " + str(mean_sq_rel_error_scan))


    # 5. PLOT THE RELATIVE ERRORS AS A FUNCTION OF TRUE_T_HOT
    t_hot_array = []
    for folder, results in fit_results.items():
        if "true_params" not in results:
            continue
        true_t_hot = results["true_params"]["t_hot"]
        
        if "3exp" not in results or "scan" not in results or results["3exp"] is None or results["scan"] is None:
            continue
        relative_error_3exp = (results["3exp"]["t_hot"] - true_t_hot)/true_t_hot
        relative_error_scan = (results["scan"]["t_hot"] - true_t_hot)/true_t_hot
        
        t_hot_array.append({"true_t_hot": true_t_hot, "3exp": relative_error_3exp, "scan": relative_error_scan})
    
    t_hot_array = sorted(t_hot_array, key=lambda x: x["3exp"])[:-20]
    t_hot_array = sorted(t_hot_array, key=lambda x: x["scan"])[:-20]
    
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = np.linspace(0, len(t_hot_array), len(t_hot_array))
    true_t_hots = [t_hots["true_t_hot"] for t_hots in t_hot_array]
    # ax.scatter(x, [t_hots["3exp"] for t_hots in t_hot_array], label="3exp")
    ax.scatter(true_t_hots, [t_hots["3exp"] for t_hots in t_hot_array], label="3-exp. Jacquelin", s=10, zorder=3, marker="x")
    ax.scatter(true_t_hots, [t_hots["scan"] for t_hots in t_hot_array], label="Scanning", s=10, zorder=4, marker="o")
    ax.set_xlabel(r"$T_\mathrm{hot,manual}$ [keV]")
    ax.set_ylabel("Relative error")
    ax.grid(zorder=0)
    ax.legend()
    plt.show()
    
    # Plot THe residuals as histograms
    fig, ax = plt.subplots()
    ax.hist([t_hots["3exp"] for t_hots in t_hot_array], bins=30, label="3-exp. Jacquelin")
    plt.show() 
    
    fig, ax = plt.subplots()
    ax.hist([t_hots["scan"] for t_hots in t_hot_array if t_hots["scan"] is not None], bins=30, label="Scanning method")
    plt.show() 