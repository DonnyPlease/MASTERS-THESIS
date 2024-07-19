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
    # 1. LOAD ALL FIT_RESULTS FROM THE FOLDER
    fit_results = []
    for file in os.listdir(FOLDER_SAVE_RESULTS):
        fit_results.append(load_fit_results(FOLDER_SAVE_RESULTS+file))
    
    # 2. EXTRACT T_HOT FROM THE FIT_RESULTS
    t_hot_array = []
    for fit in fit_results:
        if "true_params" not in fit:
            continue
        t_hots = {"3exp": fit["3exp"]["t_hot"] if fit["3exp"] is not None else None,
                  "2exp": fit["2exp"]["t_hot"] if fit["2exp"] is not None else None,
                  "nlsq": fit["nlsq"]["t_hot"] if fit["nlsq"] is not None else None,
                  "true_t_hot": fit["true_params"]["t_hot"] if fit["true_params"] is not None else None
        }
        t_hot_array.append(t_hots)
        
    t_hot_array = [t for t in t_hot_array if t["true_t_hot"] is not None]
        
    # 3. NORMALIZE THE T_HOT TO TRUE_T_HOT
    for t_hots in t_hot_array:
        t_hot = t_hots["true_t_hot"]
        t_hots["3exp"] = -(t_hot - t_hots["3exp"])/t_hot if t_hots["3exp"] is not None else None
        t_hots["2exp"] = -(t_hot - t_hots["2exp"])/t_hot if t_hots["2exp"] is not None else None
        t_hots["nlsq"] = -(t_hot - t_hots["nlsq"])/t_hot if t_hots["nlsq"] is not None else None
     
    # 4. Sort the t_hot_array by the true_t_hot
    t_hot_array = sorted(t_hot_array, key=lambda x: x["true_t_hot"])
    
    means = {"3exp": np.mean([t_hots["3exp"] for t_hots in t_hot_array if t_hots["3exp"] is not None]),
             "2exp": np.mean([t_hots["2exp"] for t_hots in t_hot_array if t_hots["2exp"] is not None]),
             "nlsq": np.mean([t_hots["nlsq"] for t_hots in t_hot_array if t_hots["nlsq"] is not None])
    }
    
    mses = {"3exp": np.mean([t_hots["3exp"]**2 for t_hots in t_hot_array if t_hots["3exp"] is not None]),
            "2exp": np.mean([t_hots["2exp"]**2 for t_hots in t_hot_array if t_hots["2exp"] is not None]),
            "nlsq": np.mean([t_hots["nlsq"]**2 for t_hots in t_hot_array if t_hots["nlsq"] is not None])
    }
    print ("Means: ", means)
    print("MSEs: ", mses)
    
    # 5. PLOT THE T_HOT_ARRAY
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = np.linspace(0, len(t_hot_array), len(t_hot_array))
    true_t_hots = [t_hots["true_t_hot"] for t_hots in t_hot_array]
    # ax.scatter(x, [t_hots["3exp"] for t_hots in t_hot_array], label="3exp")
    ax.scatter(true_t_hots, [t_hots["2exp"] for t_hots in t_hot_array], label="2-exp. Jacquelin", s=5, zorder=3)
    ax.scatter(true_t_hots, [t_hots["nlsq"] for t_hots in t_hot_array], label="Non-linear least squares", s=5, zorder=4)
    ax.set_xlabel("Simulation index")
    ax.set_ylabel("Relative error")
    ax.grid(zorder=0)
    ax.legend()
    plt.show()
    
    # Plot THe residuals as histograms
    fig, ax = plt.subplots()
    ax.hist([t_hots["2exp"] for t_hots in t_hot_array], bins=30, label="2-exp")
    plt.show() 
    
    fig, ax = plt.subplots()
    ax.hist([t_hots["nlsq"] for t_hots in t_hot_array if t_hots["nlsq"] is not None], bins=30, label="2-exp")
    plt.show() 
        