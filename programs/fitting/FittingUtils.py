import sys, os
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from fit_exp_jacquelin import FitExp
from dataset.Dataset import DatasetUtils

def reverse_normalize(histogram, fits_results, max_hist):
    histogram = (histogram[0], histogram[1]*max_hist, histogram[2])
    for key in fits_results:
        if key == "true_params":
            continue
        if fits_results[key] is not None:
            fits_results[key]["a1"] *= max_hist
            fits_results[key]["a2"] *= max_hist
            if "a3" in fits_results[key]:
                fits_results[key]["a3"] *= max_hist
    if fits_results["3exp"] is not None:
        fits_results["3exp"]["n_hot"] *= max_hist
    
    if fits_results["2exp"] is not None:
        fits_results["2exp"]["n_hot"] *= max_hist
        fits_results["2exp"]["a1_stdev"] *= max_hist
        fits_results["2exp"]["a2_stdev"] *= max_hist
        fits_results["2exp"]["n_hot_stdev"] *= max_hist
    if fits_results["nlsq"] is not None:
        fits_results["nlsq"]["a0"] *= max_hist
        fits_results["nlsq"]["n_hot"] *= max_hist
        fits_results["nlsq"]["a1_stdev"] *= max_hist
        fits_results["nlsq"]["a2_stdev"] *= max_hist
        fits_results["nlsq"]["n_hot_stdev"] *= max_hist
    
        
    return histogram, fits_results

def _fit_using_three_exponentials(histogram):
    x, y = histogram[0], histogram[1]
    fit_exp_jacquelin = FitExp(x, y, exp_count=3, include_constant=False)
    try:
        fit_results = fit_exp_jacquelin.fit(verbose=False)
        fit_results["t_hot"], fit_results["n_hot"], _, _ = get_t_hot_and_n_hot(fit_results, False)
    except Exception as e:
        print("3-exp fit failed for histogram {} because: ".format(histogram[2]) + str(e))
        fit_results = None

    return fit_results
    
def _fit_using_two_exponentials(histogram):
    x, y = histogram[0], histogram[1]
    fit_exp_jacquelin = FitExp(x, y, exp_count=2, include_constant=False)
    try:
        fit_results = fit_exp_jacquelin.fit(verbose=False)
        errors = fit_exp_jacquelin.std_errors
        fit_results["a1_stdev"], fit_results["b1_stdev"], fit_results["a2_stdev"], fit_results["b2_stdev"] = errors[0], errors[1], errors[2], errors[3]
        fit_results["t_hot"], fit_results["n_hot"], fit_results["t_hot_stdev"], fit_results["n_hot_stdev"] = get_t_hot_and_n_hot(fit_results, True)
        
    except Exception as e:
        print("2-exp fit failed for histogram {} because: ".format(histogram[2]) + str(e))
        fit_results = None

    return fit_results

def exp2(x, a, b, c, d, bias=0):
        return a * np.exp(b * x) + c * np.exp(d * x) + bias

def _fit_using_nlsq(histogram, initial_guess):
    x, y = histogram[0], histogram[1]
    weights = np.sqrt(y)
    
    popt, pcov = curve_fit(exp2, x, y, p0=initial_guess, sigma=weights)
    return popt, pcov

def find_new_start_index(histogram, start, end, params):
    hist = cut_histogram(histogram, start, end)
    if params is None:
        return 0
    a1, a2, a3, b1, b2, b3 = params["a1"], params["a2"], params["a3"], params["b1"], params["b2"], params["b3"]
    if b1>0 or b2>0 or b3>0:
        print("One of the exponents is positive. Might cause problems.")
    # find two smallest exponents and corresponting amplitudes
    if a1 < a2 and a1 < a3:
        a1, a2, a3 = a1, a2, a3
        b1, b2, b3 = b1, b2, b3
    elif a2 < a1 and a2 < a3:
        a1, a2, a3 = a2, a1, a3
        b1, b2, b3 = b2, b1, b3
    else:
        a1, a2, a3 = a3, a1, a2
        b1, b2, b3 = b3, b1, b2
        
    critical_energy = np.log(a1/a2)/(b2-b1)
    params["critical_energy"] = critical_energy
    start_index = np.argmax(hist[0]>critical_energy)  # index of the first bigger x value than critical energy
    return start_index + 25

def exp3(x, a1, b1, a2, b2, a3, b3, bias=0):
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x) + a3 * np.exp(b3 * x) + bias

def plot_fit(histogram, fits_params, fit_type, save_path):
    match fit_type:
        case "3exp":
            if "3exp" not in fits_params or fits_params["3exp"] is None:
                print("Cannot plot 3exp fit for histogram {} because the fit failed.".format(histogram[2]["filename"]))
                return
            _plot_fit_3exp(histogram, fits_params["3exp"], save_path)
        case "2exp":
            if "2exp" not in fits_params or fits_params["2exp"] is None:
                print("Cannot plot 2exp fit for histogram {} because the fit failed.".format(histogram[2]["filename"]))
                return
            _plot_fit_2exp(histogram, fits_params["2exp"], save_path)
        case "nlsq":
            if "nlsq" not in fits_params or fits_params["nlsq"] is None:
                print("Cannot plot nlsq fit for histogram {} because the fit failed.".format(histogram[2]["filename"]))
                return
            _plot_fit_nlsq(histogram, fits_params["nlsq"], save_path)

def _plot_fit_3exp(histogram, fit_params, save_path):
    plt.clf()
    y_fit = exp3(histogram[0], fit_params["a1"], fit_params["b1"], fit_params["a2"], fit_params["b2"], fit_params["a3"], fit_params["b3"])
    plt.scatter(histogram[0], histogram[1], c="black", s=10, label="Data", zorder = 4)
    plt.plot(histogram[0], y_fit, c='red', label=r"$N$ = $a_1\mathrm{e}^{b_1 E} + a_2\mathrm{e}^{b_2 E}+ a_3\mathrm{e}^{b_3 E}$", zorder=6)
    plt.axvline(fit_params["critical_energy"], color='blue', linestyle='--', label=r"E$_\mathrm{crit}$"+" = {:.2f} keV".format(fit_params["critical_energy"]), zorder=8)
    plt.yscale("log")
    plt.xlabel("E [keV]")
    plt.ylabel("N")
    plt.grid(True, zorder=0)
    _add_t_hot_to_legend(fit_params["t_hot"], None)
    plt.savefig(PATH_TO_PROJECT+save_path+histogram[2]["filename"]+".pdf")
    
def _plot_fit_2exp(histogram, fit_params, save_path):
    plt.clf()
    y_fit = exp2(histogram[0], fit_params["a1"], fit_params["b1"], fit_params["a2"], fit_params["b2"])
    plt.scatter(histogram[0], histogram[1], c="black", s=10, label="Simulation data", zorder=4)
    plt.plot(histogram[0], y_fit, c='red', label=r"$N$ = $a_1$"+f"exp({fit_params["b1"]:.2f}.E) + "+r"$a_2$"+f"exp({fit_params["b2"]:.2f}.E) ", zorder=5)
    plt.yscale("log")
    plt.xlabel("E [keV]")
    plt.ylabel("N")
    _add_t_hot_to_legend(fit_params["t_hot"], fit_params["t_hot_stdev"])
    plt.grid(True, zorder=0)
    plt.savefig(PATH_TO_PROJECT+save_path+histogram[2]["filename"]+".pdf")

def _plot_fit_nlsq(histogram, fit_params,save_path):
    plt.clf()
    y_fit = exp2(histogram[0], fit_params["a1"], fit_params["b1"], fit_params["a2"], fit_params["b2"], fit_params["a0"])
    plt.scatter(histogram[0], histogram[1], c="black", s=10, label="Simulation data",zorder=4)
    plt.plot(histogram[0], y_fit, c='red', label=r"$N$ = $a_1$"+f"exp({fit_params["b1"]:.2f}.E) + "+r"$a_2$"+f"exp({fit_params["b2"]:.2f}.E) + {fit_params['a0']:.2f}", zorder=6)
    plt.yscale("log")
    plt.xlabel("E [keV]")
    plt.ylabel("N")
    _add_t_hot_to_legend(fit_params["t_hot"], fit_params["t_hot_stdev"])
    plt.grid(True, zorder=0)
    plt.savefig(PATH_TO_PROJECT+save_path+histogram[2]["filename"]+".pdf")

def _add_t_hot_to_legend(t_hot, t_hot_stdev):
    handles, labels = plt.gca().get_legend_handles_labels()
    custom_line = Line2D([0], [0], color='white', lw=2, linestyle='--')
    handles.append(custom_line)
    label = r"$T_\mathrm{hot}$" +" = {:.2f} ± {:.2f} keV".format(t_hot, t_hot_stdev) if t_hot_stdev is not None else r"$T_\mathrm{hot}$" +" = {:.2f} keV".format(t_hot)
    labels.append(label)
    plt.legend(handles=handles, labels=labels)

def plot_fit_OLD(histogram, fit_params, max_hist, fit_stdev=None, save_suffix=""):
    plt.clf()
    y_fit = max_hist*exp2(histogram[0], fit_params["a1"], fit_params["b1"], fit_params["a2"], fit_params["b2"])
    plt.scatter(histogram[0], histogram[1]*max_hist, c="black", s=10, label="Simulation data")
    plt.plot(histogram[0], y_fit, c='red', label="N = {:.1e}exp({:.2f}E) + {:.1e}exp({:.2f}E)".format(max_hist*fit_params["a1"], fit_params["b1"], max_hist*fit_params["a2"], fit_params["b2"]))
    plt.yscale("log")
    plt.xlabel("E [keV]")
    plt.ylabel("N")
    
    # Get the current legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    custom_line = Line2D([0], [0], color='red', lw=2, linestyle='--')
    # Append the custom line to the handles and labels
    handles.append(custom_line)
    t_hot, n_hot, t_hot_stdev, n_hot_stdev = get_t_hot_and_n_hot(fit_params, fit_stdev)
    labels.append(r"$T_\mathrm{hot}$" +" = {:.2f} ± {:.2f} keV".format(t_hot, t_hot_stdev if t_hot_stdev is not None else 0))
    # Add the combined legend to the plot
    plt.grid()
    plt.legend(handles=handles, labels=labels)
    plt.savefig(PATH_TO_PROJECT+"histograms/data/fitting_plots/"+histogram[2]["filename"]+f"_{save_suffix}.pdf")
    print("Fit for histogram {} saved.".format(histogram[2]["filename"]))

def get_t_hot_and_n_hot(fit_params, std_errors=False):
    # T_hot is calculated as the minus inverse of the highest negative exponent
    b1, b2 = fit_params["b1"], fit_params["b2"]
    params = [b1, b2]
    if "b3" in fit_params:
        b3 = fit_params["b3"]
        params.append(b3)
    
    # Find the largest number smaller than zero and its index
    negative_numbers = [(num, idx) for idx, num in enumerate(params) if num < 0]
    if negative_numbers:
        max_number, index_of_max = max(negative_numbers)
    else: return None, None, None, None  
    
    t_hot = -1/max_number
    n_hot = fit_params["a"+str(index_of_max+1)]
    if std_errors == False:
        return t_hot, n_hot, None, None
    std_t_hot = fit_params["b"+str(index_of_max+1)+"_stdev"]/(max_number**2)
    std_n_hot = fit_params["a"+str(index_of_max+1)+"_stdev"]
    return t_hot, n_hot, std_t_hot, std_n_hot      
    
def cut_histogram(histogram, start_index, end_index):
    return (histogram[0][start_index:end_index], histogram[1][start_index:end_index], histogram[2])
     
def _try_fit(histogram, start, end, fit_function):
    cut_hist = cut_histogram(histogram, start, end)
    fit_results = fit_function(cut_hist)
    fail_count = 0
    while (fit_results is None) and fail_count < 10:
        start, end = start + 5, end - 5 
        cut_hist = cut_histogram(histogram, start, end)
        fit_results = fit_function(cut_hist)
        fail_count += 1
    return fit_results, start, end
    
def print_fit_results(fit_results):
    nested_dict = json.dumps(fit_results, indent=4)
    print(nested_dict)
    
def save_fit_results(fit_results, save_path="histograms/data/fitting_results/fit_params/"):
    json.dump(fit_results, open(PATH_TO_PROJECT+save_path+f"{fit_results["histogram"]["filename"]}.json", "w"), indent=4)

def load_fit_results(filename):
    return json.load(open(filename, "r"))
 
def calculate_mse_2exp(histogram, fit_params, start, end):
    y_fit = exp2(histogram[0][start:end], fit_params["a1"], fit_params["b1"], fit_params["a2"], fit_params["b2"])
    mse = np.mean(((histogram[1][start:end] - y_fit)**2)/histogram[1][start:end])
    return mse

def calculate_mse_nlsq(histogram, fit_params, start, end):
    y_fit = exp2(histogram[0][start:end], fit_params["a1"], fit_params["b1"], fit_params["a2"], fit_params["b2"], fit_params["a0"])
    mse = np.mean(((histogram[1][start:end] - y_fit)**2)/histogram[1][start:end])
    return mse
   
def calculate_mse_3exp(histogram, fit_params, start, end):
    y_fit = exp3(histogram[0][start:end], fit_params["a1"], fit_params["b1"], fit_params["a2"], fit_params["b2"], fit_params["a3"], fit_params["b3"])
    mse = np.mean(((histogram[1][start:end] - y_fit)**2)/histogram[1][start:end])
    return mse
 
def calculate_mses(histogram, fit_results):
    if fit_results["2exp"] is not None:
        fit_results["2exp"]["mse"] = calculate_mse_2exp(histogram, fit_results["2exp"], fit_results["2exp"]["start_index"], fit_results["2exp"]["end_index"])
    if fit_results["3exp"] is not None:
        fit_results["3exp"]["mse"] = calculate_mse_3exp(histogram, fit_results["3exp"], fit_results["3exp"]["start_index"], fit_results["3exp"]["end_index"])
    if fit_results["nlsq"] is not None:
        fit_results["nlsq"]["mse"] = calculate_mse_nlsq(histogram, fit_results["nlsq"], fit_results["nlsq"]["start_index"], fit_results["nlsq"]["end_index"])

def load_results_from_final_dataset(histogram, fit_results):
    try:
        final_dataset = DatasetUtils.load_final_dataset(PATH_TO_PROJECT+"dataset/")
    except:
        fit_results["true_params"] = None
        return fit_results
    
    key = (str(histogram[2]["i"]).replace("+",""), "{:.2f}".format(histogram[2]["l"]), str(int(histogram[2]["a"])))
    print(key)
    if key in final_dataset:
        fit_results["true_params"] = {"t_hot": float(final_dataset[key].t_hot),
                                      "t_hot_stdev": float(final_dataset[key].t_hot_stdev)}
    return fit_results
 
def fit_hot_temperature(histogram):
    final_params = {}
    start, end = 0, len(histogram[0])-1
    
    # 1. FIT USING THREE EXPONENTIALS
    fit_results, start, end = _try_fit(histogram, start, end, _fit_using_three_exponentials)  
    final_params["3exp"] = fit_results
    if fit_results is not None:
        final_params["3exp"]["start_index"], final_params["3exp"]["end_index"] = int(start), int(end)

    # 2. FIND NEW START INDEX
    new_start_index = find_new_start_index(histogram, start, end, fit_results) 
    start = start + new_start_index
    
    # 3. FIT USING TWO EXPONENTIALS
    fit_results, start, end = _try_fit(histogram, start, end, _fit_using_two_exponentials)
    final_params["2exp"] = fit_results 
    if fit_results is None:
        return final_params
    final_params["2exp"]["start_index"], final_params["2exp"]["end_index"] = int(start), int(end)
    
    # 4. USE THE RESULTS FROM THE LAST FIT AS INITIAL GUESS FOR NLSQ
    initial_guess = [fit_results["a1"], fit_results["b1"], fit_results["a2"], fit_results["b2"],0]
    try:
        cut_hist = cut_histogram(histogram, start, end)
        popt, pcov = _fit_using_nlsq(cut_hist, initial_guess)
        std_errors = np.sqrt(np.diag(pcov))
        fit_stdev = {"a1": std_errors[0], "b1": std_errors[1], "a2": std_errors[2], "b2": std_errors[3], "a0": std_errors[4]}
        fit_results = {"a1": popt[0], "b1": popt[1], "a2": popt[2], "b2": popt[3], "a0": popt[4]}
        fit_results["a1_stdev"], fit_results["b1_stdev"], fit_results["a2_stdev"], fit_results["b2_stdev"], fit_results["a0_stdev"] = std_errors
        fit_results["start_index"], fit_results["end_index"] = int(start), int(end)
        fit_results["t_hot"], fit_results["n_hot"], fit_results["t_hot_stdev"], fit_results["n_hot_stdev"] = get_t_hot_and_n_hot(fit_results, True)
        final_params["nlsq"] = fit_results
    except:
        final_params["nlsq"] = None
        print("NLSQ fit failed for histogram {}.".format(histogram[2]))
    
    # 5. LOAD RESULTS FROM FINAL DATASET
    final_params = load_results_from_final_dataset(histogram, final_params)
    
    calculate_mses(histogram, final_params)    
    
    return final_params 