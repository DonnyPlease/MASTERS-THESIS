import sys, os
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import numpy as np
from matplotlib import pyplot as plt
import json

# CONSTANTS
FOLDER_NAME = "data/"
PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/'+FOLDER_NAME+'spectra/'

from histograms.HistogramUtils import load_histograms
from fitting.FittingUtils import load_results_from_final_dataset

def extract_results(fit_results):
    results = {}
    results["t_hot"] = fit_results[0][0]
    results["n_hot"] = fit_results[0][1]
    results["sse"] = fit_results[1][0]
    return results

def scan_fit(x,y):
    all_results = []
    for i in range(0, len(x)-150):
        x_fit = x[i:i+150]
        y_fit = y[i:i+150]
        
        # 1. FIT THE HISTOGRAM LINEARLY USING LSQ
        # MAKE bins and counts 2D
        
        A = np.hstack([x_fit.reshape(-1, 1), np.ones((len(x_fit), 1))])
        fit_results = np.linalg.lstsq(A, np.log(y_fit), rcond=None)
        params = fit_results[0]
        residuals = np.log(y_fit) - A @ params
          
        # 2. EXTRACT THE RESULTS
        results = extract_results(fit_results)
        results["R2"] = 1 - np.sum(results["sse"]**2)/np.sum((np.log(y) - np.mean(np.log(y)))**2)
        results["mse"] = results["sse"]/len(x_fit)
        results["start"] = x_fit[0]
        results["end"] = x_fit[-1]
        
        
        # Calculate the standard deviation of the parameters
        residual_variance = np.sum(residuals**2) / (len(x_fit) - 2)
        cov_matrix = residual_variance * np.linalg.inv(A.T @ A)
        param_stdev = np.sqrt(np.diag(cov_matrix))
        
        results["t_hot_stdev"] = param_stdev[0]
        results["n_hot_stdev"] = param_stdev[1]
        
        results["t_hot"] = -1/results["t_hot"]
        results["t_hot_stdev"] = results["t_hot_stdev"]*results["t_hot"]**2
        
        results["n_hot"] = np.exp(results["n_hot"])
        results["n_hot_stdev"] = results["n_hot_stdev"]*results["n_hot"]
        
        all_results.append(results)
        
    return all_results


if __name__ == "__main__":
    histograms = load_histograms(PATH_TO_HISTOGRAMS)
    for i in range(0, len(histograms), 1):
        print(i, histograms[i][3])
        histogram = histograms[i]
        bins, counts, params, folder = histogram
        
        # 1. NORMALIZE THE HISTOGRAM
        max_hist = np.max(counts)
        counts = counts/max_hist
        
        # 2. SCAN THE HISTOGRAM
        results = scan_fit(bins, counts)
        # 3. FIND FIT WITH MAXIMUM T
        results = max(results, key=lambda x: x["t_hot"])
        results["n_hot"] = results["n_hot"]*max_hist
        results["n_hot_stdev"] = results["n_hot_stdev"]*max_hist
        
        # 4. PLOT THE RESULTS
        A = np.hstack([bins.reshape(-1, 1), np.ones((len(bins), 1))])
        b = np.array([-1/results["t_hot"], np.log(results["n_hot"])])
        
        plt.clf()
        
        plt.scatter(bins, counts*max_hist, label="Simulation data", zorder=8, color="black")
        label_t = r"$T_\mathrm{hot} =\,$"+ "{:.2f} ".format(results["t_hot"]) + r"$\pm$" + " {:.2f} keV".format(results["t_hot_stdev"]) 
        
        base, exponent = "{:.2e}".format(results["n_hot"]).split("e")
        base = float(base)
        exponent = int(exponent)
        
        base_stdev, exponent_stdev = "{:.2e}".format(results["n_hot_stdev"]).split("e")
        base_stdev = float(base_stdev)
        exponent_stdev = int(exponent_stdev)
        
        
        label_n = "\n" + r"$N_\mathrm{0} =\,$" +r"${0}\cdot10^{{{1}}}$".format(base,exponent) + r"$\,\pm\,$" + r"${0}\cdot10^{{{1}}}$".format(base_stdev,exponent_stdev)
        plt.plot(bins, np.exp(A @ b), color="red", label=label_t + label_n, zorder=9)
        # plot vertical lines for so that the limits are the same as if only the histogram was plotted
        plt.axvline(x=results["start"], color="black", linestyle="--", zorder=1)
        plt.axvline(x=results["end"], color="black", linestyle="--", zorder=1)
        plt.grid(True, zorder=0)
        plt.xlabel(r"$E [\mathrm{keV}]$")
        plt.ylabel(r"$N$")
        plt.yscale("log")
        plt.legend()
        
        ax, fig = plt.gca(), plt.gcf()
        all_axes = fig.get_axes()
        for axis in all_axes:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
                all_axes[-1].add_artist(legend)
        
        plt.savefig("C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/data/alternative_fitting/plots/"+folder+".pdf")
        
        fitting_results = {}
        fitting_results["best_linear_fit"] = results
        fitting_results = load_results_from_final_dataset(histogram, fitting_results)
        fitting_results["histogram"] = params
        
        with open("C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/data/alternative_fitting/results/"+folder+".json", "w") as f:
            json.dump(fitting_results, f, indent=4)
        
    
