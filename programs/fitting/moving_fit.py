import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

from helpful_functions import load_histogram, create_filename

HISTOGRAMS_SOURCE_FOLDER_PATH = 'old_data/trimmed_histograms/'

def exponential_function(x,a,b,c):
    return a+b*np.exp(-x*c)

def moving_exp_fit(bins, counts, step=10, max_range=100):
    n = bins.shape[0]
    print("size",n)
    count = 0
    histogram_of_temeperatures = []
    for i in range(0, n-10, 10):
        for j in range(i+10, n, 10):
            if (j-i > max_range): break
            count +=1
            bins_to_fit = bins[i:j]
            counts_to_fit = counts[i:j]
            
            popt, pcov = curve_fit(exponential_function,bins_to_fit,counts_to_fit)

            a, b, c = popt
            temperature = 1/c
            histogram_of_temeperatures.append(temperature)
            
    print(count)
    return histogram_of_temeperatures

def moving_linear_fit(bins, counts, step=20, max_range=200):
    n = bins.shape[0]
    print("size", n)
    count = 0
    histogram_of_temperatures = []
    for i in range(0, n-step, step):
        for j in range(i+step, n, step):
            if (j-i > max_range): break
            count +=1
            bins_to_fit = bins[i:j]
            counts_to_fit = counts[i:j]
            
            X = np.column_stack((np.ones_like(bins_to_fit), bins_to_fit))

            cintercept, slope = np.linalg.lstsq(X, counts_to_fit, rcond=None)[0]
            
            histogram_of_temperatures.append(-1/slope)
            
    return histogram_of_temperatures
    

if __name__ == "__main__":
    filename = create_filename(18,1.00,0)
    bins, counts = load_histogram(filename=HISTOGRAMS_SOURCE_FOLDER_PATH+filename)
    
    log_counts = np.log(counts)
    
    counts  = counts/1e10
    temperatures = moving_linear_fit(bins, log_counts, step = 10, max_range=150)
    temperatures = [t for t in temperatures if t>0 and t<1500]
    print(len(temperatures))
    print(temperatures)
    plt.hist(temperatures,50)
    plt.show()
    