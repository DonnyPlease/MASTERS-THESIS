import sys, os
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import numpy as np
from scipy.optimize import curve_fit
from fit_exp_jacquelin import FitExp

def _fit_using_three_exponentials(histogram):
    x, y = histogram[0], histogram[1]
    fit_exp_jacquelin = FitExp(x, y, exp_count=3, include_constant=False)
    try:
        fit_results = fit_exp_jacquelin.fit(verbose=False)
    except Exception as e:
        print("3-exp fit failed for histogram {} because: ".format(histogram[2]) + str(e))
        fit_results = None

    return fit_results
    
def _fit_using_two_exponentials(histogram):
    x, y = histogram[0], histogram[1]
    fit_exp_jacquelin = FitExp(x, y, exp_count=2, include_constant=False)
    try:
        fit_results = fit_exp_jacquelin.fit(verbose=False)
    except Exception as e:
        print("2-exp fit failed for histogram {} because: ".format(histogram[2]) + str(e))
        fit_results = None

    return fit_results

def _fit_using_nlsq(histogram, initial_guess):
    x, y = histogram[0], histogram[1]
    def func(x, a, b, c, d):
        return a * np.exp(b * x) + c * np.exp(d * x)
    popt, pcov = curve_fit(func, x, y, p0=initial_guess)
    return popt, pcov

def fit_hot_temperature(histogram):
    # 1. FIT USING THREE EXPONENTIALS
    fit_results = _fit_using_three_exponentials(histogram)
    fail_count = 0
    while (fit_results is None) and fail_count < 10:
        histogram = (histogram[0][5:-5], histogram[1][5:-5], histogram[2])
        fit_results = _fit_using_three_exponentials(histogram)
        fail_count += 1
    
    if fit_results is None:
        return fit_results
    
    # 2. FIND NEW START INDEX AND CUT THE BEGINNING
    a1, a2, a3, b1, b2, b3 = fit_results["a1"], fit_results["a2"], fit_results["a3"], fit_results["b1"], fit_results["b2"], fit_results["b3"]
    
    
    # 3. FIT USING TWO EXPONENTIALS

    
    # 4. USE THE RESULTS FROM THE LAST FIT AS INITIAL GUESS FOR NLSQ # 3    # 3. FIT USING TWO EXPONENTIALS
    
    return fit_results