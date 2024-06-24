import sys, os
IMPORT_PATH = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/fitting/'
sys.path.append(IMPORT_PATH)

import numpy as np
from matplotlib import pyplot as plt
from helpful_functions import load_histogram, load_parameters
from helpful_functions import create_filenames, create_filename
from helpful_functions import plot_histogram
from helpful_functions import custom_rmse
from logs import logger

from Dataset import DatasetRecord, DatasetUtils

HISTOGRAMS_SOURCE_FOLDER_PATH = 'old_data/trimmed_histograms/'
FITTED_HISTOGRAMS_FOLDER_NAME = 'fitting/fitted_histograms_without_constant/'

from fit_exp_jacquelin import FitExp

def fit_reduced(bins, counts, original_bins, original_counts, exp_count, file_name, plot=True):
    t_hot = 0
    rmse = 0
    success = True
    include_constant = False
    jacquelin_fit = FitExp(bins, counts, exp_count=exp_count, constant=include_constant)
    try:
        params = jacquelin_fit.fit(verbose=False)
        rmse = custom_rmse(jacquelin_fit, bins, counts)
        print("RMSE for ", exp_count, " exp. starting with energy ", bins[0], "  : ", rmse)
        
        highest_negative_exponent = 1
        if include_constant:
            highest_negative_exponent = max([p for p in params[2::2] if p < 0])
        else:
            highest_negative_exponent = max([p for p in params[1::2] if p < 0])
            
        if highest_negative_exponent is None:
            raise Exception("Fit is bad - it has no nonnegative temperature.")
        t_hot = -1./highest_negative_exponent
    except:
        success = False
        
    start_index = 1
    try:
        if include_constant:
            start_index = np.log(params[1]/params[3])/(params[4]-params[2]) # log(a1/a2)/(b2-b1)
        else:
            start_index = np.log(params[0]/params[2])/(params[3]-params[1])+20 # log(a1/a2)/(b2-b1)+
        start_index = int(start_index)
    except:
        print("new start index could not be calculated")
        start_index = 1

    # Plot the histogram and the fitted function and save it
    if plot:
        try:
            plot_histogram(original_bins, original_counts, 
                           jacquelin_fit,
                           t_hot, 
                           file_name,
                           exp_count,
                           vertical_at=[start_index,bins[0],bins[-1]])
            print("suucessfuly plotted and saved")
        except:
            print("Something happened when trying to plot:")
            print("\t\tname:\t", file_name)
            print("\t\texp_count:\t", exp_count)

    try:
        if exp_count == 2 and not include_constant:
            with open('dataset/auto_fit.txt','a') as dataset:
                one_data = DatasetRecord()
                one_data.I = file_name.split('_')[-4]
                
                one_data.L = file_name.split('_')[-3]
                one_data.L = one_data.L[:1] + '.' + one_data.L[1:]
                
                one_data.alpha = file_name.split('_')[-2]
                one_data.t_hot = str(t_hot)
                one_data.min_energy = str(bins[0])
                one_data.max_energy = str(bins[-1])
                one_data.type = 'j2_wo'
                one_data.a = str(params[0])
                one_data.b = str(params[1])
                one_data.c = str(params[2])
                one_data.d = str(params[3])
                one_data.e = str(0)
                one_data.f = str(0)
                one_data.g = str(0)
                
                dataset.write(one_data.to_text())
    except:
        print("some error 444")
        
    return start_index, t_hot, rmse, success
        
def cut_histogram_from_left(bins, counts, start_index):
    return bins[start_index:], counts[start_index:]

def cut_histogram_from_right(bins, counts, percentage_of_max_energy):
    condition = bins < percentage_of_max_energy*bins[-1]
    return bins[condition], counts[condition]

def log_failed_fits(failed_fits):
    for f in failed_fits:
        logger.log("failed.txt", str(f))
    
def fit_one_histogram(folder_name, sequence = [3,2], cut_each_iteration_percentage=0.10):    
    success_list = []
    # Load histogram and set start_index
    bins, counts = load_histogram(HISTOGRAMS_SOURCE_FOLDER_PATH + folder_name)
    
    minimum = np.min(counts)
    counts = np.array(counts) - minimum
     
    start_index = 0
    for i, exp_count in enumerate(sequence):
        print(i, " ", exp_count, " ", start_index)
        
        bins_for_fit, counts_for_fit = cut_histogram_from_left(bins, counts, start_index)
        
        bins_for_fit, counts_for_fit = cut_histogram_from_right(bins_for_fit, counts_for_fit, 1-i*cut_each_iteration_percentage)
        
        # Cut from left by 20 in case the previous fit was not successful
        for j in range(len(success_list)):
            if not success_list[-j-1][0]:
                bins_for_fit, counts_for_fit = cut_histogram_from_left(bins_for_fit, counts_for_fit, 10*(j+1))
                print("Moving start")
            else: 
                break
        
        histogram_name = FITTED_HISTOGRAMS_FOLDER_NAME+folder_name[:-1]+'_{}.pdf'.format(i)
        start_index_new, t_hot, rmse, success = fit_reduced(bins_for_fit, 
                                            counts_for_fit,
                                            bins,
                                            counts,
                                            exp_count=exp_count, 
                                            file_name=histogram_name,
                                            plot=True)
        if exp_count == 2:
            pass
        else:
            start_index = start_index_new
            
        with open("logs.txt", "a") as logs:
            logs.write("{}\t,{:.4f}\t,{:.4f}\t,{}\n".format(histogram_name, t_hot, rmse, start_index))

        success_list.append((success,exp_count))
        
    return success_list

def fit(files_names, target_folder = ""):
    temp_file = open(target_folder+"dataset.txt","w")
    
    # For every set of parameters, load the histogram and fit it
    # with decreasing number of bins (start with start_index, which is 
    # updated after every iteration)) 
    failed_fits = []
    for file_name in files_names:  # For every histogram 
        print("Currently fitting histogram from: " + file_name)
        success_list = fit_one_histogram(file_name)
        for (success, count) in success_list:
            if not success:
                failed_fits.append(file_name + f"{count}")       
    
    log_failed_fits(failed_fits)
              
    print("Failed: ", len(failed_fits)) 
    

if __name__ == "__main__":
    print("This is reduce_histogram.py file.")
        
            
