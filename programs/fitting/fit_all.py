import sys
IMPORT_PATH = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(IMPORT_PATH)

import numpy as np
from matplotlib import pyplot as plt

from helpful_functions import load_histogram, load_parameters
from helpful_functions import create_filenames, create_filename
from helpful_functions import plot_histogram
from helpful_functions import custom_rmse

from dataset.Dataset import DatasetRecord, DatasetUtils
from fit_exp_jacquelin_refactored import FitExp

HISTOGRAMS_SOURCE_FOLDER_PATH = 'old_data/trimmed_histograms/'
FITTED_HISTOGRAMS_FOLDER_NAME = 'fitting/fitted_histograms_without_constant/'

def get_t_hot_and_error(params, std_errors, exp_count):
    # Get the temperature of the hottest plasma
    t_hot = 0
    t_hot_error = 0
    
    highest_negative_exponent = max([p for p in params[2::2] if p < 0])
    if highest_negative_exponent is None:
        raise Exception("Fit is bad - it has no nonnegative temperature.")
        
    t_hot = -1./highest_negative_exponent
     
    if exp_count == 2:
        t_hot_error = 1./t_hot**2 * std_errors[2::2][np.argmax([p for p in params[2::2] if p < 0])]
    
    return t_hot, t_hot_error

def try_plot_histogram(original_bins, original_counts, jacquelin_fit, t_hot, file_name, exp_count, vertical_at):
    try:
        plot_histogram(original_bins, original_counts, 
                        jacquelin_fit,
                        t_hot, 
                        file_name,
                        exp_count,
                        vertical_at=vertical_at)
        print("suucessfuly plotted and saved")
    except:
        print("Something happened when trying to plot:")
        print("\t\tname:\t", file_name)
        print("\t\texp_count:\t", exp_count)  
    

def fit_reduced(bins, counts, original_bins, original_counts, exp_count, file_name, plot=True):
    t_hot = 0
    rmse = 0
    success = True
    include_constant = False
    jacquelin_fit = FitExp(bins, counts, exp_count=exp_count, include_constant=include_constant)
    try:
        print("fit init successful")
        params = jacquelin_fit.fit(verbose=False)
        std_errors = jacquelin_fit.std_errors
        rmse = custom_rmse(jacquelin_fit, bins, counts)
        print("RMSE for ", exp_count, " exp. starting with energy ", bins[0], "  : ", rmse)
        
        t_hot, t_hot_error = get_t_hot_and_error(params, std_errors, exp_count)
           
    except:
        success = False
        
    start_index = 1
    try:
        start_index = np.log(params[1]/params[3])/(params[4]-params[2]) # log(a1/a2)/(b2-b1)
        start_index = int(start_index)
    except:
        print("new start index could not be calculated")
        start_index = 1

    # Plot the histogram and the fitted function and save it
    if plot:
        try_plot_histogram()

    try:
        if exp_count == 2 and not include_constant:
            with open('dataset/auto_fit2.txt','a') as dataset:
                one_data = DatasetRecord()
                one_data.I = file_name.split('_')[-4]
                
                one_data.L = file_name.split('_')[-3]
                one_data.L = one_data.L[:1] + '.' + one_data.L[1:]
                
                one_data.alpha = file_name.split('_')[-2]
                one_data.t_hot = str(t_hot)
                one_data.t_hot_stdev = str(t_hot_error)
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

def cut_from_left_and_right(bins, counts, start_index, percentage_of_max_energy):
    bins, counts = cut_histogram_from_left(bins, counts, start_index)
    bins, counts = cut_histogram_from_right(bins, counts, percentage_of_max_energy)
    return bins, counts

def fit_one_histogram(folder_name, sequence = [3,2], cut_each_iteration_percentage=0.10):    
    success_list = []
    
    # Load histogram
    bins, counts = load_histogram(HISTOGRAMS_SOURCE_FOLDER_PATH + folder_name)
    
    # offset by minimum ??? not sure why legal
    minimum = np.min(counts)
    counts = np.array(counts) - minimum
     
    start_index = 0
    
    for i, exp_count in enumerate(sequence):
        print(i, " ", exp_count, " ", start_index)
        move_start_index_by = 0 if success_list == [] or success_list[0] else 20
        bins_for_fit, counts_for_fit = cut_from_left_and_right(bins, counts, start_index + move_start_index_by, 1-i*cut_each_iteration_percentage)
        
        histogram_name = FITTED_HISTOGRAMS_FOLDER_NAME+folder_name[:-1]+'_{}.pdf'.format(i)
        start_index_new, t_hot, rmse, success = fit_reduced(bins_for_fit, 
                                            counts_for_fit,
                                            bins,
                                            counts,
                                            exp_count=exp_count, 
                                            file_name=histogram_name,
                                            plot=False)

        start_index = start_index_new
            
        with open("logs.txt", "a") as logs:
            logs.write("{}\t,{:.4f}\t,{:.4f}\t,{}\n".format(histogram_name, t_hot, rmse, start_index))

        success_list.append((success,exp_count))
        
    return success_list

if __name__ == "__main__":
    params = load_parameters('old_data/params.txt')
    files_names = create_filenames(params)
    
    # For every set of parameters, load the histogram and fit it
    # with decreasing number of bins (start with start_index, which is 
    # updated after every iteration)) 
    failed = []
    for i in range(len(files_names)):  # For every histogram 
        print(files_names[i])
        
        success_list = fit_one_histogram(files_names[i])
        
        for (success, count) in success_list:
            if not success:
                failed.append(files_names[i] + f"{count}")       
        
    print("FAILED:")
    with open("failed.txt","w") as file:    
        for f in failed:  
            file.write(str(f))    
            file.write('\n')  
            print(f)
            
    print("\n")
              
    print("Failed: ", len(failed)) 
        
            
