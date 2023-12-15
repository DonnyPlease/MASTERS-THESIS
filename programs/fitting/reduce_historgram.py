import numpy as np
from matplotlib import pyplot as plt
from helpful_functions import load_histogram, load_parameters, create_filenames
from helpful_functions import plot_histogram
PREFIX = 'old_data/trimmed_histograms/'

from fit_exp_jacquelin import FitExp

def fit_reduced(new_bins,new_counts,exp_count,name,failed,plot=True):
    try:
        f = FitExp(new_bins, new_counts, exp_count=exp_count, constant=True)
        params = f.fit(verbose=True)
    except:
        failed.append((files_names[i],exp_count))
        
    # Plot the histogram and the fitted function and save it
    if plot:
        try:
            t_hot = -  1./max(params[2::2])
            plot_histogram(bins, counts, f.predict(bins),t_hot, name,vertical_at=start_index)
            print("suucessfuly plotted and saved")
        except:
            print("Something happened when trying to plot:")
            print("\t\tname:\t", name)
            print("\t\texp_count:\t", exp_count)
    try:
        start_index = np.log(params[1]/params[3])/(params[4]-params[2]) # log(a1/a2)/(b2-b1)
        start_index = int(start_index)
    except:
        print("new start index could not be calculated")
        
    return start_index
        



if __name__ == "__main__":
    params = load_parameters('old_data/params.txt')
    files_names = create_filenames(params)
    print(PREFIX + files_names[0])
    
    temp_file = open('dataset.txt','w')
    
    # For every set of parameters, load the histogram and fit it
    # with decreasing number of bins (start with start_index, which is 
    # updated after every iteration)) 
    failed = []
    for i in range(len(files_names)):  # For every histogram
        if i > 20: break
        print(files_names[i])
        
        # Load histogram
        bins, counts = load_histogram(PREFIX + files_names[i])
        name = 'fitting/fitted_histograms/'+files_names[i][:-1]+'.pdf'

        start_index = 0
        start_index = fit_reduced(bins[start_index:],counts[start_index:],exp_count=3,name=name,failed=failed,plot=False)
        start_index = fit_reduced(bins[start_index:],counts[start_index:],exp_count=2,name=name,failed=failed,plot=True)
       
    
    with open("failed.txt","w") as file:    
        for f in failed:  
            file.write(str(f))      
            print(f)
    print("Failed: ", len(failed)) 
        
            
