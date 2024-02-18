import numpy as np
from matplotlib import pyplot as plt
from helpful_functions import load_histogram, load_parameters, create_filenames
from helpful_functions import plot_histogram
from helpful_functions import custom_rmse
PREFIX = 'old_data/trimmed_histograms/'

from fit_exp_jacquelin import FitExp

def fit_reduced(new_bins,new_counts,exp_count,name,failed,plots,plot=True):
    t_hot = 0
    rmse = 0
    try:
        f = FitExp(new_bins, new_counts, exp_count=exp_count, constant=True)
        params = f.fit(verbose=False)
        rmse = custom_rmse(f,new_bins,new_counts)
        print("RMSE for ", exp_count, " exp. starting with energy ", new_bins[0], "  : ",rmse)
        t_hot = - 1./max(params[2::2])
    except:
        failed.append((files_names[i],exp_count))
        
    start_index = 1
    try:
        start_index = np.log(params[1]/params[3])/(params[4]-params[2]) # log(a1/a2)/(b2-b1)
        start_index = int(start_index)
    except:
        print("new start index could not be calculated")
        start_index = 0

    
    # Plot the histogram and the fitted function and save it
    if plot:
        try:
            t_hot = -  1./max(params[2::2])
            plot_histogram(bins, counts, f.predict(bins),t_hot, name,exp_count,vertical_at=start_index)
            print("suucessfuly plotted and saved")
            plots.append((name,exp_count))
        except:
            print("Something happened when trying to plot:")
            print("\t\tname:\t", name)
            print("\t\texp_count:\t", exp_count)

        
    return start_index, t_hot, rmse
        



if __name__ == "__main__":
    params = load_parameters('old_data/params.txt')
    files_names = create_filenames(params)
    print(PREFIX + files_names[0])
    
    temp_file = open('dataset.txt','w')
    
    
    # For every set of parameters, load the histogram and fit it
    # with decreasing number of bins (start with start_index, which is 
    # updated after every iteration)) 
    failed = []
    plots =[]
    for i in range(len(files_names)):  # For every histogram
        # if i > 20: break
        print(files_names[i])
        
        # Load histogram
        bins, counts = load_histogram(PREFIX + files_names[i])

        start_index = 0
        
        name = 'fitting/fitted_histograms/'+files_names[i][:-1]+'_3.pdf'
        start_index, t_hot, rmse = fit_reduced(bins[start_index:],counts[start_index:],
                                  exp_count=3,name=name,failed=failed,plots=plots,plot=True)
        
        with open("logs.txt","a") as logs:
            logs.write("{},{},{},{}\n".format(name,t_hot,rmse,start_index))
        
        name = 'fitting/fitted_histograms/'+files_names[i][:-1]+'_2.pdf'
        start_index, t_hot, rmse = fit_reduced(bins[start_index:],counts[start_index:],
                                  exp_count=2,name=name,failed=failed,plots=plots,plot=True)
        
        with open("logs.txt","a") as logs:
            logs.write("{},{},{},{}\n".format(name,t_hot,rmse,start_index))

       
    print("FAILED:")
    with open("failed.txt","w") as file:    
        for f in failed:  
            file.write(str(f))    
            file.write('\n')  
            print(f)
            
    print("\n")
    
    print("PLOTTED:")      
    with open("plotted.txt","w") as file:
        for p in plots:
            file.write(str(p))
            file.write('\n')  
            print(p)
            
    print("Failed: ", len(failed)) 
        
            
