import numpy as np
from matplotlib import pyplot as plt
from helpful_functions import load_histogram, load_parameters, create_filenames
from helpful_functions import plot_histogram
PREFIX = 'old_data/trimmed_histograms/'

from fit_exp_jacquelin import FitExp

if __name__ == "__main__":
    params = load_parameters('old_data/params.txt')
    files_names = create_filenames(params)
    print(PREFIX + files_names[0])
    
    temp_file = open('dataset.txt','w')
    
    # For every set of parameters, load the histogram and fit it
    # with decreasing number of bins (start with start_index, which is 
    # updated after every iteration)) 
    failed = []
    for i in range(len(files_names)):
        print(files_names[i])
        bins, counts = load_histogram(PREFIX + files_names[i])
        name = 'fitting/fitted_histograms/'+files_names[i][:-1]+'.pdf'

        start_index = 0
        exp_count = 3
        for _ in range(exp_count):
            # Fit the histogram with the current number of bins and current
            # number of exponentials
            try:
                f = FitExp(bins[start_index:], counts[start_index:], exp_count=2, constant=True)
                params = f.fit(verbose=True)
            except:
                failed.append((files_names[i],exp_count))
                break
                
            # Plot the histogram and the fitted function and save it
            try:
                f.plot(log=True, residuals=True, show=False, save=True, save_name=name)
                print("suucessfuly plotted and saved")
            except:
                print("Something happened probably after plotting.")
            
            # If the number of exponentials is 1, plot the whole histogram
            if exp_count == 1:
                try:
                    f = FitExp(bins[start_index:], counts[start_index:], exp_count=2, constant=True)
                    params = f.fit(verbose=True)
                except:
                    failed.append((files_names[i],exp_count))
                    break
                
                try:
                    t_hot = -  1./max(params[2::2])
                    plot_histogram(bins, counts, f.predict(bins),t_hot, name,vertical_at=start_index)
                except:
                    print("Failed to plot the final histogram")
            
            elif exp_count>1:  # Else reduce the number of bins and try again
                try:
                    start_index = np.log(params[1]/params[3])/(params[4]-params[2]) # log(a1/a2)/(b2-b1)
                    start_index = int(start_index)
                except:
                    print("Failed to calculate new start_index")
                exp_count -= 1
                print("start_index: ", start_index)
                print("exp_count: ", exp_count)
        
            
    print(failed)
    print("Failed: ", len(failed)) 
        
            
