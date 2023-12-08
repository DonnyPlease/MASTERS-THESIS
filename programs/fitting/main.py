import numpy as np
from matplotlib import pyplot as plt
from helpful_functions import load_histogram, load_parameters, create_filenames
PREFIX = 'old_data/original_histograms/'
from fit_exp_jacquelin import FitExp

if __name__ == "__main__":
    params = load_parameters('old_data/params.txt')
    
    files_names = create_filenames(params)
    print(PREFIX + files_names[0])
    bins, counts = load_histogram(PREFIX + files_names[0])
    
    temp_file = open('dataset.txt','w')
    f = FitExp(bins[165:-1], counts[165:], exp_count=1, constant=True)
    
    name = 'histogram_test.pdf'
    
    
    try:
        params = f.fit(verbose=True)
        #new_x = np.log(params[1]/params[3])/(params[4]-params[2]) # log(a1/a2)/(b2-b1)
        #print(new_x)
        #highest_temperature = max(params[2], params[4])
        f.plot(log=True, residuals=True, show=True, save=True, save_name='fitted_histograms/'+name)
        if highest_temperature > 0:
            raise Exception('highest temperature')
        temp_file.write('{},{},{},{:.8f}\n'.format(i,l,a,highest_temperature))
    except:
        pass
    
    plt.plot(bins[:-1], counts)
    plt.yscale('log')
    plt.show()