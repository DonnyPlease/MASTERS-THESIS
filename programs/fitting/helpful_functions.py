import numpy as np
import matplotlib.pyplot as plt


def load_histogram(filename):
    """
    Load a histogram from a file. The file, name of which is passed
    as an argument, contains two files: bins.txt and counts.txt. The first one
    contains the bins of the histogram, the second one contains the counts.
    """
    bins = np.loadtxt(filename+'bins.txt')
    counts = np.loadtxt(filename+'counts.txt')
    return bins, counts
        
def load_parameters(filename):
    """
    Load the parameters from a file. The parameters are stored in a file
    if format: intensity,characteristic_length,angle (in each line). 
    """
    return np.loadtxt(filename, delimiter=',')

def create_filename(x,y,z):
    """
    This function creates a filename from the parameters of the simulation.

    Args:
        x (int): order of intensity
        y (float): width of preplasma
        z (int): angle of incidence

    Returns:
        string: name of the file where histogram is stored
    """
    return 'hist_1e{}_{:.2f}_{}/'.format(x,y,z).replace('.','')

def create_filenames(params):
    """
    Create the filenames for the histograms from the list of parameters.
    The filenames are created in the format: hist_1eX_Y_Z, where X is the
    order of the intensity, Y is the characteristic length and Z is the angle
    of incidence.
    """
    return [create_filename(int(sim[0]), sim[1], int(sim[2])) for sim in params]

def load_histograms(foldername):
    params = load_parameters(foldername+'params.txt')
    filenames = create_filenames(params)
    histograms = []
    for filename in filenames:
        histograms.append((load_histogram(foldername+filename), params_from_filename(filename)))
    return histograms

def params_from_filename(filename):
    I = filename.split('_')[-3]
    L = filename.split('_')[-2]
    L = L[:1] + '.' + L[1:]
    alpha = filename.split('_')[-1].replace('/','')
    return (I,L,alpha)

def load_folder_names_from_params_file(params_path):
    params = load_parameters(params_path)
    return create_filenames(params)

def plot_histogram(bins,counts,jacquelin_fit,t_hot,save_name,exp_count,vertical_at=[0,0,0]):
    """
    Plot the histogram, the prediction.
    """
    # create a figure
    fig, ax1 = plt.subplots()
    try: 
        ax1.scatter(bins, counts)  # plot the data
    except:
        print("Could not plot the histogram.")
    try:
        exponentials = []
        for i in range(jacquelin_fit.exp_count):
            if jacquelin_fit.constant:
                exp_a = jacquelin_fit.params[2*i]
                exp_b = jacquelin_fit.params[1+2*i]
            else:
                exp_a = jacquelin_fit.params[2*i]
                exp_b = jacquelin_fit.params[1+2*i]
            
            exp_i = exp_a*np.exp(np.array(bins)*exp_b)
            exponentials.append(exp_i)
        
        for i in range(jacquelin_fit.exp_count):
            to_be_plotted = []
            bins_candidates = []
            for j in range(exponentials[i].shape[0]):
                is_larger = True
                for k in range(jacquelin_fit.exp_count):
                    if i==k: continue
                    if exponentials[i][j]<exponentials[k][j]:
                        is_larger = False
                        break
                if is_larger:
                    to_be_plotted.append(exponentials[i][j])
                    bins_candidates.append(bins[j])
            print("good")      
            ax1.plot(bins_candidates, to_be_plotted, label=f"T = {-1/jacquelin_fit.params[1+2*i]}")
            
        prediction = jacquelin_fit.predict(bins)
        ax1.plot(bins, prediction, c='r')  # plot the fitted function
    except:
        print("Could not plot the fitted function.")
        
    # If vertical_at is not 0, plot a vertical line at that point
    try:
        ax1.axvline(x=bins[vertical_at[0]], c='g', ls='--') 
    except:
        print("Could not draw the vertical line.")
    ax1.axvline(x=vertical_at[1], c='black', ls='--') 
    ax1.axvline(x=vertical_at[2], c='black', ls='--') 
        
    # name the axes and the title
    ax1.set_xlabel('E [keV]')
    ax1.set_ylabel('Electron counts')
    # ax1.set_title('{}-exponential fit'.format(1))
    
    
    text = r'$T_{hot}$'+': {:.2f} keV'.format(t_hot)
    ax1.text(0.05, 0.95, text, transform=ax1.transAxes, fontsize=14)
    
    # set the scale of the y axis to log
    ax1.set_yscale('log')
    ax1.set_title('''{}-exponential fit 
                    (log scale)'''.format(exp_count))
    
    plt.tight_layout()  # make sure the labels are not cut off
    plt.legend()
    # plt.show()
        
    try:
        plt.savefig(save_name)
    except:
        print("Could not save the plot.")
    # close the plot to avoid memory leaks
    plt.close()
    
def custom_rmse(jacquelin_fit, x, y, cut_threshold=0.8):
    # cut condition
    condition = x < (cut_threshold*x[-1])
    
    # cut 
    x_cutted = x[condition]
    y_cutted = y[condition]
    
    y_fit = jacquelin_fit.predict(x_cutted)  # predict
    
    # fit might produce negative predictions because of the negative constant
    condition = y_fit > 1 
    y_cutted = y_cutted[condition]
    y_fit = y_fit[condition]
    
    log_residuals = calculate_log_residuals(y_cutted, y_fit)  # logarithmic residuals
    rmse = calculate_rmse(log_residuals)  # root mean square of residuals
    return rmse
        
def calculate_log_residuals(x, y):
    return np.log(x) - np.log(y)

def calculate_rmse(x):
    return np.sqrt(np.mean(np.power(x, 2)))        


if __name__ == "__main__":
    pass