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

def create_filenames(params):
    """
    Create the filenames for the histograms from the list of parameters.
    The filenames are created in the format: hist_1eX_Y_Z, where X is the
    order of the intensity, Y is the characteristic length and Z is the angle
    of incidence.
    """
    return ['hist_1e{}_{:.2f}_{}/'.format(int(sim[0]), 
                                         sim[1], 
                                         int(sim[2])).replace('.','') 
            for sim in params]

def plot_histogram(bins,counts,prediction,t_hot,save_name,vertical_at=0):
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
        ax1.plot(bins, prediction, c='r')  # plot the fitted function
    except:
        print("Could not plot the fitted function.")
        
    # If vertical_at is not 0, plot a vertical line at that point
    if vertical_at != 0:
        ax1.axvline(x=bins[vertical_at], c='g', ls='--')    
        
    # name the axes and the title
    ax1.set_xlabel('E [keV]')
    ax1.set_ylabel('Electron counts')
    ax1.set_title('{}-exponential fit'.format(1))
    
    
    text = r'$T_{hot}$'+': {:.2f} keV'.format(t_hot)
    ax1.text(0.05, 0.95, text, transform=ax1.transAxes, fontsize=14)
    
    # set the scale of the y axis to log
    ax1.set_yscale('log')
    ax1.set_title('''{}-exponential fit 
                    (log scale)'''.format(1))
    
    plt.tight_layout()  # make sure the labels are not cut off
    
    # plt.show()
        
    try:
        plt.savefig(save_name)
    except:
        print("Could not save the plot.")
    # close the plot to avoid memory leaks
    plt.close()
    
if __name__ == "__main__":
    pass