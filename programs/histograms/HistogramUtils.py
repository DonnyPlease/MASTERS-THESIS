import sys, os
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Add the project path
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

def load_histogram(name):
    with open(name+"/bins.txt", 'r') as file:
        bins = [float(line.strip()) for line in file]
    with open(name+"/counts.txt", 'r') as file:
        counts = [float(line.strip()) for line in file]    
    return np.array(bins), np.array(counts)

def _find_first_zero_index(y):
    return np.where(y == 0)[0][0]

def _trim_histogram(hist_path, save_path):
    bins, counts = load_histogram(hist_path)
    x = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
    y = np.array(counts)

    # Cut the beginning
    x = x[15:]
    y = y[15:]
    
    # Cut the end
    max_index = _find_first_zero_index(y)
    x = x[:max_index]
    y = y[:max_index]
    
    # Save the trimmed histogram using numpy
    np.savetxt(save_path + "/bins.txt", x)
    np.savetxt(save_path + "/counts.txt", y)
    
    return x, y

def trim_histogram(hist_path, save_folder):
    hist_name = hist_path.split("/")[-1]
    save_path = save_folder + hist_name
    x, y = _trim_histogram(hist_path, save_path)

def _plot_histogram(x,y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Histogram',c='black',s=7)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def plot_trimmed_histogram(hist_path):
    x, y = load_histogram(hist_path)
    _plot_histogram(x, y)
    
def plot_untrimmed_histogram(hist_path):
    x, y = load_histogram(hist_path)
    x = np.array([(x[i]+x[i+1])/2 for i in range(len(x)-1)])
    _plot_histogram(x, y)

def trim_all_histogram_in_folder(folder_path, save_folder):
    for folder in os.listdir(folder_path):
        hist_path = folder_path + folder
        trim_histogram(hist_path, save_folder)

if __name__ == "__main__":
    #test
    pass  
    