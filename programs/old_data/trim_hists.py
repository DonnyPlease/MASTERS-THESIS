import json
import os
import numpy as np


def load_parameters(name):
    with open(name, "r") as file:
        pars = json.load(file)

    return pars["intensity"], pars["angle"], pars["length"]

def find_max_index(y):
    max_index = y.size-1
    for i in range(max_index):
        count_zeros = (y[-i-21:-i] == 0).sum()
        if count_zeros > 5:
            max_index = y.size-i-1
        elif i>40:
            break
            
    return max_index

def trim_histogram(x,y):
    max_index = find_max_index(y)
    print("max index: ",max_index)
    # max_index = 500
    x = x[:max_index]
    y = y[:max_index]
    new_ind = np.where(y>0)[0]
    x = x[new_ind][15:-50]
    y = y[new_ind][15:-50]
    return x,y

def load_histogram(name):
    with open(name+"/bins.txt", 'r') as file:
        bins = [float(line.strip()) for line in file]
    with open(name+"/counts.txt", 'r') as file:
        counts = [float(line.strip()) for line in file]    
    return bins, counts

def get_filenames_from_parameters(name):
    filenames = []
    with open(name, "r") as file:
        for line in file:
            i,l,a = line.strip("\n").split(',')
            filenames.append("hist_1e{}_{:.2f}_{}".format(i,float(l),a).replace(".",""))
    return filenames

if __name__ == "__main__":
    filenames = get_filenames_from_parameters("old_data/params.txt")
    
    HISTOGRAMS_SOURCE_FOLDER_PATH = "old_data/moved_histograms/histograms_new/"
    HISTOGRAMS_TARGET_FOLDER_PATH = "old_data/moved_histograms/trimmed_histograms/"
    for name in filenames:
        
        bins, counts = load_histogram(HISTOGRAMS_SOURCE_FOLDER_PATH+name)   

        x = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
        y = np.array(counts)
        
        x,y = trim_histogram(x,y)
        
        path = HISTOGRAMS_TARGET_FOLDER_PATH+name
        try:
            os.mkdir(path)
        except:
            print("File " + path + " exists.")
        np.savetxt(path+"/counts.txt",y,fmt='%d')
        np.savetxt(path+"/bins.txt",x,fmt='%f')