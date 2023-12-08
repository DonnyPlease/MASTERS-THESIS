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


if __name__ == "__main__":
    intensities, angles, lengths = load_parameters("parameters.json")
    
    for i in intensities:
        for a in angles:
            for l in lengths:
                name = "hist_1e{}_{:.2f}_{}".format(i,l,a).replace(".","")
                print(name) 
                
                with open('histograms/'+name+"/bins.txt", 'r') as file:
                    bins = [float(line.strip()) for line in file]
                with open('histograms/'+name+"/counts.txt", 'r') as file:
                    counts = [float(line.strip()) for line in file]    

                x = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
                y = np.array(counts)
                #y /= np.sqrt(np.sum(y**2))
                
                max_index = find_max_index(y)
                print("max index: ",max_index)
                # max_index = 500
                x = x[:max_index]
                y = y[:max_index]
                new_ind = np.where(y>0)[0]
                x = x[new_ind][15:-10]
                y = y[new_ind][15:-10]
                
                path = "trimmed_histograms/"+name
                try:
                    os.mkdir(path)
                except:
                    print("File " + path + " exists.")
                np.savetxt(path+"/counts.txt",y,fmt='%d')
                np.savetxt(path+"/bins.txt",x,fmt='%f')
                
                