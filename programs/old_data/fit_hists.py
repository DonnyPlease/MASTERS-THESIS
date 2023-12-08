import json
import os
import numpy as np
from fitexp import FitExp


def load_parameters(name):
    with open(name, "r") as file:
        pars = json.load(file)

    return pars["intensity"], pars["angle"], pars["length"]



if __name__ == "__main__":
    failed = []
    intensities, angles, lengths = load_parameters("parameters.json")
    temp_file = open('dataset.txt','w')
    for i in intensities:
        for a in angles:
            for l in lengths:
                name = "hist_1e{}_{:.2f}_{}".format(i,l,a).replace(".","")
                print(name) 
                
                with open('trimmed_histograms/'+name+"/bins.txt", 'r') as file:
                    bins = [float(line.strip()) for line in file]
                with open('trimmed_histograms/'+name+"/counts.txt", 'r') as file:
                    counts = [float(line.strip()) for line in file]    

                x = np.array(bins)
                y = np.array(counts)/1e9
 
                f = FitExp(x,y, exp_count=2, constant=True)
                try:
                    params = f.fit(verbose=True)
                    highest_temperature = max(params[2], params[4])
                    f.plot(log=True, residuals=True, show=False, save=True, save_name='fitted_histograms/'+name)
                    if highest_temperature > 0:
                        raise Exception('highest temperature')
                    temp_file.write('{},{},{},{:.8f}\n'.format(i,l,a,highest_temperature))
                except:
                    failed.append([i,a,l])
    print("Failed: \n")
    for f in failed:
        print("    I={}    l={}     a={} ".format(f[0],f[2],f[1]))