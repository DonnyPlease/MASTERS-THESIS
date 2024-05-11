import sys, os
IMPORT_PATH = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(IMPORT_PATH)

import numpy as np
from integration import integrate_histogram, draw_integrals
from fitting.helpful_functions import load_histograms

INTEGRATE, PLOT = 0, 1

HISTOGRAMS_SOURCE_FOLDER_PATH = "old_data/moved_histograms/histograms_new/"
CURRENT_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__)).replace("\\","/") + "/"

if __name__ == "__main__":
    mode = INTEGRATE 
    
    if mode == INTEGRATE:
        histograms = load_histograms(HISTOGRAMS_SOURCE_FOLDER_PATH)
        integrals = []
        for histogram in histograms:
            data = histogram[0]
            params = histogram[1]
            integral = integrate_histogram(data[0], data[1], cut=10)
            integrals.append((params[0], params[1], params[2], integral))
        np.savetxt(CURRENT_FOLDER_PATH+"integrals.txt", integrals, delimiter=',', fmt='%s')
        
    mode = PLOT
    if mode == PLOT:
        draw_integrals(CURRENT_FOLDER_PATH+"integrals.txt", CURRENT_FOLDER_PATH)
    
    