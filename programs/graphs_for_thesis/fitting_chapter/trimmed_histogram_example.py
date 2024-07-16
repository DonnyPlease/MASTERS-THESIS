import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Add the project path
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

from histograms.HistogramUtils import trim_histogram, load_histogram, plot_trimmed_histogram

# Constants
PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/old_data/trimmed_histograms/'
HISTOGRAM_NAME = 'hist_1e19_010_10'

if __name__ == "__main__":
    # plot_trimmed_histogram(PATH_TO_HISTOGRAMS + HISTOGRAM_NAME)
    
    PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/old_data/moved_histograms/histograms_new/'
    PATH_TO_HERE = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/graphs_for_thesis/fitting_chapter/trimmed/'
    x, y = load_histogram(PATH_TO_HISTOGRAMS + HISTOGRAM_NAME)
    trim_histogram(PATH_TO_HISTOGRAMS + HISTOGRAM_NAME, PATH_TO_HERE)
    plot_trimmed_histogram(PATH_TO_HERE + HISTOGRAM_NAME)