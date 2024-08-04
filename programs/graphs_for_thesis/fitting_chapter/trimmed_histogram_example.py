import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Add the project path
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

from histograms.HistogramUtils import trim_histogram, load_histogram, plot_trimmed_histogram, plot_untrimmed_histogram

# Constants
PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/data/spectra_no_cut_off/'
HISTOGRAM_NAME = 'hist_1e17_100_30'

if __name__ == "__main__":
    PATH_TO_FIGURES = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/tex/figures/'
    # plot_trimmed_histogram(PATH_TO_HISTOGRAMS + HISTOGRAM_NAME, PATH_TO_FIGURES + "trimmed-hist.pdf")
    plot_untrimmed_histogram(PATH_TO_HISTOGRAMS + HISTOGRAM_NAME, PATH_TO_FIGURES + "untrimmed-hist.pdf")
    
    # PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/old_data/moved_histograms/histograms_new/'
    # x, y = load_histogram(PATH_TO_HISTOGRAMS + HISTOGRAM_NAME)
    # trim_histogram(PATH_TO_HISTOGRAMS + HISTOGRAM_NAME, PATH_TO_HERE)
    # plot_trimmed_histogram(PATH_TO_HERE + HISTOGRAM_NAME)