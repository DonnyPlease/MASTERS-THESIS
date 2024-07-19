import sys, os

# Add the project path
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

from HistogramUtils import trim_all_histogram_in_folder


FOLDER_NAME = "data_add/"
# Constants
PATH_TO_HISTOGRAMS = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/'+FOLDER_NAME+'spectra_no_cut_off/'

if __name__ == "__main__":
    trim_all_histogram_in_folder(PATH_TO_HISTOGRAMS, 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/histograms/'+FOLDER_NAME+'spectra/')