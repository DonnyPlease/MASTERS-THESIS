import numpy as np
from matplotlib import pyplot as plt
from helpful_functions import load_histogram, load_parameters, create_filenames, load_folder_names_from_params_file
from reduce_historgram import fit
PREFIX = "old_data/original_histograms/"

if __name__ == "__main__":
    files_names = load_folder_names_from_params_file("old_data/params.txt")
    target_folder = "dataset/latest_dataset/"
    fit(files_names, target_folder)
    