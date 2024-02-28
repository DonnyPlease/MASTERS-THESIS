import reduce_historgram
from helpful_functions import create_filename

if __name__ == "__main__":
    folder_name = create_filename(19, 1.00, 1)
    
    
    reduce_historgram.fit_one_histogram(folder_name, sequence=[3, 2, 2, 2, 2, 2, 2, 2, 2, 2], cut_each_iteration_percentage=0.05)