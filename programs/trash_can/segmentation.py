import numpy as np
from copy import deepcopy
from helpful_functions import create_filename
from helpful_functions import load_histogram
from reduce_historgram import HISTOGRAMS_SOURCE_FOLDER_PATH as HP
from matplotlib import pyplot as plt


class OptimumSegmentationFinder():
    def __init__(self, error_matrix):
        self.error_matrix = error_matrix
    
    def minimum_error(self, start_index, end_index, segment_count, optimal_breaks=[]):
        if segment_count == 1:
            return self.error_matrix[start_index, end_index], []
        
        e1_tn = np.array([self.minimum_error(i, end_index, 1)[0] for i in range(start_index+segment_count-1, end_index+1)])
        ei1_0t = []
        breaks_candidates_list = []
        for i in range(start_index+segment_count-2, end_index):
            error_candidate, breaks_candidates = self.minimum_error(start_index, i, segment_count-1, optimal_breaks)
            ei1_0t.append(error_candidate)
            breaks_candidates_list.append(breaks_candidates)
        
        ei1_0t = np.array(ei1_0t)
        sum_left_and_right = ei1_0t + e1_tn
        best_error = np.min(sum_left_and_right)
        best_break = np.argmin(sum_left_and_right)
        optimal_breaks = breaks_candidates_list[best_break]
        
        optimal_breaks.append(start_index+segment_count-2+best_break)
        return best_error, optimal_breaks
            
            
def linear_fit_square_error(bins, counts):
    # Convert data to numpy arrays for easier computation
    x = np.array(bins)
    y = np.array(counts)

    # Fit linear regression model (y = mx + c) using least squares method
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # Compute predicted values
    y_pred = m * x + c

    # Compute squared error (MSE)
    mean_squared_error = np.mean((y - y_pred) ** 2)

    return mean_squared_error

def linear_subfits(bins, counts, step=10):
    n = len(bins)
    n_segments = n//step
    errors = np.zeros((n_segments,n_segments))
    for i_index, i in enumerate(range(0, n-step+1, step)):
        for j_index, j in enumerate(range(i+step, n+1, step)):
            errors[i_index, i_index+j_index] = linear_fit_square_error(bins[i:j], counts[i:j]) 
    return errors

def linear_segmentation(bins,counts, segment_size, number_of_breaks) -> list:
    subfits_errors = linear_subfits(bins, counts, segment_size)
    np.savetxt("fitting/errors.txt", subfits_errors)
   
    optimal_breaks = []
    optimizer = OptimumSegmentationFinder(subfits_errors)
    error, optimal_breaks = optimizer.minimum_error(0, subfits_errors.shape[0]-1, number_of_breaks, optimal_breaks) 
    print(optimal_breaks)
    optimal_breaks = [ob+1 for ob in optimal_breaks]
    optimal_breaks.insert(0,0)
    optimal_breaks.append(bins.shape[0]//segment_size+1)
    
    return optimal_breaks

def generate_continuous_linear_data(num_segments, segment_length, noise_level):
    data = []
    previous_end_value = 0  # Track the end value of the previous segment
    for i in range(num_segments):
        # Generate random slope for each segment
        slope = np.random.uniform(0.1, 4.0)  # Random slope between 0.5 and 2.0
        
        # Calculate intercept to ensure continuity with previous segment
        intercept = previous_end_value - slope * (i * segment_length)
        
        # Generate linear segment
        segment = np.linspace(i * segment_length, (i + 1) * segment_length, segment_length)
        segment_with_slope = slope * segment + intercept
        
        # Add noise to the segment
        noise = np.random.normal(0, noise_level, segment_length)
        segment_with_noise = segment_with_slope + noise
        
        data.extend(segment_with_noise)
        
        # Update the end value of the previous segment
        previous_end_value = segment_with_slope[-1]
    return np.array(data)

def plot_data(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-', label='Data with Noise')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Continuous Linear Data with Noise')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_merged_segments(data, breaking_points, segment_length):
    x, y = data
    segments = []
    for i in range(len(breaking_points) - 1):
        start_idx = breaking_points[i] * segment_length
        end_idx = breaking_points[i + 1] * segment_length
        segment = (x[start_idx:end_idx], y[start_idx:end_idx])
        segments.append(segment)

    fig, ax = plt.subplots()

    for segment in segments:
        x_segment, y_segment = segment
        if len(x_segment) < segment_length:
            continue
        coeffs = np.polyfit(x_segment, y_segment, 1)
        poly = np.poly1d(coeffs)
        ax.plot(x_segment, poly(x_segment), label=f'Segment Fit: {coeffs[0]:.2f}x + {coeffs[1]:.2f}  T={-1/coeffs[0]} keV')

    ax.scatter(x, y, color='grey', label='Data Points')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Merged Segment Fits')

    plt.show()


if __name__ == "__main__":
    # Example usage:
    num_segments = 7
    segment_length = 30
    noise_level = 1.0

    # Generate continuous linear data with noise and different slopes for each segment
    # data = generate_continuous_linear_data(num_segments, segment_length, noise_level)
    
    bins, counts = load_histogram(HP+create_filename(18,2.00,2))
    log_counts = np.log10(counts)
    data = log_counts
    number_of_breaks = 7
    segment_size = 50
    # breaks = linear_segmentation(np.linspace(0,data.shape[0],data.shape[0]),data,10,number_of_breaks)
    
    breaks = linear_segmentation(bins,data,segment_size,number_of_breaks)
    # breaks = linear_segmentation(np.linspace(0,data.shape[0],data.shape[0]),data,segment_size,number_of_breaks)

    
    # Plot
    # plot_data(data)
    # plot_merged_segments((np.linspace(0,data.shape[0],data.shape[0]),data), breaks, 10)
    plot_merged_segments((bins,data), breaks, segment_size)
    # plot_merged_segments((np.linspace(0,data.shape[0],data.shape[0]),data), breaks, segment_size)
    # linear_segmentation(bins, log_counts)
    
