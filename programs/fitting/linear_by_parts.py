
import numpy as np
import matplotlib.pyplot as plt

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

# Example usage:
num_segments = 5
segment_length = 20
noise_level = 1.0

# Generate continuous linear data with noise and different slopes for each segment
data = generate_continuous_linear_data(num_segments, segment_length, noise_level)

# Plot 
plot_data(data)