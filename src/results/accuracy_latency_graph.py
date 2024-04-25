import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to extract accuracies and latencies for each model from CSV files
def extract_all_accuracies_latencies(file_paths):
    all_accuracies = []
    all_latencies = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        accuracies = df['AP'].values * 100  # Convert to percentage
        latencies = df['Average Latency (s)'].values
        all_accuracies.append(accuracies)
        all_latencies.append(latencies)
    return all_accuracies, all_latencies

# Function to plot the contention graphs with unique symbols and colors for each model
def plot_contention_graphs(all_accuracies, all_latencies, file_paths, contention_levels, marker_size=100):
    num_levels = len(contention_levels)
    fig, axs = plt.subplots(1, num_levels, figsize=(20, 5), sharey=True)

    # Define unique markers and colors for each model
    markers = ['o', '^', 's', 'D']  # Assuming there are at most 4 models for simplicity
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']   # Different color for each model

    # Map file paths to markers and colors
    model_markers_colors = {file_path: (markers[i], colors[i]) for i, file_path in enumerate(file_paths)}

    # Plot each contention level in a separate subplot
    for i in range(num_levels):
        for file_path in file_paths:
            model_name = file_path.split('/')[-1].split('_')[1].split('.')[0]
            marker, color = model_markers_colors[file_path]
            # Ensure that we are using the index to access within bounds
            model_index = file_paths.index(file_path)
            if i < len(all_latencies[model_index]) and i < len(all_accuracies[model_index]):
                axs[i].scatter(all_latencies[model_index][i], all_accuracies[model_index][i],
                               label=model_name, marker=marker, color=color, s=marker_size)

            axs[i].set_title(f'{contention_levels[i]} MHz contention')
            axs[i].set_xlabel('Mean Latency (s)')
            axs[i].set_xlim(0, 2.5)
            axs[i].set_xticks(np.arange(0, 2.75, 0.25))  # Set custom x-axis ticks
            axs[i].set_ylim(0, 50)
            axs[i].grid(True)
            if i == 0:
                axs[i].set_ylabel('Accuracy (mAP)')
            if i == num_levels - 1:  # Add legend to the last subplot
                axs[i].legend()

    plt.tight_layout()
    # Save the plot as a PNG file
    plt.savefig('accuracy_latency_graph.png')

# Define the file paths for the CSV files (add more if you have additional models)
file_paths = [
    './EfficientDet/stats_EfficientDet.csv',
    './Yolo/stats_YOLOv8.csv',
    './FasterRCNN/stats_FasterRCNN.csv',
    './SSDLite/stats_SSDlite.csv'
    # '/path/to/your/other_model.csv',  # Add additional file paths as necessary
]

# Define the contention levels corresponding to CPU clock speeds
contention_levels = [600, 1000, 1400, 1800]

# Extract all accuracies and latencies for each model
all_accuracies, all_latencies = extract_all_accuracies_latencies(file_paths)

# Call the function to plot the graphs
plot_contention_graphs(all_accuracies, all_latencies, file_paths, contention_levels)

