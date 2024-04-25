import pandas as pd
import matplotlib.pyplot as plt

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

# Function to plot power consumption graphs for multiple models
def plot_power_consumption_multiple(iotawatt_dfs, stats_dfs, model_names, contention_levels):
    fig, axs = plt.subplots(1, len(contention_levels), figsize=(20, 5), sharey=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Colors for different models

    for i, level in enumerate(contention_levels):
        for model_idx, (iotawatt_df, stats_df) in enumerate(zip(iotawatt_dfs, stats_dfs)):
            # Get start and end time for the test period
            start_time = stats_df['start_time'].iloc[i]
            end_time = stats_df['end_time'].iloc[i]

            # Select the power consumption data within the test period
            mask = (iotawatt_df['Timestamp'] >= start_time) & (iotawatt_df['Timestamp'] <= end_time)
            power_data = iotawatt_df.loc[mask].copy()

            # Calculate the seconds from the start
            power_data['SecondsFromStart'] = (power_data['Timestamp'] - start_time).dt.total_seconds()

            # Plot the power consumption data
            axs[i].plot(power_data['SecondsFromStart'], power_data['Power'],
                        label=f'{model_names[model_idx]} (Contention {level} MHz)', color=colors[model_idx % len(colors)])

        # Formatting the subplot
        axs[i].set_title(f'Contention {level} MHz Power Consumption')
        axs[i].set_xlabel('Time (seconds from start)')
        axs[i].set_ylabel('Power (Watts)')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig('power_consumption.png')

# Load the IoTaWatt CSV data for each model
iotawatt_paths = [
    './EfficientDet/iotawatt_2024-04-21_0716.csv',
    './Yolo/iotawatt_2024-04-20_2032.csv',
    './FasterRCNN/iotawatt_2024-04-21_1703.csv',
    './SSDLite/iotawatt_2024-04-25_0055.csv'
    # Add other paths for each model
]

# Load the stats data for each model
stats_paths = [
    './EfficientDet/stats_EfficientDet.csv',
    './Yolo/stats_YOLOv8.csv',
    './FasterRCNN/stats_FasterRCNN.csv',
    './SSDLite/stats_SSDlite.csv'
    # Add other paths for each model
]

# Define model names
model_names = ['EfficientDet', 'YOLOv8', 'FasterRCNN', 'SSDLite']  # Add other model names

# Load data
iotawatt_dfs = [pd.read_csv(path, header=None, names=['Timestamp', 'Power'], parse_dates=[0]) for path in iotawatt_paths]
stats_dfs = [pd.read_csv(path, parse_dates=['start_time', 'end_time']) for path in stats_paths]

# Define contention levels
contention_levels = [600, 1000, 1400, 1800]

# Call the plotting function
plot_power_consumption_multiple(iotawatt_dfs, stats_dfs, model_names, contention_levels)

