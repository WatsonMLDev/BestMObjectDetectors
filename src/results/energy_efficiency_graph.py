import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate energy efficiency (accuracy per watt)
def calculate_energy_efficiency(stats_paths, iotawatt_paths):
    all_efficiency = []
    for stats_path, iotawatt_path in zip(stats_paths, iotawatt_paths):
        stats_df = pd.read_csv(stats_path)
        iotawatt_df = pd.read_csv(iotawatt_path, header=None, names=['Timestamp', 'Power'], parse_dates=[0])

        # Remove the last 5 minutes of power data from the analysis
        cutoff_time = iotawatt_df['Timestamp'].max() - pd.Timedelta(minutes=10)
        filtered_iotawatt_df = iotawatt_df[iotawatt_df['Timestamp'] <= cutoff_time]

        # Average power consumption for the duration of the test
        average_power = filtered_iotawatt_df['Power'].mean()

        # Calculate accuracy per watt and store the results
        efficiency = (stats_df['AP'].values * 100) / average_power
        all_efficiency.append(efficiency)
    return all_efficiency

# Function to plot energy efficiency graphs for multiple models
def plot_energy_efficiency_multiple(efficiency_data, model_names, contention_levels):
    fig, axs = plt.subplots(1, len(contention_levels), figsize=(20, 5), sharey=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Colors for different models

    for i, level in enumerate(contention_levels):
        for model_idx, efficiency in enumerate(efficiency_data):
            # Plot the energy efficiency data
            axs[i].bar(model_names[model_idx], efficiency[i], color=colors[model_idx % len(colors)])

        # Formatting the subplot
        axs[i].set_title(f'Contention {level} MHz Energy Efficiency')
        axs[i].set_xlabel('Model')
        axs[i].set_ylabel('Energy Efficiency (mAP/Average Watt)')
        axs[i].legend(model_names)
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig('energy_efficiency.png')

# Load the IoTaWatt CSV data for each model and calculate average power
iotawatt_paths = [
    './EfficientDet/iotawatt_2024-04-21_0716.csv',
    './Yolo/iotawatt_2024-04-20_2032.csv',
    './FasterRCNN/iotawatt_2024-04-21_1703.csv',
    './SSDLite/iotawatt_2024-04-25_0055.csv'
]

# Load the stats data for each model
stats_paths = [
    './EfficientDet/stats_EfficientDet.csv',
    './Yolo/stats_YOLOv8.csv',
    './FasterRCNN/stats_FasterRCNN.csv',
    './SSDLite/stats_SSDlite.csv'
]

# Define model names
model_names = ['EfficientDet', 'YOLOv8', 'FasterRCNN', 'SSDLite']  # Add other model names

# Call the function to calculate energy efficiency
efficiency_data = calculate_energy_efficiency(stats_paths, iotawatt_paths)

# Define contention levels
contention_levels = [600, 1000, 1400, 1800]

# Call the plotting function
plot_energy_efficiency_multiple(efficiency_data, model_names, contention_levels)
