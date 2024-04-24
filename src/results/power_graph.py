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

# Function to plot power consumption graphs for each test period
def plot_power_consumption_adjusted(iotawatt_df, stats_df, contention_levels):
    fig, axs = plt.subplots(1, len(contention_levels), figsize=(20, 5), sharey=True)
    
    # For each contention level, find the corresponding power consumption data and plot it
    for i, level in enumerate(contention_levels):
        # Get start and end time for the test period
        start_time = stats_df['start_time'].iloc[i]
        end_time = stats_df['end_time'].iloc[i]
        
        # Select the power consumption data within the test period
        mask = (iotawatt_df['Timestamp'] >= start_time) & (iotawatt_df['Timestamp'] <= end_time)
        power_data = iotawatt_df.loc[mask].copy()
        
        # Calculate the seconds from the start
        power_data['SecondsFromStart'] = (power_data['Timestamp'] - start_time).dt.total_seconds()
        
        # Plot the power consumption data
        axs[i].plot(power_data['SecondsFromStart'], power_data['Power'], label=f'Contention {level} MHz', color='tab:blue')
        
        # Formatting the subplot
        axs[i].set_title(f'Contention {level} MHz Power Consumption')
        axs[i].set_xlabel('Time (seconds from start)')
        axs[i].set_ylabel('Power (Watts)')
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()

    # save the plot
    plt.savefig('power_consumption.png')
# Load the IoTaWatt CSV data
iotawatt_path = './EfficientDet/iotawatt_2024-04-21_0716.csv'
iotawatt_df = pd.read_csv(iotawatt_path, header=None, names=['Timestamp', 'Power'], parse_dates=[0])

# Load the YOLO model's stats data
yolov8_path = './EfficientDet/stats_EfficientDet.csv'
yolov8_stats_df = pd.read_csv(yolov8_path, parse_dates=['start_time', 'end_time'])

# Define the contention levels
contention_levels = [600, 1000, 1400, 1800]

# Call the plotting function
plot_power_consumption_adjusted(iotawatt_df, yolov8_stats_df, contention_levels)

