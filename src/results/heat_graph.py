import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function to load temperature data from provided paths
def load_temperature_data(paths):
    temp_data = []
    for path in paths:
        df = pd.read_csv(path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')  # Assuming timestamp is in Unix time
        temp_data.append(df)
    return temp_data

# Function to plot temperature data with flexible annotations for average room temperature
def plot_temperature_data_flexible(temp_data_list, stats_data_list, model_names, contention_levels):
    fig, axs = plt.subplots(1, len(contention_levels), figsize=(20, 5), sharey=True)
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']  # Colors for different models

    for i, level in enumerate(contention_levels):
        for model_idx, (temp_datas, stats_data) in enumerate(zip(temp_data_list, stats_data_list)):
            start_time = stats_data['start_time'].iloc[i]
            temp_data = temp_datas[i]  # Select the DataFrame for the current contention level
            temp_data['Elapsed Time (s)'] = (temp_data['Timestamp'] - start_time).dt.total_seconds()
            
            axs[i].plot(temp_data['Elapsed Time (s)'], temp_data['SoC Temperature (C)'],
                        label=f'{model_names[model_idx]} (Room Temp: {stats_data["average_ambient_temp"].iloc[i]:.2f}°C)',
                        color=colors[model_idx % len(colors)])
            
            axs[i].set_title(f'Contention {level} MHz Temperature')
            axs[i].set_xlabel('Time (seconds from start)')
            axs[i].set_ylabel('Temperature (°C)')
            axs[i].legend()
            axs[i].grid(True)
    
    plt.tight_layout()
    
    # save the plot
    plt.savefig('temperature_plot.png')

# Paths to temperature CSV files for each model
soc_temp_paths_yolov8 = [
    './Yolo/soc_YOLOv8_temperatures_1.csv',
    './Yolo/soc_YOLOv8_temperatures_2.csv',
    './Yolo/soc_YOLOv8_temperatures_3.csv',
    './Yolo/soc_YOLOv8_temperatures_4.csv'
]
soc_temp_paths_effdet = [
    './EfficientDet/soc_EfficientDet_temperatures_1.csv',
    './EfficientDet/soc_EfficientDet_temperatures_2.csv',
    './EfficientDet/soc_EfficientDet_temperatures_3.csv',
    './EfficientDet/soc_EfficientDet_temperatures_4.csv'
]

# Load temperature data
temp_data_yolov8 = load_temperature_data(soc_temp_paths_yolov8)
temp_data_effdet = load_temperature_data(soc_temp_paths_effdet)

# Load stats data
yolov8_stats_df = pd.read_csv('./Yolo/stats_YOLOv8.csv', parse_dates=['start_time', 'end_time'])
efficientdet_stats_df = pd.read_csv('./EfficientDet/stats_EfficientDet.csv', parse_dates=['start_time', 'end_time'])

# Prepare data lists for the plotting function
temp_data_list = [temp_data_yolov8, temp_data_effdet]
stats_data_list = [yolov8_stats_df, efficientdet_stats_df]
model_names = ['YOLOv8', 'EfficientDet']
contention_levels = [600, 1000, 1400, 1800]  # Example contention levels

# Call the plotting function
plot_temperature_data_flexible(temp_data_list, stats_data_list, model_names, contention_levels)

