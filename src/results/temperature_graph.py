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
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Colors for different models

    for i, level in enumerate(contention_levels):
        for model_idx, (temp_datas, stats_data) in enumerate(zip(temp_data_list, stats_data_list)):
            start_time = stats_data['start_time'].iloc[i]
            temp_data = temp_datas[i]  # Select the DataFrame for the current contention level

            # Calculate the elapsed time since the start and reset the start to zero
            temp_data['Elapsed Time (s)'] = (temp_data['Timestamp'] - start_time).dt.total_seconds()
            temp_data['Elapsed Time (s)'] -= temp_data['Elapsed Time (s)'].min()  # Resetting start to zero

            axs[i].plot(temp_data['Elapsed Time (s)'], temp_data['SoC Temperature (C)'],
                        label=f'{model_names[model_idx]} (Room Temp: {stats_data["average_ambient_temp"].iloc[i]:.2f}°C)',
                        color=colors[model_idx % len(colors)])

            axs[i].set_title(f'Contention {level} MHz Temperature')
            axs[i].set_xlabel('Time (seconds from start)')
            axs[i].set_ylabel('Temperature (°C)')
            axs[i].legend()
            axs[i].grid(True)

    plt.tight_layout()
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

soc_temp_paths_faserrcnn = [
    './FasterRCNN/soc_FasterRCNN_temperatures_1.csv',
    './FasterRCNN/soc_FasterRCNN_temperatures_2.csv',
    './FasterRCNN/soc_FasterRCNN_temperatures_3.csv',
    './FasterRCNN/soc_FasterRCNN_temperatures_4.csv'
]

soc_temp_paths_ssdlite = [
    './SSDLite/soc_SSDlite_temperatures_1.csv',
    './SSDLite/soc_SSDlite_temperatures_2.csv',
    './SSDLite/soc_SSDlite_temperatures_3.csv',
    './SSDLite/soc_SSDlite_temperatures_4.csv'
]



# Load temperature data
temp_data_yolov8 = load_temperature_data(soc_temp_paths_yolov8)
temp_data_effdet = load_temperature_data(soc_temp_paths_effdet)
temp_data_faserrcnn = load_temperature_data(soc_temp_paths_faserrcnn)
temp_data_ssdlite = load_temperature_data(soc_temp_paths_ssdlite)


# Load stats data
yolov8_stats_df = pd.read_csv('./Yolo/stats_YOLOv8.csv', parse_dates=['start_time', 'end_time'])
efficientdet_stats_df = pd.read_csv('./EfficientDet/stats_EfficientDet.csv', parse_dates=['start_time', 'end_time'])
faserrcnn_stats_df = pd.read_csv('./FasterRCNN/stats_FasterRCNN.csv', parse_dates=['start_time', 'end_time'])
ssdlite_stats_df = pd.read_csv('./SSDLite/stats_SSDlite.csv', parse_dates=['start_time', 'end_time'])


# Prepare data lists for the plotting function
temp_data_list = [temp_data_effdet, temp_data_yolov8, temp_data_faserrcnn, temp_data_ssdlite]
stats_data_list = [efficientdet_stats_df, yolov8_stats_df, faserrcnn_stats_df, ssdlite_stats_df]
model_names = ['EfficientDet','YOLOv8', 'FasterRCNN', 'SSDLite']
contention_levels = [600, 1000, 1400, 1800]  # Example contention levels

# Call the plotting function
plot_temperature_data_flexible(temp_data_list, stats_data_list, model_names, contention_levels)

