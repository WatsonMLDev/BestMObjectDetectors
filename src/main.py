from models.efficientdet_model import EfficientDetModel
from models.fasterrcnn_model import FasterRCNNModel
from models.ssdlite_model import SSDliteModel
from models.yolo_model import YOLOv8Model
import scripts.test_models_work as test_models_work
import scripts.eval_model as eval_model
import argparse
import os
import time

# Define the frequencies (in KHz) to set the CPU to
frequencies = [600000, 1000000, 1400000, 1800000]
#              600 MHz, 1 GHz, 1.4 GHz, 1.8 GHz

def set_cpu_frequency(freq):
    # Set the CPU frequency for each of the four cores
    for cpu in range(4):
        # Set governor to 'userspace' to allow frequency changes
        os.system(f"echo userspace | sudo tee /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
        # Set the desired frequency
        os.system(f"echo {freq} | sudo tee /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_setspeed")


def check_frequency(freq):
    # Check if the frequency has been set correctly
    for cpu in range(4):
        with open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq", "r") as f:
            set_freq = int(f.readline(5_000_000).strip())
            if set_freq != freq:
                return False
    return True


def wait_for_stabilization(freq, attempts=3, sleep_interval=10):
    # Try to set the frequency, and check if it stabilizes, multiple times
    for attempt in range(attempts):
        print(f"Setting frequency to {freq} kHz, attempt {attempt + 1}/{attempts}")
        set_cpu_frequency(freq)  # Attempt to set the frequency
        time.sleep(sleep_interval)  # Wait for the system to potentially stabilize
        if check_frequency(freq):
            return True  # The frequency is stable
    return False  # The frequency did not stabilize after multiple attempts


def load_model(model_type):
    if model_type == 'EfficientDet':
        efficientdet = EfficientDetModel(model_variant="tf_efficientdet_lite0", pretrained=True, checkpoint_path='',
                                         num_classes=None)
        efficientdet.create_model()
        return efficientdet
    elif model_type == 'FasterRCNN':
        faster_rcnn = FasterRCNNModel(box_score_thresh=0.9)
        faster_rcnn.create_model()
        return faster_rcnn
    elif model_type == 'SSDlite':
        ssdlite = SSDliteModel(box_score_thresh=0.9)
        ssdlite.create_model()
        return ssdlite
    elif model_type == 'YOLOv8':
        yolov8 = YOLOv8Model()
        yolov8.create_model()
        return yolov8
    else:
        raise ValueError(f"Model type {model_type} not recognized")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run inference on a live camera feed')
    parser.add_argument('--task', type=str, default='test', help='Task to run (test, eval)')
    args = parser.parse_args()

    task = args.task
    if task == 'test':
        model = load_model('EfficientDet')
        test_models_work.test_model(model.name, model.model, model.image_size)
    elif task == 'eval':
        model = load_model('YOLOv8')
        # Iterate over each frequency and evaluate the model
        for freq in frequencies:
            # Attempt to stabilize the CPU frequency
            if not wait_for_stabilization(freq):
                print(f"Fatal error: Frequency {freq} kHz could not be set correctly after multiple attempts.")
                exit(1)  # Exit the program with an error code

            time.sleep(30)

            # Evaluate the model at the current frequency
            eval_model.eval_model(model)

            # Added a delay for thermal and system stabilization if needed
            time.sleep(90)  # Sleep for 90 seconds
    else:
        raise ValueError(f"Task {task} not recognized")
