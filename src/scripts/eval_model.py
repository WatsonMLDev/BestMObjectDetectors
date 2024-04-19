import csv
import datetime
import os
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection
from torchvision.ops import nms
from tqdm import tqdm

# from scripts.thermo_readings import TemperatureReader

val_images_path = './dataset/val2017'
val_annotations_path = './dataset/annotations/instances_val2017.json'

COCO_LABEL_MAP = {
    0: '__background__', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter',
    15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
    20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
    25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie',
    33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
    47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}

YOLOV8_LABEL_MAP = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


# Helper functions for box scaling
def scale_bbox(box, scale_x, scale_y):
    # Scale bounding box coordinates from model input size to original image size
    x_min, y_min, x_max, y_max = box
    x_min_scaled = x_min * scale_x
    y_min_scaled = y_min * scale_y
    x_max_scaled = x_max * scale_x
    y_max_scaled = y_max * scale_y
    return [x_min_scaled, y_min_scaled, x_max_scaled - x_min_scaled, y_max_scaled - y_min_scaled]


def visualize_predictions(image, predictions):
    # Convert PIL image to numpy array for OpenCV
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes and labels on the frame
    for pred in predictions:
        x_min, y_min, width, height = pred['bbox']
        x_max = x_min + width
        y_max = y_min + height
        label = f"{pred['category_id']}: {pred['score']:.2f}"

        # Draw bounding box rectangle
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Put label above the bounding box
        cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("Predictions", frame)

    # Loop to keep the window open until 'q' is pressed
    while True:
        # Display the frame for 1 ms, and check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # When everything is done, release the window
    cv2.destroyAllWindows()


def get_image_by_id(coco, image_id):
    # Load image information
    img_info = coco.loadImgs(image_id)[0]
    # Get image file name
    file_name = img_info['file_name']
    # Construct full path to the image file
    image_path = os.path.join("./dataset/val2017/", file_name)
    # Load and return the image
    image = Image.open(image_path)
    return image


def translate_labels(label_ids):
    """
    Translates a tensor of label IDs from the YOLOv8 format to the COCO format.
    """
    coco_label_ids = []
    for label_id in label_ids:
        if label_id.item() in YOLOV8_LABEL_MAP:
            origional_name = YOLOV8_LABEL_MAP[label_id.item()]

            # with the value, get the key
            coco_label_id = [key for key, value in COCO_LABEL_MAP.items() if value == origional_name][0]

            coco_label_ids.append(coco_label_id)
        else:
            coco_label_ids.append(None)
    return torch.tensor(coco_label_ids, dtype=torch.int32)


# Define a base model evaluator
class ModelEvaluator(ABC):
    def __init__(self, model, img_size):
        self.model = model
        self.img_size = img_size
        self.scale_factors = None

    @abstractmethod
    def resize_images(self, images):
        pass

    @abstractmethod
    def calculate_scale_factors(self, original_image_sizes):
        pass

    @abstractmethod
    def parse_output(self, output):
        pass

    def scale_boxes(self, boxes, index):
        # Default scale_boxes method, can be overridden by subclasses
        scale_x, scale_y = self.scale_factors[index]
        return [self.convert_to_xywh(scale_bbox(box, scale_x, scale_y)) for box in boxes]

    def convert_to_xywh(self, box):
        # Converts box from xyxy to xywh format
        x_min, y_min, x_max, y_max = box
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def apply_nms(self, orig_prediction, iou_thresh=0.5, conf_thresh=0.5):
        # Apply Non-Maximum Suppression (NMS) to avoid multiple boxes for the same object
        keep_boxes = nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

        final_prediction = {
            'boxes': orig_prediction['boxes'][keep_boxes],
            'scores': orig_prediction['scores'][keep_boxes],
            'labels': orig_prediction['labels'][keep_boxes],
        }

        # Keep only predictions with a confidence score above the threshold
        keep_scores = final_prediction['scores'] > conf_thresh
        final_prediction = {
            'boxes': final_prediction['boxes'][keep_scores],
            'scores': final_prediction['scores'][keep_scores],
            'labels': final_prediction['labels'][keep_scores],
        }

        return final_prediction

    def prepare_predictions(self, targets, outputs):
        predictions = []
        for index, (target, output) in enumerate(zip(targets, outputs)):
            output = self.parse_output(output)
            boxes = self.scale_boxes(output['boxes'], index)
            # boxes = output['boxes']
            scores = output['scores'].cpu().numpy().tolist()
            labels = output['labels'].cpu().numpy().tolist()

            image_id = target['image_id'].item()
            pred = [{
                'image_id': image_id,
                'category_id': labels[i],
                'bbox': boxes[i],
                'score': scores[i]
            } for i in range(len(boxes))]
            predictions.extend(pred)
        return predictions


# Define a concrete evaluator for EfficientDet
class EfficientDetEvaluator(ModelEvaluator):
    def resize_images(self, images):
        return torch.cat(
            [F.interpolate(image.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False) for image in
             images], dim=0)

    def calculate_scale_factors(self, original_image_sizes):
        return [(size[1] / self.img_size[0], size[0] / self.img_size[1]) for size in original_image_sizes]

    def parse_output(self, output):
        boxes = output[:, :4]
        scores = output[:, 4]
        labels = output[:, 5].int()
        output = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
        return self.apply_nms(output, iou_thresh=0.7, conf_thresh=0.7)

    def scale_boxes(self, boxes, index):
        scale_x, scale_y = self.scale_factors[index]
        return [scale_bbox(box, scale_x, scale_y) for box in boxes]


class YoloEvaluator(ModelEvaluator):
    def resize_images(self, images):
        if isinstance(self.img_size, int):
            target_size = (self.img_size, self.img_size)
        else:
            target_size = self.img_size

        resized_images = [F.interpolate(image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False) for
                          image in images]
        images_tensor = torch.cat(resized_images, dim=0)
        return images_tensor

    def calculate_scale_factors(self, original_image_sizes):
        return [(size[1] / self.img_size[0], size[0] / self.img_size[1]) for size in original_image_sizes]

    def parse_output(self, output):
        boxes = output.boxes.xyxy
        scores = output.boxes.conf
        labels = translate_labels(output.boxes.cls.int())
        output = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
        return output

    def scale_boxes(self, boxes, index):
        scale_x, scale_y = self.scale_factors[index]
        return [scale_bbox(box, scale_x, scale_y) for box in boxes]


# Define a default evaluator for other models
class DefaultModelEvaluator(ModelEvaluator):
    def resize_images(self, images):
        # Other models might not need resizing, so we stack them as they are
        return torch.stack([image for image in images])

    def calculate_scale_factors(self, original_image_sizes):
        # Other models might not resize, so scale factors are 1
        return [(1, 1) for _ in original_image_sizes]

    def parse_output(self, output):
        return output  # Assuming the output is already in the desired format

    def scale_boxes(self, boxes, index):
        # No scaling needed if images were not resized
        return [self.convert_to_xywh(box) for box in boxes]


def eval_model(model_class):
    model_type = model_class.name
    model, img_size = model_class.model, model_class.image_size

    #get date and time of evaluation
    start_of_test = datetime.datetime.now()

    # Initialize the validation dataset and loader
    val_dataset = CocoDetection(val_images_path, val_annotations_path, transform=T.ToTensor())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize COCO ground truth API and predictions list
    coco_gt = COCO(val_annotations_path)
    coco_predictions = []

    # Select the appropriate evaluator based on the model_type
    if model_type == 'EfficientDet':
        evaluator = EfficientDetEvaluator(model, img_size)
    elif model_type == 'YOLOv8':
        evaluator = YoloEvaluator(model, img_size)
        evaluator.model.to('cpu')
    else:
        evaluator = DefaultModelEvaluator(model, img_size)


    total_time = 0  # To accumulate the inference time for each image
    num_images = 0  # Total number of images processed

    # Start the temperature reader in a separate thread
    temp_reader = TemperatureReader(cs_pin=15, sck=14, data_pin=18, units="f")
    temp_reader.start()

    # Iterate over the dataset to evaluate the model
    for image, targets in tqdm(val_loader, desc="Evaluating", unit="images"):
        original_image_sizes = [img.shape[-2:] for img in image]
        image = evaluator.resize_images(image)
        evaluator.scale_factors = evaluator.calculate_scale_factors(original_image_sizes)

        # Start time
        start_time = time.time()

        with torch.no_grad():
            if model_type == 'YOLOv8':
                outputs = model.predict(image, verbose=False, conf=0.5, iou=0.5)
            else:
                outputs = model(image)

        # End time
        end_time = time.time()

        # Calculate the time taken and accumulate
        time_taken = end_time - start_time
        total_time += time_taken
        num_images += 1  # Assuming batch_size=1

        # Prepare predictions using the evaluator
        batch_predictions = evaluator.prepare_predictions(targets, outputs)
        coco_predictions.extend(batch_predictions)

        # Visualize the predictions
        # visualize_predictions(get_image_by_id(coco_gt, coco_predictions[-1]["image_id"]), batch_predictions)

    # Stop the temperature reader
    temp_reader.stop()

    #get the ending date and time of evaluation
    end_of_test = datetime.datetime.now()

    average_latency = total_time / num_images
    average_fps = num_images / total_time

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(coco_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract COCO summary metrics
    coco_stats = coco_eval.stats

    # Save statistics to a CSV file
    stats_file = f'results/stats_{model_type}.csv'
    file_exists = os.path.isfile(stats_file)
    with open(stats_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if the file is new
            writer.writerow(['Model', 'Average Latency (s)', 'Average FPS',
                             'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'start_time', 'end_time', 'average_soc_temp'])
        # Write data
        writer.writerow([model_class.name, f"{average_latency:.3f}", f"{average_fps:.2f}",
                         f"{coco_stats[0]:.3f}", f"{coco_stats[1]:.3f}", f"{coco_stats[2]:.3f}",
                         f"{coco_stats[3]:.3f}", f"{coco_stats[4]:.3f}", f"{coco_stats[5]:.3f}", start_of_test, end_of_test, temp_reader.average_soc_temp])


