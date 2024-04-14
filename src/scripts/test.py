from abc import ABC, abstractmethod
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm
import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from torchvision.ops import nms
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os

from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np
import cv2

val_images_path = './dataset/val2017'
val_annotations_path = './dataset/annotations/instances_val2017.json'

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
    # Convert PIL image to tensor

    # Assuming 'image' is your PIL image
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    while True:
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return


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

    def prepare_predictions(self, targets, outputs):
        predictions = []
        for index, (target, output) in enumerate(zip(targets, outputs)):
            output = self.parse_output(output)
            boxes = self.scale_boxes(output['boxes'], index)
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
        return torch.cat([F.interpolate(image.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False) for image in images], dim=0)

    def calculate_scale_factors(self, original_image_sizes):
        return [(size[1] / self.img_size[0], size[0] / self.img_size[1]) for size in original_image_sizes]

    def parse_output(self, output):
        boxes = output[:, :4]
        scores = output[:, 4]
        labels = output[:, 5].int()
        output =  {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
        return self.apply_nms(output)

    def scale_boxes(self, boxes, index):
        scale_x, scale_y = self.scale_factors[index]
        return [scale_bbox(box, scale_x, scale_y) for box in boxes]

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

    # Initialize the validation dataset and loader
    val_dataset = CocoDetection(val_images_path, val_annotations_path, transform=T.ToTensor())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize COCO ground truth API and predictions list
    coco_gt = COCO(val_annotations_path)
    coco_predictions = []

    # Select the appropriate evaluator based on the model_type
    if model_type == 'EfficientDet':
        evaluator = EfficientDetEvaluator(model, img_size)
    else:
        evaluator = DefaultModelEvaluator(model, img_size)

    # Iterate over the dataset to evaluate the model
    for images, targets in tqdm(val_loader, desc="Evaluating", unit="batch"):
        original_image_sizes = [img.shape[-2:] for img in images]
        images = evaluator.resize_images(images)
        evaluator.scale_factors = evaluator.calculate_scale_factors(original_image_sizes)

        with torch.no_grad():
            outputs = model(images)

        # Prepare predictions using the evaluator
        batch_predictions = evaluator.prepare_predictions(targets, outputs)
        coco_predictions.extend(batch_predictions)

        # Visualize the predictions
        # if len(coco_predictions) < 1000000000:  # adjust this number to print as many predictions as you want to check
        #     visualize_predictions(get_image_by_id(coco_gt, coco_predictions[0]["image_id"]), batch_predictions)
        #     return

    # Further processing of coco_predictions can be done here as needed
    # This could include saving the predictions, evaluating metrics, etc.

    # Optionally return the predictions if needed
    # Convert predictions to COCO detection results
    coco_dt = coco_gt.loadRes(coco_predictions)



    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# Usage example:
# coco_predictions = eval_model(your_model_class_instance)

