import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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


def apply_nms(orig_prediction, iou_thresh=0.5, conf_thresh=0.5):
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

def eval_model(model_class):
    model_type = model_class.name
    model, img_size = model_class.model, model_class.image_size

    # Create the COCO validation dataset
    val_dataset = CocoDetection(val_images_path, val_annotations_path, transform=T.ToTensor())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize COCO ground truth API
    coco_gt = COCO(val_annotations_path)
    # Initialize COCO predictions list
    coco_predictions = []

    def xyxy_to_xywh(box):
        x_min, y_min, x_max, y_max = box
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def scale_bbox(box, scale_x, scale_y):
        # Scale bounding box coordinates from model input size to original image size
        x_min, y_min, x_max, y_max = box
        x_min_scaled = x_min * scale_x
        y_min_scaled = y_min * scale_y
        x_max_scaled = x_max * scale_x
        y_max_scaled = y_max * scale_y
        return [x_min_scaled, y_min_scaled, x_max_scaled - x_min_scaled, y_max_scaled - y_min_scaled]

    # Iterate over validation data with a progress bar
    for images, targets in tqdm(val_loader, desc="Evaluating", unit="batch"):

        original_image_sizes = [img.shape[-2:] for img in images]  # Get the original image sizes before resizing

        if model_type == 'EfficientDet':
            # Assuming images are PyTorch tensors
            images_resized = [F.interpolate(image.unsqueeze(0), size=img_size, mode='bilinear', align_corners=False) for image in images]
            images = torch.cat(images_resized, dim=0)

            # Calculate scale factors for each image
            scale_factors = [(original_image_size[1] / img_size[0], original_image_size[0] / img_size[1]) for original_image_size in original_image_sizes]

        else:
            # For other models, just send to device without resizing
            images = torch.stack([image for image in images])


        with torch.no_grad():
            outputs = model(images)

        for index, (target, output) in enumerate(zip(targets, outputs)):
            if model_type == 'EfficientDet':
                # Parse the tensor output into a dictionary
                boxes = output[:, :4]
                scores = output[:, 4]
                labels = output[:, 5].int()
                output = {
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                }
                output = apply_nms(output)

                # Scale boxes from resized to original size
                scale_x, scale_y = scale_factors[index]
                boxes = [scale_bbox(box, scale_x, scale_y) for box in output["boxes"].numpy()]
            else:
                # For other models, the output is already in the expected format
                boxes = output["boxes"].cpu().numpy()
                boxes = [xyxy_to_xywh(box) for box in boxes]


            # Extract and process each detection
            image_id = target["image_id"].item()
            scores = output["scores"].cpu().numpy().tolist()
            labels = output["labels"].cpu().numpy().tolist()

            # Prepare COCO predictions
            coco_pred = [{
                "image_id": image_id,
                "category_id": labels[i],
                "bbox": boxes[i],  # Use the scaled boxes for predictions
                "score": scores[i]
            } for i in range(len(boxes))]

            coco_predictions.extend(coco_pred)

            # Visualize the predictions
            # if len(coco_predictions) < 10:  # adjust this number to print as many predictions as you want to check
            #     visualize_predictions(get_image_by_id(coco_gt, coco_predictions[0]["image_id"]), coco_pred)
            #     return

    # Convert predictions to COCO detection results
    coco_dt = coco_gt.loadRes(coco_predictions)

    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()