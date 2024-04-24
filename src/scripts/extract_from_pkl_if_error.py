import csv
import os
import pickle

import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection
from torchvision.ops import nms


def load_and_process_predictions(temp_predictions_file, iou_thresh=0.5, conf_thresh=0.5):
    predictions = []
    with open(temp_predictions_file, 'rb') as f:
        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                break  # End of file reached
            length = int.from_bytes(length_bytes, byteorder='big')
            data = f.read(length)
            if not data:
                break  # Ensure that data read matches the length expected
            prediction = pickle.loads(data)
            # Apply NMS and thresholding to each prediction immediately after loading
            nms_prediction = apply_nms_to_predictions(prediction, iou_thresh, conf_thresh)
            predictions.extend(nms_prediction)  # Extend with processed predictions
    return predictions


def perform_coco_evaluation(coco_gt, coco_predictions):
    try:
        coco_dt = coco_gt.loadRes(coco_predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats
    except Exception as e:
        print(f"Error during COCO evaluation: {e}")
        return [0, 0, 0, 0, 0, 0]


def save_results(coco_stats, model_type, start_of_test, end_of_test, average_ambient_temp, average_latency,
                 average_fps):
    stats_file = f'stats_{model_type}.csv'
    file_exists = os.path.isfile(stats_file)
    with open(stats_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if the file is new
            writer.writerow(['Model', 'Average Latency (s)', 'Average FPS',
                             'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'start_time', 'end_time',
                             'average_ambient_temp'])
        # Write data
        writer.writerow([model_type, f"{average_latency}", f"{average_fps}",
                         f"{coco_stats[0]:.3f}", f"{coco_stats[1]:.3f}", f"{coco_stats[2]:.3f}",
                         f"{coco_stats[3]:.3f}", f"{coco_stats[4]:.3f}", f"{coco_stats[5]:.3f}", start_of_test,
                         end_of_test, average_ambient_temp])

# Convert the existing predictions to tensors
def prepare_for_nms(predictions):
    boxes = torch.stack([torch.tensor(pred['bbox']) for pred in predictions])
    scores = torch.tensor([pred['score'] for pred in predictions])
    labels = torch.tensor([pred['category_id'] for pred in predictions])
    return boxes, scores, labels

# Apply NMS to the predictions
def apply_nms_to_predictions(predictions, iou_thresh=0.5, conf_thresh=0.5):
    boxes, scores, labels = prepare_for_nms(predictions)

    # Apply NMS
    keep = nms(boxes, scores, iou_thresh)

    # Select only the detections that were kept by NMS
    nms_predictions = []
    for idx in keep:
        if scores[idx] >= conf_thresh:
            nms_predictions.append({
                'bbox': boxes[idx].tolist(),  # Convert tensor back to list
                'score': scores[idx].item(),  # Convert tensor to single float value
                'category_id': labels[idx].item()  # Convert tensor to single int value
            })

    return nms_predictions


val_images_path = '../dataset/val2017'
val_annotations_path = '../dataset/annotations/instances_val2017.json'

# Initialize the validation dataset and loader
val_dataset = CocoDetection(val_images_path, val_annotations_path, transform=T.ToTensor())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize COCO ground truth API
coco_gt = COCO(val_annotations_path)


coco_predictions = load_and_process_predictions("../results/ssdlite/predictions_SSDlite_1.pkl")

# Continue with COCO evaluation and result saving...
coco_stats = perform_coco_evaluation(coco_gt, coco_predictions)
save_results(coco_stats, "model_type", "start_of_test", "datetime.datetime.now()", "temp_reader.average_ambient_temperature",
             "average_latency", "average_fps")
