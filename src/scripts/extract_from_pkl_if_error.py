import torch
from torchvision.ops import nms
import pickle
import csv
import os
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection
import torchvision.transforms as T


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_and_process_predictions(temp_predictions_file, processed_file, iou_thresh=0.5, conf_thresh=0.5):
    predictions = []
    total_entries = 0
    processed_entries = 0

    # First, get the total number of entries for progress indication
    with open(temp_predictions_file, 'rb') as f:
        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                break  # End of file reached
            total_entries += 1
            length = int.from_bytes(length_bytes, byteorder='big')
            f.read(length)  # Skip the content for now

    # Now, process each entry and repickle
    with open(temp_predictions_file, 'rb') as f, open(processed_file, 'wb') as out_f:
        for _ in range(total_entries):
            length_bytes = f.read(4)
            length = int.from_bytes(length_bytes, byteorder='big')
            data = f.read(length)
            prediction = pickle.loads(data)
            nms_prediction = apply_nms_to_predictions(prediction, iou_thresh, conf_thresh)
            predictions.extend(nms_prediction)

            # Repickle the data immediately to save space
            # processed_data = pickle.dumps(nms_prediction)
            # out_f.write(len(processed_data).to_bytes(4, byteorder='big'))
            # out_f.write(processed_data)

            # Print progress
            processed_entries += 1
            sys.stdout.write(f'\rProcessed {processed_entries}/{total_entries} entries.')
            sys.stdout.flush()

    print()  # Newline after progress output
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


def apply_nms_to_predictions(predictions, iou_thresh=0.5, conf_thresh=0.5):
    if not predictions:
        return []

    # Extract elements from prediction dictionaries
    boxes = [pred['bbox'] for pred in predictions]
    scores = [pred['score'] for pred in predictions]
    labels = [pred['category_id'] for pred in predictions]
    image_ids = [pred['image_id'] for pred in predictions]

    # Check if the list is empty after extracting
    if not boxes or not scores or not labels:
        return []

    # Convert to tensors and ensure correct dimensions and types
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    # Ensure boxes is a 2D tensor [num_boxes, 4]
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)

    # Apply NMS
    keep_indices = nms(boxes, scores, iou_thresh)

    # Filter predictions based on NMS results and confidence threshold
    nms_predictions = []
    for idx in keep_indices:
        if scores[idx] > conf_thresh:
            nms_predictions.append({
                'image_id': image_ids[idx],
                'category_id': labels[idx].item(),
                'bbox': boxes[idx].tolist(),
                'score': scores[idx].item()
            })

    return nms_predictions


val_images_path = '../dataset/val2017'
val_annotations_path = '../dataset/annotations/instances_val2017.json'

# Initialize the validation dataset and loader
val_dataset = CocoDetection(val_images_path, val_annotations_path, transform=T.ToTensor())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize COCO ground truth API
coco_gt = COCO(val_annotations_path)


# Process predictions with NMS and repickle them
processed_predictions_file = "../results/ssdlite/processed_predictions_SSDlite_5.pkl"
coco_predictions = load_and_process_predictions(
    "../results/ssdlite/processed_predictions_SSDlite_4.pkl",
    processed_predictions_file,
    iou_thresh=0.7,
    conf_thresh=0.5
)

# Continue with COCO evaluation and result saving...
coco_stats = perform_coco_evaluation(coco_gt, coco_predictions)
save_results(coco_stats, "model_type", "start_of_test", "datetime.datetime.now()", "temp_reader.average_ambient_temperature",
             "average_latency", "average_fps")
