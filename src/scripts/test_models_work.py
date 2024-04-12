from torchvision.ops import nms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import draw_bounding_boxes
import cv2
import numpy as np
import torch
from PIL import Image

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

def apply_nms(orig_prediction, iou_thresh=0.3, conf_thresh=0.5):
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

def test_model(model_type, model, img_size):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to match the input size of the model
        frame = cv2.resize(frame, img_size)

        # Convert frame to PIL, then to tensor, and move to GPU if available
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = to_tensor(frame_pil).unsqueeze(0)

        if model_type == 'YoloV8':
            # Run YOLOv8 inference on the frame
            results = model.predict(frame, verbose=False, conf=0.7, iou=0.6)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:

            # Make prediction
            with torch.no_grad():
                prediction = model(frame_tensor)[0]
                if model_type == 'EfficientDet':
                    # Extract boxes, scores, and labels from the prediction tensor
                    boxes = prediction[:, :4]
                    scores = prediction[:, 4]
                    labels = prediction[:, 5].int()

                    # Create a dictionary for the prediction in the expected format
                    prediction = {
                        'boxes': boxes,
                        'scores': scores,
                        'labels': labels
                    }
                prediction = apply_nms(prediction)

            # Prepare for drawing boxes: move tensor to CPU, convert to uint8, remove batch dimension
            frame_draw = (frame_tensor.squeeze().cpu() * 255).type(torch.uint8)

            # Draw bounding boxes (assumes labels and boxes are in prediction)
            boxes = prediction['boxes'].cpu()
            scores = prediction['scores'].cpu()

            # Combine the labels and scores in the format "label: score"
            label_names_with_scores = [
                f"{COCO_LABEL_MAP.get(i, 'unknown')}: {s:.2f}"
                for i, s in zip(prediction['labels'].cpu().numpy(), scores)
            ]

            # Note: draw_bounding_boxes currently works with CPU tensors
            box_tensor = draw_bounding_boxes(frame_draw, boxes=boxes, labels=label_names_with_scores, colors="blue", width=5)

            # Convert tensor to PIL Image, then to numpy array for OpenCV display
            box_pil = to_pil_image(box_tensor)
            box_np = np.array(box_pil)
            box_np = cv2.cvtColor(box_np, cv2.COLOR_RGB2BGR)

            # Display the image
            cv2.imshow('Detection', box_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()