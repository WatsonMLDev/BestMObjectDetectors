import cv2
import numpy as np
import torch
from PIL import Image
# rcnn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
# ssdlite
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.ops import nms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import draw_bounding_boxes
# MobileFormer
from MobileFormer.mobile_former import mobile_former_96m
#EfficientDet
from efficientdet.effdet import create_model
#YoloV8
from ultralytics import YOLO


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

# def apply_nms(orig_prediction, iou_thresh=0.3, conf_thresh=0.5):
#     # Check if the prediction is empty
#     if orig_prediction['boxes'].nelement() == 0:
#         return orig_prediction
#
#     # Ensure boxes and scores have the correct shapes
#     boxes = orig_prediction['boxes'].view(-1, 4)
#     scores = orig_prediction['scores'].view(-1)
#
#     # Apply Non-Maximum Suppression (NMS)
#     keep_boxes = nms(boxes, scores, iou_thresh)
#
#     final_prediction = {
#         'boxes': boxes[keep_boxes],
#         'scores': scores[keep_boxes],
#         'labels': orig_prediction['labels'][keep_boxes],
#     }
#
#     # Keep only predictions with a confidence score above the threshold
#     keep_scores = final_prediction['scores'] > conf_thresh
#     final_prediction = {
#         'boxes': final_prediction['boxes'][keep_scores],
#         'scores': final_prediction['scores'][keep_scores],
#         'labels': final_prediction['labels'][keep_scores],
#     }
#
#     return final_prediction



def load_model(model_name):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # Force CPU for now

    if model_name == 'EfficientDet':

        args_model = 'tf_efficientdet_lite0'  # replace with your chosen variant
        args_pretrained = True
        args_checkpoint = ''  # replace with path to your checkpoint, if you have one
        args_num_classes = None

        # Create the model
        extra_args = {}
        model = create_model(
            args_model,
            bench_task='predict',
            num_classes=args_num_classes,
            pretrained=args_pretrained,
            checkpoint_path=args_checkpoint,
            **extra_args,
            **extra_args,
        )

        # model_config = model.config
        # input_config = resolve_input_config({}, model_config)

        model.to(device)
        model.eval()

        image_size = (320, 320)
        return model, device, image_size

    elif model_name == 'FasterRCNN':
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.9)
        model = model.to(device)  # Move model to the appropriate device
        model.eval()

        image_size = (320, 320)
        return model, device, image_size

    elif model_name == 'SSDlite':
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=.9)
        model = model.to(device)  # Move model to the appropriate device
        model.eval()

        image_size = (320, 320)
        return model, device, image_size

    # elif model_name == 'MobileFormer':
    #     model = mobile_former_96m(pretrained=False)
    #     # Load pre-trained weights
    #     weights_path = './mobileformer/mobileformer_weights.pth'
    #     model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    #     return model

    elif model_name == 'YoloV8':
        model = YOLO('./files/yolov8n.pt')  # load an official model
        model = model.to(device)  # Move model to the appropriate device
        image_size = (640,640)
        return model, device, image_size

    else:
        raise ValueError("Model not supported")


# Main loop to capture webcam feed and process frames
def main():
    model_type = 'YoloV8'
    # Assuming load_model is defined elsewhere and includes device in its return values
    model, device, img_size = load_model(model_type)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to match the input size of the model
        frame = cv2.resize(frame, img_size)

        # Convert frame to PIL, then to tensor, and move to GPU if available
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = to_tensor(frame_pil).unsqueeze(0).to(device)

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


if __name__ == "__main__":
    main()
