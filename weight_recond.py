import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from tqdm import tqdm
import torch.nn.functional as F

# rcnn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
# ssdlite
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
# MobileFormer
from MobileFormer.mobile_former import mobile_former_96m
#EfficientDet
from efficientdet.effdet import create_model
#YoloV8
from ultralytics import YOLO


def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")  # Force CPU for now

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

model_type = 'EfficientDet'
# Assuming load_model is defined elsewhere and includes device in its return values
model, device, img_size = load_model(model_type)

# Specify the path to the COCO validation images and annotation file
val_images_path = './Dataset/val2017'
val_annotations_path = './Dataset/annotations/instances_val2017.json'

# Create the COCO validation dataset
val_dataset = CocoDetection(val_images_path, val_annotations_path, transform=T.ToTensor())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Initialize COCO ground truth API
coco_gt = COCO(val_annotations_path)
# Initialize COCO predictions list
coco_predictions = []

def xyxy_to_xywh(box):
    x_min, y_min, x_max, y_max = box
    return [x_min, y_min, x_max - x_min, y_max - y_min]

# Iterate over validation data with a progress bar
for images, targets in tqdm(val_loader, desc="Evaluating", unit="batch"):
    if model_type == 'EfficientDet':
        # Assuming images are PyTorch tensors
        images_resized = [F.interpolate(image.unsqueeze(0), size=img_size, mode='bilinear', align_corners=False) for image in images]
        images = torch.cat(images_resized, dim=0).to(device)
    else:
        # For other models, just send to device without resizing
        images = torch.stack([image.to(device) for image in images])

    with torch.no_grad():
        outputs = model(images)

    for target, output in zip(targets, outputs):
        image_id = target["image_id"].item()  # Assuming `image_id` is available in target

        # Convert boxes to XYWH format as expected by COCO
        boxes = output["boxes"].cpu().numpy()
        boxes = [xyxy_to_xywh(box) for box in boxes]

        scores = output["scores"].cpu().numpy().tolist()
        labels = output["labels"].cpu().numpy().tolist()

        # Prepare COCO predictions
        coco_pred = [{
            "image_id": image_id,
            "category_id": labels[i],
            "bbox": boxes[i],
            "score": scores[i]
        } for i in range(len(boxes))]

        coco_predictions.extend(coco_pred)

# Convert predictions to COCO detection results
coco_dt = coco_gt.loadRes(coco_predictions)

# Run COCO evaluation
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()