from models.efficientdet_model import EfficientDetModel
from models.fasterrcnn_model import FasterRCNNModel
from models.ssdlite_model import SSDliteModel
from models.yolo_model import YOLOv8Model
import scripts.test_models_work as test_models_work
import scripts.eval_model as eval_model
import argparse


def load_model(model_type):
    if model_type == 'EfficientDet':
        efficientdet = EfficientDetModel(model_variant="tf_efficientdet_lite0", pretrained=True, checkpoint_path='', num_classes=None)
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
        eval_model.eval_model(model)
    else:
        raise ValueError(f"Task {task} not recognized")