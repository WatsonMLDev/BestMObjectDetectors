from ultralytics import YOLO

class YOLOv8Model:
    def __init__(self, model_path='./model_repo/yolov8n.pt'):
        self.model_path = model_path
        self.model = None
        self.image_size = (640, 640)
        self.name = 'YOLOv8'

    def create_model(self):
        self.model = YOLO(self.model_path)
