import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

class FasterRCNNModel:
    def __init__(self, box_score_thresh=0.9):
        self.box_score_thresh = box_score_thresh
        self.model = None
        self.image_size = (320, 320)
        self.name = 'FasterRCNN'

    def create_model(self):
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=self.box_score_thresh)
        self.model.eval()

