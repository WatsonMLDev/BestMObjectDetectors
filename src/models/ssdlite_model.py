from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

class SSDliteModel:
    def __init__(self, box_score_thresh=0.9):
        self.box_score_thresh = box_score_thresh
        self.model = None
        self.image_size = (320, 320)
        self.name = 'SSDlite'

    def create_model(self):
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=self.box_score_thresh)
        self.model.eval()

