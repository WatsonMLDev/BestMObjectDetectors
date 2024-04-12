from model_repo.efficientdet.effdet import create_model

class EfficientDetModel:
    def __init__(self, model_variant='tf_efficientdet_lite0', pretrained=True, checkpoint_path='', num_classes=None):
        self.model_variant = model_variant
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.image_size = (320, 320)
        self.model = None
        self.name = 'EfficientDet'

    def create_model(self):
        extra_args = {}
        self.model = create_model(
            self.model_variant,
            bench_task='predict',
            num_classes=self.num_classes,
            pretrained=self.pretrained,
            checkpoint_path=self.checkpoint_path,
            **extra_args
        )
        self.model.eval()  # Set the model to evaluation mode





