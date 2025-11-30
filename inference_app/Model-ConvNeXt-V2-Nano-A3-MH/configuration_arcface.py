from transformers import PretrainedConfig

class ArcFaceConfig(PretrainedConfig):
    model_type = "custom_arcface"

    def __init__(
        self,
        backbone="resnet50",
        num_classes=1000,
        arcface_s=30.0,
        arcface_m=0.50,
        arcface_k=1,
        **kwargs,
    ):
        self.backbone = backbone
        self.num_classes = num_classes
        self.arcface_s = arcface_s
        self.arcface_m = arcface_m
        self.arcface_k = arcface_k
        super().__init__(**kwargs)
