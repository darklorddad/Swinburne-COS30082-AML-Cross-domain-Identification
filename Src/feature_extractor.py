
import torch
from torchvision import transforms
from PIL import Image
import timm

class DinoV2FeatureExtractor:
    def __init__(self, model_name='facebook/dinov2-base', device=None):
        """
        Initializes the DINOv2 feature extractor.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=True).to(self.device)
        self.model.eval()

        # Get the data configuration associated with the model
        data_config = timm.data.resolve_data_config(self.model.default_cfg)
        self.transform = timm.data.create_transform(**data_config)

    def extract_features(self, image_path):
        """
        Extracts features from a single image.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.forward_features(image_tensor)
                # Global average pooling
                features = features.mean(dim=1)

            return features.cpu().numpy()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
