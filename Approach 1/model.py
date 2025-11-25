import torch.nn as nn
import timm
import config

def create_resnet50_model(num_classes):
    """
    Creates a ResNet50 model using the 'timm' library.
    Uses transfer learning by freezing the feature extraction layers.
    """
    # Create a pre-trained ResNet50 model from timm
    # Disable Stochastic Depth for ResNet50 to allow for better convergence
    model = timm.create_model('resnet50', pretrained=True, drop_path_rate=0.0)

    # Full Fine-Tuning: We do NOT freeze the backbone anymore.
    # Differential learning rates in main.py will handle the stability.

    # Get the number of input features for the classifier
    # In timm's ResNet50 model, the classifier is named 'fc'
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer with our custom head
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

def create_convnextv2_model(num_classes):
    """
    Creates a ConvNeXtV2 Tiny model using the 'timm' library.
    Uses transfer learning by freezing the feature extraction layers.
    """
    # Create a pre-trained ConvNeXtV2 Tiny model from timm
    model = timm.create_model('convnextv2_tiny', pretrained=True, drop_path_rate=config.DROP_PATH_RATE)

    # Full Fine-Tuning: We do NOT freeze the backbone anymore.
    
    # Get the number of input features for the classifier
    # In timm's ConvNeXt models, the classifier is typically 'head.fc'
    num_ftrs = model.head.fc.in_features

    # Replace the final fully connected layer with our custom head
    model.head.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

def create_xception_model(num_classes):
    """
    Creates an Xception model using the 'timm' library.
    Uses transfer learning by freezing the feature extraction layers.
    """
    # Create a pre-trained Xception model from timm
    # Note: Xception does not support drop_path_rate in timm
    model = timm.create_model('xception', pretrained=True)

    # Full Fine-Tuning: We do NOT freeze the backbone anymore.

    # Get the number of input features for the classifier
    # In timm's Xception model, the classifier is named 'fc'
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer with our custom head
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model