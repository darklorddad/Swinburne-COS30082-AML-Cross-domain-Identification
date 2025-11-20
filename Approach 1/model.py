import torch.nn as nn
import timm

def create_resnet50_model(num_classes):
    """
    Creates a ResNet50 model using the 'timm' library.
    Uses transfer learning by freezing the feature extraction layers.
    """
    # Create a pre-trained ResNet50 model from timm
    model = timm.create_model('resnet50', pretrained=True)

    # Freeze all parameters in the feature extractor
    for param in model.parameters():
        param.requires_grad = False

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
    model = timm.create_model('convnextv2_tiny', pretrained=True)

    # Freeze all parameters in the feature extractor
    for param in model.parameters():
        param.requires_grad = False
    
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
    model = timm.create_model('xception', pretrained=True)

    # Freeze all parameters in the feature extractor
    for param in model.parameters():
        param.requires_grad = False

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