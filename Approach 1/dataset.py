import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PlantDataset(Dataset):
    """Custom Dataset for Plant Species Identification from a pandas DataFrame."""

    def __init__(self, root_dir, dataframe, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            dataframe (pandas.DataFrame): DataFrame with image paths and class IDs.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.image_list = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Assumes the dataframe has columns 'image_path' and 'class_id'
        img_path = self.image_list.iloc[idx]['image_path']
        label = int(self.image_list.iloc[idx]['class_id'])
        
        full_img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(full_img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """Returns a dictionary of transformations for training and validation/testing."""
    # ResNet50/ConvNeXt/Xception were pre-trained on ImageNet, so we use the stats.
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms