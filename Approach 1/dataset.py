import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import config

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
    # Calculate resize dimensions based on config.IMAGE_SIZE
    train_size = config.IMAGE_SIZE
    val_resize = int(config.IMAGE_SIZE / 0.875) # Standard practice: resize to ~1.14x and crop

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(train_size),
            transforms.RandomHorizontalFlip(),
            # RandAugment is a powerful automated augmentation strategy
            # Reduced magnitude from 9 to 5 to prevent excessive distortion
            transforms.RandAugment(num_ops=2, magnitude=5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # RandomErasing randomly occludes parts of the image
            # Disabled for now as it might be too aggressive combined with other regularizations
            # transforms.RandomErasing(p=0.25) 
        ]),
        'val': transforms.Compose([
            transforms.Resize(val_resize),
            transforms.CenterCrop(train_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms