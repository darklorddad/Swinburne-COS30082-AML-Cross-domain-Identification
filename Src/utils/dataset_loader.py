"""
Custom Dataset Loaders for Plant Identification

Provides PyTorch Dataset classes for loading plant images from different
directory structures (balanced, original, test sets).
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json


class PlantDataset(Dataset):
    """
    Generic plant dataset loader.

    Directory structure:
        root/
            class_1/
                img1.jpg
                img2.jpg
            class_2/
                img1.jpg
                ...
    """

    def __init__(self, root_dir, transform=None, class_to_idx=None):
        """
        Args:
            root_dir (str): Path to dataset root directory
            transform (callable, optional): Transform to apply to images
            class_to_idx (dict, optional): Mapping of class names to indices
        """
        self.root_dir = root_dir
        self.transform = transform

        # Collect all images and labels
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))

        # Create class to index mapping
        if class_to_idx is None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx

        # Collect all image paths
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_counts(self):
        """Return dictionary of class counts"""
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts


class PlantTestDataset(Dataset):
    """
    Test dataset loader for images without labels in subdirectories.

    Directory structure:
        test/
            img1.jpg
            img2.jpg
            ...
    """

    def __init__(self, root_dir, transform=None, groundtruth_file=None):
        """
        Args:
            root_dir (str): Path to test images directory
            transform (callable, optional): Transform to apply to images
            groundtruth_file (str, optional): Path to groundtruth.txt
        """
        self.root_dir = root_dir
        self.transform = transform

        # Collect all image paths
        self.image_paths = []
        for img_name in sorted(os.listdir(root_dir)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(root_dir, img_name))

        # Load groundtruth labels if available
        self.labels = None
        if groundtruth_file and os.path.exists(groundtruth_file):
            self.labels = self._load_groundtruth(groundtruth_file)

    def _load_groundtruth(self, groundtruth_file):
        """Load groundtruth labels"""
        labels_dict = {}
        with open(groundtruth_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_name, label = parts
                    labels_dict[img_name] = label
        return labels_dict

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Return image with label if available
        if self.labels:
            img_name = os.path.basename(img_path)
            label = self.labels.get(img_name, -1)  # -1 if not found
            return image, label, img_path
        else:
            return image, img_path


class FeatureDataset(Dataset):
    """
    Dataset for pre-extracted features (for traditional ML training).
    """

    def __init__(self, features, labels):
        """
        Args:
            features (numpy.ndarray or torch.Tensor): Feature vectors
            labels (numpy.ndarray or torch.Tensor): Labels
        """
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features).float()
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels).long()

        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_train_transforms(image_size=224):
    """
    Get training data augmentation transforms.

    Args:
        image_size (int): Target image size

    Returns:
        torchvision.transforms.Compose: Training transforms
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size=224):
    """
    Get validation/test transforms (no augmentation).

    Args:
        image_size (int): Target image size

    Returns:
        torchvision.transforms.Compose: Validation transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_minimal_transforms(image_size=224):
    """
    Get minimal transforms for GPU augmentation pipeline.
    Only converts to tensor without any augmentation or normalization.

    Args:
        image_size (int): Target image size

    Returns:
        torchvision.transforms.Compose: Minimal transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])


def load_class_names(class_file):
    """
    Load class names from classes.txt file.

    Args:
        class_file (str): Path to classes.txt

    Returns:
        list: List of class names
    """
    with open(class_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


def create_class_to_idx_mapping(class_file):
    """
    Create mapping from class ID (species ID) to index (0-99).

    Args:
        class_file (str): Path to species_list.txt from dataset

    Returns:
        dict: {class_id: index}
    """
    mapping = {}
    with open(class_file, 'r') as f:
        for idx, line in enumerate(f):
            # Parse format: "  1→105951; Maripa glabra Choisy"
            parts = line.strip().split('→')
            if len(parts) == 2:
                class_id = parts[1].split(';')[0].strip()
                mapping[class_id] = idx
    return mapping


def save_metadata(output_file, **kwargs):
    """
    Save metadata as JSON file.

    Args:
        output_file (str): Path to output JSON file
        **kwargs: Key-value pairs to save
    """
    with open(output_file, 'w') as f:
        json.dump(kwargs, f, indent=4)


def load_metadata(input_file):
    """
    Load metadata from JSON file.

    Args:
        input_file (str): Path to JSON file

    Returns:
        dict: Loaded metadata
    """
    with open(input_file, 'r') as f:
        return json.load(f)
