"""
Feature Extraction using DINOv2 Models

This script extracts features from pre-trained DINOv2 models (both plant-pretrained
and ImageNet-pretrained) for use with traditional ML classifiers.

The script supports multiple model variants:
- Plant-pretrained Base (PlantCLEF 2024)
- ImageNet Small, Base, Large

Features are saved as .npy files for later training with SVM, Random Forest, or Linear Probe.

Usage:
    python Approach_A_Feature_Extraction/extract_features.py \
        --model_type imagenet_base \
        --train_dir Dataset/balanced_train \
        --val_dir Dataset/validation \
        --test_dir Dataset/test \
        --output_dir Approach_A_Feature_Extraction/features \
        --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
import numpy as np
from tqdm import tqdm
import json
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Src.utils.dataset_loader import PlantDataset, PlantTestDataset, get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Extract features using DINOv2')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['plant_pretrained_base', 'imagenet_small', 'imagenet_base', 'imagenet_large'],
                        help='Model variant to use')
    parser.add_argument('--train_dir', type=str, default='Dataset/balanced_train',
                        help='Training images directory')
    parser.add_argument('--val_dir', type=str, default='Dataset/validation',
                        help='Validation images directory')
    parser.add_argument('--test_dir', type=str, default='Dataset/test',
                        help='Test images directory')
    parser.add_argument('--groundtruth_file', type=str, default='Dataset/list/groundtruth.txt',
                        help='Ground truth file for test set')
    parser.add_argument('--output_dir', type=str, default='Approach_A_Feature_Extraction/features',
                        help='Output directory for features')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction (increased for GPU utilization)')
    parser.add_argument('--image_size', type=int, default=518,
                        help='Input image size (DINOv2 uses 518x518)')
    parser.add_argument('--plant_model_path', type=str, default='dinov2_patch14_reg4_onlyclassifier_th/model_best.pth.tar',
                        help='Path to plant-pretrained model checkpoint')
    return parser.parse_args()


def get_model_config(model_type):
    """
    Get model configuration based on model type.

    Returns:
        dict: Configuration with model_name and feature_dim
    """
    configs = {
        'imagenet_small': {
            'model_name': 'vit_small_patch14_reg4_dinov2.lvd142m',
            'pretrained': True,
            'feature_dim': 384
        },
        'imagenet_base': {
            'model_name': 'vit_base_patch14_reg4_dinov2.lvd142m',
            'pretrained': True,
            'feature_dim': 768
        },
        'imagenet_large': {
            'model_name': 'vit_large_patch14_reg4_dinov2.lvd142m',
            'pretrained': True,
            'feature_dim': 1024
        },
        'plant_pretrained_base': {
            'model_name': 'vit_base_patch14_reg4_dinov2.lvd142m',
            'pretrained': False,  # Load custom weights
            'feature_dim': 768
        }
    }
    return configs[model_type]


def load_model(model_type, plant_model_path=None, device='cuda'):
    """
    Load DINOv2 model for feature extraction.

    Args:
        model_type (str): Model variant
        plant_model_path (str): Path to plant-pretrained checkpoint
        device (str): Device to load model on

    Returns:
        torch.nn.Module: Model ready for feature extraction
    """
    config = get_model_config(model_type)

    print(f"Loading {model_type} model...")

    # Create model with no classification head (num_classes=0 for feature extraction)
    model = timm.create_model(
        config['model_name'],
        pretrained=config['pretrained'],
        num_classes=0  # Remove classification head for feature extraction
    )

    # Load plant-pretrained weights if specified
    if model_type == 'plant_pretrained_base':
        if plant_model_path and os.path.exists(plant_model_path):
            print(f"Loading plant-pretrained weights from {plant_model_path}")
            checkpoint = torch.load(plant_model_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Remove head weights if present (we don't need them for feature extraction)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}

            # Load weights (strict=False to ignore missing head weights)
            model.load_state_dict(state_dict, strict=False)
            print("Plant-pretrained weights loaded successfully")
        else:
            print(f"WARNING: Plant model path not found: {plant_model_path}")
            print("Using ImageNet-pretrained DINOv2 base as fallback")
            model = timm.create_model(
                config['model_name'],
                pretrained=True,
                num_classes=0
            )

    model = model.to(device)
    model.eval()

    print(f"Model loaded: {config['model_name']}")
    print(f"   Feature dimension: {config['feature_dim']}")

    return model, config['feature_dim']


def extract_features(model, dataloader, device, desc="Extracting"):
    """
    Extract features from a dataloader.

    Args:
        model: Feature extractor model
        dataloader: DataLoader
        device: Device
        desc: Progress bar description

    Returns:
        tuple: (features, labels) as numpy arrays
    """
    features_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            if len(batch) == 2:
                images, labels = batch
            elif len(batch) == 3:
                images, labels, _ = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            images = images.to(device)

            # Extract features
            feats = model(images)

            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    # Concatenate all batches
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels


def main():
    args = parse_args()

    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("=" * 70)
    print(f"FEATURE EXTRACTION: {args.model_type}")
    print("=" * 70)

    # Load model
    model, feature_dim = load_model(args.model_type, args.plant_model_path, device)

    # Prepare data transforms
    transform = get_val_transforms(args.image_size)

    # Create output subdirectory for this model
    model_output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(model_output_dir, exist_ok=True)

    # Track statistics
    stats = {
        'model_type': args.model_type,
        'feature_dim': feature_dim,
        'image_size': args.image_size,
        'batch_size': args.batch_size
    }

    # Extract features from training set
    if os.path.exists(args.train_dir):
        print(f"\nProcessing training set: {args.train_dir}")
        train_dataset = PlantDataset(args.train_dir, transform=transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,  # Reduced to prevent resource exhaustion when running multiple models
            pin_memory=True
        )

        train_features, train_labels = extract_features(
            model, train_loader, device, desc="Training set"
        )

        # Save training features
        train_output = os.path.join(model_output_dir, 'train_features.npy')
        train_labels_output = os.path.join(model_output_dir, 'train_labels.npy')
        np.save(train_output, train_features)
        np.save(train_labels_output, train_labels)

        stats['train_samples'] = len(train_features)
        print(f"   Saved: {train_output}")
        print(f"   Shape: {train_features.shape}")
        print(f"   Labels: {train_labels.shape}")

    # Extract features from validation set
    if os.path.exists(args.val_dir):
        print(f"\nProcessing validation set: {args.val_dir}")
        val_dataset = PlantDataset(args.val_dir, transform=transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,  # Reduced to prevent resource exhaustion when running multiple models
            pin_memory=True
        )

        val_features, val_labels = extract_features(
            model, val_loader, device, desc="Validation set"
        )

        # Save validation features
        val_output = os.path.join(model_output_dir, 'val_features.npy')
        val_labels_output = os.path.join(model_output_dir, 'val_labels.npy')
        np.save(val_output, val_features)
        np.save(val_labels_output, val_labels)

        stats['val_samples'] = len(val_features)
        print(f"   Saved: {val_output}")
        print(f"   Shape: {val_features.shape}")
        print(f"   Labels: {val_labels.shape}")

    # Extract features from test set
    if os.path.exists(args.test_dir):
        print(f"\nProcessing test set: {args.test_dir}")
        test_dataset = PlantTestDataset(
            args.test_dir,
            transform=transform,
            groundtruth_file=args.groundtruth_file if os.path.exists(args.groundtruth_file) else None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,  # Reduced to prevent resource exhaustion when running multiple models
            pin_memory=True
        )

        # Extract features (handling test dataset format)
        features_list = []
        labels_list = []
        paths_list = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test set"):
                if len(batch) == 2:
                    images, paths = batch
                    labels = torch.full((images.size(0),), -1, dtype=torch.long)  # No labels
                else:
                    images, labels, paths = batch

                images = images.to(device)
                feats = model(images)

                features_list.append(feats.cpu().numpy())
                labels_list.append(labels.numpy())
                paths_list.extend(paths)

        test_features = np.concatenate(features_list, axis=0)
        test_labels = np.concatenate(labels_list, axis=0)

        # Save test features
        test_output = os.path.join(model_output_dir, 'test_features.npy')
        test_labels_output = os.path.join(model_output_dir, 'test_labels.npy')
        test_paths_output = os.path.join(model_output_dir, 'test_paths.txt')

        np.save(test_output, test_features)
        np.save(test_labels_output, test_labels)
        with open(test_paths_output, 'w') as f:
            f.write('\n'.join(paths_list))

        stats['test_samples'] = len(test_features)
        print(f"   Saved: {test_output}")
        print(f"   Shape: {test_features.shape}")
        print(f"   Labels: {test_labels.shape}")

    # Save metadata
    metadata_file = os.path.join(model_output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(stats, f, indent=4)

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"Output directory: {model_output_dir}")
    print(f"Feature dimension: {feature_dim}")
    if 'train_samples' in stats:
        print(f"   Training samples: {stats['train_samples']}")
    if 'val_samples' in stats:
        print(f"   Validation samples: {stats['val_samples']}")
    if 'test_samples' in stats:
        print(f"   Test samples: {stats['test_samples']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
