"""
GPU-Accelerated Data Augmentation using Kornia

This module provides GPU-accelerated data augmentations using the kornia library,
which performs transformations on GPU tensors, reducing CPU bottlenecks during training.
"""

import torch
import torch.nn as nn
try:
    import kornia as K
    import kornia.augmentation as KA
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: kornia not installed. Install with: pip install kornia")


class GPUAugmentation(nn.Module):
    """
    GPU-accelerated augmentation pipeline using Kornia.

    This performs augmentations on GPU tensors, reducing CPU preprocessing overhead.
    Use this as a model wrapper or apply directly to batched tensors on GPU.
    """

    def __init__(self, image_size=518, training=True):
        """
        Args:
            image_size (int): Target image size
            training (bool): Whether to apply training augmentations
        """
        super().__init__()

        if not KORNIA_AVAILABLE:
            raise ImportError("kornia is required for GPU augmentations. Install with: pip install kornia")

        self.training_mode = training
        self.image_size = image_size

        if training:
            # Training augmentations (applied on GPU)
            self.augmentations = nn.Sequential(
                KA.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0
                ),
                KA.RandomHorizontalFlip(p=0.5),
                KA.RandomVerticalFlip(p=0.3),
                KA.RandomRotation(degrees=30.0, p=0.7),
                KA.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=0.8
                ),
                KA.RandomGaussianBlur(
                    kernel_size=(3, 3),
                    sigma=(0.1, 2.0),
                    p=0.3
                ),
                # Normalize using ImageNet stats
                KA.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225])
                )
            )
        else:
            # Validation augmentations (just resize and normalize)
            self.augmentations = nn.Sequential(
                KA.Resize(size=(image_size, image_size)),
                KA.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225])
                )
            )

    def forward(self, x):
        """
        Apply augmentations to batch of images.

        Args:
            x (torch.Tensor): Batch of images [B, C, H, W] in range [0, 1]

        Returns:
            torch.Tensor: Augmented images [B, C, H, W]
        """
        return self.augmentations(x)


class AugmentedModel(nn.Module):
    """
    Wrapper that applies GPU augmentations before the model.

    This is useful for applying augmentations during training without
    modifying the data loading pipeline.
    """

    def __init__(self, model, augmentation):
        """
        Args:
            model (nn.Module): The base model
            augmentation (GPUAugmentation): GPU augmentation module
        """
        super().__init__()
        self.augmentation = augmentation
        self.model = model

    def forward(self, x):
        """
        Apply augmentations then forward through model.

        Args:
            x (torch.Tensor): Input batch [B, C, H, W]

        Returns:
            torch.Tensor: Model output
        """
        # Only apply augmentations during training
        if self.training and self.augmentation.training_mode:
            x = self.augmentation(x)
        else:
            # Still normalize during validation
            x = self.augmentation(x)

        return self.model(x)


def test_gpu_augmentation():
    """Test GPU augmentation pipeline"""
    if not KORNIA_AVAILABLE:
        print("Kornia not available, skipping test")
        return

    print("Testing GPU Augmentation Pipeline...")

    # Create dummy batch
    batch = torch.randn(4, 3, 518, 518).cuda()

    # Training augmentation
    train_aug = GPUAugmentation(image_size=518, training=True).cuda()
    augmented = train_aug(batch)
    print(f"✓ Training augmentation: {batch.shape} -> {augmented.shape}")

    # Validation augmentation
    val_aug = GPUAugmentation(image_size=518, training=False).cuda()
    augmented = val_aug(batch)
    print(f"✓ Validation augmentation: {batch.shape} -> {augmented.shape}")

    print("✓ GPU augmentation test passed!")


if __name__ == '__main__':
    test_gpu_augmentation()
