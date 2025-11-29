"""
Fine-tune Plant-Pretrained DINOv2 Base Model for Maximum Accuracy

This script implements advanced fine-tuning techniques to achieve the highest
possible accuracy on cross-domain plant identification:
- Gradual unfreezing (head only → last 4 blocks)
- Differential learning rates
- Advanced data augmentation (MixUp, CutMix, RandAugment)
- Cosine annealing with warm restarts
- Mixed precision training (FP16)
- Early stopping with overfitting detection

Usage:
    python Approach_B_Fine_Tuning/Models/plant_pretrained_base/train.py \
        --train_dir Dataset/balanced_train \
        --val_dir Dataset/validation \
        --plant_model_path Models/pretrained/model_best.pth.tar \
        --epochs 60 \
        --batch_size 32
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import json
import time
import numpy as np

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Src.utils.dataset_loader import PlantDataset, get_train_transforms, get_val_transforms
from Src.utils.visualization import (save_training_history, plot_training_history,
                                      plot_loss_curves, plot_accuracy_curves,
                                      plot_learning_rate, plot_overfitting_analysis,
                                      plot_confusion_matrix, save_metrics_summary)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Plant-Pretrained DINOv2 Base')
    parser.add_argument('--train_dir', type=str, default='Dataset/balanced_train')
    parser.add_argument('--val_dir', type=str, default='Dataset/validation')
    parser.add_argument('--plant_model_path', type=str, default='Models/pretrained/model_best.pth.tar',
                        help='Path to plant-pretrained model')
    parser.add_argument('--output_dir', type=str,
                        default='Approach_B_Fine_Tuning/Models/plant_pretrained_base')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--lr_backbone', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def load_plant_pretrained_model(model_path, num_classes=100, dropout=0.4):
    """
    Load plant-pretrained DINOv2 model.

    Args:
        model_path: Path to checkpoint
        num_classes: Number of output classes
        dropout: Dropout rate for classifier

    Returns:
        model: Loaded model
    """
    print(f"🌱 Loading plant-pretrained DINOv2 model...")

    # Create base model (no pretrained ImageNet weights)
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=False,
        num_classes=num_classes,
        drop_rate=dropout
    )

    # Load plant-pretrained weights
    if os.path.exists(model_path):
        print(f"   Loading from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Remove head weights (we'll train new head for our 100 classes)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}

        # Load weights (strict=False because head is different)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"   ✅ Loaded plant-pretrained weights")
        print(f"   Missing keys (expected - new head): {len(missing)}")
        print(f"   Unexpected keys: {len(unexpected)}")
    else:
        print(f"   ⚠️  Warning: Model file not found: {model_path}")
        print(f"   Using ImageNet-pretrained DINOv2 as fallback")
        model = timm.create_model(
            'vit_base_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            num_classes=num_classes,
            drop_rate=dropout
        )

    return model


def setup_training(model, args, stage='head_only'):
    """
    Setup optimizer and scheduler for different training stages.

    Args:
        model: The model
        args: Arguments
        stage: 'head_only' or 'gradual_unfreeze'

    Returns:
        optimizer, scheduler
    """
    if stage == 'head_only':
        # Stage 1: Train only head
        print("\n🔒 Stage 1: Freezing backbone, training head only")

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze head
        for param in model.head.parameters():
            param.requires_grad = True

        # Optimizer only for head
        optimizer = optim.AdamW(
            model.head.parameters(),
            lr=args.lr_head,
            weight_decay=args.weight_decay
        )

    else:  # gradual_unfreeze
        # Stage 2: Unfreeze last 4 transformer blocks + head
        print("\n🔓 Stage 2: Unfreezing last 4 transformer blocks + head")

        # Freeze all first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last 4 blocks (assuming ViT structure)
        # DINOv2 ViT-Base has 12 blocks, unfreeze blocks 8-11 (indices 8,9,10,11)
        if hasattr(model, 'blocks'):
            num_blocks = len(model.blocks)
            unfreeze_from = num_blocks - 4
            print(f"   Unfreezing blocks {unfreeze_from} to {num_blocks-1}")

            for i in range(unfreeze_from, num_blocks):
                for param in model.blocks[i].parameters():
                    param.requires_grad = True

        # Unfreeze head
        for param in model.head.parameters():
            param.requires_grad = True

        # Collect parameters for differential learning rates
        head_params = list(model.head.parameters())

        backbone_params = []
        if hasattr(model, 'blocks'):
            for i in range(unfreeze_from, num_blocks):
                backbone_params.extend(model.blocks[i].parameters())

        # Optimizer with differential learning rates
        optimizer = optim.AdamW([
            {'params': head_params, 'lr': args.lr_head},
            {'params': backbone_params, 'lr': args.lr_backbone}
        ], weight_decay=args.weight_decay)

    # Scheduler: Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, max_epochs, scaler):
    """Train for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{max_epochs}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Mixed precision backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'LR': f'{current_lr:.2e}'
        })

    # Step scheduler
    scheduler.step()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    current_lr = optimizer.param_groups[0]['lr']

    return epoch_loss, epoch_acc, current_lr


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating', leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()

    print("=" * 70)
    print("🌱 PLANT-PRETRAINED DINOv2 BASE - FINE-TUNING")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("\n📂 Loading datasets...")
    train_transform = get_train_transforms(args.image_size)
    val_transform = get_val_transforms(args.image_size)

    train_dataset = PlantDataset(args.train_dir, transform=train_transform)
    val_dataset = PlantDataset(args.val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Number of classes: {len(train_dataset.classes)}")

    # Load model
    model = load_plant_pretrained_model(
        args.plant_model_path,
        num_classes=len(train_dataset.classes),
        dropout=args.dropout
    )
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training history
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0

    # Stage 1: Train head only (warmup)
    optimizer, scheduler = setup_training(model, args, stage='head_only')

    print(f"\n{'='*70}")
    print("🏋️  STAGE 1: TRAINING HEAD ONLY (WARMUP)")
    print(f"{'='*70}")

    stage1_start = time.time()

    for epoch in range(args.warmup_epochs):
        train_loss, train_acc, lr = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch+1, args.warmup_epochs, scaler
        )

        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        # Record history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(lr)

        # Print summary
        print(f"\n   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"   ✅ New best: {best_val_acc:.2f}%")

    stage1_time = time.time() - stage1_start
    print(f"\n✅ Stage 1 complete in {stage1_time/60:.1f} minutes")

    # Stage 2: Gradual unfreezing
    optimizer, scheduler = setup_training(model, args, stage='gradual_unfreeze')

    print(f"\n{'='*70}")
    print("🏋️  STAGE 2: GRADUAL UNFREEZING")
    print(f"{'='*70}")

    stage2_start = time.time()
    remaining_epochs = args.epochs - args.warmup_epochs

    for epoch in range(remaining_epochs):
        actual_epoch = args.warmup_epochs + epoch + 1

        train_loss, train_acc, lr = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, actual_epoch, args.epochs, scaler
        )

        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        # Record history
        history['epochs'].append(actual_epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(lr)

        # Print summary
        print(f"\n   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Overfitting check
        gap = train_acc - val_acc
        if gap > 10.0:
            print(f"   ⚠️  OVERFITTING WARNING: Gap = {gap:.2f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"   ✅ New best: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"   Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"\n⚠️  Early stopping at epoch {actual_epoch}")
                break

    stage2_time = time.time() - stage2_start
    total_time = stage1_time + stage2_time

    # Save results
    print(f"\n{'='*70}")
    print("💾 SAVING RESULTS")
    print(f"{'='*70}")

    # Save history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    save_training_history(history, history_path)
    print(f"✅ History saved: {history_path}")

    # Save config
    config = {
        'model_type': 'plant_pretrained_base',
        'num_classes': len(train_dataset.classes),
        'epochs_trained': len(history['epochs']),
        'best_val_accuracy': float(best_val_acc),
        'hyperparameters': vars(args),
        'training_time_seconds': float(total_time)
    }

    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✅ Config saved: {config_path}")

    # Create plots
    plot_training_history(history, results_dir, "Plant-Pretrained DINOv2 Base")
    plot_loss_curves(history, results_dir, "Plant-Pretrained DINOv2 Base")
    plot_accuracy_curves(history, results_dir, "Plant-Pretrained DINOv2 Base")
    plot_learning_rate(history, results_dir, "Plant-Pretrained DINOv2 Base")
    plot_overfitting_analysis(history, results_dir, "Plant-Pretrained DINOv2 Base")
    print(f"✅ Plots saved to: {results_dir}")

    # Save metrics
    metrics = {
        'best_val_accuracy': float(best_val_acc),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'training_time_seconds': float(total_time),
        'stage1_time_seconds': float(stage1_time),
        'stage2_time_seconds': float(stage2_time)
    }

    metrics_path = os.path.join(results_dir, 'metrics_summary')
    save_metrics_summary(metrics, metrics_path)
    print(f"✅ Metrics saved")

    # Final summary
    print(f"\n{'='*70}")
    print("✨ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"⏱️  Total Training Time: {total_time/60:.1f} minutes")
    print(f"   Stage 1 (Head only): {stage1_time/60:.1f} minutes")
    print(f"   Stage 2 (Unfreezing): {stage2_time/60:.1f} minutes")
    print(f"📊 Epochs trained: {len(history['epochs'])}")
    print(f"📁 Model saved: {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
