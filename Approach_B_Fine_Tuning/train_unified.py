"""
Unified Fine-Tuning Script for All DINOv2 Variants

This unified script can train any DINOv2 variant with the same advanced techniques:
- Plant-pretrained Base
- ImageNet Small, Base, Large

Usage Examples:
    # Plant-pretrained Base
    python Approach_B_Fine_Tuning/train_unified.py \
        --model_type plant_pretrained_base \
        --plant_model_path Models/pretrained/model_best.pth.tar

    # ImageNet Small
    python Approach_B_Fine_Tuning/train_unified.py \
        --model_type imagenet_small

    # ImageNet Base
    python Approach_B_Fine_Tuning/train_unified.py \
        --model_type imagenet_base

    # ImageNet Large
    python Approach_B_Fine_Tuning/train_unified.py \
        --model_type imagenet_large \
        --batch_size 16
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Src.utils.dataset_loader import PlantDataset, get_train_transforms, get_val_transforms
from Src.utils.visualization import (save_training_history, plot_training_history,
                                      plot_loss_curves, plot_accuracy_curves,
                                      plot_learning_rate, plot_overfitting_analysis,
                                      save_metrics_summary)


# Model configurations
MODEL_CONFIGS = {
    'plant_pretrained_base': {
        'model_name': 'vit_base_patch14_reg4_dinov2.lvd142m',
        'pretrained': False,  # Load custom plant weights
        'display_name': 'Plant-Pretrained DINOv2 Base'
    },
    'imagenet_small': {
        'model_name': 'vit_small_patch14_reg4_dinov2.lvd142m',
        'pretrained': True,
        'display_name': 'ImageNet DINOv2 Small'
    },
    'imagenet_base': {
        'model_name': 'vit_base_patch14_reg4_dinov2.lvd142m',
        'pretrained': True,
        'display_name': 'ImageNet DINOv2 Base'
    },
    'imagenet_large': {
        'model_name': 'vit_large_patch14_reg4_dinov2.lvd142m',
        'pretrained': True,
        'display_name': 'ImageNet DINOv2 Large'
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='Unified DINOv2 Fine-Tuning')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['plant_pretrained_base', 'imagenet_small', 'imagenet_base', 'imagenet_large'])
    parser.add_argument('--train_dir', type=str, default='Dataset/balanced_train')
    parser.add_argument('--val_dir', type=str, default='Dataset/validation')
    parser.add_argument('--plant_model_path', type=str, default='Models/pretrained/model_best.pth.tar')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: Approach_B_Fine_Tuning/Models/<model_type>)')
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


def load_model(model_type, plant_model_path, num_classes, dropout):
    """Load DINOv2 model based on type"""
    config = MODEL_CONFIGS[model_type]

    print(f"🔧 Loading {config['display_name']}...")

    # Create model
    model = timm.create_model(
        config['model_name'],
        pretrained=config['pretrained'],
        num_classes=num_classes,
        drop_rate=dropout
    )

    # Load custom plant weights if needed
    if model_type == 'plant_pretrained_base' and os.path.exists(plant_model_path):
        print(f"   Loading plant-pretrained weights from: {plant_model_path}")
        checkpoint = torch.load(plant_model_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}

        model.load_state_dict(state_dict, strict=False)
        print(f"   ✅ Plant-pretrained weights loaded")
    elif config['pretrained']:
        print(f"   ✅ ImageNet-pretrained weights loaded from timm")

    return model


def setup_training(model, args, stage='head_only'):
    """Setup optimizer and scheduler"""
    if stage == 'head_only':
        print("\n🔒 Stage 1: Freezing backbone, training head only")

        for param in model.parameters():
            param.requires_grad = False

        for param in model.head.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(
            model.head.parameters(),
            lr=args.lr_head,
            weight_decay=args.weight_decay
        )

    else:  # gradual_unfreeze
        print("\n🔓 Stage 2: Unfreezing last 4 transformer blocks + head")

        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, 'blocks'):
            num_blocks = len(model.blocks)
            unfreeze_from = num_blocks - 4
            print(f"   Unfreezing blocks {unfreeze_from} to {num_blocks-1}")

            for i in range(unfreeze_from, num_blocks):
                for param in model.blocks[i].parameters():
                    param.requires_grad = True

        for param in model.head.parameters():
            param.requires_grad = True

        head_params = list(model.head.parameters())
        backbone_params = []
        if hasattr(model, 'blocks'):
            for i in range(unfreeze_from, num_blocks):
                backbone_params.extend(model.blocks[i].parameters())

        optimizer = optim.AdamW([
            {'params': head_params, 'lr': args.lr_head},
            {'params': backbone_params, 'lr': args.lr_backbone}
        ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, max_epochs, scaler):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{max_epochs}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'LR': f'{current_lr:.2e}'
        })

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

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating', leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def main():
    args = parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = f'Approach_B_Fine_Tuning/Models/{args.model_type}'

    config = MODEL_CONFIGS[args.model_type]

    print("=" * 70)
    print(f"🚀 {config['display_name'].upper()} - FINE-TUNING")
    print("=" * 70)
    print(f"Model type: {args.model_type}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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

    print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"   Classes: {len(train_dataset.classes)}")

    # Load model
    model = load_model(args.model_type, args.plant_model_path,
                      len(train_dataset.classes), args.dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler()

    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    best_val_acc = 0.0
    patience_counter = 0

    # Stage 1: Head only
    optimizer, scheduler = setup_training(model, args, stage='head_only')

    print(f"\n{'='*70}")
    print("🏋️  STAGE 1: TRAINING HEAD ONLY")
    print(f"{'='*70}")

    stage1_start = time.time()

    for epoch in range(args.warmup_epochs):
        train_loss, train_acc, lr = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch+1, args.warmup_epochs, scaler
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(lr)

        print(f"\n   Train: {train_loss:.4f} | {train_acc:.2f}%")
        print(f"   Val: {val_loss:.4f} | {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"   ✅ Best: {best_val_acc:.2f}%")

    stage1_time = time.time() - stage1_start

    # Stage 2: Gradual unfreezing
    optimizer, scheduler = setup_training(model, args, stage='gradual_unfreeze')

    print(f"\n{'='*70}")
    print("🏋️  STAGE 2: GRADUAL UNFREEZING")
    print(f"{'='*70}")

    stage2_start = time.time()

    for epoch in range(args.epochs - args.warmup_epochs):
        actual_epoch = args.warmup_epochs + epoch + 1

        train_loss, train_acc, lr = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, actual_epoch, args.epochs, scaler
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['epochs'].append(actual_epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(lr)

        print(f"\n   Train: {train_loss:.4f} | {train_acc:.2f}%")
        print(f"   Val: {val_loss:.4f} | {val_acc:.2f}%")

        gap = train_acc - val_acc
        if gap > 10.0:
            print(f"   ⚠️  Overfitting: {gap:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"   ✅ Best: {best_val_acc:.2f}%")
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

    save_training_history(history, os.path.join(args.output_dir, 'training_history.json'))

    config_data = {
        'model_type': args.model_type,
        'model_name': config['model_name'],
        'num_classes': len(train_dataset.classes),
        'epochs_trained': len(history['epochs']),
        'best_val_accuracy': float(best_val_acc),
        'hyperparameters': vars(args),
        'training_time_seconds': float(total_time)
    }

    with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as f:
        json.dump(config_data, f, indent=4)

    # Plots
    plot_training_history(history, results_dir, config['display_name'])
    plot_loss_curves(history, results_dir, config['display_name'])
    plot_accuracy_curves(history, results_dir, config['display_name'])
    plot_learning_rate(history, results_dir, config['display_name'])
    plot_overfitting_analysis(history, results_dir, config['display_name'])

    # Metrics
    metrics = {
        'best_val_accuracy': float(best_val_acc),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'training_time_seconds': float(total_time)
    }

    save_metrics_summary(metrics, os.path.join(results_dir, 'metrics_summary'))

    print(f"\n{'='*70}")
    print("✨ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"🏆 Best Val Acc: {best_val_acc:.2f}%")
    print(f"⏱️  Time: {total_time/60:.1f} min")
    print(f"📊 Epochs: {len(history['epochs'])}")
    print(f"📁 Saved: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
