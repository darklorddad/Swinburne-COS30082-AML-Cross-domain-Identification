"""
Train Linear Probe (PyTorch) on DINOv2 Features

This script trains a simple linear classifier on pre-extracted DINOv2 features
using PyTorch with AdamW optimizer and label smoothing.

Usage:
    python Approach_A_Feature_Extraction/train_linear_probe.py \
        --features_dir Approach_A_Feature_Extraction/features/imagenet_base \
        --output_dir Approach_A_Feature_Extraction/results/linear_imagenet_base \
        --epochs 100 \
        --batch_size 128
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Src.utils.visualization import (save_training_history, save_metrics_summary,
                                      plot_training_history, plot_overfitting_analysis)
from Src.utils.detailed_evaluation import (
    load_class_categories,
    load_groundtruth,
    categorize_test_samples,
    compute_category_metrics,
    display_detailed_breakdown,
    save_detailed_results
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Linear Probe on DINOv2 features')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing extracted features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    return parser.parse_args()


def load_features(features_dir):
    """Load pre-extracted features"""
    print(f"📂 Loading features from {features_dir}")

    train_features = np.load(os.path.join(features_dir, 'train_features.npy'))
    train_labels = np.load(os.path.join(features_dir, 'train_labels.npy'))
    val_features = np.load(os.path.join(features_dir, 'val_features.npy'))
    val_labels = np.load(os.path.join(features_dir, 'val_labels.npy'))

    print(f"   ✅ Training: {train_features.shape}")
    print(f"   ✅ Validation: {val_features.shape}")

    # Convert to PyTorch tensors
    train_features = torch.from_numpy(train_features).float()
    train_labels = torch.from_numpy(train_labels).long()
    val_features = torch.from_numpy(val_features).float()
    val_labels = torch.from_numpy(val_labels).long()

    return train_features, train_labels, val_features, val_labels


def load_test_features(features_dir):
    """Load test features and labels"""
    test_features = np.load(os.path.join(features_dir, 'test_features.npy'))
    test_labels = np.load(os.path.join(features_dir, 'test_labels.npy'))
    return test_features, test_labels


class LinearProbe(nn.Module):
    """Simple linear classifier"""
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='Validating', leave=False):
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_linear_probe(X_train, y_train, X_val, y_val, args):
    """
    Train linear probe with early stopping.

    Returns:
        tuple: (model, history, training_time)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Get dimensions
    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    print(f"\n🧠 Creating Linear Probe...")
    print(f"   Input dimension: {input_dim}")
    print(f"   Number of classes: {num_classes}")

    # Create model
    model = LinearProbe(input_dim, num_classes).to(device)

    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training history
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"\n🏋️  Training Linear Probe...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Label smoothing: {args.label_smoothing}")
    print(f"   Early stopping patience: {args.patience}")

    start_time = time.time()

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Record history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Check for overfitting
        gap = train_acc - val_acc
        if gap > 10.0:
            print(f"   ⚠️  OVERFITTING WARNING: Gap = {gap:.2f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"   ✅ New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"   Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break

    training_time = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n✅ Training complete in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")

    return model, history, training_time, best_val_acc


def save_model_and_results(model, history, training_time, best_val_acc, output_dir, args):
    """Save model and results"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 Saving model and results to {output_dir}")

    # Save model
    model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"   ✅ Model saved: {model_path}")

    # Save training config
    config = {
        'input_dim': model.fc.in_features,
        'num_classes': model.fc.out_features,
        'epochs_trained': len(history['epochs']),
        'best_val_accuracy': float(best_val_acc),
        'hyperparameters': {
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'label_smoothing': args.label_smoothing,
            'batch_size': args.batch_size,
            'max_epochs': args.epochs,
            'patience': args.patience
        },
        'training_time_seconds': float(training_time)
    }

    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"   ✅ Config saved: {config_path}")

    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    save_training_history(history, history_path)
    print(f"   ✅ History saved: {history_path}")

    # Create results subdirectory
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Plot training curves
    plot_training_history(history, results_dir, model_name="Linear Probe")
    print(f"   ✅ Training curves saved")

    # Plot overfitting analysis
    plot_overfitting_analysis(history, results_dir, model_name="Linear Probe", threshold=10.0)
    print(f"   ✅ Overfitting analysis saved")

    # Save metrics summary
    metrics = {
        'best_val_accuracy': float(best_val_acc),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'training_time_seconds': float(training_time),
        'epochs_trained': len(history['epochs'])
    }

    metrics_path = os.path.join(results_dir, 'metrics_summary')
    save_metrics_summary(metrics, metrics_path)
    print(f"   ✅ Metrics summary saved")

    # Evaluate on test set with domain breakdown
    print(f"\n{'='*70}")
    print("📊 TEST SET EVALUATION (Domain Breakdown)")
    print(f"{'='*70}")

    try:
        # Load test features
        X_test, y_test = load_test_features(args.features_dir)

        # Load domain categories
        with_pairs, without_pairs = load_class_categories(
            'Dataset/list/class_with_pairs.txt',
            'Dataset/list/class_without_pairs.txt'
        )

        # Load groundtruth
        groundtruth = load_groundtruth('Dataset/list/groundtruth.txt')

        # Get test image names (sorted to match feature order)
        image_names = sorted(list(groundtruth.keys()))

        # Predict on test set
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        X_test_tensor = torch.from_numpy(X_test).float().to(device)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probs = torch.softmax(outputs, dim=1)

        y_pred_proba_test = probs.cpu().numpy()

        # Categorize test samples
        categories = categorize_test_samples(image_names, groundtruth, with_pairs, without_pairs)

        # Compute metrics for each category
        results = {}
        for category_name, indices in categories.items():
            results[category_name] = compute_category_metrics(y_pred_proba_test, y_test, indices)

        # Display results
        display_detailed_breakdown("Test Set", results)

        # Save detailed results
        save_detailed_results(results, output_dir, "test_domain_metrics.json")
        print(f"   ✅ Domain-specific metrics saved to: {output_dir}/test_domain_metrics.json")

    except Exception as e:
        import traceback
        print(f"   ⚠️  Warning: Could not compute domain-specific metrics: {e}")
        print(f"   Full error: {traceback.format_exc()}")


def main():
    args = parse_args()

    print("=" * 70)
    print("🧠 LINEAR PROBE TRAINING")
    print("=" * 70)

    # Load features
    X_train, y_train, X_val, y_val = load_features(args.features_dir)

    # Train linear probe
    model, history, training_time, best_val_acc = train_linear_probe(
        X_train, y_train, X_val, y_val, args
    )

    # Save model and results
    save_model_and_results(model, history, training_time, best_val_acc, args.output_dir, args)

    print("\n" + "=" * 70)
    print("✨ LINEAR PROBE TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"✅ Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"⏱️  Training Time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"📊 Epochs trained: {len(history['epochs'])}")
    print("=" * 70)


if __name__ == '__main__':
    main()
