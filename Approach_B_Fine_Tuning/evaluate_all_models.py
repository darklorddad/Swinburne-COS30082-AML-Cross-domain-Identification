"""
Evaluate All Approach B Fine-Tuned Models on Test Set

This script evaluates all fine-tuned DINOv2 models on the test set and calculates
Top-1, Top-5, and average per-class accuracy.

Usage:
    python Approach_B_Fine_Tuning/evaluate_all_models.py \
        --models_dir Approach_B_Fine_Tuning/Models \
        --test_dir Dataset/test \
        --groundtruth_file Dataset/list/groundtruth.txt \
        --output_file Approach_B_evaluation_results.json
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
import numpy as np
from tqdm import tqdm
import json
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Src.utils.dataset_loader import PlantTestDataset, get_val_transforms
from Src.utils.visualization import plot_confusion_matrix, save_metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate all Approach B models')
    parser.add_argument('--models_dir', type=str, default='Approach_B_Fine_Tuning/Models')
    parser.add_argument('--test_dir', type=str, default='Dataset/test')
    parser.add_argument('--groundtruth_file', type=str, default='Dataset/list/groundtruth.txt')
    parser.add_argument('--output_file', type=str, default='Approach_B_evaluation_results.json')
    parser.add_argument('--classes_file', type=str, default='classes.txt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    return parser.parse_args()


def load_class_names(classes_file):
    """Load class names"""
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return None


def load_model(model_dir, device):
    """Load a fine-tuned model"""
    # Load config
    config_path = os.path.join(model_dir, 'training_config.json')
    if not os.path.exists(config_path):
        return None, None

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_type = config['model_type']
    model_name = config['model_name']
    num_classes = config['num_classes']

    # Create model
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    # Load weights
    model_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        return None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, config


def calculate_top_k_accuracy(y_true, y_pred_proba, k=5):
    """Calculate top-k accuracy"""
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return correct.mean()


def calculate_per_class_accuracy(y_true, y_pred):
    """Calculate average per-class accuracy"""
    classes = np.unique(y_true)
    per_class_acc = []

    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean()
            per_class_acc.append(class_acc)

    return np.mean(per_class_acc) if per_class_acc else 0.0


def evaluate_model(model, dataloader, device, model_name):
    """Evaluate a model on test set"""
    print(f"\n📊 Evaluating {model_name}...")

    all_preds = []
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            if len(batch) == 2:
                images, paths = batch
                # No labels available
                continue
            else:
                images, labels, paths = batch

            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    if len(all_labels) == 0:
        print("   ⚠️  No labels found in test set")
        return None

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate metrics
    top1_acc = (all_preds == all_labels).mean()
    top5_acc = calculate_top_k_accuracy(all_labels, all_probs, k=5)
    avg_per_class_acc = calculate_per_class_accuracy(all_labels, all_preds)

    metrics = {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'avg_per_class_accuracy': float(avg_per_class_acc)
    }

    print(f"   ✅ Top-1: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"   ✅ Top-5: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"   ✅ Avg Per-Class: {avg_per_class_acc:.4f} ({avg_per_class_acc*100:.2f}%)")

    return metrics, all_preds, all_labels


def find_models(models_dir):
    """Find all trained models"""
    models = []

    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)

        if not os.path.isdir(model_path):
            continue

        config_path = os.path.join(model_path, 'training_config.json')
        model_file = os.path.join(model_path, 'best_model.pth')

        if os.path.exists(config_path) and os.path.exists(model_file):
            models.append({
                'name': model_name,
                'path': model_path
            })

    return models


def main():
    args = parse_args()

    print("=" * 70)
    print("🧪 APPROACH B: FINE-TUNED MODELS EVALUATION")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load class names
    class_names = load_class_names(args.classes_file)
    if class_names:
        print(f"✅ Loaded {len(class_names)} class names")

    # Find models
    models = find_models(args.models_dir)
    print(f"\n🔍 Found {len(models)} trained models:")
    for model_info in models:
        print(f"   - {model_info['name']}")

    # Load test dataset
    print(f"\n📂 Loading test dataset from {args.test_dir}")
    test_transform = get_val_transforms(args.image_size)
    test_dataset = PlantTestDataset(
        args.test_dir,
        transform=test_transform,
        groundtruth_file=args.groundtruth_file
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f"   Test samples: {len(test_dataset)}")

    # Evaluate each model
    all_results = {}

    for model_info in models:
        model_name = model_info['name']
        model_path = model_info['path']

        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")

        # Load model
        model, config = load_model(model_path, device)

        if model is None:
            print(f"   ❌ Failed to load model")
            continue

        # Evaluate
        try:
            result = evaluate_model(model, test_loader, device, model_name)

            if result is None:
                continue

            metrics, y_pred, y_true = result

            # Add model info
            metrics['model_name'] = model_name
            metrics['model_type'] = config['model_type']

            all_results[model_name] = metrics

            # Save confusion matrix
            results_subdir = os.path.join(model_path, 'results')
            os.makedirs(results_subdir, exist_ok=True)

            if class_names:
                plot_confusion_matrix(
                    y_true,
                    y_pred,
                    class_names,
                    results_subdir,
                    model_name=model_name,
                    normalize=True
                )
                print(f"   ✅ Confusion matrix saved")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    # Save results
    print(f"\n{'='*70}")
    print("💾 SAVING RESULTS")
    print(f"{'='*70}")

    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"✅ Results saved: {args.output_file}")

    # Create comparison table
    if all_results:
        df = pd.DataFrame.from_dict(all_results, orient='index')
        df = df.sort_values('top1_accuracy', ascending=False)

        csv_file = args.output_file.replace('.json', '.csv')
        df.to_csv(csv_file)
        print(f"✅ CSV saved: {csv_file}")

        print(f"\n{'='*70}")
        print("📊 COMPARISON TABLE")
        print(f"{'='*70}\n")

        print(df[['model_type', 'top1_accuracy', 'top5_accuracy', 'avg_per_class_accuracy']].to_string())

        # Best model
        best_model = df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model.name}")
        print(f"   Type: {best_model['model_type']}")
        print(f"   Top-1: {best_model['top1_accuracy']:.4f} ({best_model['top1_accuracy']*100:.2f}%)")
        print(f"   Top-5: {best_model['top5_accuracy']:.4f} ({best_model['top5_accuracy']*100:.2f}%)")

    print(f"\n{'='*70}")
    print("✨ EVALUATION COMPLETE!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
