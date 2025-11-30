"""
Evaluate All Approach A Classifiers on Test Set

This script evaluates all trained classifiers (SVM, Linear Probe, Logistic Regression)
on the test set and calculates Top-1, Top-5, and average per-class accuracy.

Usage:
    python Approach_A_Feature_Extraction/evaluate_classifiers.py \
        --results_dir Approach_A_Feature_Extraction/results \
        --features_base Approach_A_Feature_Extraction/features \
        --output_file Approach_A_evaluation_results.json
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Src.utils.visualization import plot_confusion_matrix, save_metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate all Approach A classifiers')
    parser.add_argument('--results_dir', type=str, default='Approach_A_Feature_Extraction/results',
                        help='Directory containing trained models')
    parser.add_argument('--features_base', type=str, default='Approach_A_Feature_Extraction/features',
                        help='Base directory for features')
    parser.add_argument('--output_file', type=str, default='Approach_A_evaluation_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--classes_file', type=str, default='classes.txt',
                        help='File containing class names')
    return parser.parse_args()


class LinearProbe(nn.Module):
    """Linear probe model (must match training script)"""
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def load_class_names(classes_file):
    """Load class names"""
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return None


def load_test_features(features_dir):
    """Load test features and labels"""
    test_features = np.load(os.path.join(features_dir, 'test_features.npy'))
    test_labels = np.load(os.path.join(features_dir, 'test_labels.npy'))
    return test_features, test_labels


def calculate_top_k_accuracy(y_true, y_pred_proba, k=5):
    """
    Calculate top-k accuracy.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (N x num_classes)
        k: Top-k

    Returns:
        float: Top-k accuracy
    """
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

    return np.mean(per_class_acc)


def evaluate_sklearn_model(model_path, X_test, y_test, model_name):
    """Evaluate scikit-learn model (SVM or Logistic Regression)"""
    print(f"\n📊 Evaluating {model_name}...")

    # Load model
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    top1_acc = accuracy_score(y_test, y_pred)
    top5_acc = calculate_top_k_accuracy(y_test, y_pred_proba, k=5)
    avg_per_class_acc = calculate_per_class_accuracy(y_test, y_pred)

    metrics = {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'avg_per_class_accuracy': float(avg_per_class_acc)
    }

    print(f"   ✅ Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"   ✅ Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"   ✅ Avg Per-Class Accuracy: {avg_per_class_acc:.4f} ({avg_per_class_acc*100:.2f}%)")

    return metrics, y_pred, y_pred_proba


def evaluate_pytorch_model(model_path, config_path, X_test, y_test, model_name):
    """Evaluate PyTorch linear probe model"""
    print(f"\n📊 Evaluating {model_name}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model
    model = LinearProbe(config['input_dim'], config['num_classes'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Convert to tensor
    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    # Predict
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, y_pred_tensor = outputs.max(1)

    y_pred = y_pred_tensor.cpu().numpy()
    y_pred_proba = probs.cpu().numpy()

    # Calculate metrics
    top1_acc = accuracy_score(y_test, y_pred)
    top5_acc = calculate_top_k_accuracy(y_test, y_pred_proba, k=5)
    avg_per_class_acc = calculate_per_class_accuracy(y_test, y_pred)

    metrics = {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'avg_per_class_accuracy': float(avg_per_class_acc)
    }

    print(f"   ✅ Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"   ✅ Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"   ✅ Avg Per-Class Accuracy: {avg_per_class_acc:.4f} ({avg_per_class_acc*100:.2f}%)")

    return metrics, y_pred, y_pred_proba


def find_models(results_dir):
    """Find all trained models in results directory"""
    models = []

    for model_dir in os.listdir(results_dir):
        model_path_full = os.path.join(results_dir, model_dir)

        if not os.path.isdir(model_path_full):
            continue

        # Check for different model types
        sklearn_model = os.path.join(model_path_full, 'best_model.joblib')
        pytorch_model = os.path.join(model_path_full, 'best_model.pth')
        config_file = os.path.join(model_path_full, 'training_config.json')

        if os.path.exists(sklearn_model):
            # Extract info from directory name (e.g., "svm_imagenet_base")
            parts = model_dir.split('_')
            classifier_type = parts[0]  # svm, rf
            feature_type = '_'.join(parts[1:])  # imagenet_base, plant_pretrained_base, etc.

            models.append({
                'name': model_dir,
                'type': 'sklearn',
                'classifier': classifier_type.upper(),
                'features': feature_type,
                'model_path': sklearn_model,
                'config_path': config_file if os.path.exists(config_file) else None
            })

        elif os.path.exists(pytorch_model) and os.path.exists(config_file):
            parts = model_dir.split('_')
            classifier_type = parts[0]  # linear
            feature_type = '_'.join(parts[1:])

            models.append({
                'name': model_dir,
                'type': 'pytorch',
                'classifier': 'Linear Probe',
                'features': feature_type,
                'model_path': pytorch_model,
                'config_path': config_file
            })

    return models


def main():
    args = parse_args()

    print("=" * 70)
    print("🧪 APPROACH A: CLASSIFIER EVALUATION")
    print("=" * 70)

    # Load class names
    class_names = load_class_names(args.classes_file)
    if class_names:
        print(f"✅ Loaded {len(class_names)} class names")

    # Find all models
    models = find_models(args.results_dir)
    print(f"\n🔍 Found {len(models)} trained models")

    for model_info in models:
        print(f"   - {model_info['name']} ({model_info['classifier']} on {model_info['features']})")

    # Evaluate each model
    all_results = {}

    for model_info in models:
        model_name = model_info['name']
        feature_type = model_info['features']

        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")

        # Load test features
        features_dir = os.path.join(args.features_base, feature_type)

        if not os.path.exists(features_dir):
            print(f"⚠️  Warning: Features directory not found: {features_dir}")
            continue

        X_test, y_test = load_test_features(features_dir)
        print(f"📂 Loaded test features: {X_test.shape}")

        # Evaluate model
        try:
            if model_info['type'] == 'sklearn':
                metrics, y_pred, y_pred_proba = evaluate_sklearn_model(
                    model_info['model_path'],
                    X_test,
                    y_test,
                    model_name
                )
            else:  # pytorch
                metrics, y_pred, y_pred_proba = evaluate_pytorch_model(
                    model_info['model_path'],
                    model_info['config_path'],
                    X_test,
                    y_test,
                    model_name
                )

            # Add model info to metrics
            metrics['model_name'] = model_name
            metrics['classifier'] = model_info['classifier']
            metrics['features'] = feature_type

            all_results[model_name] = metrics

            # Save confusion matrix
            results_subdir = os.path.join(args.results_dir, model_name, 'results')
            os.makedirs(results_subdir, exist_ok=True)

            if class_names:
                plot_confusion_matrix(
                    y_test,
                    y_pred,
                    class_names,
                    results_subdir,
                    model_name=model_name,
                    normalize=True
                )
                print(f"   ✅ Confusion matrix saved")

        except Exception as e:
            print(f"   ❌ Error evaluating {model_name}: {e}")
            continue

    # Save all results
    print(f"\n{'='*70}")
    print("💾 Saving Results")
    print(f"{'='*70}")

    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"✅ Results saved to: {args.output_file}")

    # Create comparison table
    if all_results:
        df = pd.DataFrame.from_dict(all_results, orient='index')
        df = df.sort_values('top1_accuracy', ascending=False)

        # Save as CSV
        csv_file = args.output_file.replace('.json', '.csv')
        df.to_csv(csv_file)
        print(f"✅ CSV table saved to: {csv_file}")

        # Print comparison table
        print(f"\n{'='*70}")
        print("📊 COMPARISON TABLE")
        print(f"{'='*70}\n")

        print(df[['classifier', 'features', 'top1_accuracy', 'top5_accuracy', 'avg_per_class_accuracy']].to_string())

        # Find best model
        best_model = df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model.name}")
        print(f"   Classifier: {best_model['classifier']}")
        print(f"   Features: {best_model['features']}")
        print(f"   Top-1 Accuracy: {best_model['top1_accuracy']:.4f} ({best_model['top1_accuracy']*100:.2f}%)")
        print(f"   Top-5 Accuracy: {best_model['top5_accuracy']:.4f} ({best_model['top5_accuracy']*100:.2f}%)")

    print(f"\n{'='*70}")
    print("✨ EVALUATION COMPLETE!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
