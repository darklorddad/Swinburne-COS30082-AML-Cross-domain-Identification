"""
Evaluate Approach A Classifiers with Domain-Specific Breakdown

This script evaluates all trained classifiers (SVM, Random Forest, Linear Probe, Logistic Regression)
on the test set with metrics broken down by:
1. Overall - All 207 test samples
2. With Pairs - 60 classes with both herbarium + field training images
3. Without Pairs - 40 classes with herbarium-only training images

Metrics computed:
- Top-1 Accuracy
- Top-5 Accuracy
- Mean Reciprocal Rank (MRR)

Usage:
    python Approach_A_Feature_Extraction/evaluate_classifiers_detailed.py \
        --results_dir Approach_A_Feature_Extraction/results \
        --features_base Approach_A_Feature_Extraction/features \
        --with_pairs Dataset/list/class_with_pairs.txt \
        --without_pairs Dataset/list/class_without_pairs.txt \
        --groundtruth Dataset/list/groundtruth.txt \
        --output_dir Approach_A_Feature_Extraction/evaluation_results
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Src.utils.detailed_evaluation import (
    load_class_categories,
    compute_mrr,
    compute_top_k_accuracy,
    categorize_test_samples,
    compute_category_metrics,
    display_evaluation_table,
    display_detailed_breakdown,
    save_detailed_results,
    load_groundtruth,
    save_summary_table
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Approach A with domain breakdown')
    parser.add_argument('--results_dir', type=str, default='Approach_A_Feature_Extraction/results',
                        help='Directory containing trained models')
    parser.add_argument('--features_base', type=str, default='Approach_A_Feature_Extraction/features',
                        help='Base directory for features')
    parser.add_argument('--with_pairs', type=str, default='Dataset/list/class_with_pairs.txt',
                        help='File with class IDs that have both domains')
    parser.add_argument('--without_pairs', type=str, default='Dataset/list/class_without_pairs.txt',
                        help='File with class IDs that have herbarium only')
    parser.add_argument('--groundtruth', type=str, default='Dataset/list/groundtruth.txt',
                        help='Test set ground truth file')
    parser.add_argument('--output_dir', type=str, default='Approach_A_Feature_Extraction/evaluation_results',
                        help='Output directory for results')
    return parser.parse_args()


class LinearProbe(nn.Module):
    """Linear probe model (must match training script)"""
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def load_test_features_and_paths(features_dir, groundtruth):
    """
    Load test features, labels, and image paths.

    Args:
        features_dir: Directory containing feature files
        groundtruth: Dict mapping image names to class IDs

    Returns:
        tuple: (features, labels, image_paths)
    """
    test_features = np.load(os.path.join(features_dir, 'test_features.npy'))
    test_labels = np.load(os.path.join(features_dir, 'test_labels.npy'))

    # Get image names from groundtruth (sorted to match feature order)
    image_paths = sorted(list(groundtruth.keys()))

    return test_features, test_labels, image_paths


def evaluate_sklearn_model_detailed(model_path, X_test, y_test, image_names,
                                    groundtruth, with_pairs, without_pairs,
                                    class_id_to_idx, model_name):
    """
    Evaluate scikit-learn model with domain breakdown.

    Args:
        model_path: Path to trained model
        X_test: Test features
        y_test: Test labels (indices)
        image_names: List of test image names
        groundtruth: Dict mapping image names to class IDs
        with_pairs: Set of class IDs with both domains
        without_pairs: Set of class IDs with herbarium only
        class_id_to_idx: Dict mapping class IDs to indices
        model_name: Model name for display

    Returns:
        dict: Metrics for all three categories
    """
    print(f"\n[*] Evaluating {model_name}...")

    # Load model
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Categorize test samples
    categories = categorize_test_samples(image_names, groundtruth, with_pairs, without_pairs)

    # Compute metrics for each category
    results = {}
    for category_name, indices in categories.items():
        results[category_name] = compute_category_metrics(y_pred_proba, y_test, indices)

    # Display results
    display_detailed_breakdown(model_name, results)

    return results


def evaluate_pytorch_model_detailed(model_path, config_path, X_test, y_test, image_names,
                                    groundtruth, with_pairs, without_pairs,
                                    class_id_to_idx, model_name):
    """
    Evaluate PyTorch linear probe model with domain breakdown.

    Args:
        Similar to evaluate_sklearn_model_detailed

    Returns:
        dict: Metrics for all three categories
    """
    print(f"\n[*] Evaluating {model_name}...")

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

    y_pred_proba = probs.cpu().numpy()

    # Categorize test samples
    categories = categorize_test_samples(image_names, groundtruth, with_pairs, without_pairs)

    # Compute metrics for each category
    results = {}
    for category_name, indices in categories.items():
        results[category_name] = compute_category_metrics(y_pred_proba, y_test, indices)

    # Display results
    display_detailed_breakdown(model_name, results)

    return results


def find_models(results_dir):
    """
    Find all trained models in results directory.

    Returns:
        list: List of dicts with model information
    """
    models = []

    if not os.path.exists(results_dir):
        return models

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
            if len(parts) >= 2:
                classifier_type = parts[0]  # svm, rf, logistic
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
            # For linear_probe models, skip first 2 parts to get feature type
            if len(parts) >= 3 and parts[0] == 'linear' and parts[1] == 'probe':
                feature_type = '_'.join(parts[2:])  # imagenet_base, plant_pretrained_base, etc.
            elif len(parts) >= 2:
                feature_type = '_'.join(parts[1:])
            else:
                continue

            models.append({
                'name': model_dir,
                'type': 'pytorch',
                'classifier': 'Linear Probe',
                'features': feature_type,
                'model_path': pytorch_model,
                'config_path': config_file
            })

    return models


def create_class_id_to_idx_mapping(groundtruth):
    """
    Create mapping from class ID (string) to index (0-99).

    Args:
        groundtruth: Dict mapping image names to class IDs

    Returns:
        dict: Mapping from class ID to index
    """
    unique_class_ids = sorted(set(groundtruth.values()))
    class_id_to_idx = {class_id: idx for idx, class_id in enumerate(unique_class_ids)}
    return class_id_to_idx


def map_labels_to_indices(labels, image_names, groundtruth, class_id_to_idx):
    """
    Map class ID labels to indices for samples where we have groundtruth.

    Args:
        labels: Original labels (may not correspond to correct class IDs)
        image_names: List of image names
        groundtruth: Dict mapping image names to class IDs
        class_id_to_idx: Mapping from class ID to index

    Returns:
        numpy array: Mapped labels as indices
    """
    mapped_labels = []
    for img_name in image_names:
        class_id = groundtruth.get(img_name)
        if class_id:
            idx = class_id_to_idx[class_id]
            mapped_labels.append(idx)
        else:
            mapped_labels.append(-1)  # Unknown

    return np.array(mapped_labels)


def main():
    args = parse_args()

    print("\n" + "=" * 90)
    print("  APPROACH A: DETAILED CLASSIFIER EVALUATION")
    print("=" * 90)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load class categories
    print("\n[*] Loading class categories...")
    with_pairs, without_pairs = load_class_categories(args.with_pairs, args.without_pairs)
    print(f"   [+] With pairs (both domains): {len(with_pairs)} classes")
    print(f"   [+] Without pairs (herbarium only): {len(without_pairs)} classes")

    # Load ground truth
    print(f"\n[*] Loading ground truth: {args.groundtruth}")
    groundtruth = load_groundtruth(args.groundtruth)
    print(f"   [+] Loaded {len(groundtruth)} test labels")

    # Create class ID to index mapping
    class_id_to_idx = create_class_id_to_idx_mapping(groundtruth)
    print(f"   [+] Created mapping for {len(class_id_to_idx)} unique classes")

    # Find all models
    print(f"\n[*] Searching for trained models in: {args.results_dir}")
    models = find_models(args.results_dir)

    if len(models) == 0:
        print("[!] No trained models found!")
        print(f"   Make sure models exist in: {args.results_dir}")
        print("   Expected structure: results/<model_name>/best_model.[joblib|pth]")
        return

    print(f"[+] Found {len(models)} trained models:")
    for model_info in models:
        print(f"   • {model_info['name']} ({model_info['classifier']} on {model_info['features']})")

    # Evaluate each model
    all_results = {}

    for model_info in tqdm(models, desc="Evaluating models"):
        model_name = model_info['name']
        feature_type = model_info['features']

        print(f"\n{'=' * 90}")
        print(f"MODEL: {model_name}")
        print('=' * 90)

        # Load test features for this feature extractor
        features_dir = os.path.join(args.features_base, feature_type)

        if not os.path.exists(features_dir):
            print(f"[!] Features not found: {features_dir}")
            print("   Skipping this model...")
            continue

        try:
            X_test, y_test_original, image_names = load_test_features_and_paths(features_dir, groundtruth)
            print(f"   [+] Loaded test features: {X_test.shape}")
            print(f"   [+] Test samples: {len(image_names)}")

            # Map labels to correct indices based on groundtruth
            y_test = map_labels_to_indices(y_test_original, image_names, groundtruth, class_id_to_idx)

            # Evaluate based on model type
            if model_info['type'] == 'sklearn':
                results = evaluate_sklearn_model_detailed(
                    model_info['model_path'],
                    X_test, y_test, image_names,
                    groundtruth, with_pairs, without_pairs,
                    class_id_to_idx, model_name
                )
            else:  # pytorch
                results = evaluate_pytorch_model_detailed(
                    model_info['model_path'],
                    model_info['config_path'],
                    X_test, y_test, image_names,
                    groundtruth, with_pairs, without_pairs,
                    class_id_to_idx, model_name
                )

            # Store results
            all_results[model_name] = results

            # Save individual model results
            model_output_file = os.path.join(args.output_dir, f"{model_name}_detailed.json")
            with open(model_output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"   [+] Saved detailed results: {model_output_file}")

        except Exception as e:
            print(f"   [!] Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Display summary table
    if all_results:
        print("\n" + "=" * 90)
        display_evaluation_table(all_results, title="APPROACH A - EVALUATION SUMMARY")

        # Save complete results
        detailed_results_file = os.path.join(args.output_dir, 'detailed_results.json')
        save_detailed_results(all_results, detailed_results_file)

        # Save summary table
        summary_table_file = os.path.join(args.output_dir, 'summary_table.txt')
        save_summary_table(all_results, summary_table_file)

        # Print best models
        print("\n[*] TOP PERFORMING MODELS:")
        print("-" * 90)

        # Sort by overall top-1 accuracy
        sorted_models = sorted(all_results.items(),
                             key=lambda x: x[1]['overall']['top1'],
                             reverse=True)

        for i, (model_name, metrics) in enumerate(sorted_models[:5], 1):
            overall = metrics['overall']
            print(f"  {i}. {model_name}")
            print(f"     Top-1: {overall['top1']:.2f}%  |  Top-5: {overall['top5']:.2f}%  |  MRR: {overall['mrr']:.10f}")

        print("=" * 90)

        print(f"\n[+] Evaluation complete!")
        print(f"[*] Results saved in: {args.output_dir}")
        print(f"   • detailed_results.json - Full metrics for all models")
        print(f"   • summary_table.txt - Formatted summary table")
        print(f"   • {len(models)} individual model result files")
    else:
        print("\n[!] No models were successfully evaluated.")
        print("   Check error messages above for details.")


if __name__ == '__main__':
    main()
