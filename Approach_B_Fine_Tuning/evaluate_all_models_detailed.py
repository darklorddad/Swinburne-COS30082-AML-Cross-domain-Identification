"""
Evaluate Approach B Fine-Tuned Models with Domain-Specific Breakdown

This script evaluates all fine-tuned DINOv2 models on the test set with metrics
broken down by:
1. Overall - All 207 test samples
2. With Pairs - 60 classes with both herbarium + field training images
3. Without Pairs - 40 classes with herbarium-only training images

Metrics computed:
- Top-1 Accuracy
- Top-5 Accuracy
- Mean Reciprocal Rank (MRR)

Usage:
    python Approach_B_Fine_Tuning/evaluate_all_models_detailed.py \
        --models_dir Approach_B_Fine_Tuning/Models \
        --test_dir Dataset/test \
        --with_pairs Dataset/list/class_with_pairs.txt \
        --without_pairs Dataset/list/class_without_pairs.txt \
        --groundtruth Dataset/list/groundtruth.txt \
        --output_dir Approach_B_Fine_Tuning/evaluation_results
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Src.utils.dataset_loader import PlantTestDataset, get_val_transforms
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
    parser = argparse.ArgumentParser(description='Evaluate Approach B with domain breakdown')
    parser.add_argument('--models_dir', type=str, default='Approach_B_Fine_Tuning/Models',
                        help='Directory containing fine-tuned models')
    parser.add_argument('--test_dir', type=str, default='Dataset/test',
                        help='Test images directory')
    parser.add_argument('--with_pairs', type=str, default='Dataset/list/class_with_pairs.txt',
                        help='File with class IDs that have both domains')
    parser.add_argument('--without_pairs', type=str, default='Dataset/list/class_without_pairs.txt',
                        help='File with class IDs that have herbarium only')
    parser.add_argument('--groundtruth', type=str, default='Dataset/list/groundtruth.txt',
                        help='Test set ground truth file')
    parser.add_argument('--output_dir', type=str, default='Approach_B_Fine_Tuning/evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    return parser.parse_args()


def load_model(model_dir, device):
    """
    Load a fine-tuned model.

    Args:
        model_dir: Directory containing model checkpoint and config
        device: torch device

    Returns:
        tuple: (model, config) or (None, None) if failed
    """
    # Load config
    config_path = os.path.join(model_dir, 'training_config.json')
    if not os.path.exists(config_path):
        print(f"   [!] Config not found: {config_path}")
        return None, None

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_type = config.get('model_type', 'unknown')
    model_name = config.get('model_name')
    num_classes = config.get('num_classes', 100)

    if not model_name:
        print(f"   [!] Model name not found in config")
        return None, None

    # Create model
    try:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )
    except Exception as e:
        print(f"   [!] Failed to create model: {e}")
        return None, None

    # Load weights
    model_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"   [!] Model weights not found: {model_path}")
        return None, None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"   [+] Loaded model: {model_type}")
        return model, config
    except Exception as e:
        print(f"   [!] Failed to load weights: {e}")
        return None, None


def evaluate_model_detailed(model, dataloader, groundtruth, with_pairs,
                            without_pairs, class_id_to_idx, device, model_name):
    """
    Evaluate fine-tuned model with domain breakdown.

    Args:
        model: Fine-tuned model
        dataloader: Test data loader
        groundtruth: Dict mapping image names to class IDs
        with_pairs: Set of class IDs with both domains
        without_pairs: Set of class IDs with herbarium only
        class_id_to_idx: Mapping from class ID to index
        device: torch device
        model_name: Model name for display

    Returns:
        dict: Metrics for all three categories
    """
    print(f"\n[*] Evaluating {model_name}...")

    all_probs = []
    all_labels = []
    all_image_names = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            if len(batch) == 2:
                images, paths = batch
                # Create placeholder labels (-1)
                labels = torch.full((images.size(0),), -1, dtype=torch.long)
            else:
                images, labels, paths = batch

            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_image_names.extend([os.path.basename(p) for p in paths])

    # Concatenate results
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Map labels to correct indices based on groundtruth
    correct_labels = []
    for img_name in all_image_names:
        class_id = groundtruth.get(img_name)
        if class_id and class_id in class_id_to_idx:
            idx = class_id_to_idx[class_id]
            correct_labels.append(idx)
        else:
            correct_labels.append(-1)

    correct_labels = np.array(correct_labels)

    # Filter out samples with unknown labels
    valid_mask = correct_labels != -1
    if valid_mask.sum() == 0:
        print("   [!] No valid labels found")
        return None

    all_probs = all_probs[valid_mask]
    correct_labels = correct_labels[valid_mask]
    valid_image_names = [img_name for img_name, valid in zip(all_image_names, valid_mask) if valid]

    print(f"   [+] Valid samples: {len(correct_labels)}")

    # Categorize test samples
    categories = categorize_test_samples(valid_image_names, groundtruth, with_pairs, without_pairs)

    # Compute metrics for each category
    results = {}
    for category_name, indices in categories.items():
        results[category_name] = compute_category_metrics(all_probs, correct_labels, indices)

    # Display results
    display_detailed_breakdown(model_name, results)

    return results


def find_models(models_dir):
    """
    Find all fine-tuned models in models directory.

    Args:
        models_dir: Base models directory

    Returns:
        list: List of dicts with model information
    """
    models = []

    if not os.path.exists(models_dir):
        return models

    for model_dir_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_dir_name)

        if not os.path.isdir(model_path):
            continue

        # Check for model checkpoint and config
        checkpoint_path = os.path.join(model_path, 'best_model.pth')
        config_path = os.path.join(model_path, 'training_config.json')

        if os.path.exists(checkpoint_path) and os.path.exists(config_path):
            models.append({
                'name': model_dir_name,
                'path': model_path
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


def main():
    args = parse_args()

    print("\n" + "=" * 90)
    print("  APPROACH B: DETAILED FINE-TUNED MODEL EVALUATION")
    print("=" * 90)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

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

    # Prepare test dataset
    print(f"\n[*] Loading test dataset: {args.test_dir}")
    transform = get_val_transforms(args.image_size)
    test_dataset = PlantTestDataset(
        args.test_dir,
        transform=transform,
        groundtruth_file=args.groundtruth
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"   [+] Test dataset: {len(test_dataset)} images")

    # Find all models
    print(f"\n[*] Searching for fine-tuned models in: {args.models_dir}")
    models = find_models(args.models_dir)

    if len(models) == 0:
        print("[!] No fine-tuned models found!")
        print(f"   Make sure models exist in: {args.models_dir}")
        print("   Expected structure: Models/<model_name>/best_model.pth")
        return

    print(f"[+] Found {len(models)} fine-tuned models:")
    for model_info in models:
        print(f"   • {model_info['name']}")

    # Evaluate each model
    all_results = {}

    for model_info in models:
        model_name = model_info['name']
        model_path = model_info['path']

        print(f"\n{'=' * 90}")
        print(f"MODEL: {model_name}")
        print('=' * 90)

        try:
            # Load model
            model, config = load_model(model_path, device)

            if model is None:
                print(f"   [!] Failed to load model")
                continue

            # Evaluate
            results = evaluate_model_detailed(
                model, test_loader, groundtruth,
                with_pairs, without_pairs,
                class_id_to_idx, device, model_name
            )

            if results is None:
                print(f"   [!] Evaluation failed")
                continue

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
        display_evaluation_table(all_results, title="APPROACH B - EVALUATION SUMMARY")

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

        for i, (model_name, metrics) in enumerate(sorted_models, 1):
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
