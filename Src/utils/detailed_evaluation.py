"""
Detailed Evaluation Utilities with Domain-Specific Metrics

Provides shared utilities for computing Top-1, Top-5, and MRR (Mean Reciprocal Rank)
metrics with breakdown by domain categories (overall, with_pairs, without_pairs).

Key Functions:
- load_class_categories: Load class category files
- compute_mrr: Calculate Mean Reciprocal Rank
- compute_top_k_accuracy: Calculate Top-K accuracy
- categorize_test_samples: Group samples by domain category
- compute_category_metrics: Calculate metrics for a specific category
- display_evaluation_table: Show formatted results table
- save_detailed_results: Save results to JSON

Author: Cross-Domain Plant Identification Team
Date: 2025-11-26
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Set
import json
from tabulate import tabulate


def load_class_categories(with_pairs_file: str, without_pairs_file: str) -> Tuple[Set[str], Set[str]]:
    """
    Load class IDs for with_pairs and without_pairs categories.

    Args:
        with_pairs_file: Path to class_with_pairs.txt (classes with both herbarium + field)
        without_pairs_file: Path to class_without_pairs.txt (classes with herbarium only)

    Returns:
        tuple: (with_pairs_set, without_pairs_set) containing class IDs as strings

    Example:
        with_pairs, without_pairs = load_class_categories(
            'Dataset/list/class_with_pairs.txt',
            'Dataset/list/class_without_pairs.txt'
        )
    """
    with_pairs = set()
    without_pairs = set()

    # Load with_pairs
    if os.path.exists(with_pairs_file):
        with open(with_pairs_file, 'r') as f:
            for line in f:
                class_id = line.strip()
                if class_id:  # Skip empty lines
                    with_pairs.add(class_id)

    # Load without_pairs
    if os.path.exists(without_pairs_file):
        with open(without_pairs_file, 'r') as f:
            for line in f:
                class_id = line.strip()
                if class_id:  # Skip empty lines
                    without_pairs.add(class_id)

    return with_pairs, without_pairs


def compute_mrr(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR measures where the correct answer appears in the ranked prediction list:
    - Rank 1 (top prediction): 1/1 = 1.0
    - Rank 2 (second prediction): 1/2 = 0.5
    - Rank 3 (third prediction): 1/3 = 0.333
    ...

    Args:
        predictions: Array of shape (N, num_classes) containing logits or probabilities
        labels: Array of shape (N,) containing true class indices

    Returns:
        float: Mean Reciprocal Rank (0-1), higher is better

    Example:
        >>> predictions = np.array([[0.1, 0.7, 0.2], [0.3, 0.1, 0.6]])
        >>> labels = np.array([1, 2])
        >>> mrr = compute_mrr(predictions, labels)
        >>> print(mrr)  # 1.0 (both correct predictions are rank 1)
    """
    # Convert to numpy if torch tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Sort predictions in descending order (highest score first)
    sorted_indices = np.argsort(predictions, axis=1)[:, ::-1]

    reciprocal_ranks = []

    for i in range(len(labels)):
        true_label = labels[i]

        # Find rank of true label (1-indexed)
        rank_position = np.where(sorted_indices[i] == true_label)[0][0] + 1

        # Reciprocal rank
        reciprocal_ranks.append(1.0 / rank_position)

    mrr = np.mean(reciprocal_ranks)
    return float(mrr)


def compute_top_k_accuracy(predictions: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """
    Compute Top-K accuracy.

    Top-K accuracy is the percentage of samples where the true class appears
    in the top-K predictions.

    Args:
        predictions: Array of shape (N, num_classes) containing logits or probabilities
        labels: Array of shape (N,) containing true class indices
        k: Number of top predictions to consider (default: 5)

    Returns:
        float: Top-K accuracy as percentage (0-100)

    Example:
        >>> predictions = np.array([[0.1, 0.7, 0.2], [0.3, 0.1, 0.6]])
        >>> labels = np.array([1, 0])
        >>> top1 = compute_top_k_accuracy(predictions, labels, k=1)
        >>> top2 = compute_top_k_accuracy(predictions, labels, k=2)
        >>> print(f"Top-1: {top1}%, Top-2: {top2}%")
        Top-1: 50.0%, Top-2: 100.0%
    """
    # Convert to numpy if torch tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get top-k indices
    top_k_indices = np.argsort(predictions, axis=1)[:, -k:]

    # Check if true label is in top-k for each sample
    correct = np.array([labels[i] in top_k_indices[i] for i in range(len(labels))])

    accuracy = correct.mean() * 100
    return float(accuracy)


def categorize_test_samples(image_names: List[str],
                            groundtruth: Dict[str, str],
                            with_pairs: Set[str],
                            without_pairs: Set[str]) -> Dict[str, List[int]]:
    """
    Group test samples by category (overall, with_pairs, without_pairs).

    Args:
        image_names: List of test image filenames
        groundtruth: Dict mapping image names to class IDs
        with_pairs: Set of class IDs with both herbarium + field training
        without_pairs: Set of class IDs with herbarium-only training

    Returns:
        dict: {
            'overall': [0, 1, 2, ...],      # All indices
            'with_pairs': [0, 2, 5, ...],   # Indices for with_pairs classes
            'without_pairs': [1, 3, 4, ...]  # Indices for without_pairs classes
        }

    Example:
        >>> image_names = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        >>> groundtruth = {'img1.jpg': '12518', 'img2.jpg': '105951', 'img3.jpg': '12518'}
        >>> with_pairs = {'12518'}
        >>> without_pairs = {'105951'}
        >>> categories = categorize_test_samples(image_names, groundtruth, with_pairs, without_pairs)
        >>> print(categories)
        {'overall': [0, 1, 2], 'with_pairs': [0, 2], 'without_pairs': [1]}
    """
    categories = {
        'overall': [],
        'with_pairs': [],
        'without_pairs': []
    }

    for idx, img_name in enumerate(image_names):
        # Get class ID for this image
        class_id = groundtruth.get(img_name)

        if class_id is None:
            continue

        # Overall always includes all samples
        categories['overall'].append(idx)

        # Categorize by domain pairing
        if class_id in with_pairs:
            categories['with_pairs'].append(idx)
        elif class_id in without_pairs:
            categories['without_pairs'].append(idx)

    return categories


def compute_category_metrics(predictions: np.ndarray,
                             labels: np.ndarray,
                             indices: List[int]) -> Dict[str, float]:
    """
    Compute Top-1, Top-5, and MRR for a specific category.

    Args:
        predictions: Array of shape (N, num_classes) containing all predictions
        labels: Array of shape (N,) containing all true labels
        indices: List of sample indices belonging to this category

    Returns:
        dict: {
            'N': int,        # Number of samples
            'top1': float,   # Top-1 accuracy (%)
            'top5': float,   # Top-5 accuracy (%)
            'mrr': float     # Mean Reciprocal Rank (0-1)
        }

    Example:
        >>> predictions = np.array([[0.1, 0.7, 0.2], [0.3, 0.1, 0.6], [0.8, 0.1, 0.1]])
        >>> labels = np.array([1, 2, 0])
        >>> indices = [0, 2]  # Only evaluate samples 0 and 2
        >>> metrics = compute_category_metrics(predictions, labels, indices)
        >>> print(metrics)
        {'N': 2, 'top1': 100.0, 'top5': 100.0, 'mrr': 1.0}
    """
    if len(indices) == 0:
        return {
            'N': 0,
            'top1': 0.0,
            'top5': 0.0,
            'mrr': 0.0
        }

    # Filter predictions and labels for this category
    cat_predictions = predictions[indices]
    cat_labels = labels[indices]

    # Compute metrics
    top1_acc = compute_top_k_accuracy(cat_predictions, cat_labels, k=1)
    top5_acc = compute_top_k_accuracy(cat_predictions, cat_labels, k=5)
    mrr = compute_mrr(cat_predictions, cat_labels)

    return {
        'N': len(indices),
        'top1': top1_acc,
        'top5': top5_acc,
        'mrr': mrr
    }


def display_evaluation_table(results_dict: Dict[str, Dict[str, Dict[str, float]]],
                             title: str = "EVALUATION RESULTS") -> None:
    """
    Display formatted evaluation table with Top-1, Top-5, MRR columns.

    Args:
        results_dict: Dictionary with model results
            {
                'model_name': {
                    'overall': {'N': 207, 'top1': 99.5, 'top5': 100.0, 'mrr': 0.995},
                    'with_pairs': {...},
                    'without_pairs': {...}
                },
                ...
            }
        title: Table title (default: "EVALUATION RESULTS")

    Example:
        >>> results = {
        ...     'svm_imagenet_base': {
        ...         'overall': {'N': 207, 'top1': 99.52, 'top5': 100.0, 'mrr': 0.9952}
        ...     }
        ... }
        >>> display_evaluation_table(results)
    """
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90 + "\n")

    # Prepare table data (show overall metrics for each model)
    table_data = []
    for model_name, metrics in results_dict.items():
        overall = metrics.get('overall', {})
        table_data.append([
            model_name,
            overall.get('N', 0),
            f"{overall.get('top1', 0):.2f}%",
            f"{overall.get('top5', 0):.2f}%",
            f"{overall.get('mrr', 0):.10f}"
        ])

    headers = ['Model', 'N', 'Top-1 Acc', 'Top-5 Acc', 'MRR']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print("\n" + "=" * 90)


def display_detailed_breakdown(model_name: str,
                               metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Display detailed breakdown for a single model showing all three categories.

    Args:
        model_name: Name of the model
        metrics: Dictionary with metrics for all categories
            {
                'overall': {'N': 207, 'top1': 99.5, 'top5': 100.0, 'mrr': 0.995},
                'with_pairs': {...},
                'without_pairs': {...}
            }

    Example:
        >>> metrics = {
        ...     'overall': {'N': 207, 'top1': 99.52, 'top5': 100.0, 'mrr': 0.9952},
        ...     'with_pairs': {'N': 153, 'top1': 99.35, 'top5': 100.0, 'mrr': 0.9934},
        ...     'without_pairs': {'N': 54, 'top1': 100.0, 'top5': 100.0, 'mrr': 1.0}
        ... }
        >>> display_detailed_breakdown('svm_imagenet_base', metrics)
    """
    print(f"\n{'=' * 90}")
    print(f"MODEL: {model_name}")
    print('=' * 90)
    print()

    # Prepare table data
    table_data = [
        ['Overall',
         metrics['overall']['N'],
         f"{metrics['overall']['top1']:.2f}%",
         f"{metrics['overall']['top5']:.2f}%",
         f"{metrics['overall']['mrr']:.10f}"],
        ['With Pairs',
         metrics['with_pairs']['N'],
         f"{metrics['with_pairs']['top1']:.2f}%",
         f"{metrics['with_pairs']['top5']:.2f}%",
         f"{metrics['with_pairs']['mrr']:.10f}"],
        ['Without Pairs',
         metrics['without_pairs']['N'],
         f"{metrics['without_pairs']['top1']:.2f}%",
         f"{metrics['without_pairs']['top5']:.2f}%",
         f"{metrics['without_pairs']['mrr']:.10f}"]
    ]

    headers = ['Category', 'N', 'Top-1 Acc', 'Top-5 Acc', 'MRR']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()

    # Cross-domain analysis
    diff = metrics['with_pairs']['top1'] - metrics['without_pairs']['top1']
    print("CROSS-DOMAIN ANALYSIS:")
    if abs(diff) < 1.0:
        print(f"  [+] Excellent generalization (only {abs(diff):.2f}% difference)")
    elif abs(diff) < 5.0:
        print(f"  [*] Good generalization ({abs(diff):.2f}% difference)")
    else:
        print(f"  [!] Notable performance gap ({abs(diff):.2f}% difference)")
    print()


def save_detailed_results(results: Dict[str, Dict[str, Dict[str, float]]],
                         output_path: str) -> None:
    """
    Save detailed evaluation results to JSON file.

    Args:
        results: Dictionary with evaluation results for all models
        output_path: Path to output JSON file

    Example:
        >>> results = {
        ...     'svm_imagenet_base': {
        ...         'overall': {'N': 207, 'top1': 99.52, 'top5': 100.0, 'mrr': 0.9952}
        ...     }
        ... }
        >>> save_detailed_results(results, 'evaluation_results/detailed_results.json')
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"[+] Results saved to: {output_path}")


def load_groundtruth(groundtruth_file: str) -> Dict[str, str]:
    """
    Load test set ground truth labels.

    Args:
        groundtruth_file: Path to groundtruth.txt file

    Returns:
        dict: Mapping from image name to class ID
            {'img1.jpg': '12518', 'img2.jpg': '105951', ...}

    Example:
        >>> gt = load_groundtruth('Dataset/list/groundtruth.txt')
        >>> print(gt['1745.jpg'])
        '105951'
    """
    groundtruth = {}

    with open(groundtruth_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_path, class_id = parts
                img_name = os.path.basename(img_path)
                groundtruth[img_name] = class_id

    return groundtruth


def save_summary_table(results: Dict[str, Dict[str, Dict[str, float]]],
                       output_path: str) -> None:
    """
    Save summary table to text file.

    Args:
        results: Dictionary with evaluation results
        output_path: Path to output text file

    Example:
        >>> results = {...}
        >>> save_summary_table(results, 'evaluation_results/summary_table.txt')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        # Write header
        f.write("=" * 90 + "\n")
        f.write("  EVALUATION RESULTS - SUMMARY TABLE\n")
        f.write("=" * 90 + "\n\n")

        # Prepare table data
        table_data = []
        for model_name, metrics in results.items():
            overall = metrics.get('overall', {})
            table_data.append([
                model_name,
                overall.get('N', 0),
                f"{overall.get('top1', 0):.2f}%",
                f"{overall.get('top5', 0):.2f}%",
                f"{overall.get('mrr', 0):.10f}"
            ])

        headers = ['Model', 'N', 'Top-1 Acc', 'Top-5 Acc', 'MRR']
        table_str = tabulate(table_data, headers=headers, tablefmt='grid')

        f.write(table_str + "\n")
        f.write("\n" + "=" * 90 + "\n")

    print(f"[+] Summary table saved to: {output_path}")
