"""
Comprehensive Visualization Module for Approach A Classifiers

This module provides visualization functions for SVM, Random Forest, and Linear Probe
classifiers trained on DINOv2 features.

Visualizations include:
- Confusion Matrix
- Per-Class Accuracy
- Top-K Accuracy
- Precision-Recall Curves
- t-SNE/UMAP Feature Visualization
- GridSearch Parameter Analysis

Author: Auto-generated visualization module
Date: 2025-11-06
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str],
                          output_path: str,
                          title: str = "Confusion Matrix"):
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
        title: Plot title
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 18))

    # Plot heatmap
    sns.heatmap(cm_normalized,
                annot=False,  # Too many classes to annotate
                fmt='.2f',
                cmap='Blues',
                cbar_kws={'label': 'Normalized Count'},
                square=True,
                ax=ax)

    # Set labels
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Set tick labels (show every 10th class for readability)
    tick_positions = list(range(0, len(class_names), 10))
    tick_labels = [class_names[i] if i < len(class_names) else '' for i in tick_positions]

    ax.set_xticks([i + 0.5 for i in tick_positions])
    ax.set_yticks([i + 0.5 for i in tick_positions])
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   Confusion matrix saved: {output_path}")


def plot_per_class_accuracy(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: List[str],
                            output_path: str,
                            title: str = "Per-Class Accuracy"):
    """
    Plot per-class accuracy as a bar chart.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
        title: Plot title
    """
    num_classes = len(class_names)
    per_class_acc = []

    # Calculate accuracy for each class
    for i in range(num_classes):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)

    # Sort by accuracy
    sorted_indices = np.argsort(per_class_acc)
    sorted_acc = [per_class_acc[i] for i in sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Create color map (red for low, green for high)
    colors = plt.cm.RdYlGn([acc for acc in sorted_acc])

    # Plot horizontal bar chart
    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_acc, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{acc:.2%}',
                ha='left', va='center', fontsize=6, fontweight='bold')

    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=6)
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.05)

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add summary statistics
    mean_acc = np.mean(per_class_acc)
    std_acc = np.std(per_class_acc)
    ax.axvline(mean_acc, color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {mean_acc:.2%} ± {std_acc:.2%}')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   Per-class accuracy saved: {output_path}")


def plot_topk_accuracy(y_true: np.ndarray,
                      y_pred_proba: np.ndarray,
                      class_names: List[str],
                      output_path: str,
                      title: str = "Top-K Accuracy",
                      k_values: List[int] = [1, 3, 5, 10]):
    """
    Plot Top-K accuracy for different K values.

    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities (n_samples, n_classes)
        class_names: List of class names
        output_path: Path to save the plot
        title: Plot title
        k_values: List of K values to compute
    """
    topk_accuracies = []

    # Calculate Top-K accuracy for each K
    for k in k_values:
        # Get top-k predictions
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]

        # Check if true label is in top-k
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        accuracy = correct.mean()
        topk_accuracies.append(accuracy)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(k_values)))
    bars = ax.bar(range(len(k_values)), topk_accuracies, color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, topk_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Set labels
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f'Top-{k}' for k in k_values], fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.05)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   Top-K accuracy saved: {output_path}")

    return dict(zip([f'top_{k}' for k in k_values], topk_accuracies))


def plot_precision_recall_curves(y_true: np.ndarray,
                                 y_pred_proba: np.ndarray,
                                 class_names: List[str],
                                 output_path: str,
                                 title: str = "Precision-Recall Curves",
                                 max_classes_to_plot: int = 20):
    """
    Plot precision-recall curves for selected classes.

    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        class_names: List of class names
        output_path: Path to save the plot
        title: Plot title
        max_classes_to_plot: Maximum number of classes to plot
    """
    num_classes = len(class_names)

    # Select classes to plot (evenly spaced)
    if num_classes > max_classes_to_plot:
        step = num_classes // max_classes_to_plot
        classes_to_plot = list(range(0, num_classes, step))[:max_classes_to_plot]
    else:
        classes_to_plot = list(range(num_classes))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot PR curves for selected classes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes_to_plot)))

    avg_precisions = []
    for i, class_idx in enumerate(classes_to_plot):
        # One-vs-rest
        y_true_binary = (y_true == class_idx).astype(int)
        y_scores = y_pred_proba[:, class_idx]

        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
        avg_precision = average_precision_score(y_true_binary, y_scores)
        avg_precisions.append(avg_precision)

        # Plot
        label = f'{class_names[class_idx][:20]} (AP={avg_precision:.2f})'
        ax.plot(recall, precision, color=colors[i], lw=1.5, alpha=0.7, label=label)

    # Set labels
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\nMean AP: {np.mean(avg_precisions):.3f}',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   Precision-Recall curves saved: {output_path}")


def plot_feature_visualization(features: np.ndarray,
                               labels: np.ndarray,
                               class_names: List[str],
                               output_path: str,
                               title: str = "Feature Visualization (t-SNE)",
                               method: str = 'tsne',
                               max_samples: int = 2000):
    """
    Plot 2D visualization of features using t-SNE or UMAP.

    Args:
        features: Feature vectors (n_samples, n_features)
        labels: True labels
        class_names: List of class names
        output_path: Path to save the plot
        title: Plot title
        method: 'tsne' or 'umap'
        max_samples: Maximum samples to plot (for speed)
    """
    # Subsample if too many samples
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    print(f"   Computing {method.upper()} projection...")

    # Compute 2D projection
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    elif method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            print("   UMAP not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    else:
        raise ValueError(f"Unknown method: {method}")

    features_2d = reducer.fit_transform(features)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot each class
    num_classes = len(class_names)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.sum() > 0:
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[colors[class_idx]],
                      label=class_names[class_idx] if class_idx % 10 == 0 else '',
                      alpha=0.6, s=20, edgecolors='none')

    # Set labels
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n({max_samples} samples)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=8, ncol=3, markerscale=2)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   Feature visualization saved: {output_path}")


def plot_gridsearch_heatmap(grid_search_results: Dict,
                            output_path: str,
                            title: str = "GridSearch Parameter Analysis"):
    """
    Plot heatmap of GridSearch results.

    Args:
        grid_search_results: Results from GridSearchCV
        output_path: Path to save the plot
        title: Plot title
    """
    # Extract params and scores
    params = grid_search_results['params']
    scores = grid_search_results['mean_test_scores']

    # Get parameter names
    param_names = list(params[0].keys()) if params else []

    if len(param_names) < 2:
        print("   Skipping GridSearch heatmap (need at least 2 parameters)")
        return

    # Create pivot table for first two parameters
    param1_name = param_names[0]
    param2_name = param_names[1]

    # Extract unique values
    param1_values = sorted(set(str(p[param1_name]) for p in params))
    param2_values = sorted(set(str(p[param2_name]) for p in params))

    # Create score matrix
    score_matrix = np.zeros((len(param2_values), len(param1_values)))
    for i, p2 in enumerate(param2_values):
        for j, p1 in enumerate(param1_values):
            # Find matching score
            for idx, p in enumerate(params):
                if str(p[param1_name]) == p1 and str(p[param2_name]) == p2:
                    score_matrix[i, j] = scores[idx]
                    break

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap
    im = ax.imshow(score_matrix, cmap='YlGnBu', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(param1_values)))
    ax.set_yticks(np.arange(len(param2_values)))
    ax.set_xticklabels(param1_values, rotation=45, ha='right')
    ax.set_yticklabels(param2_values)

    # Labels
    ax.set_xlabel(param1_name, fontsize=12, fontweight='bold')
    ax.set_ylabel(param2_name, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CV Accuracy', fontsize=11, fontweight='bold')

    # Add text annotations
    for i in range(len(param2_values)):
        for j in range(len(param1_values)):
            text = ax.text(j, i, f'{score_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   GridSearch heatmap saved: {output_path}")


def generate_all_visualizations(model_dir: str,
                                features_dir: Optional[str] = None,
                                classifier_type: str = 'svm') -> Dict:
    """
    Generate all visualizations for a trained classifier.

    Args:
        model_dir: Directory containing model and results
        features_dir: Directory containing features (for t-SNE visualization)
        classifier_type: Type of classifier ('svm', 'random_forest', 'linear_probe')

    Returns:
        Dictionary with paths to generated visualizations
    """
    print(f"\n Generating visualizations for {model_dir}...")

    # Create visualization directory
    viz_dir = os.path.join(model_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Load predictions and labels
    val_predictions = np.load(os.path.join(model_dir, 'val_predictions.npy'))

    # Load class names if available
    class_names_path = os.path.join(model_dir, 'class_names.json')
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    else:
        # Infer number of classes
        num_classes = len(np.unique(val_predictions))
        class_names = [f'Class_{i}' for i in range(num_classes)]

    # Load true labels from features directory if available
    if features_dir and os.path.exists(features_dir):
        val_labels = np.load(os.path.join(features_dir, 'val_labels.npy'))
    else:
        print("   Warning: Cannot find val_labels.npy, some visualizations will be skipped")
        return {}

    viz_paths = {}

    # 1. Confusion Matrix
    cm_path = os.path.join(viz_dir, 'confusion_matrix.png')
    plot_confusion_matrix(val_labels, val_predictions, class_names, cm_path,
                         title=f'Confusion Matrix - {classifier_type.upper()}')
    viz_paths['confusion_matrix'] = cm_path

    # 2. Per-Class Accuracy
    per_class_path = os.path.join(viz_dir, 'per_class_accuracy.png')
    plot_per_class_accuracy(val_labels, val_predictions, class_names, per_class_path,
                           title=f'Per-Class Accuracy - {classifier_type.upper()}')
    viz_paths['per_class_accuracy'] = per_class_path

    # 3. Top-K Accuracy (if probabilities available)
    proba_path = os.path.join(model_dir, 'val_predictions_proba.npy')
    if os.path.exists(proba_path):
        val_proba = np.load(proba_path)
        topk_path = os.path.join(viz_dir, 'topk_accuracy.png')
        topk_metrics = plot_topk_accuracy(val_labels, val_proba, class_names, topk_path,
                                         title=f'Top-K Accuracy - {classifier_type.upper()}')
        viz_paths['topk_accuracy'] = topk_path

        # 4. Precision-Recall Curves
        pr_path = os.path.join(viz_dir, 'precision_recall_curves.png')
        plot_precision_recall_curves(val_labels, val_proba, class_names, pr_path,
                                    title=f'Precision-Recall Curves - {classifier_type.upper()}')
        viz_paths['precision_recall'] = pr_path
    else:
        print("   Skipping Top-K and PR curves (no probability predictions found)")

    # 5. Feature Visualization (if features available)
    if features_dir and os.path.exists(features_dir):
        val_features_path = os.path.join(features_dir, 'val_features.npy')
        if os.path.exists(val_features_path):
            val_features = np.load(val_features_path)
            tsne_path = os.path.join(viz_dir, 'feature_tsne_visualization.png')
            plot_feature_visualization(val_features, val_labels, class_names, tsne_path,
                                      title=f'Feature Visualization (t-SNE) - {classifier_type.upper()}',
                                      method='tsne')
            viz_paths['feature_tsne'] = tsne_path

    # 6. GridSearch Heatmap (for SVM and Random Forest)
    if classifier_type in ['svm', 'random_forest']:
        config_path = os.path.join(model_dir, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

            if 'cv_results' in config:
                heatmap_path = os.path.join(viz_dir, 'gridsearch_heatmap.png')
                plot_gridsearch_heatmap(config['cv_results'], heatmap_path,
                                       title=f'GridSearch Results - {classifier_type.upper()}')
                viz_paths['gridsearch_heatmap'] = heatmap_path

    print(f"\n All visualizations generated in {viz_dir}")
    return viz_paths
