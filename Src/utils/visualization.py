"""
Visualization Utilities for Training and Evaluation

Provides functions for creating training plots, confusion matrices,
and performance visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
import os


def plot_training_history(history, output_dir, model_name="model"):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history (dict): Training history with keys:
            - train_loss, val_loss
            - train_acc, val_acc
            - epochs (list of epoch numbers)
        output_dir (str): Directory to save plots
        model_name (str): Model name for plot titles
    """
    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{model_name} - Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(history, output_dir, model_name="model"):
    """Plot detailed loss curves"""
    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.title(f'{model_name} - Loss Curves', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_curves(history, output_dir, model_name="model"):
    """Plot detailed accuracy curves"""
    epochs = history['epochs']
    train_acc = history['train_acc']
    val_acc = history['val_acc']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)
    plt.title(f'{model_name} - Accuracy Curves', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_rate(history, output_dir, model_name="model"):
    """Plot learning rate schedule"""
    if 'learning_rates' not in history:
        return

    epochs = history['epochs']
    lrs = history['learning_rates']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, 'g-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Learning Rate', fontsize=13)
    plt.title(f'{model_name} - Learning Rate Schedule', fontsize=15, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_overfitting_analysis(history, output_dir, model_name="model", threshold=10.0):
    """
    Plot overfitting analysis with train-val gap.

    Args:
        history (dict): Training history
        output_dir (str): Output directory
        model_name (str): Model name
        threshold (float): Overfitting threshold (default: 10% gap)
    """
    epochs = history['epochs']
    train_acc = history['train_acc']
    val_acc = history['val_acc']

    # Calculate gap
    gap = [t - v for t, v in zip(train_acc, val_acc)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Train vs Val accuracy
    axes[0].plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    axes[0].plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title(f'{model_name} - Train vs Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Overfitting gap
    axes[1].plot(epochs, gap, 'purple', linewidth=2, label='Train-Val Gap')
    axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold}%)')
    axes[1].fill_between(epochs, 0, gap, where=np.array(gap) > threshold,
                         alpha=0.3, color='red', label='Overfitting Region')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy Gap (%)', fontsize=12)
    axes[1].set_title(f'{model_name} - Overfitting Analysis', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Add text annotation for max gap
    max_gap_idx = np.argmax(gap)
    max_gap = gap[max_gap_idx]
    axes[1].annotate(f'Max Gap: {max_gap:.2f}%',
                     xy=(epochs[max_gap_idx], max_gap),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, model_name="model", normalize=True):
    """
    Plot confusion matrix.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): List of class names
        output_dir (str): Output directory
        model_name (str): Model name
        normalize (bool): Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    # For large number of classes, create a simplified view
    if len(class_names) > 20:
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, cmap='Blues', cbar=True, square=True,
                    xticklabels=False, yticklabels=False, linewidths=0.1)
        plt.ylabel('True Label', fontsize=13)
        plt.xlabel('Predicted Label', fontsize=13)
        title = f'{model_name} - Confusion Matrix'
        if normalize:
            title += ' (Normalized)'
        plt.title(title, fontsize=15, fontweight='bold')
    else:
        plt.figure(figsize=(15, 13))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True, square=True, linewidths=0.5)
        plt.ylabel('True Label', fontsize=13)
        plt.xlabel('Predicted Label', fontsize=13)
        title = f'{model_name} - Confusion Matrix'
        if normalize:
            title += ' (Normalized)'
        plt.title(title, fontsize=15, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_classification_report(y_true, y_pred, class_names, output_file):
    """
    Save classification report to text file.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): List of class names
        output_file (str): Output file path
    """
    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(len(class_names))],
                                  digits=4, zero_division=0)

    with open(output_file, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)


def save_metrics_summary(metrics, output_file):
    """
    Save metrics summary to JSON and text file.

    Args:
        metrics (dict): Dictionary of metrics
        output_file (str): Output file path (without extension)
    """
    # Save as JSON
    with open(f"{output_file}.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save as formatted text
    with open(f"{output_file}.txt", 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("METRICS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")


def create_comparison_plot(results_dict, output_file, metric='top1_acc'):
    """
    Create bar plot comparing multiple models.

    Args:
        results_dict (dict): {model_name: metrics_dict}
        output_file (str): Output file path
        metric (str): Metric to compare ('top1_acc', 'top5_acc', etc.)
    """
    models = list(results_dict.keys())
    values = [results_dict[m].get(metric, 0) for m in models]

    plt.figure(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = plt.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(range(len(models)), models, rotation=45, ha='right', fontsize=9)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def save_training_history(history, output_file):
    """
    Save training history to JSON file.

    Args:
        history (dict): Training history
        output_file (str): Output JSON file path
    """
    with open(output_file, 'w') as f:
        json.dump(history, f, indent=4)


def load_training_history(input_file):
    """
    Load training history from JSON file.

    Args:
        input_file (str): Input JSON file path

    Returns:
        dict: Training history
    """
    with open(input_file, 'r') as f:
        return json.load(f)
