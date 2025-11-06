"""
Train SVM Classifier with GridSearch on DINOv2 Features

This script trains a Support Vector Machine classifier on pre-extracted DINOv2 features
using GridSearchCV for hyperparameter optimization.

Usage:
    python Approach_A_Feature_Extraction/train_svm.py \
        --features_dir Approach_A_Feature_Extraction/features/imagenet_base \
        --output_dir Approach_A_Feature_Extraction/results/svm_imagenet_base \
        --n_jobs -1
"""

import os
import sys
import argparse
import numpy as np
import joblib
import json
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train SVM on DINOv2 features')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing extracted features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for trained model')

    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs for GridSearch (-1 = all CPUs, 1 = sequential)')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='Number of cross-validation folds')

    args = parser.parse_args()

    return args


def load_features(features_dir):
    """Load pre-extracted features"""
    print(f"📂 Loading features from {features_dir}")

    # Load training features
    train_features = np.load(os.path.join(features_dir, 'train_features.npy'))
    train_labels = np.load(os.path.join(features_dir, 'train_labels.npy'))

    print(f"   ✅ Training features: {train_features.shape}")
    print(f"   ✅ Training labels: {train_labels.shape}")

    # Load validation features
    val_features = np.load(os.path.join(features_dir, 'val_features.npy'))
    val_labels = np.load(os.path.join(features_dir, 'val_labels.npy'))

    print(f"   ✅ Validation features: {val_features.shape}")
    print(f"   ✅ Validation labels: {val_labels.shape}")

    return train_features, train_labels, val_features, val_labels


def train_svm(X_train, y_train, n_jobs=-1, cv_folds=3):
    """
    Train SVM with GridSearchCV for hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        n_jobs: Number of parallel jobs
        cv_folds: Number of CV folds

    Returns:
        GridSearchCV: Fitted grid search object
    """
    print("\n🔧 Setting up SVM GridSearch...")

    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'linear']
    }

    print("   Parameter grid:")
    for key, values in param_grid.items():
        print(f"      {key}: {values}")

    # Create SVM classifier
    svm = SVC(
        class_weight='balanced',  # Handle class imbalance
        probability=True,          # Enable probability estimates for Top-5
        random_state=42,
        verbose=False
    )

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=2,
        scoring='accuracy',
        return_train_score=True
    )

    print(f"\n🏋️  Training SVM with {cv_folds}-fold cross-validation...")
    print(f"   Total combinations: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])}")
    print(f"   Using {n_jobs} parallel jobs")

    # Train
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"\n✅ Training complete in {training_time:.2f} seconds")
    print(f"\n🏆 Best parameters:")
    for key, value in grid_search.best_params_.items():
        print(f"   {key}: {value}")
    print(f"\n   Best CV accuracy: {grid_search.best_score_:.4f}")

    return grid_search, training_time


def evaluate_svm(model, X_val, y_val):
    """Evaluate SVM on validation set"""
    print("\n📊 Evaluating on validation set...")

    # Predict
    y_pred = model.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)

    print(f"   ✅ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return accuracy, y_pred


def save_model_and_results(grid_search, output_dir, training_time, val_accuracy, y_val, y_pred):
    """Save trained model and results"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 Saving model and results to {output_dir}")

    # Save best model
    model_path = os.path.join(output_dir, 'best_model.joblib')
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"   ✅ Model saved: {model_path}")

    # Save full grid search results
    grid_path = os.path.join(output_dir, 'grid_search.joblib')
    joblib.dump(grid_search, grid_path)
    print(f"   ✅ Grid search saved: {grid_path}")

    # Save training configuration and results
    config = {
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'val_accuracy': float(val_accuracy),
        'training_time_seconds': float(training_time),
        'cv_results': {
            'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
            'params': [str(p) for p in grid_search.cv_results_['params']]
        }
    }

    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"   ✅ Config saved: {config_path}")

    # Generate classification report
    num_classes = len(np.unique(y_val))
    report = classification_report(
        y_val,
        y_pred,
        target_names=[f"Class_{i}" for i in range(num_classes)],
        digits=4,
        zero_division=0
    )

    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("SVM Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Best CV Score: {grid_search.best_score_:.4f}\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)

    print(f"   ✅ Classification report saved: {report_path}")

    # Save predictions
    predictions_path = os.path.join(output_dir, 'val_predictions.npy')
    np.save(predictions_path, y_pred)
    print(f"   ✅ Predictions saved: {predictions_path}")


def main():
    args = parse_args()

    print("=" * 70)
    print("🤖 SVM TRAINING WITH GRIDSEARCH")
    print("=" * 70)

    # Load features
    X_train, y_train, X_val, y_val = load_features(args.features_dir)

    # Train SVM
    grid_search, training_time = train_svm(X_train, y_train, args.n_jobs, args.cv_folds)

    # Evaluate on validation set
    val_accuracy, y_pred = evaluate_svm(grid_search.best_estimator_, X_val, y_val)

    # Save model and results
    save_model_and_results(grid_search, args.output_dir, training_time, val_accuracy, y_val, y_pred)

    print("\n" + "=" * 70)
    print("✨ SVM TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🏆 Best CV Score: {grid_search.best_score_:.4f}")
    print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"⏱️  Training Time: {training_time:.2f} seconds")
    print("=" * 70)


if __name__ == '__main__':
    main()
