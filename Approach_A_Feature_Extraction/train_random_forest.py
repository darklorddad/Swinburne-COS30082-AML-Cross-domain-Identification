"""
Train Random Forest Classifier on DINOv2 Features

This script trains a Random Forest classifier on pre-extracted DINOv2 features
using GridSearchCV for hyperparameter optimization.

Usage:
    python Approach_A_Feature_Extraction/train_random_forest.py \
        --features_dir Approach_A_Feature_Extraction/features/imagenet_base \
        --output_dir Approach_A_Feature_Extraction/results/rf_imagenet_base \
        --n_jobs -1
"""

import os
import sys
import argparse
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train Random Forest on DINOv2 features')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing extracted features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for trained model')

    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all CPUs, 1 = sequential)')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='Number of cross-validation folds')

    args = parser.parse_args()

    return args


def load_features(features_dir):
    """Load pre-extracted features"""
    print(f"📂 Loading features from {features_dir}")

    train_features = np.load(os.path.join(features_dir, 'train_features.npy'))
    train_labels = np.load(os.path.join(features_dir, 'train_labels.npy'))
    val_features = np.load(os.path.join(features_dir, 'val_features.npy'))
    val_labels = np.load(os.path.join(features_dir, 'val_labels.npy'))

    print(f"   ✅ Training: {train_features.shape}")
    print(f"   ✅ Validation: {val_features.shape}")

    return train_features, train_labels, val_features, val_labels


def train_random_forest(X_train, y_train, n_jobs=-1, cv_folds=3):
    """
    Train Random Forest with GridSearchCV.

    Args:
        X_train: Training features
        y_train: Training labels
        n_jobs: Number of parallel jobs
        cv_folds: Number of CV folds

    Returns:
        GridSearchCV: Fitted grid search object
    """
    print("\n🌲 Setting up Random Forest GridSearch...")

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [20, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    print("   Parameter grid:")
    for key, values in param_grid.items():
        print(f"      {key}: {values}")

    # Create Random Forest classifier
    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=1,  # Set to 1 because GridSearchCV will parallelize
        verbose=0
    )

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=2,
        scoring='accuracy',
        return_train_score=True
    )

    total_combinations = (len(param_grid['n_estimators']) *
                         len(param_grid['max_depth']) *
                         len(param_grid['min_samples_split']) *
                         len(param_grid['min_samples_leaf']) *
                         len(param_grid['max_features']))

    print(f"\n🏋️  Training Random Forest with {cv_folds}-fold cross-validation...")
    print(f"   Total combinations: {total_combinations}")
    print(f"   Using {n_jobs} parallel jobs")

    # Train
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"\n✅ Training complete in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"\n🏆 Best parameters:")
    for key, value in grid_search.best_params_.items():
        print(f"   {key}: {value}")
    print(f"\n   Best CV accuracy: {grid_search.best_score_:.4f}")

    return grid_search, training_time


def evaluate_rf(model, X_val, y_val):
    """Evaluate Random Forest on validation set"""
    print("\n📊 Evaluating on validation set...")

    # Predict
    y_pred = model.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)

    print(f"   ✅ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Feature importance analysis
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print(f"\n   🔍 Feature importance statistics:")
        print(f"      Mean: {importances.mean():.6f}")
        print(f"      Std: {importances.std():.6f}")
        print(f"      Max: {importances.max():.6f}")
        print(f"      Min: {importances.min():.6f}")

    return accuracy, y_pred


def save_model_and_results(grid_search, output_dir, training_time, val_accuracy, y_val, y_pred):
    """Save trained model and results"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 Saving model and results to {output_dir}")

    # Save best model
    model_path = os.path.join(output_dir, 'best_model.joblib')
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"   ✅ Model saved: {model_path}")

    # Save full grid search
    grid_path = os.path.join(output_dir, 'grid_search.joblib')
    joblib.dump(grid_search, grid_path)
    print(f"   ✅ Grid search saved: {grid_path}")

    # Save training configuration
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

    # Add feature importances if available
    if hasattr(grid_search.best_estimator_, 'feature_importances_'):
        config['feature_importances'] = {
            'mean': float(grid_search.best_estimator_.feature_importances_.mean()),
            'std': float(grid_search.best_estimator_.feature_importances_.std()),
            'max': float(grid_search.best_estimator_.feature_importances_.max()),
            'min': float(grid_search.best_estimator_.feature_importances_.min())
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
        f.write("Random Forest Classification Report\n")
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
    print("🌲 RANDOM FOREST TRAINING WITH GRIDSEARCH")
    print("=" * 70)

    # Load features
    X_train, y_train, X_val, y_val = load_features(args.features_dir)

    # Train Random Forest
    grid_search, training_time = train_random_forest(X_train, y_train, args.n_jobs, args.cv_folds)

    # Evaluate on validation set
    val_accuracy, y_pred = evaluate_rf(grid_search.best_estimator_, X_val, y_val)

    # Save model and results
    save_model_and_results(grid_search, args.output_dir, training_time, val_accuracy, y_val, y_pred)

    print("\n" + "=" * 70)
    print("✨ RANDOM FOREST TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🏆 Best CV Score: {grid_search.best_score_:.4f}")
    print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"⏱️  Training Time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print("=" * 70)


if __name__ == '__main__':
    main()
