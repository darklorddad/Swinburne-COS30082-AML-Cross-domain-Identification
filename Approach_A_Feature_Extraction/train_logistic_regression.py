"""
Train Logistic Regression Classifier with GridSearch on DINOv2 Features

This script trains a Logistic Regression classifier on pre-extracted DINOv2 features
using GridSearchCV for hyperparameter optimization.

Usage:
    python Approach_A_Feature_Extraction/train_logistic_regression.py \
        --features_dir Approach_A_Feature_Extraction/features/imagenet_base \
        --output_dir Approach_A_Feature_Extraction/results/logistic_regression_imagenet_base \
        --n_jobs -1
"""

import os
import sys
import argparse
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train Logistic Regression on DINOv2 features')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing extracted features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for trained model')

    # Windows with Python 3.13 has multiprocessing issues, use n_jobs=1 by default
    default_n_jobs = 1 if sys.platform == 'win32' else -1
    parser.add_argument('--n_jobs', type=int, default=default_n_jobs,
                        help='Number of parallel jobs for GridSearch (-1 = all CPUs, 1 = sequential)')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='Number of cross-validation folds')

    args = parser.parse_args()

    # Warn if using parallel processing on Windows
    if sys.platform == 'win32' and args.n_jobs != 1:
        print(f"\n⚠️  WARNING: n_jobs={args.n_jobs} may cause errors on Windows with Python 3.13")
        print(f"   Recommended: n_jobs=1 for stability")
        print(f"   Will attempt parallel processing but may fallback to sequential if it fails...\n")

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


def train_logistic_regression(X_train, y_train, n_jobs=-1, cv_folds=3):
    """
    Train Logistic Regression with GridSearchCV for hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        n_jobs: Number of parallel jobs
        cv_folds: Number of CV folds

    Returns:
        GridSearchCV: Fitted grid search object
    """
    print("\n🔧 Setting up Logistic Regression GridSearch...")

    # Define parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],  # L2 only for lbfgs solver
        'solver': ['lbfgs'],  # Fast and efficient
        'max_iter': [1000]  # Enough for convergence
    }

    print("   Parameter grid:")
    for key, values in param_grid.items():
        print(f"      {key}: {values}")

    # Create Logistic Regression classifier
    lr = LogisticRegression(
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        verbose=0,
        n_jobs=1  # Set to 1 because GridSearchCV will parallelize
    )

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=2,
        scoring='accuracy',
        return_train_score=True
    )

    print(f"\n🏋️  Training Logistic Regression with {cv_folds}-fold cross-validation...")
    print(f"   Total combinations: {len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['solver']) * len(param_grid['max_iter'])}")
    print(f"   Using {n_jobs} parallel jobs")

    # Train with automatic fallback for Windows multiprocessing issues
    start_time = time.time()
    try:
        # Try with requested parallel processing
        grid_search.fit(X_train, y_train)
    except (ModuleNotFoundError, OSError, RuntimeError) as e:
        # If multiprocessing fails on Windows, fallback to sequential
        if 'posixsubprocess' in str(e) or 'multiprocessing' in str(e) or '_posix' in str(e):
            print(f"\n❌ Multiprocessing failed: {str(e)[:80]}...")
            print(f"⚙️  Falling back to sequential processing (n_jobs=1)...\n")
            # Recreate GridSearch with n_jobs=1
            grid_search = GridSearchCV(
                estimator=lr,
                param_grid=param_grid,
                cv=cv_folds,
                n_jobs=1,  # Sequential - works on all platforms
                verbose=2,
                scoring='accuracy',
                return_train_score=True
            )
            grid_search.fit(X_train, y_train)
        else:
            raise  # Re-raise if it's a different error
    training_time = time.time() - start_time

    print(f"\n✅ Training complete in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"\n🏆 Best parameters:")
    for key, value in grid_search.best_params_.items():
        print(f"   {key}: {value}")
    print(f"\n   Best CV accuracy: {grid_search.best_score_:.4f}")

    return grid_search, training_time


def evaluate_logistic_regression(model, X_val, y_val):
    """Evaluate Logistic Regression on validation set"""
    print("\n📊 Evaluating on validation set...")

    # Predict
    y_pred = model.predict(X_val)

    # Predict probabilities (for Top-K accuracy and other visualizations)
    y_pred_proba = model.predict_proba(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)

    print(f"   ✅ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Coefficient analysis
    if hasattr(model, 'coef_'):
        coef = model.coef_
        print(f"\n   🔍 Coefficient statistics:")
        print(f"      Shape: {coef.shape}")
        print(f"      Mean abs: {np.abs(coef).mean():.6f}")
        print(f"      Std: {coef.std():.6f}")
        print(f"      Max: {coef.max():.6f}")
        print(f"      Min: {coef.min():.6f}")

    return accuracy, y_pred, y_pred_proba


def save_model_and_results(grid_search, output_dir, training_time, val_accuracy, y_val, y_pred, y_pred_proba, features_dir):
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

    # Add coefficient statistics if available
    if hasattr(grid_search.best_estimator_, 'coef_'):
        coef = grid_search.best_estimator_.coef_
        config['coefficient_stats'] = {
            'shape': list(coef.shape),
            'mean_abs': float(np.abs(coef).mean()),
            'std': float(coef.std()),
            'max': float(coef.max()),
            'min': float(coef.min())
        }

    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"   ✅ Config saved: {config_path}")

    # Get class names from dataset
    num_classes = len(np.unique(y_val))
    class_names = [f"Class_{i}" for i in range(num_classes)]

    # Try to load actual class names from dataset
    try:
        dataset_dir = os.path.join(os.path.dirname(features_dir), '..', 'Dataset', 'balanced_train')
        if os.path.exists(dataset_dir):
            actual_class_names = sorted([d for d in os.listdir(dataset_dir)
                                        if os.path.isdir(os.path.join(dataset_dir, d))])
            if len(actual_class_names) == num_classes:
                class_names = actual_class_names
    except:
        pass  # Use default Class_0, Class_1, etc.

    # Save class names
    class_names_path = os.path.join(output_dir, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"   ✅ Class names saved: {class_names_path}")

    # Generate classification report
    report = classification_report(
        y_val,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Logistic Regression Classification Report\n")
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

    # Save prediction probabilities
    proba_path = os.path.join(output_dir, 'val_predictions_proba.npy')
    np.save(proba_path, y_pred_proba)
    print(f"   ✅ Prediction probabilities saved: {proba_path}")

    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    try:
        from visualize_classifier import generate_all_visualizations
        generate_all_visualizations(output_dir, features_dir, classifier_type='logistic_regression')
    except Exception as e:
        print(f"   ⚠️  Warning: Could not generate visualizations: {e}")


def main():
    args = parse_args()

    print("=" * 70)
    print("📈 LOGISTIC REGRESSION TRAINING WITH GRIDSEARCH")
    print("=" * 70)

    # Load features
    X_train, y_train, X_val, y_val = load_features(args.features_dir)

    # Train Logistic Regression
    grid_search, training_time = train_logistic_regression(X_train, y_train, args.n_jobs, args.cv_folds)

    # Evaluate on validation set
    val_accuracy, y_pred, y_pred_proba = evaluate_logistic_regression(grid_search.best_estimator_, X_val, y_val)

    # Save model and results
    save_model_and_results(grid_search, args.output_dir, training_time, val_accuracy, y_val, y_pred, y_pred_proba, args.features_dir)

    print("\n" + "=" * 70)
    print("✨ LOGISTIC REGRESSION TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🏆 Best CV Score: {grid_search.best_score_:.4f}")
    print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"⏱️  Training Time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print("=" * 70)


if __name__ == '__main__':
    main()
