
import os
import numpy as np
import argparse
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

def load_features(feature_dir):
    """
    Loads features and labels from a directory.
    """
    features = []
    labels = []
    class_names = sorted(os.listdir(feature_dir))
    for class_name in tqdm(class_names, desc="Loading features"):
        class_path = os.path.join(feature_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for feature_file in os.listdir(class_path):
            feature_path = os.path.join(class_path, feature_file)
            feature = np.load(feature_path)
            features.append(feature.flatten())
            labels.append(class_name)
    return np.array(features), np.array(labels), class_names

def train_svm_gridsearch(feature_dir, results_dir, random_state=42):
    """
    Trains an SVM classifier with GridSearchCV and saves the results.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    X, y, class_names = load_features(feature_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    # Train model with GridSearchCV
    print("Training SVM with GridSearchCV...")
    svc = SVC(probability=True, random_state=random_state)
    grid_search = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best model
    best_svm = grid_search.best_estimator_

    # Evaluate model
    print("Evaluating model...")
    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=np.unique(y_test).astype(str))

    # Save results
    with open(os.path.join(results_dir, "report.txt"), "w") as f:
        f.write(f"SVM Classifier with GridSearchCV Results\n")
        f.write(f"Random State: {random_state}\n\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:")
        f.write(report)

    # Save model
    model_path = os.path.join(results_dir, "svm_gridsearch_model.joblib")
    joblib.dump(best_svm, model_path)

    print(f"Results and model saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an SVM classifier with GridSearchCV.")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory of features to use for training.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save the results and model.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    args = parser.parse_args()

    train_svm_gridsearch(args.feature_dir, args.results_dir, args.random_state)
