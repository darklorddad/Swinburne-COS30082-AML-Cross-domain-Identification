import os
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

def train_rf(feature_dir, results_dir, n_estimators=100, random_state=42):
    """
    Trains a Random Forest classifier and saves the results.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    X, y, class_names = load_features(feature_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Train model
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=np.unique(y_test).astype(str))

    # Save results
    with open(os.path.join(results_dir, "report.txt"), "w") as f:
        f.write(f"Random Forest Classifier Results\n")
        f.write(f"N-Estimators: {n_estimators}\n")
        f.write(f"Random State: {random_state}\n\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save model
    model_path = os.path.join(results_dir, "random_forest_model.joblib")
    joblib.dump(rf, model_path)

    print(f"Results and model saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest classifier.")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory of features to use for training.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save the results and model.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    args = parser.parse_args()

    train_rf(args.feature_dir, args.results_dir, args.n_estimators, args.random_state)
