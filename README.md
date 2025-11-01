# High-Accuracy Plant Identification using DINOv2 Feature Extraction

This project uses a DINOv2 model as a feature extractor for cross-domain plant identification. The extracted features are then used to train traditional machine learning models like SVM and Random Forest.

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have Python 3.8+ and the following libraries installed:

```bash
pip install torch torchvision timm scikit-learn joblib tqdm gradio numpy
```

### 2. Project Structure

```
Swinburne-COS30082-AML-Cross-Domain-Identification/
├── Dataset/
│   ├── train/
│   ├── validation/
│   └── balanced_train/
├── results/
│   ├── DINOv2_FeatureExtractor_RF/
│   └── DINOv2_FeatureExtractor_SVM_GridSearch/
├── Src/
│   ├── data_balancing.py
│   ├── feature_extractor.py
│   ├── extract_all_features.py
│   ├── train_random_forest.py
│   └── train_svm_gridsearch.py
├── app.py
└── README.md
```

### 3. How to Run

Follow these steps to run the project:

**Step 1: Balance the Dataset**

This script creates a balanced training set from the original training data.

```bash
python Src/data_balancing.py --train_path Dataset/train --balanced_train_path Dataset/balanced_train --samples_per_class 100
```

**Step 2: Extract Features**

This script extracts features from the balanced training set and the validation set using the DINOv2 model.

*   **For the training set:**

    ```bash
    python Src/extract_all_features.py --image_dir Dataset/balanced_train --feature_dir Dataset/features/train
    ```

*   **For the validation set:**

    ```bash
    python Src/extract_all_features.py --image_dir Dataset/validation --feature_dir Dataset/features/validation
    ```

**Step 3: Train the Models**

*   **Train the Random Forest model:**

    ```bash
    python Src/train_random_forest.py --feature_dir Dataset/features/train --results_dir results/DINOv2_FeatureExtractor_RF
    ```

*   **Train the SVM model with GridSearchCV:**

    ```bash
    python Src/train_svm_gridsearch.py --feature_dir Dataset/features/train --results_dir results/DINOv2_FeatureExtractor_SVM_GridSearch
    ```

**Step 4: Run the Gradio Web App**

This will launch a web interface where you can upload an image and get predictions.

```bash
python app.py
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.