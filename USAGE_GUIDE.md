# Usage Guide: Approach A - Feature Extraction

This guide shows you how to use the Approach A scripts step-by-step.

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Balance the dataset (creates 16,000 train + 4,000 val images):
```bash
python Src/data_balancing.py
```

3. (Optional) Explore the dataset:
```bash
python Src/data_exploration.py
```

## Step 1: Extract Features

Extract features from all 4 DINOv2 variants:

### ImageNet Small
```bash
python Approach_A_Feature_Extraction/extract_features.py \
    --model_type imagenet_small \
    --train_dir Dataset/balanced_train \
    --val_dir Dataset/validation \
    --test_dir Dataset/test \
    --batch_size 32
```

### ImageNet Base
```bash
python Approach_A_Feature_Extraction/extract_features.py \
    --model_type imagenet_base \
    --train_dir Dataset/balanced_train \
    --val_dir Dataset/validation \
    --test_dir Dataset/test \
    --batch_size 32
```

### ImageNet Large
```bash
python Approach_A_Feature_Extraction/extract_features.py \
    --model_type imagenet_large \
    --train_dir Dataset/balanced_train \
    --val_dir Dataset/validation \
    --test_dir Dataset/test \
    --batch_size 16
```

### Plant-Pretrained Base (after downloading PlantCLEF model)
```bash
# First download the model from Zenodo:
mkdir -p Models/pretrained
# wget https://zenodo.org/records/10848263/files/...
# Extract model_best.pth.tar to Models/pretrained/

python Approach_A_Feature_Extraction/extract_features.py \
    --model_type plant_pretrained_base \
    --plant_model_path Models/pretrained/model_best.pth.tar \
    --train_dir Dataset/balanced_train \
    --val_dir Dataset/validation \
    --test_dir Dataset/test \
    --batch_size 32
```

**Output**: Features saved to `Approach_A_Feature_Extraction/features/<model_type>/`

## Step 2: Train Classifiers

For EACH feature set, train 3 classifiers (SVM, Random Forest, Linear Probe).

### Example: Train all classifiers on ImageNet Base features

#### SVM
```bash
python Approach_A_Feature_Extraction/train_svm.py \
    --features_dir Approach_A_Feature_Extraction/features/imagenet_base \
    --output_dir Approach_A_Feature_Extraction/results/svm_imagenet_base \
    --n_jobs -1
```

#### Random Forest
```bash
python Approach_A_Feature_Extraction/train_random_forest.py \
    --features_dir Approach_A_Feature_Extraction/features/imagenet_base \
    --output_dir Approach_A_Feature_Extraction/results/rf_imagenet_base \
    --n_jobs -1
```

#### Linear Probe
```bash
python Approach_A_Feature_Extraction/train_linear_probe.py \
    --features_dir Approach_A_Feature_Extraction/features/imagenet_base \
    --output_dir Approach_A_Feature_Extraction/results/linear_imagenet_base \
    --epochs 100 \
    --batch_size 128
```

### Repeat for all 4 feature sets:
- `imagenet_small`
- `imagenet_base`
- `imagenet_large`
- `plant_pretrained_base`

**Total models to train**: 4 feature sets × 3 classifiers = **12 models**

## Step 3: Evaluate All Models

After training all 12 models:

```bash
python Approach_A_Feature_Extraction/evaluate_classifiers.py \
    --results_dir Approach_A_Feature_Extraction/results \
    --features_base Approach_A_Feature_Extraction/features \
    --output_file Approach_A_evaluation_results.json \
    --classes_file classes.txt
```

**Output**:
- `Approach_A_evaluation_results.json` - All metrics in JSON
- `Approach_A_evaluation_results.csv` - Comparison table
- Confusion matrices in each model's results folder

## Expected Results Format

The evaluation will produce:

```json
{
  "svm_imagenet_base": {
    "top1_accuracy": 0.8234,
    "top5_accuracy": 0.9567,
    "avg_per_class_accuracy": 0.8102,
    "classifier": "SVM",
    "features": "imagenet_base"
  },
  ...
}
```

And a comparison CSV:

| Model | Classifier | Features | Top-1 Acc | Top-5 Acc | Avg/Class |
|-------|------------|----------|-----------|-----------|-----------|
| svm_plant_pretrained_base | SVM | plant_pretrained_base | 0.8567 | 0.9678 | 0.8423 |
| ... | ... | ... | ... | ... | ... |

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 16` or `--batch_size 8`

### Feature Extraction Takes Too Long
- Use GPU if available
- ImageNet Large is slower due to model size

### SVM/RF Training is Slow
- Use fewer CV folds: `--cv_folds 2`
- Reduce parameter grid in the training scripts
- Use `--n_jobs -1` to use all CPU cores

### Model Files Too Large
- Features are ~500MB per model - delete after training classifiers
- Trained classifiers are small (~10-100MB each)

## Next Steps

After completing Approach A:
1. Implement Approach B (Fine-Tuning) for maximum accuracy
2. Create Gradio app to deploy all models
3. Compare Approach A vs Approach B results
