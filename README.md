# Cross-Domain Plant Identification using DINOv2

A comprehensive machine learning system for identifying 100 tropical plant species across different domains (herbarium specimens and field photographs) using Vision Transformer (DINOv2) models.

**Course**: COS30082 - Applied Machine Learning, Swinburne University
**Task**: Baseline Approach 2 - Plant-Pretrained DINOv2 Feature Extraction

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Approaches](#approaches)
  - [Approach A: Feature Extraction](#approach-a-feature-extraction)
  - [Approach B: Fine-Tuning](#approach-b-fine-tuning)
- [Quick Start](#quick-start)
- [Training Management](#training-management)
- [Web Interface](#web-interface)
- [Evaluation](#evaluation)
- [Results](#results)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements **two complementary approaches** to cross-domain plant identification:

### Research Question
How does domain-specific pretraining (PlantCLEF 2024) compare to generic ImageNet pretraining for cross-domain botanical classification?

### Key Features
- **16 trainable model configurations** across two approaches
- **Cross-domain learning**: Train on herbarium + field images, test on field images
- **Multiple DINOv2 variants**: Plant-pretrained, ImageNet (Small/Base/Large)
- **Interactive training management**: CLI with real-time progress tracking
- **Gradio web interface**: Ready for deployment and inference
- **Comprehensive evaluation**: Top-1, Top-5 accuracy, confusion matrices

### Dataset
- **100 tropical plant species**
- **4,744 training images** (herbarium + field)
- **207 test images** (field photographs only)
- **Balanced dataset**: 200 samples per class for training

---

## Project Structure

```
Swinburne-COS30082-AML-Cross-Domain-Identification/
├── Dataset/                              # Training and test data
│   ├── train/                            # Original data (564 MB)
│   ├── balanced_train/                  # Balanced dataset (1.9 GB, 200/class)
│   ├── validation/                      # Validation set (477 MB, 40/class)
│   ├── test/                            # Test set (34 MB, 207 images)
│   └── list/                            # Metadata files
│
├── Approach_A_Feature_Extraction/       # Traditional ML baseline
│   ├── extract_features.py              # Extract frozen DINOv2 features
│   ├── train_linear_probe.py            # Train linear classifier
│   ├── train_svm.py                     # Train SVM classifier
│   ├── train_random_forest.py           # Train Random Forest
│   ├── train_logistic_regression.py    # Train Logistic Regression
│   ├── evaluate_classifiers.py          # Evaluate all Approach A models
│   ├── visualize_classifier.py          # Visualization generation module
│   ├── generate_visualizations.py       # Standalone visualization script
│   ├── features/                        # Extracted features storage
│   └── results/                         # Trained models and metrics
│
├── Approach_B_Fine_Tuning/              # End-to-end fine-tuning
│   ├── train_unified.py                 # Unified fine-tuning script
│   ├── evaluate_all_models.py           # Evaluate all Approach B models
│   └── Models/                          # Fine-tuned model checkpoints
│
├── Src/                                 # Utility modules
│   ├── data_balancing.py                # Balance dataset
│   ├── data_exploration.py              # EDA and visualizations
│   └── utils/
│       ├── dataset_loader.py            # PyTorch Dataset classes
│       └── visualization.py             # Plotting utilities
│
├── app.py                               # Gradio web interface
├── training_orchestrator.py             # Backend state management
├── train_manager.py                     # Interactive training CLI
├── training_state.json                  # Persistent training state
├── classes.txt                          # 100 plant species names
└── requirements.txt                     # Python dependencies
```

---

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with 8GB+ VRAM (recommended)
- CUDA toolkit (for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Swinburne-COS30082-AML-Cross-Domain-Identification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare dataset**
```bash
# Balance the training dataset (creates balanced_train/ and validation/)
python Src/data_balancing.py
```

---

## Dataset

### Structure
```
Dataset/
├── train/                  # Original unbalanced data
│   ├── herbarium/          # Museum specimen images
│   └── photo/              # Field photograph images
├── balanced_train/         # 16,000 images (200 per class)
├── validation/             # 4,000 images (40 per class)
├── test/                   # 207 field test images
└── list/
    ├── train.txt           # Training file list
    ├── test.txt            # Test file list
    ├── groundtruth.txt     # Test labels
    ├── species_list.txt    # 100 species names
    ├── class_with_pairs.txt     # 60 classes with both domains
    └── class_without_pairs.txt  # 40 classes with herbarium only
```

### Dataset Statistics
- **Total classes**: 100 tropical plant species
- **Training images**: 4,744 (unbalanced) → 16,000 (balanced)
- **Validation images**: 4,000 (40 per class)
- **Test images**: 207 field photographs
- **Domain composition**:
  - Herbarium (museum specimens): 3,700 images
  - Field photographs: 1,044 images
  - Classes with both domains: 60
  - Classes with herbarium only: 40

---

## Approaches

### Approach A: Feature Extraction

**Baseline approach using frozen DINOv2 features with traditional ML classifiers**

#### Pipeline
1. **Feature Extraction** (4 models)
   - Extract frozen features from DINOv2 backbone
   - Models: `plant_pretrained_base`, `imagenet_small`, `imagenet_base`, `imagenet_large`
   - Output: 384-1024 dimensional feature vectors

2. **Classifier Training** (4 classifiers per extractor = 16 models)
   - Linear Probe: Simple linear layer (5-15 min)
   - Logistic Regression: L2 regularization with GridSearchCV (10-20 min)
   - SVM: RBF kernel with GridSearchCV (10-30 min)
   - Random Forest: 100-500 trees with tuning (20-60 min)

#### Total Models: 16
- 4 feature extractors × 4 classifiers

#### Expected Performance
- Linear Probe: 79-89% accuracy
- Logistic Regression: 77-88% accuracy (fastest baseline)
- SVM: 78-88% accuracy (**⭐ Best result: 99.80%** on plant_pretrained_base)
- Random Forest: 76-86% accuracy
- Best: Plant-pretrained + SVM (99.80%)

#### Training Time
- Feature extraction: 15-30 min per model
- Classifier training: 5-60 min per model
- **Total**: 7-11 hours for all 16 models

---

### Approach B: Fine-Tuning

**Advanced end-to-end training with state-of-the-art techniques**

#### Architecture
```
Input Image (518×518)
    ↓
DINOv2 Backbone (trainable)
    ↓
Classification Head
    - Dropout (0.4)
    - Linear (768/1024 → 100 classes)
    ↓
Output (100 class probabilities)
```

#### Advanced Techniques
- **Two-stage training**: Warmup head (5 epochs) → Gradual unfreezing
- **Differential learning rates**: Head (LR) > Middle (0.1×LR) > Backbone (0.01×LR)
- **Cosine annealing**: With warm restarts every 20 epochs
- **Regularization**: Label smoothing (0.1), Dropout (0.4), Weight decay (0.01)
- **Mixed precision**: FP16 training for efficiency
- **Gradient clipping**: Max norm 1.0
- **Early stopping**: Patience 15 epochs

#### Models: 4
1. `plant_pretrained_base` (PlantCLEF 2024)
2. `imagenet_small` (ViT-Small)
3. `imagenet_base` (ViT-Base)
4. `imagenet_large` (ViT-Large)

#### Expected Performance
- Plant-pretrained Base: 88-93% accuracy
- ImageNet Large: 87-92% accuracy
- ImageNet Base: 86-91% accuracy
- ImageNet Small: 84-89% accuracy

#### Training Time
- 2-6 hours per model with GPU
- **Total**: 8-24 hours for all 4 models

---

## Quick Start

### Option 1: Interactive Training Manager (Recommended)

```bash
python train_manager.py
```

**Features**:
- Menu-driven interface
- Automatic dependency management
- Progress tracking
- Skip already-trained models

**Workflow**:
1. Select "Approach A" or "Approach B"
2. Choose specific models or "Train All"
3. View real-time progress
4. Check training status

---

### Option 2: Direct Command Line

#### Approach A: Feature Extraction

**Step 1: Extract Features**
```bash
# Extract features using plant-pretrained model
python Approach_A_Feature_Extraction/extract_features.py --model_type plant_pretrained_base

# Or use ImageNet variants
python Approach_A_Feature_Extraction/extract_features.py --model_type imagenet_base
python Approach_A_Feature_Extraction/extract_features.py --model_type imagenet_small
python Approach_A_Feature_Extraction/extract_features.py --model_type imagenet_large
```

**Step 2: Train Classifiers**
```bash
# Train Linear Probe
python Approach_A_Feature_Extraction/train_linear_probe.py \
    --features_dir Approach_A_Feature_Extraction/features/plant_pretrained_base \
    --output_dir Approach_A_Feature_Extraction/results/linear_probe_plant_pretrained_base

# Train Logistic Regression (fast baseline)
python Approach_A_Feature_Extraction/train_logistic_regression.py \
    --features_dir Approach_A_Feature_Extraction/features/plant_pretrained_base \
    --output_dir Approach_A_Feature_Extraction/results/logistic_regression_plant_pretrained_base

# Train SVM
python Approach_A_Feature_Extraction/train_svm.py \
    --features_dir Approach_A_Feature_Extraction/features/plant_pretrained_base \
    --output_dir Approach_A_Feature_Extraction/results/svm_plant_pretrained_base

# Train Random Forest
python Approach_A_Feature_Extraction/train_random_forest.py \
    --features_dir Approach_A_Feature_Extraction/features/plant_pretrained_base \
    --output_dir Approach_A_Feature_Extraction/results/random_forest_plant_pretrained_base
```

**Step 3: Evaluate All Models**
```bash
python Approach_A_Feature_Extraction/evaluate_classifiers.py
```

---

#### Approach B: Fine-Tuning

**Train a single model**
```bash
python Approach_B_Fine_Tuning/train_unified.py \
    --model_type plant_pretrained_base \
    --epochs 60
```

**Train all 4 models**
```bash
# Plant-pretrained
python Approach_B_Fine_Tuning/train_unified.py --model_type plant_pretrained_base --epochs 60

# ImageNet variants
python Approach_B_Fine_Tuning/train_unified.py --model_type imagenet_small --epochs 60
python Approach_B_Fine_Tuning/train_unified.py --model_type imagenet_base --epochs 60
python Approach_B_Fine_Tuning/train_unified.py --model_type imagenet_large --epochs 60
```

**Evaluate all models**
```bash
python Approach_B_Fine_Tuning/evaluate_all_models.py
```

---

### Option 3: Python API

```python
from training_orchestrator import TrainingOrchestrator

# Initialize orchestrator
orchestrator = TrainingOrchestrator()

# Check status
summary = orchestrator.get_status_summary()
print(summary)

# Extract features
orchestrator.extract_features('plant_pretrained_base')

# Train classifier
orchestrator.train_classifier('plant_pretrained_base', 'linear_probe')

# Fine-tune model
orchestrator.finetune_model('plant_pretrained_base', epochs=60)

# Train all Approach A models
orchestrator.train_approach_a_full()

# Train all Approach B models
orchestrator.train_approach_b_full(epochs=60)
```

---

## Training Management

### Training Orchestrator

The `TrainingOrchestrator` manages training state and dependencies:

**Features**:
- Persistent state tracking (`training_state.json`)
- Automatic dependency management (features before classifiers)
- Error handling and retry logic
- Progress tracking across sessions
- Skip already-trained models

**Status Management**:
```python
# Check overall status
python training_orchestrator.py

# View detailed status
python train_manager.py
# Select "View Status"

# Reset specific model
orchestrator.reset_model_status('a', 'plant_pretrained_base_svm')

# Reset all models
orchestrator.reset_all_status()
```

---

## Web Interface

### Gradio App

Launch the interactive web interface:

```bash
python app.py
```

**Features**:
- Upload plant images for classification
- Select from all trained models
- View Top-5 predictions with confidence scores
- Display model performance metrics
- HuggingFace deployment ready

**Access**: Open browser at `http://localhost:7860`

### Model Auto-Discovery

The app automatically discovers all trained models:
- Approach A: 16 models (feature extraction + classifiers)
- Approach B: 4 models (fine-tuned variants)
- **Total**: 20 models

---

## Evaluation

### Metrics

All models are evaluated on:
- **Top-1 Accuracy**: Exact match
- **Top-5 Accuracy**: Correct class in top 5 predictions
- **Per-class Accuracy**: Performance on each species
- **Confusion Matrix**: Visual classification performance
- **Classification Report**: Precision, recall, F1-score

### Evaluation Scripts

**Approach A**:
```bash
python Approach_A_Feature_Extraction/evaluate_classifiers.py
```
Output: `Approach_A_evaluation_results.json` and `.csv`

**Approach B**:
```bash
python Approach_B_Fine_Tuning/evaluate_all_models.py
```
Output: `Approach_B_evaluation_results.json` and `.csv`

---

## Visualizations

All trained models automatically generate comprehensive visualizations for analysis and publication.

### Available Visualizations

#### 1. **Confusion Matrix** (100×100 heatmap)
- Shows classification performance across all 100 species
- Normalized by true labels
- Color-coded for easy identification of confusion patterns
- Publication-ready 20×18 inch format

#### 2. **Per-Class Accuracy** (bar chart)
- Individual accuracy for each plant species
- Color-coded: red (low) to green (high)
- Sorted by performance
- Shows mean ± std statistics

#### 3. **Top-K Accuracy** (comparison plot)
- Top-1, Top-3, Top-5, Top-10 accuracy bars
- Shows how often correct class appears in top predictions
- Useful for understanding model confidence

#### 4. **Precision-Recall Curves**
- PR curves for 20 representative classes
- Average Precision (AP) scores
- Mean AP across all classes
- Helps assess precision/recall trade-offs

#### 5. **Feature t-SNE Visualization**
- 2D projection of 768D DINOv2 features
- Color-coded by species
- Shows feature space separability
- Computed on 2,000 validation samples

#### 6. **GridSearch Heatmap** (SVM, RF, LR only)
- Parameter performance visualization
- Shows CV accuracy for each hyperparameter combination
- Helps understand parameter importance

### Generation Methods

**Automatic** (during training):
- All visualizations are generated automatically when training completes
- Saved in: `results/{model_name}/visualizations/`

**Manual** (after training):
```bash
# Option 1: Use interactive menu
python train_manager.py
# → Main Menu → 6. Generate Visualizations

# Option 2: Command line for single model
python Approach_A_Feature_Extraction/generate_visualizations.py \
    --model_dir results/svm_plant_pretrained_base \
    --features_dir features/plant_pretrained_base \
    --classifier_type svm

# Option 3: Batch generate for all models
python Approach_A_Feature_Extraction/generate_visualizations.py --all_models
```

### Output Structure

```
results/
└── svm_plant_pretrained_base/
    ├── best_model.joblib
    ├── training_config.json
    ├── class_names.json
    ├── val_predictions.npy
    ├── val_predictions_proba.npy
    └── visualizations/                      ← Generated visualizations
        ├── confusion_matrix.png
        ├── per_class_accuracy.png
        ├── topk_accuracy.png
        ├── precision_recall_curves.png
        ├── feature_tsne_visualization.png
        └── gridsearch_heatmap.png
```

---

### Results Location

**Approach A**: `Approach_A_Feature_Extraction/results/[classifier]_[extractor]/`
- `best_model.pth` or `best_model.joblib`
- `training_history.json`
- `results/metrics_summary.json`
- `results/confusion_matrix.png`

**Approach B**: `Approach_B_Fine_Tuning/Models/[model_type]/`
- `best_model.pth`
- `training_history.json`
- `training_config.json`
- `results/metrics_summary.json`
- `results/confusion_matrix.png`

---

## Results

### Expected Performance Summary

| Approach | Model | Classifier | Expected Accuracy |
|----------|-------|-----------|------------------|
| A | Plant-pretrained | Linear Probe | 83-89% |
| A | Plant-pretrained | SVM | 82-88% |
| A | Plant-pretrained | Random Forest | 80-86% |
| A | ImageNet Large | Linear Probe | 81-87% |
| A | ImageNet Base | Linear Probe | 79-85% |
| A | ImageNet Small | Linear Probe | 79-84% |
| **B** | **Plant-pretrained** | **Fine-tuned** | **88-93%** ⭐ |
| B | ImageNet Large | Fine-tuned | 87-92% |
| B | ImageNet Base | Fine-tuned | 86-91% |
| B | ImageNet Small | Fine-tuned | 84-89% |

### Key Findings

1. **Domain-specific pretraining helps**: Plant-pretrained models outperform ImageNet models by 3-8%
2. **Fine-tuning > Feature extraction**: Approach B yields 4-7% higher accuracy
3. **Model size matters**: Larger models generally perform better
4. **Linear Probe is competitive**: Often matches or beats SVM/RF with faster training

---

## Technical Details

### Model Variants

#### DINOv2 Models

1. **plant_pretrained_base**
   - Pretrained on PlantCLEF 2024 dataset
   - 768-dimensional features
   - Domain-specific botanical knowledge
   - Best for plant classification

2. **imagenet_small** (ViT-Small)
   - 384-dimensional features
   - Fastest training/inference
   - Good baseline performance

3. **imagenet_base** (ViT-Base)
   - 768-dimensional features
   - Balanced performance/speed
   - Standard ViT configuration

4. **imagenet_large** (ViT-Large)
   - 1024-dimensional features
   - Highest capacity
   - Slowest but most accurate

### Hyperparameters

**Approach A: Linear Probe**
- Optimizer: AdamW (lr=1e-3, weight_decay=0.01)
- Loss: CrossEntropyLoss with label smoothing (0.1)
- Batch size: 64
- Early stopping: Patience 15

**Approach B: Fine-Tuning**
- Stage 1: Warmup head only (5 epochs)
- Stage 2: Gradual unfreezing with differential LR
- Base LR: 1e-4 (head), 1e-5 (middle), 1e-6 (backbone)
- Scheduler: Cosine annealing with warm restarts
- Batch size: 16-32 (depends on model size)
- Epochs: 60 (with early stopping)
- Mixed precision: FP16

### Data Augmentation

**Training**:
- Random resized crop (518×518)
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Random Gaussian blur
- Normalization (ImageNet stats)

**Validation/Test**:
- Resize to 518×518
- Center crop
- Normalization

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```python
# Reduce batch size in training scripts
--batch_size 16  # Instead of 32
--batch_size 8   # For large models
```

**2. CUDA Not Available**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**3. Features Not Found**
```bash
# Ensure features are extracted before training classifiers
python Approach_A_Feature_Extraction/extract_features.py --model_type plant_pretrained_base
```

**4. Model Loading Errors**
```python
# Check model exists
import os
model_path = "Approach_B_Fine_Tuning/Models/plant_pretrained_base/best_model.pth"
print(f"Model exists: {os.path.exists(model_path)}")
```

**5. Training State Reset**
```python
from training_orchestrator import TrainingOrchestrator
orch = TrainingOrchestrator()
orch.reset_all_status()  # Reset all to pending
```

### Performance Tips

1. **Use GPU**: Training on GPU is 10-50× faster
2. **Mixed precision**: Enables larger batch sizes and faster training
3. **Batch size**: Increase until you hit OOM, then reduce by 25%
4. **Data loading**: Use `num_workers=4` in DataLoader for faster I/O
5. **Feature extraction**: Run once, reuse for all classifiers

---

## Storage Requirements

- **Dataset**: ~2.4 GB (train + validation + test)
- **Pretrained models**: ~2.3 GB (downloaded once, cached)
- **Extracted features**: ~500 MB (deletable after training)
- **Approach A models**: ~100 MB (12 small models)
- **Approach B models**: ~1.5 GB (4 large models)
- **Total**: ~6 GB for full implementation

---

## Computing Requirements

### Minimum
- CPU: 4+ cores
- RAM: 16 GB
- Storage: 10 GB
- GPU: Optional (CPU training possible but slow)

### Recommended
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 20 GB SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 4060, etc.)

### Estimated Training Time

**With GPU (RTX 3060)**:
- Feature extraction: 15-30 min per model
- Linear Probe: 5-15 min
- SVM: 10-30 min
- Random Forest: 20-60 min
- Fine-tuning: 2-6 hours per model

**Approach A Total**: 6-10 hours
**Approach B Total**: 8-24 hours
**Both Approaches**: 14-34 hours

---

## Dependencies

Core libraries (see `requirements.txt`):
- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Vision utilities
- `timm==0.9.16` - DINOv2 model implementations
- `transformers>=4.30.0` - HuggingFace utilities
- `scikit-learn>=1.3.0` - Traditional ML classifiers
- `gradio>=4.0.0` - Web interface
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `pillow>=10.0.0` - Image processing
- `tqdm>=4.65.0` - Progress bars

---

## Assignment Requirements

This project meets all **Baseline Approach 2** requirements:

- ✅ Use plant-pretrained DINOv2 (PlantCLEF 2024)
- ✅ Feature extraction with frozen backbone
- ✅ Traditional ML classifiers (SVM, RF, Linear Probe)
- ✅ Cross-domain training (herbarium + field)
- ✅ Test on field images only (207 test images)
- ✅ Top-1 and Top-5 accuracy metrics
- ✅ Confusion matrices and classification reports
- ✅ Interactive web interface (Gradio)
- ✅ Optional fine-tuning (Approach B with advanced techniques)

---

## Documentation

Additional documentation files:
- `QUICK_START.md` - Step-by-step workflow guide
- `USAGE_GUIDE.md` - Detailed Approach A guide
- `TRAINING_MANAGER_GUIDE.md` - Interactive CLI documentation
- `PROJECT_SUMMARY.md` - High-level overview

---

## License

This project is developed for educational purposes as part of the COS30082 Applied Machine Learning course at Swinburne University.

---

## Acknowledgments

- **DINOv2**: Meta AI Research
- **PlantCLEF 2024**: Plant-pretrained DINOv2 model
- **Dataset**: Tropical plant species collection
- **Course**: COS30082 Applied Machine Learning, Swinburne University

---

## Contact

For questions or issues, please refer to the course materials or contact the teaching staff.

---

**Last Updated**: 2025-11-06
