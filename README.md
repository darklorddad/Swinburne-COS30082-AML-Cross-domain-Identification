# Baseline Approach 2: DINOv2 for Cross-Domain Plant Identification

This repository implements Baseline Approach 2 for cross-domain plant species identification using DINOv2 models. The project compares plant-pretrained vs ImageNet-pretrained DINOv2 models using two methodologies:

1. **Approach A (Feature Extraction)**: DINOv2 as frozen feature extractor + traditional ML classifiers
2. **Approach B (Fine-Tuning)**: Full fine-tuning of DINOv2 models for maximum accuracy

## 📁 Project Structure

 Baseline-Approach_2/
 ├── Src/
 │   ├── data_balancing.py              # Create balanced dataset (200/class, 80/20 split)
 │   ├── data_exploration.py            # Dataset analysis & visualization
 │   └── utils/
 │       ├── dataset_loader.py          # Custom dataset classes
 │       └── visualization.py           # Plotting utilities
 │
 ├── Approach_A_Feature_Extraction/     # Assignment Required Baseline
 │   ├── extract_features.py            # Extract embeddings from all models
 │   ├── train_svm.py                   # SVM + GridSearch
 │   ├── train_random_forest.py         # Random Forest classifier
 │   ├── train_linear_probe.py          # PyTorch linear classifier
 │   ├── evaluate_classifiers.py        # Test all classifiers
 │   ├── features/                      # Saved feature embeddings
 │   │   ├── plant_pretrained_base/
 │   │   ├── imagenet_base/
 │   │   └── [other variants...]
 │   └── results/                       # Models & metrics
 │       ├── svm_plant_pretrained/
 │       │   ├── best_model.joblib
 │       │   ├── metrics_summary.txt
 │       │   └── confusion_matrix.png
 │       ├── rf_plant_pretrained/
 │       └── [other combinations...]
 │
 ├── Approach_B_Fine_Tuning/            # Maximum Accuracy Goal
 │   ├── Models/
 │   │   ├── plant_pretrained_base/
 │   │   │   ├── train.py
 │   │   │   ├── best_model.pth
 │   │   │   ├── training_config.json
 │   │   │   ├── training_history.json
 │   │   │   └── results/
 │   │   │       ├── accuracy_plot.png
 │   │   │       ├── loss_plot.png
 │   │   │       ├── lr_schedule.png
 │   │   │       ├── overfitting_analysis.png
 │   │   │       ├── confusion_matrix.png
 │   │   │       └── metrics_summary.txt
 │   │   ├── imagenet_small/            [same structure]
 │   │   ├── imagenet_base/             [same structure]
 │   │   └── imagenet_large/            [same structure]
 │   └── evaluate_all_models.py         # Compare all fine-tuned models
 │
 ├── Dataset/                           # Prepared datasets
 │   ├── balanced_train/                # 200 samples/class, 80% = 160
 │   ├── validation/                    # 20% = 40 samples/class
 │   └── test/                          # 207 field images (from original)
 │
 ├── app.py                             # Gradio web interface
 ├── classes.txt                        # Species ID → Name mapping
 ├── requirements.txt
 └── README.md

 ---
 🎯 Implementation Phases

 PHASE 1: Dataset Preparation (30 min)

 1.1 Extract Dataset

 - Unzip AML_project_herbarium_dataset.zip
 - Parse list/species_list.txt → classes.txt (100 species)
 - Parse list/train.txt, list/test.txt

 1.2 Data Exploration (Src/data_exploration.py)

 - Visualize herbarium vs field samples
 - Class distribution histogram
 - Domain imbalance analysis
 - Sample images grid (5×5)
 - Save EDA visualizations

 1.3 Balance Dataset (Src/data_balancing.py)

 - Target: 200 samples per class
 - Strategy:
   - If class has >200: random sample 200
   - If class has <200: duplicate randomly to reach 200
 - Split: 160 train / 40 validation (80/20)
 - Preserve domain labels: Track herbarium vs field
 - Create Dataset/balanced_train/ and Dataset/validation/

 ---
 PHASE 2A: Feature Extraction Approach (Assignment Baseline)

 2A.1 Extract Features (extract_features.py)

 Models to extract from (6 total combinations):

 | Pretraining       | Variant | Model Source            |
 |-------------------|---------|-------------------------|
 | Plant (PlantCLEF) | Base    | Zenodo (download 2.3GB) |
 | ImageNet          | Small   | facebook/dinov2-small   |
 | ImageNet          | Base    | facebook/dinov2-base    |
 | ImageNet          | Large   | facebook/dinov2-large   |

 Process:
 1. Load model in eval mode (frozen)
 2. Remove classification head (num_classes=0 for feature extraction)
 3. Process all images: balanced_train (16,000) + validation (4,000) + test (207)
 4. Extract 768-dim (Base/Small) or 1024-dim (Large) embeddings
 5. Save as .npy files with labels

 Output:
 features/
 ├── plant_pretrained_base_train.npy      # (16000, 768)
 ├── plant_pretrained_base_val.npy        # (4000, 768)
 ├── plant_pretrained_base_test.npy       # (207, 768)
 ├── imagenet_small_train.npy
 └── [all combinations...]

 2A.2 Train Traditional ML Classifiers

 For EACH feature set (6 variants × 3 classifiers = 18 models):

 A) SVM with GridSearch (train_svm.py)

 param_grid = {
     'C': [0.1, 1, 10, 100],
     'gamma': ['scale', 'auto', 0.001, 0.01],
     'kernel': ['rbf', 'linear']
 }
 - 3-fold cross-validation
 - Class weights: 'balanced'
 - Save best model as .joblib

 B) Random Forest (train_random_forest.py)

 params = {
     'n_estimators': [100, 200, 500],
     'max_depth': [20, 50, None],
     'min_samples_split': [2, 5, 10]
 }
 - GridSearchCV with 3-fold CV
 - Class weights: 'balanced'

 C) Linear Probe (PyTorch) (train_linear_probe.py)

 - Single linear layer: nn.Linear(feature_dim, 100)
 - AdamW optimizer (lr=1e-3)
 - CrossEntropyLoss with label smoothing
 - 50 epochs max, early stopping (patience=10)

 2A.3 Evaluate All Classifiers (evaluate_classifiers.py)

 - Test on 207 field test images
 - Calculate: Top-1 acc, Top-5 acc, Avg per-class acc
 - Generate confusion matrices
 - Create comparison table (CSV + markdown)

 ---
 PHASE 2B: Fine-Tuning Approach (Maximum Accuracy)

 2B.1 Model Training Scripts

 Create 4 training scripts (one per model variant):

 1. Approach_B_Fine_Tuning/Models/plant_pretrained_base/train.py
 2. Approach_B_Fine_Tuning/Models/imagenet_small/train.py
 3. Approach_B_Fine_Tuning/Models/imagenet_base/train.py
 4. Approach_B_Fine_Tuning/Models/imagenet_large/train.py

 2B.2 Training Configuration (for Maximum Accuracy)

 Architecture Setup:
 # Load pretrained model
 model = timm.create_model(model_name, pretrained=True, num_classes=100)

 # Replace classifier head
 model.head = nn.Linear(model.embed_dim, 100)

 Advanced Fine-Tuning Strategy:

 Stage 1: Head Only (10 epochs)

 - Freeze backbone completely
 - Train only classification head
 - LR: 1e-3
 - Warmup: 2 epochs

 Stage 2: Gradual Unfreezing (30-50 epochs)

 - Unfreeze last 4 transformer blocks
 - Differential LR:
   - Head: 1e-3
   - Last 4 blocks: 1e-4
   - Earlier blocks: frozen
 - Cosine annealing with warm restarts

 Optimizer & Scheduler:
 optimizer = AdamW(
     [
         {'params': head_params, 'lr': 1e-3},
         {'params': backbone_params, 'lr': 1e-4}
     ],
     weight_decay=0.05
 )

 scheduler = CosineAnnealingWarmRestarts(
     optimizer, T_0=10, T_mult=2, eta_min=1e-6
 )

 Data Augmentation:
 train_transform = transforms.Compose([
     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.3),
     transforms.RandomRotation(30),
     transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
     transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
 ])

 Regularization:
 - Dropout: 0.3-0.5 in classifier head
 - Label smoothing: 0.1
 - MixUp: alpha=0.2
 - Gradient clipping: max_norm=1.0
 - Weight decay: 0.05

 Training Loop Features:
 # Professional progress bar
 pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{max_epochs}')
 for images, labels in pbar:
     # Training step...

     # Update progress bar
     pbar.set_postfix({
         'Loss': f'{loss.item():.4f}',
         'Acc': f'{acc:.2f}%',
         'LR': f'{current_lr:.2e}'
     })

 # After each epoch:
 # Epoch 15/100: 100%|████████| 160/160 [02:34<00:00, 1.04it/s]
 # Train Loss: 0.245 | Train Acc: 92.3% | Val Loss: 0.389 | Val Acc: 88.7%

 Overfitting Detection:
 gap = train_acc - val_acc
 if gap > 10.0:
     print(f"⚠️  OVERFITTING WARNING: Gap = {gap:.2f}%")

 Early Stopping:
 - Monitor: validation loss
 - Patience: 15 epochs
 - Save best checkpoint only

 Mixed Precision:
 scaler = torch.cuda.amp.GradScaler()
 with torch.cuda.amp.autocast():
     outputs = model(images)
     loss = criterion(outputs, labels)

 2B.3 Save Training Artifacts

 Per model folder:
 - best_model.pth - Only best checkpoint (state_dict only)
 - training_config.json - All hyperparameters
 - training_history.json - Loss/acc per epoch
 - results/accuracy_plot.png - Train vs val accuracy
 - results/loss_plot.png - Train vs val loss
 - results/lr_schedule.png - Learning rate over time
 - results/overfitting_analysis.png - Gap plot with 10% threshold
 - results/confusion_matrix.png - On validation set
 - results/metrics_summary.txt - Top-1, Top-5, Avg per-class

 ---
 PHASE 3: Comprehensive Evaluation

 3.1 Test All Models on 207 Field Test Images

 Approach A: 18 models (6 feature extractors × 3 classifiers)Approach B: 4 models (4 fine-tuned variants)Total:
  22 models to evaluate

 3.2 Metrics Calculation

 # Top-1 Accuracy
 correct = predictions.argmax(1) == labels
 top1_acc = correct.float().mean()

 # Top-5 Accuracy
 _, top5_pred = predictions.topk(5, dim=1)
 top5_acc = (top5_pred == labels.unsqueeze(1)).any(1).float().mean()

 # Average Per-Class Accuracy
 per_class_acc = []
 for class_id in range(100):
     mask = labels == class_id
     if mask.sum() > 0:
         class_acc = correct[mask].float().mean()
         per_class_acc.append(class_acc)
 avg_per_class = sum(per_class_acc) / len(per_class_acc)

 3.3 Comparison Table

 Generate comprehensive CSV/markdown table:

 | Approach    | Pretraining | Variant | Classifier | Top-1 Acc | Top-5 Acc | Avg/Class | Model Size |
 |-------------|-------------|---------|------------|-----------|-----------|-----------|------------|
 | Feature Ext | Plant       | Base    | SVM        | 85.0%     | 95.2%     | 83.4%     | 12 MB      |
 | Feature Ext | ImageNet    | Base    | Linear     | 82.1%     | 93.8%     | 80.5%     | 8 MB       |
 | Fine-Tuning | Plant       | Base    | -          | 92.3%     | 98.1%     | 90.7%     | 330 MB     |
 | ...         | ...         | ...     | ...        | ...       | ...       | ...       | ...        |

 ---
 PHASE 4: Gradio Web Application

 4.1 App Features (app.py)

 UI Components:

 1. Model Selector Dropdown
   - All 22 trained models
   - Format: "Approach | Pretraining | Variant | Classifier"
   - Example: "Fine-Tuning | Plant | Base | -"
 2. Image Upload
   - Accept: JPG, PNG
   - Display preview
 3. Prediction Output
   - Top 5 predictions with:
       - Rank (1-5)
     - Species name (not just ID)
     - Confidence % with progress bar
     - Class ID
   - Model metrics display:
       - Top-1 Accuracy: XX.X%
     - Avg Per-Class Accuracy: XX.X%
 4. Design
   - Clean, professional CSS (similar to bird example)
   - Dark gradient background
   - Glassmorphism cards
   - Responsive layout

 Class Name Mapping:
 # Load classes.txt (parsed from species_list.txt)
 class_names = {
     0: "Acer campestre",
     1: "Acer platanoides",
     # ... 100 species
 }

 # Display format
 f"#{class_id}: {class_names[class_id]}"

 Model Loading Strategy:
 @lru_cache(maxsize=5)  # Cache last 5 models
 def load_model(model_path, model_type):
     if model_type == 'pytorch':
         model = load_pytorch_model(model_path)
     elif model_type == 'joblib':
         model = joblib.load(model_path)
     return model

 4.2 HuggingFace Deployment

 - Create app.py with minimal dependencies
 - Include requirements.txt
 - Upload best 3-5 models only (space constraint)
 - Add README.md with usage instructions

 ---
 PHASE 5: Space Optimization

 Storage Management:

 | Item                  | Strategy                          | Expected Size |
 |-----------------------|-----------------------------------|---------------|
 | PlantCLEF weights     | Download once, cache              | 2.3 GB        |
 | Feature files (.npy)  | Delete after training classifiers | ~500 MB       |
 | Traditional ML models | Keep all (small)                  | ~100 MB       |
 | Fine-tuned PyTorch    | Best checkpoint only (state_dict) | ~1.2 GB       |
 | Training logs (.json) | Compress or delete old            | ~50 MB        |
 | Visualizations (.png) | Keep all                          | ~100 MB       |
 | Total                 |                                   | ~4 GB         |

 Cleanup Script:
 # After training:
 - Delete intermediate checkpoints
 - Clear CUDA cache: torch.cuda.empty_cache()
 - Compress feature files: np.savez_compressed()
 - Remove duplicate data

 ---
 PHASE 6: Documentation

 6.1 README.md Sections

 1. Project overview
 2. Dataset preparation instructions
 3. How to run Approach A (feature extraction)
 4. How to run Approach B (fine-tuning)
 5. How to evaluate models
 6. How to launch Gradio app
 7. Results summary table
 8. HuggingFace deployment guide
 9. Requirements & setup

 6.2 Requirements.txt

 torch>=2.0.0
 torchvision>=0.15.0
 timm==0.9.16
 transformers>=4.30.0
 gradio>=4.0.0
 scikit-learn>=1.3.0
 joblib>=1.3.0
 tqdm>=4.65.0
 pillow>=10.0.0
 matplotlib>=3.7.0
 seaborn>=0.12.0
 pandas>=2.0.0
 numpy>=1.24.0

 ---
 🎯 Expected Outcomes

 Performance Predictions:

 Approach A (Feature Extraction):
 - Plant-pretrained models: 80-88% top-1 accuracy
 - ImageNet models: 75-82% top-1 accuracy
 - Best classifier: SVM or Linear Probe

 Approach B (Fine-Tuning):
 - Plant-pretrained base: 88-93% top-1 accuracy
 - ImageNet large: 86-91% top-1 accuracy
 - Best overall: Plant-pretrained with fine-tuning

 Key Insight: Plant-pretrained models should outperform ImageNet by 3-8% due to domain specificity.

 ---
 ⏱️ Timeline Estimate

 | Phase | Task                                | Time        |
 |-------|-------------------------------------|-------------|
 | 1     | Dataset prep + exploration          | 1 hour      |
 | 2A    | Feature extraction (6 models)       | 2 hours     |
 | 2A    | Train classifiers (18 models)       | 3 hours     |
 | 2B    | Fine-tune plant-pretrained base     | 4 hours     |
 | 2B    | Fine-tune ImageNet small/base/large | 12 hours    |
 | 3     | Evaluate all 22 models              | 1 hour      |
 | 4     | Gradio app development              | 2 hours     |
 | 5     | Space optimization                  | 0.5 hours   |
 | 6     | Documentation                       | 1 hour      |
 | Total |                                     | ~26.5 hours |

 (Most time is GPU training - can run overnight)

 ---
 ✅ Final Deliverables Checklist

 - Src/data_balancing.py - Dataset balancing
 - Src/data_exploration.py - EDA visualizations
 - Approach_A_Feature_Extraction/ - 18 traditional ML models
 - Approach_B_Fine_Tuning/ - 4 fine-tuned DINOv2 models
 - app.py - Gradio web interface with all 22 models
 - classes.txt - Species names mapping
 - requirements.txt - All dependencies
 - README.md - Comprehensive documentation
 - Comparison table showing plant vs ImageNet performance
 - All training visualizations (accuracy, loss, overfitting)