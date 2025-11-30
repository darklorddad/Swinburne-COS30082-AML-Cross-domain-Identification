# Quick Start Guide: Complete Workflow

This guide shows the complete workflow from dataset preparation to model deployment.

## 📋 Prerequisites

```bash
pip install -r requirements.txt
```

## 🎯 Complete Workflow

### Step 1: Prepare Dataset (Required)

```bash
# Balance the dataset to 200 samples per class
python Src/data_balancing.py
```

**Output**:
- `Dataset/balanced_train/` (16,000 images)
- `Dataset/validation/` (4,000 images)

### Step 2: Optional - Explore Data

```bash
python Src/data_exploration.py
```

**Output**: Visualizations in `EDA_Results/`

---

## 🔬 Approach A: Feature Extraction (Assignment Baseline)

### A1. Extract Features

Extract features from 4 DINOv2 variants.

**Important:** DINOv2 uses **518×518 images** (default). Adjust batch size based on your GPU memory.

```bash
# ImageNet Small (batch_size 16 recommended)
python Approach_A_Feature_Extraction/extract_features.py --model_type imagenet_small --batch_size 16

# ImageNet Base (batch_size 16 recommended)
python Approach_A_Feature_Extraction/extract_features.py --model_type imagenet_base --batch_size 16

# ImageNet Large (batch_size 8 recommended, use 4 if OOM)
python Approach_A_Feature_Extraction/extract_features.py --model_type imagenet_large --batch_size 8

# Plant-pretrained Base (after downloading PlantCLEF model)
python Approach_A_Feature_Extraction/extract_features.py \
    --model_type plant_pretrained_base \
    --plant_model_path Models/pretrained/model_best.pth.tar \
    --batch_size 16
```

**If CUDA Out of Memory:** Reduce batch size (16→8→4→2)

### A2. Train Classifiers

For EACH feature set, train 3 classifiers. Example with `imagenet_base`:

```bash
# SVM
python Approach_A_Feature_Extraction/train_svm.py \
    --features_dir Approach_A_Feature_Extraction/features/imagenet_base \
    --output_dir Approach_A_Feature_Extraction/results/svm_imagenet_base

# Random Forest
python Approach_A_Feature_Extraction/train_random_forest.py \
    --features_dir Approach_A_Feature_Extraction/features/imagenet_base \
    --output_dir Approach_A_Feature_Extraction/results/rf_imagenet_base

# Linear Probe
python Approach_A_Feature_Extraction/train_linear_probe.py \
    --features_dir Approach_A_Feature_Extraction/features/imagenet_base \
    --output_dir Approach_A_Feature_Extraction/results/linear_imagenet_base
```

Repeat for all 4 feature sets: `imagenet_small`, `imagenet_base`, `imagenet_large`, `plant_pretrained_base`

**Total**: 4 × 3 = 12 models

### A3. Evaluate All Approach A Models

```bash
python Approach_A_Feature_Extraction/evaluate_classifiers.py
```

**Output**:
- `Approach_A_evaluation_results.json`
- `Approach_A_evaluation_results.csv`
- Confusion matrices in each model's results folder

---

## 🚀 Approach B: Fine-Tuning (Maximum Accuracy)

Train 4 fine-tuned models using the unified script:

### B1. Plant-Pretrained Base

```bash
python Approach_B_Fine_Tuning/train_unified.py \
    --model_type plant_pretrained_base \
    --plant_model_path Models/pretrained/model_best.pth.tar \
    --epochs 60 \
    --batch_size 32
```

### B2. ImageNet Small

```bash
python Approach_B_Fine_Tuning/train_unified.py \
    --model_type imagenet_small \
    --epochs 60 \
    --batch_size 32
```

### B3. ImageNet Base

```bash
python Approach_B_Fine_Tuning/train_unified.py \
    --model_type imagenet_base \
    --epochs 60 \
    --batch_size 32
```

### B4. ImageNet Large

```bash
python Approach_B_Fine_Tuning/train_unified.py \
    --model_type imagenet_large \
    --epochs 60 \
    --batch_size 16
```

**Training Time**: 2-6 hours per model (GPU recommended)

### B5. Evaluate All Approach B Models

```bash
python Approach_B_Fine_Tuning/evaluate_all_models.py
```

**Output**:
- `Approach_B_evaluation_results.json`
- `Approach_B_evaluation_results.csv`
- Confusion matrices

---

## 🌐 Web Application

Launch the Gradio interface to test all trained models:

```bash
python app.py
```

The app will:
- Auto-detect all trained models from both approaches
- Provide a dropdown to select any model
- Show Top-5 predictions with species names
- Display model performance metrics

**Features**:
- Upload plant images for classification
- Get real-time predictions
- View confidence scores
- See model information

---

## 📊 Expected Results

### Approach A (Feature Extraction):

| Pretraining | Classifier | Expected Top-1 Acc |
|-------------|------------|-------------------|
| Plant | SVM | 82-88% |
| Plant | Random Forest | 80-86% |
| Plant | Linear Probe | 83-89% |
| ImageNet | SVM | 78-84% |
| ImageNet | Random Forest | 76-82% |
| ImageNet | Linear Probe | 79-85% |

### Approach B (Fine-Tuning):

| Model | Expected Top-1 Acc |
|-------|-------------------|
| Plant-pretrained Base | 88-93% |
| ImageNet Small | 84-89% |
| ImageNet Base | 86-91% |
| ImageNet Large | 87-92% |

**Key Insight**: Plant-pretrained models consistently outperform ImageNet-pretrained by 3-8% due to domain specificity.

---

## 🎓 Training Tips

### Memory Management:
- **CUDA OOM**: Reduce `--batch_size` (try 16, then 8)
- **CPU Training**: Use `--num_workers 0` to avoid multiprocessing issues

### Speed Up Training:
- Use GPU if available
- Reduce `--epochs` for quick experiments (try 20-30)
- Use fewer CV folds for Approach A: `--cv_folds 2`

### Best Practices:
1. Start with ImageNet Base (good balance of speed and accuracy)
2. Train Approach A first (faster, baseline results)
3. Then train Approach B for maximum accuracy
4. Compare results between approaches

---

## 📁 Project Structure After Training

```
Baseline-Approach_2/
├── Dataset/
│   ├── balanced_train/        ✅ 16,000 images
│   ├── validation/            ✅ 4,000 images
│   └── test/                  ✅ 207 images
│
├── Approach_A_Feature_Extraction/
│   ├── features/              ✅ Extracted features (.npy)
│   ├── results/               ✅ 12 trained models
│   └── *.py                   ✅ Training scripts
│
├── Approach_B_Fine_Tuning/
│   ├── Models/
│   │   ├── plant_pretrained_base/  ✅ best_model.pth + results
│   │   ├── imagenet_small/         ✅ best_model.pth + results
│   │   ├── imagenet_base/          ✅ best_model.pth + results
│   │   └── imagenet_large/         ✅ best_model.pth + results
│   └── *.py                        ✅ Training scripts
│
├── app.py                     ✅ Gradio web app
├── requirements.txt           ✅
└── README.md                  ✅
```

---

## 🚨 Troubleshooting

### "Input height (224) doesn't match model (518)" Error
- **Solution**: DINOv2 uses 518×518 images (now the default)
- The script has been updated to use `--image_size 518` by default
- You don't need to specify it manually anymore

### CUDA Out of Memory during feature extraction
- **Solution**: Reduce batch size progressively
  ```bash
  # Try these in order until it works:
  --batch_size 16  # First try
  --batch_size 8   # If still fails
  --batch_size 4   # If still fails
  --batch_size 2   # Last resort
  ```
- 518×518 images use **~5.5x more memory** than 224×224

### "No trained models found" in app.py
- Train at least one model first using Approach A or B scripts
- Check that model files exist in the expected directories

### Feature extraction is slow
- Use GPU: Models will automatically use CUDA if available
- Expected time: 5-10 min with GPU, 30-60 min with CPU per model

### SVM/RF training takes forever
- Use `--n_jobs -1` to parallelize across all CPU cores
- Reduce `--cv_folds` from 3 to 2
- Consider using only Linear Probe for faster experiments

### Fine-tuning crashes with CUDA OOM
- Reduce `--batch_size` to 16 or 8
- Use smaller model: `imagenet_small` instead of `large`
- Reduce `--image_size` to 224 (not recommended, but works)

---

## 🎯 Minimum Viable Workflow

If you're short on time, here's the minimum to get working results:

1. **Prepare data**: `python Src/data_balancing.py`

2. **Train ONE model** (Approach B ImageNet Base - good accuracy, reasonable speed):
   ```bash
   python Approach_B_Fine_Tuning/train_unified.py --model_type imagenet_base --epochs 30
   ```

3. **Launch app**:
   ```bash
   python app.py
   ```

This gives you a working demo in ~2-3 hours!

---

## 🏆 For Maximum Accuracy

1. Complete Approach A (12 models)
2. Complete Approach B (4 models)
3. Compare all 16 models
4. Use best performer for deployment

**Best model typically**: Plant-pretrained Base (Approach B)

---

## 📞 Need Help?

See `README.md` for detailed documentation or `USAGE_GUIDE.md` for Approach A specifics.
