# Quick Start - Optimized Training for ImageNet Base

## 🚀 TL;DR - Run Optimized Training Now

```bash
# Windows (double-click or run in terminal)
Approach_B_Fine_Tuning\run_optimized_imagenet_base.bat

# Linux/Mac
python Approach_B_Fine_Tuning/train_unified.py \
    --model_type imagenet_base \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --image_size 448 \
    --num_workers 8 \
    --use_gpu_aug \
    --profile
```

**Expected Results**:
- Training time: ~7-8 hours (35-40% faster than baseline)
- Best validation accuracy: 98.5-99.5% (< 1% drop from baseline)
- GPU utilization: 90%+

---

## 📊 What Was Optimized?

| Optimization | Improvement | Quality Impact |
|-------------|-------------|----------------|
| **Gradient Accumulation** | +5-10% speed | None (same effective batch) |
| **Increased Workers (8)** | +10-15% speed | None |
| **Image Size (448)** | +25-30% speed | <1-2% accuracy |
| **GPU Augmentation** | +10-20% speed | None (same augmentations) |
| **TOTAL** | **35-70% faster** | **<1-2% accuracy drop** |

---

## 🔧 What Changed?

### Before (Baseline)
```bash
--batch_size 32              # Full batch
--gradient_accumulation_steps 1
--image_size 518             # Large images
--num_workers 4              # Conservative
# No GPU augmentation
# No profiling
```

### After (Optimized)
```bash
--batch_size 16              # Smaller batches
--gradient_accumulation_steps 2  # Accumulate to 32
--image_size 448             # Optimized size
--num_workers 8              # More workers
--use_gpu_aug                # GPU augmentation (kornia)
--profile                    # Performance monitoring
```

**Key Point**: Effective batch size remains 32 (16 × 2 = 32), maintaining training stability!

---

## 📝 Three Ways to Run

### 1. Full Training (60 Epochs) - Recommended

```bash
# Windows
Approach_B_Fine_Tuning\run_optimized_imagenet_base.bat

# Linux/Mac
bash Approach_B_Fine_Tuning/run_optimized_imagenet_base.sh
```

**Time**: ~7-8 hours
**Output**: `Approach_B_Fine_Tuning/Models/imagenet_base/`

---

### 2. Quick Test (5 Epochs) - For Validation

```bash
# Windows
Approach_B_Fine_Tuning\run_optimized_test.bat

# Linux/Mac
python Approach_B_Fine_Tuning/train_unified.py \
    --model_type imagenet_base \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --image_size 448 \
    --num_workers 8 \
    --use_gpu_aug \
    --profile \
    --epochs 5 \
    --warmup_epochs 5 \
    --output_dir Approach_B_Fine_Tuning/Models/imagenet_base_optimized_test
```

**Time**: ~35-40 minutes
**Output**: `Approach_B_Fine_Tuning/Models/imagenet_base_optimized_test/`

---

### 3. Baseline Comparison (3 Epochs) - For Benchmarking

```bash
# Windows
Approach_B_Fine_Tuning\run_baseline_profile.bat

# Linux/Mac
python Approach_B_Fine_Tuning/train_unified.py \
    --model_type imagenet_base \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --image_size 518 \
    --num_workers 4 \
    --profile \
    --epochs 3 \
    --warmup_epochs 3 \
    --output_dir Approach_B_Fine_Tuning/Models/imagenet_base_baseline_profile
```

**Time**: ~25-30 minutes
**Output**: `Approach_B_Fine_Tuning/Models/imagenet_base_baseline_profile/`

---

## 🎯 Next Steps

### After Training Completes

1. **Check Results**:
```bash
# View training history
cat Approach_B_Fine_Tuning/Models/imagenet_base/training_config.json

# View best validation accuracy
grep "best_val_accuracy" Approach_B_Fine_Tuning/Models/imagenet_base/training_config.json
```

2. **View Profiler Results**:
```bash
# TensorBoard
tensorboard --logdir Approach_B_Fine_Tuning/Models/imagenet_base/profiler_logs

# Open http://localhost:6006 in browser
```

3. **Compare with Baseline** (if you ran both):
```bash
# Baseline
cat Approach_B_Fine_Tuning/Models/imagenet_base_baseline_profile/training_config.json

# Optimized
cat Approach_B_Fine_Tuning/Models/imagenet_base_optimized_test/training_config.json
```

---

## 🛠️ Customization

### Adjust Workers (Based on CPU)
```bash
--num_workers 12  # If you have 12+ CPU cores
--num_workers 6   # If you have 6-8 CPU cores
```

### Reduce Memory Usage (If OOM)
```bash
--batch_size 12 --gradient_accumulation_steps 3  # Effective = 36
# or
--batch_size 8 --gradient_accumulation_steps 4   # Effective = 32
```

### Disable GPU Augmentation (If Issues)
```bash
# Simply remove --use_gpu_aug flag
python Approach_B_Fine_Tuning/train_unified.py \
    --model_type imagenet_base \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --image_size 448 \
    --num_workers 8 \
    --profile
```

### Adjust Image Size
```bash
--image_size 384  # Even faster, minimal accuracy impact
--image_size 512  # Slower but potentially higher accuracy
```

---

## 📚 More Information

- **Full Documentation**: `Approach_B_Fine_Tuning/OPTIMIZATION_GUIDE.md`
- **Code**: `Approach_B_Fine_Tuning/train_unified.py`
- **GPU Augmentation**: `Src/utils/gpu_augmentation.py`

---

## ❓ Troubleshooting

### "Out of Memory" Error
```bash
# Solution 1: Reduce batch size
--batch_size 8 --gradient_accumulation_steps 4

# Solution 2: Reduce image size
--image_size 384
```

### "kornia not found" Error
```bash
# Install kornia
pip install kornia

# Or disable GPU augmentation
# Remove --use_gpu_aug flag
```

### Training Too Slow
```bash
# Enable profiling to identify bottleneck
--profile

# Check GPU utilization in profiler output
# If <70%, increase --num_workers
# If >90%, already optimal
```

---

## 📈 Expected Output

During training, you should see:

```
======================================================================
🚀 IMAGENET DINOV2 BASE - FINE-TUNING (OPTIMIZED)
======================================================================
Model type: imagenet_base
Output: Approach_B_Fine_Tuning/Models/imagenet_base
Epochs: 60 | Batch size: 16
Gradient Accumulation: 2 steps
Effective Batch Size: 32
Image Size: 448 | Num Workers: 8
Profiling: Enabled
======================================================================

Device: cuda
GPU: NVIDIA GeForce RTX 4050 Laptop GPU

📂 Loading datasets...
   🚀 Using GPU-accelerated augmentation (kornia)
   Train: 16000 | Val: 4000
   Classes: 101

🔧 Loading ImageNet DINOv2 Base...
   ✅ ImageNet-pretrained weights loaded from timm

🔒 Stage 1: Freezing backbone, training head only
   Trainable: 77,908 / 86,736,228 (0.09%)

======================================================================
🏋️  STAGE 1: TRAINING HEAD ONLY
======================================================================
📊 Profiler enabled - will profile first 50 steps

Epoch 1/5: 100%|██████████| 1000/1000 [07:32<00:00, 2.21it/s, Loss=2.3456, Acc=45.32%, LR=1.0e-03]

   Train: 2.3456 | 45.32%
   Val: 2.1234 | 52.18%
   ✅ Best: 52.18%

[... training continues ...]
```

---

## ✅ Success Criteria

Training is successful if:
- ✅ GPU utilization > 85% (check profiler)
- ✅ No "Out of Memory" errors
- ✅ Validation accuracy > 95% by epoch 20-30
- ✅ Training completes in ~7-8 hours
- ✅ Final accuracy > 98%

---

**Ready to start? Run the batch file or Python command above!**

For detailed information, see `OPTIMIZATION_GUIDE.md`
