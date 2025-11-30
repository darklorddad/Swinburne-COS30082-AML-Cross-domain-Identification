# Training Manager - Quick Guide

## 🎯 Overview

The Training Manager is an **interactive terminal interface** that makes it easy to train all your models without memorizing commands. It provides:

- ✅ **Simple menu navigation** - No complex commands to remember
- ✅ **Individual model selection** - Train only what you need
- ✅ **Automatic resume** - Skips already-trained models
- ✅ **Progress tracking** - See what's done at a glance
- ✅ **Batch operations** - Train multiple models automatically
- ✅ **State management** - Never lose track of your progress

---

## 🚀 Quick Start

### Launch the Training Manager

```bash
python train_manager.py
```

That's it! The interactive menu will guide you through everything.

---

## 📋 Main Menu

When you launch, you'll see:

```
🌱 PLANT IDENTIFICATION - TRAINING MANAGER
==========================================

📊 TRAINING STATUS
--------------------------------------------------
Approach A - Features:     ✅ 0/4  🔄 0  ❌ 0  ⏳ 4
Approach A - Classifiers:  ✅ 0/12  🔄 0  ❌ 0  ⏳ 12
Approach B - Fine-tuned:   ✅ 0/4  🔄 0  ❌ 0  ⏳ 4
--------------------------------------------------

MAIN MENU:
  1. Approach A - Feature Extraction & Classifiers
  2. Approach B - Fine-Tuning
  3. Run Full Pipeline (All 16 Models)
  4. View Detailed Status
  5. Generate Comparison Report
  6. Reset Model Status
  0. Exit

Select option (0-6):
```

---

## 🎓 Common Workflows

### Workflow 1: Train a Single Classifier (Fastest - 30-60 min)

Perfect for testing or when you only need one model.

1. Launch manager: `python train_manager.py`
2. Select: `1` (Approach A)
3. Select: `2` (Train Classifier)
4. Select feature extractor: e.g., `2` (imagenet_small)
5. Select classifier: e.g., `3` (linear_probe)
6. Wait 30-60 minutes
7. Done! ✅

**Use case**: Quick demo, testing the pipeline, or you only need one specific model.

---

### Workflow 2: Train All Approach A Models (~6-10 hours)

Trains all 12 Approach A models automatically.

1. Launch manager: `python train_manager.py`
2. Select: `1` (Approach A)
3. Select: `3` (Run All Approach A)
4. Confirm: `y`
5. Let it run overnight
6. Done! ✅

**Use case**: Complete Approach A for your assignment (assignment baseline requirement).

---

### Workflow 3: Fine-Tune a Single Model (~2-6 hours)

Train one fine-tuned model for maximum accuracy.

1. Launch manager: `python train_manager.py`
2. Select: `2` (Approach B)
3. Select: `1` (Fine-tune Single Model)
4. Select model: e.g., `1` (plant_pretrained_base)
5. Enter epochs: `60` (or press Enter for default)
6. Wait 2-6 hours
7. Done! ✅

**Use case**: You want the best model for deployment, or comparing fine-tuning vs feature extraction.

---

### Workflow 4: Complete Everything (~14-34 hours)

Train all 16 models in one go.

1. Launch manager: `python train_manager.py`
2. Select: `3` (Run Full Pipeline)
3. Confirm: `y`
4. Go do other things for 1-2 days
5. Done! ✅

**Use case**: Final project completion, comprehensive comparison, maximum coverage.

---

### Workflow 5: Custom Selection

Train exactly what you need.

**Example: Train 2 extractors × 2 classifiers = 4 models**

1. Launch manager: `python train_manager.py`
2. Select: `1` (Approach A)
3. Select: `4` (Custom Batch)
4. Enter extractors: `1,2` (plant_base, imagenet_small)
5. Enter classifiers: `1,3` (svm, linear_probe)
6. Confirm: `y`
7. Wait for completion
8. Done! ✅

**Use case**: Targeted experiments, limited time, specific comparisons.

---

## 💾 State Management & Resume

### How Resume Works

The training manager **automatically tracks** which models are trained in `training_state.json`. If you:

- ❌ Accidentally close the terminal
- 🔌 Lose power
- ⏸️ Need to stop and continue later

**Just relaunch the manager and select the same option!** It will:
- ✅ Skip models already completed
- ✅ Continue from where you left off
- ✅ No data loss

### Example Resume Scenario

```bash
# Day 1 - Start training all Approach A
python train_manager.py
> Select: 1 → 3 (Run All Approach A)
# ... trains 6 models, then you close terminal

# Day 2 - Resume
python train_manager.py
> Select: 1 → 3 (Run All Approach A)
# Automatically skips the 6 completed models
# Continues with remaining 6 models
```

### Check What's Already Trained

```bash
python train_manager.py
> Select: 4 (View Detailed Status)
```

Shows:
- ✅ Completed models (green checkmark)
- 🔄 In-progress (blue circle)
- ❌ Failed (red X)
- ⏳ Pending (hourglass)

---

## 📊 Understanding the Status Bar

```
Approach A - Features:     ✅ 2/4  🔄 1  ❌ 0  ⏳ 1
```

- **✅ 2/4**: 2 out of 4 feature extractors completed
- **🔄 1**: 1 currently in progress
- **❌ 0**: 0 failed
- **⏳ 1**: 1 pending (not started)

---

## 🔧 Advanced Features

### Retrain a Model

If you want to retrain a model that's already completed:

1. Navigate to the specific model
2. When prompted: `Model already trained. Retrain? (y/N):`
3. Type: `y`
4. It will retrain with new settings

### Generate Comparison Reports

After training multiple models:

```bash
python train_manager.py
> Select: 5 (Generate Comparison Report)
> Select: 3 (Both Reports)
```

Creates comparison tables showing accuracy of all trained models.

### Reset Status (Rarely Needed)

If state tracking gets corrupted:

```bash
python train_manager.py
> Select: 6 (Reset Model Status)
> Select: 1 (Reset All Status)
> Confirm: y
```

**Note**: This only resets the tracking state, not the actual trained model files.

---

## ⚙️ Technical Details

### What Happens Behind the Scenes

When you select a training option, the manager:

1. **Checks state**: Reads `training_state.json` to see what's done
2. **Verifies dependencies**: E.g., features must exist before training classifiers
3. **Updates status**: Marks model as "in_progress"
4. **Runs training script**: Calls your existing training scripts with correct arguments
5. **Updates state**: Marks as "completed" or "failed"
6. **Saves progress**: Writes to `training_state.json`

### Files Created

```
training_state.json         # Tracks which models are trained (auto-created)
training_orchestrator.py    # Backend logic (you have this)
train_manager.py           # Interactive menu (you have this)
```

### Integration with Existing Code

The training manager **does not modify your existing scripts**. It:
- ✅ Calls your existing training scripts
- ✅ Uses your existing data and configurations
- ✅ Saves models in the same locations
- ✅ Generates the same outputs

It's just a **convenient wrapper** around your existing workflow.

---

## 🎯 Recommended Approach for Your Assignment

### Minimum Viable Approach (1-2 days)

1. **Train 1 Approach A model** (~1 hour)
   - `train_manager.py` → 1 → 2 → Select any extractor/classifier

2. **Train 1 Approach B model** (~3 hours)
   - `train_manager.py` → 2 → 1 → Select imagenet_small

3. **Generate report** (~5 min)
   - `train_manager.py` → 5 → 3

4. **Launch Gradio app**
   ```bash
   python app.py
   ```

✅ **Result**: Working demo with 2 models, report, and web interface

---

### Full Implementation (3-5 days)

1. **Day 1**: Train all Approach A (~10 hours)
   ```bash
   python train_manager.py → 1 → 3
   ```

2. **Day 2-3**: Train all Approach B (~24 hours)
   ```bash
   python train_manager.py → 2 → 3
   ```

3. **Day 4**: Generate reports and test
   ```bash
   python train_manager.py → 5 → 3
   python app.py
   ```

✅ **Result**: Complete project with all 16 models, comprehensive comparison

---

## 🐛 Troubleshooting

### "CUDA out of memory"

**For Approach A**: Reduce batch size in training scripts

**For Approach B**:
- Use smaller model (imagenet_small instead of large)
- Close other GPU applications
- Reduce batch size in `train_unified.py`

### "Features not found"

The manager will automatically extract features if missing. If it fails:

1. Check Dataset folder exists
2. Ensure balanced dataset is created:
   ```bash
   python Src/data_balancing.py
   ```

### "Training failed"

1. View detailed status: `train_manager.py` → 4
2. Check error message
3. Common issues:
   - Missing dataset
   - CUDA out of memory
   - Package not installed

---

## 💡 Tips & Tricks

### Disk Space Management

As noted in PROJECT_SUMMARY.md, the project needs ~6GB total:
- **1.2GB**: Balanced dataset
- **2.3GB**: PlantCLEF model (downloaded once)
- **~100MB**: Approach A models (all 12!)
- **~1.5GB**: Approach B models (all 4)
- **~650MB**: Features, logs, visualizations

**Already optimized**:
- ✅ Only best checkpoints saved
- ✅ No intermediate checkpoint accumulation
- ✅ Features can be deleted after training (saves 500MB)

### Time Optimization

Train overnight or during class:
- **Approach A**: Start before bed, done by morning
- **Approach B**: Start Friday evening, done by Sunday
- **Full Pipeline**: Start on weekend, done by Tuesday

### GPU Utilization

If you have multiple GPUs or want to optimize usage:
- Train Approach A first (lighter on GPU)
- Then Approach B models one at a time
- Monitor with `nvidia-smi` in another terminal

---

## 📞 Support

If you encounter issues:

1. **Check documentation**:
   - `README.md` - Project overview
   - `QUICK_START.md` - Detailed workflow
   - `PROJECT_SUMMARY.md` - Complete implementation details

2. **View training status**:
   ```bash
   python train_manager.py → 4
   ```

3. **Check training_state.json** for error messages

4. **Run scripts directly** to see full error output:
   ```bash
   python Approach_A_Feature_Extraction/extract_features.py --help
   ```

---

## 🎉 Summary

The Training Manager makes your life easier by:

- 🎮 **Simple interface**: No command memorization
- 🔄 **Auto-resume**: Never lose progress
- 📊 **Progress tracking**: Always know what's done
- ⚡ **Flexible**: Train one model or all 16
- 🛡️ **Safe**: Checks dependencies, handles errors

**Start here**: `python train_manager.py`

**Good luck with your project! 🌱🚀**
