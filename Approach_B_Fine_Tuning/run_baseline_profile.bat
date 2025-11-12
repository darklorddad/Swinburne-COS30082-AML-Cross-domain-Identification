@echo off
REM Baseline Profiling Script - Original Settings
REM Run 3 epochs with profiling to establish baseline performance

echo ============================================================
echo BASELINE PROFILING - ImageNet Base DINOv2
echo ============================================================
echo.
echo Settings (ORIGINAL):
echo   - Batch Size: 32 (no gradient accumulation)
echo   - Image Size: 518x518
echo   - Data Workers: 4
echo   - GPU Augmentation: Disabled (CPU torchvision)
echo   - Profiling: Enabled (first warmup stage)
echo   - Epochs: 3 (for baseline measurement)
echo.
echo ============================================================
echo.

python Approach_B_Fine_Tuning/train_unified.py ^
    --model_type imagenet_base ^
    --batch_size 32 ^
    --gradient_accumulation_steps 1 ^
    --image_size 518 ^
    --num_workers 4 ^
    --profile ^
    --epochs 3 ^
    --warmup_epochs 3 ^
    --output_dir Approach_B_Fine_Tuning/Models/imagenet_base_baseline_profile

echo.
echo ============================================================
echo Baseline Profiling Complete!
echo Check: Approach_B_Fine_Tuning/Models/imagenet_base_baseline_profile/profiler_logs
echo ============================================================
pause
