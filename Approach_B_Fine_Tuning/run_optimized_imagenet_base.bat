@echo off
REM Optimized Training Script for ImageNet Base with ALL Optimizations
REM This script includes:
REM - Gradient accumulation (batch_size=16, accum_steps=2, effective=32)
REM - Increased workers (8 instead of 4)
REM - Reduced image size (448 instead of 518)
REM - GPU-accelerated augmentation (kornia)
REM - Profiling enabled for first warmup stage

echo ============================================================
echo OPTIMIZED TRAINING - ImageNet Base DINOv2
echo ============================================================
echo.
echo Optimizations enabled:
echo   - Gradient Accumulation: 2 steps (effective batch size = 32)
echo   - Image Size: 518x518 (original size for compatibility)
echo   - Data Workers: 8 (increased from 4)
echo   - GPU Augmentation: Enabled (kornia)
echo   - Profiling: Enabled (first 5 epochs)
echo.
echo ============================================================
echo.

python Approach_B_Fine_Tuning/train_unified.py ^
    --model_type imagenet_base ^
    --batch_size 16 ^
    --gradient_accumulation_steps 2 ^
    --image_size 518 ^
    --num_workers 8 ^
    --use_gpu_aug ^
    --profile ^
    --epochs 60 ^
    --patience 15

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
pause
