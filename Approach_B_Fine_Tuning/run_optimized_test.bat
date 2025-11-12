@echo off
REM Optimized Test Run - 5 Epochs with All Optimizations
REM Compare performance against baseline

echo ============================================================
echo OPTIMIZED TEST RUN - ImageNet Base DINOv2
echo ============================================================
echo.
echo Settings (OPTIMIZED):
echo   - Gradient Accumulation: 2 steps (batch=16, effective=32)
echo   - Image Size: 518x518 (original size for compatibility)
echo   - Data Workers: 8 (increased from 4)
echo   - GPU Augmentation: Enabled (kornia)
echo   - Profiling: Enabled
echo   - Epochs: 5 (for speed comparison)
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
    --epochs 5 ^
    --warmup_epochs 5 ^
    --output_dir Approach_B_Fine_Tuning/Models/imagenet_base_optimized_test

echo.
echo ============================================================
echo Optimized Test Complete!
echo Check: Approach_B_Fine_Tuning/Models/imagenet_base_optimized_test/profiler_logs
echo.
echo Compare results:
echo   Baseline: Approach_B_Fine_Tuning/Models/imagenet_base_baseline_profile
echo   Optimized: Approach_B_Fine_Tuning/Models/imagenet_base_optimized_test
echo ============================================================
pause
