#!/bin/bash

# Quick Demo Script
# This script runs a minimal workflow to get you started quickly
# Estimated time: 2-3 hours (mostly training)

echo "=========================================="
echo "🌱 Quick Demo: Baseline Approach 2"
echo "=========================================="
echo ""

# Step 1: Balance dataset
echo "📊 Step 1/3: Balancing dataset..."
echo "This will create 16,000 train + 4,000 val images"
echo ""
python Src/data_balancing.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Dataset balancing failed"
    exit 1
fi

echo ""
echo "✅ Dataset balanced successfully!"
echo ""

# Step 2: Train one model
echo "🚀 Step 2/3: Training ImageNet Base model..."
echo "This will take 2-3 hours with GPU (longer with CPU)"
echo "Training with 30 epochs for quick demo"
echo ""
python Approach_B_Fine_Tuning/train_unified.py \
    --model_type imagenet_base \
    --epochs 30 \
    --batch_size 32

if [ $? -ne 0 ]; then
    echo "❌ Error: Model training failed"
    exit 1
fi

echo ""
echo "✅ Model trained successfully!"
echo ""

# Step 3: Launch Gradio app
echo "🌐 Step 3/3: Launching Gradio app..."
echo "The app will open in your browser"
echo "Press Ctrl+C to stop the app"
echo ""
python app.py

echo ""
echo "=========================================="
echo "✨ Demo Complete!"
echo "=========================================="
