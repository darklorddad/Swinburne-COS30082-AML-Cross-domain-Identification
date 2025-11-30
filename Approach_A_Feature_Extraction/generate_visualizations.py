"""
Generate Visualizations for Trained Classifiers

This standalone script generates all visualizations for an already-trained model.
Useful for creating visualizations after training is complete.

Usage:
    # For a single model
    python Approach_A_Feature_Extraction/generate_visualizations.py \
        --model_dir Approach_A_Feature_Extraction/results/svm_plant_pretrained_base \
        --features_dir Approach_A_Feature_Extraction/features/plant_pretrained_base \
        --classifier_type svm

    # For all models in results directory
    python Approach_A_Feature_Extraction/generate_visualizations.py \
        --all_models

Author: Auto-generated visualization script
Date: 2025-11-06
"""

import os
import sys
import argparse
from pathlib import Path
from visualize_classifier import generate_all_visualizations


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for trained classifiers'
    )

    # Option 1: Single model
    parser.add_argument('--model_dir', type=str,
                       help='Directory containing trained model')
    parser.add_argument('--features_dir', type=str,
                       help='Directory containing extracted features')
    parser.add_argument('--classifier_type', type=str,
                       choices=['svm', 'random_forest', 'linear_probe', 'logistic_regression'],
                       help='Type of classifier')

    # Option 2: All models
    parser.add_argument('--all_models', action='store_true',
                       help='Generate visualizations for all trained models')

    parser.add_argument('--results_dir', type=str,
                       default='Approach_A_Feature_Extraction/results',
                       help='Results directory (when using --all_models)')
    parser.add_argument('--features_base_dir', type=str,
                       default='Approach_A_Feature_Extraction/features',
                       help='Features base directory (when using --all_models)')

    return parser.parse_args()


def infer_classifier_type(model_dir_name: str) -> str:
    """Infer classifier type from directory name"""
    if 'svm' in model_dir_name.lower():
        return 'svm'
    elif 'random_forest' in model_dir_name.lower() or 'rf' in model_dir_name.lower():
        return 'random_forest'
    elif 'linear' in model_dir_name.lower() or 'probe' in model_dir_name.lower():
        return 'linear_probe'
    elif 'logistic' in model_dir_name.lower() or 'lr' in model_dir_name.lower():
        return 'logistic_regression'
    else:
        return 'svm'  # Default


def infer_feature_extractor(model_dir_name: str) -> str:
    """Infer feature extractor from directory name"""
    extractors = ['plant_pretrained_base', 'imagenet_small', 'imagenet_base', 'imagenet_large']
    for extractor in extractors:
        if extractor in model_dir_name:
            return extractor
    return None


def generate_for_single_model(model_dir: str, features_dir: str, classifier_type: str):
    """Generate visualizations for a single model"""
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    print(f"Model directory: {model_dir}")
    print(f"Features directory: {features_dir}")
    print(f"Classifier type: {classifier_type}")
    print(f"{'='*70}\n")

    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory not found: {model_dir}")
        return False

    if not os.path.exists(features_dir):
        print(f"⚠️  Warning: Features directory not found: {features_dir}")
        print(f"   Some visualizations may be skipped")

    try:
        viz_paths = generate_all_visualizations(model_dir, features_dir, classifier_type)

        print(f"\n{'='*70}")
        print(f"✅ VISUALIZATIONS COMPLETE!")
        print(f"{'='*70}")
        print(f"Generated {len(viz_paths)} visualizations:")
        for viz_name, viz_path in viz_paths.items():
            print(f"   ✓ {viz_name}: {viz_path}")
        print(f"{'='*70}\n")

        return True

    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_for_all_models(results_dir: str, features_base_dir: str):
    """Generate visualizations for all trained models"""
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS FOR ALL MODELS")
    print(f"{'='*70}")
    print(f"Results directory: {results_dir}")
    print(f"Features base directory: {features_base_dir}")
    print(f"{'='*70}\n")

    if not os.path.exists(results_dir):
        print(f"❌ Error: Results directory not found: {results_dir}")
        return

    # Find all model directories
    model_dirs = [d for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d))]

    if not model_dirs:
        print(f"No model directories found in {results_dir}")
        return

    print(f"Found {len(model_dirs)} model directories:\n")

    successful = 0
    failed = 0

    for i, model_dir_name in enumerate(model_dirs, 1):
        print(f"\n[{i}/{len(model_dirs)}] Processing {model_dir_name}...")

        # Infer classifier type and feature extractor
        classifier_type = infer_classifier_type(model_dir_name)
        feature_extractor = infer_feature_extractor(model_dir_name)

        if feature_extractor is None:
            print(f"   ⚠️  Could not infer feature extractor from {model_dir_name}, skipping...")
            failed += 1
            continue

        model_dir = os.path.join(results_dir, model_dir_name)
        features_dir = os.path.join(features_base_dir, feature_extractor)

        # Check if model has required files
        required_files = ['val_predictions.npy']
        missing_files = [f for f in required_files
                        if not os.path.exists(os.path.join(model_dir, f))]

        if missing_files:
            print(f"   ⚠️  Missing required files: {missing_files}, skipping...")
            failed += 1
            continue

        try:
            viz_paths = generate_all_visualizations(model_dir, features_dir, classifier_type)
            print(f"   ✅ Generated {len(viz_paths)} visualizations")
            successful += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"✅ BATCH VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(model_dirs)}")
    print(f"{'='*70}\n")


def main():
    args = parse_args()

    # Validate arguments
    if args.all_models:
        # Batch mode
        generate_for_all_models(args.results_dir, args.features_base_dir)
    else:
        # Single model mode
        if not args.model_dir or not args.features_dir or not args.classifier_type:
            print("❌ Error: When not using --all_models, you must specify:")
            print("   --model_dir, --features_dir, and --classifier_type")
            print("\nExample:")
            print("   python generate_visualizations.py \\")
            print("       --model_dir results/svm_plant_pretrained_base \\")
            print("       --features_dir features/plant_pretrained_base \\")
            print("       --classifier_type svm")
            sys.exit(1)

        generate_for_single_model(args.model_dir, args.features_dir, args.classifier_type)


if __name__ == '__main__':
    main()
