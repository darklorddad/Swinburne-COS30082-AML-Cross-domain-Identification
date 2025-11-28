"""
Cross-Domain Data Balancing Script for Plant Identification

This script creates a validation split optimized for testing cross-domain generalization:
- With-Pairs classes (60): Reserve field images for validation (herb→field testing)
- Without-Pairs classes (40): Random split herbarium only

The goal is to properly evaluate DINOv2's cross-domain transfer capabilities with
metrics (MRR, Top-1, Top-5) across Overall, With-Pairs, and Without-Pairs categories.

Usage:
    python Src/data_balancing_crossdomain.py
"""

import os
import shutil
import random
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Create cross-domain validation split')
    parser.add_argument('--train_txt', type=str, default='Dataset/list/train.txt',
                        help='Path to train.txt file')
    parser.add_argument('--source_dir', type=str, default='Dataset/train',
                        help='Source training directory')
    parser.add_argument('--with_pairs_txt', type=str, default='Dataset/list/class_with_pairs.txt',
                        help='Classes with both herbarium and field')
    parser.add_argument('--without_pairs_txt', type=str, default='Dataset/list/class_without_pairs.txt',
                        help='Classes with herbarium only')
    parser.add_argument('--train_dir', type=str, default='Dataset/balanced_train_crossdomain',
                        help='Output training directory')
    parser.add_argument('--val_dir', type=str, default='Dataset/validation_crossdomain',
                        help='Output validation directory')
    parser.add_argument('--samples_per_class', type=int, default=200,
                        help='Total samples per class (train + val)')
    parser.add_argument('--val_samples', type=int, default=40,
                        help='Validation samples per class')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def load_class_categories(with_pairs_txt, without_pairs_txt):
    """Load class categories"""
    print(f"📋 Loading class categories...")

    with_pairs = set()
    with open(with_pairs_txt, 'r') as f:
        with_pairs = set(line.strip() for line in f if line.strip())

    without_pairs = set()
    with open(without_pairs_txt, 'r') as f:
        without_pairs = set(line.strip() for line in f if line.strip())

    print(f"   ✅ With-Pairs: {len(with_pairs)} classes (both domains)")
    print(f"   ✅ Without-Pairs: {len(without_pairs)} classes (herbarium only)")

    return with_pairs, without_pairs


def load_train_data(train_txt):
    """
    Load training data from train.txt file.

    Returns:
        dict: {class_id: [(image_path, domain), ...]}
    """
    class_images = defaultdict(list)

    print(f"\n📖 Reading {train_txt}...")
    with open(train_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            img_path, class_id = parts[0], parts[1]

            # Determine domain (herbarium or field/photo)
            if '/herbarium/' in img_path:
                domain = 'herbarium'
            elif '/photo/' in img_path:
                domain = 'field'
            else:
                domain = 'unknown'

            class_images[class_id].append((img_path, domain))

    return class_images


def split_crossdomain(class_images, with_pairs, without_pairs, val_samples=40, total_samples=200, seed=42):
    """
    Split dataset with cross-domain strategy.

    With-Pairs: Reserve field for validation (cross-domain testing)
    Without-Pairs: Random split (herbarium only)

    Args:
        class_images: dict of {class_id: [(img_path, domain), ...]}
        with_pairs: set of class IDs with both domains
        without_pairs: set of class IDs with herbarium only
        val_samples: number of validation samples per class
        total_samples: total samples per class (train + val)
        seed: random seed

    Returns:
        tuple: (train_dict, val_dict, statistics)
    """
    random.seed(seed)
    train_data = {}
    val_data = {}
    statistics = {
        'with_pairs': {'train': {'herbarium': 0, 'field': 0},
                       'val': {'herbarium': 0, 'field': 0}},
        'without_pairs': {'train': {'herbarium': 0, 'field': 0},
                          'val': {'herbarium': 0, 'field': 0}}
    }

    print(f"\n✂️  Cross-Domain Splitting Strategy:")
    print(f"   With-Pairs: Reserve field for validation (cross-domain testing)")
    print(f"   Without-Pairs: Random split (herbarium only)")
    print(f"   Target: {val_samples} val + {total_samples - val_samples} train per class")
    print()

    for class_id, images in tqdm(class_images.items(), desc="Splitting classes"):
        # Separate by domain
        herbarium = [img for img in images if img[1] == 'herbarium']
        field = [img for img in images if img[1] == 'field']

        # Shuffle for randomness
        random.shuffle(herbarium)
        random.shuffle(field)

        train_samples = total_samples - val_samples

        if class_id in with_pairs:
            # WITH-PAIRS: Reserve field for validation (cross-domain testing)

            # Validation: Prioritize field images
            val_field = field[:val_samples]  # Take up to val_samples field images
            val_herb_needed = val_samples - len(val_field)  # Fill remainder with herbarium
            val_herb = herbarium[:val_herb_needed]

            val_data[class_id] = val_field + val_herb

            # Training: Use remaining images
            remaining_herb = herbarium[val_herb_needed:]
            remaining_field = field[val_samples:]
            train_pool = remaining_herb + remaining_field

            # Balance training to target size with duplication
            if len(train_pool) >= train_samples:
                train_data[class_id] = random.sample(train_pool, train_samples)
            else:
                train_data[class_id] = train_pool.copy()
                while len(train_data[class_id]) < train_samples:
                    to_add = random.sample(train_pool,
                                         min(len(train_pool), train_samples - len(train_data[class_id])))
                    train_data[class_id].extend(to_add)

            # Update statistics
            category = 'with_pairs'
            for img_path, domain in val_data[class_id]:
                statistics[category]['val'][domain] += 1
            for img_path, domain in train_data[class_id]:
                statistics[category]['train'][domain] += 1

        else:
            # WITHOUT-PAIRS: Random split (herbarium only)
            # Note: Should only have herbarium, but handle gracefully
            all_images = herbarium + field
            random.shuffle(all_images)

            # Validation: First val_samples
            val_data[class_id] = all_images[:val_samples]

            # Training: Balance remaining to train_samples
            remaining = all_images[val_samples:]
            if len(remaining) >= train_samples:
                train_data[class_id] = random.sample(remaining, train_samples)
            else:
                train_data[class_id] = remaining.copy()
                while len(train_data[class_id]) < train_samples:
                    to_add = random.sample(remaining if remaining else all_images,
                                         min(len(remaining) if remaining else len(all_images),
                                             train_samples - len(train_data[class_id])))
                    train_data[class_id].extend(to_add)

            # Update statistics
            category = 'without_pairs'
            for img_path, domain in val_data[class_id]:
                statistics[category]['val'][domain] += 1
            for img_path, domain in train_data[class_id]:
                statistics[category]['train'][domain] += 1

    return train_data, val_data, statistics


def copy_images(image_dict, source_dir, dest_dir, split_name):
    """
    Copy images from source to destination directory.

    Args:
        image_dict: {class_id: [(img_path, domain), ...]}
        source_dir: root source directory
        dest_dir: root destination directory
        split_name: 'train' or 'validation' for logging
    """
    print(f"\n📁 Copying {split_name} images to {dest_dir}...")

    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)

    # Statistics
    stats = {
        'total_images': 0,
        'herbarium': 0,
        'field': 0,
        'classes': len(image_dict)
    }

    # Copy images class by class
    for class_id, images in tqdm(image_dict.items(), desc=f"Copying {split_name}"):
        # Create class directory
        class_dir = os.path.join(dest_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)

        # Copy each image
        for idx, (img_path, domain) in enumerate(images):
            source_path = os.path.join(source_dir, '/'.join(img_path.split('/')[1:]))  # Remove 'train/' prefix

            # Create unique filename: classID_idx_domain.jpg
            ext = os.path.splitext(img_path)[1]
            dest_filename = f"{class_id}_{idx:04d}_{domain}{ext}"
            dest_path = os.path.join(class_dir, dest_filename)

            # Copy file
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                stats['total_images'] += 1
                stats[domain] = stats.get(domain, 0) + 1
            else:
                print(f"⚠️  Warning: File not found: {source_path}")

    # Save metadata
    metadata = {
        'split': split_name,
        'num_classes': stats['classes'],
        'total_images': stats['total_images'],
        'herbarium_images': stats.get('herbarium', 0),
        'field_images': stats.get('field', 0)
    }

    metadata_path = os.path.join(dest_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ {split_name.capitalize()} set complete:")
    print(f"   📊 Classes: {stats['classes']}")
    print(f"   🖼️  Total images: {stats['total_images']}")
    print(f"   🏛️  Herbarium: {stats.get('herbarium', 0)} ({stats.get('herbarium', 0)/stats['total_images']*100:.1f}%)")
    print(f"   🌿 Field: {stats.get('field', 0)} ({stats.get('field', 0)/stats['total_images']*100:.1f}%)")

    return stats


def print_comparison_stats(statistics, train_stats, val_stats):
    """Print detailed comparison statistics"""
    print("\n" + "=" * 70)
    print("📊 CROSS-DOMAIN SPLIT ANALYSIS")
    print("=" * 70)

    # With-Pairs breakdown
    print("\n🔄 WITH-PAIRS CLASSES (60 classes - Cross-Domain Testing):")
    print("   Training Set:")
    wp_train_herb = statistics['with_pairs']['train']['herbarium']
    wp_train_field = statistics['with_pairs']['train']['field']
    wp_train_total = wp_train_herb + wp_train_field
    print(f"      Herbarium: {wp_train_herb} ({wp_train_herb/wp_train_total*100:.1f}%)")
    print(f"      Field: {wp_train_field} ({wp_train_field/wp_train_total*100:.1f}%)")
    print(f"      Total: {wp_train_total}")

    print("   Validation Set:")
    wp_val_herb = statistics['with_pairs']['val']['herbarium']
    wp_val_field = statistics['with_pairs']['val']['field']
    wp_val_total = wp_val_herb + wp_val_field
    print(f"      Herbarium: {wp_val_herb} ({wp_val_herb/wp_val_total*100:.1f}%)")
    print(f"      Field: {wp_val_field} ({wp_val_field/wp_val_total*100:.1f}%) ⭐")
    print(f"      Total: {wp_val_total}")
    print(f"   ✅ Strategy: Field images reserved for validation (tests herb→field)")

    # Without-Pairs breakdown
    print("\n📚 WITHOUT-PAIRS CLASSES (40 classes - Herbarium Only):")
    print("   Training Set:")
    wop_train_herb = statistics['without_pairs']['train']['herbarium']
    wop_train_field = statistics['without_pairs']['train']['field']
    wop_train_total = wop_train_herb + wop_train_field
    print(f"      Herbarium: {wop_train_herb}")
    print(f"      Field: {wop_train_field}")
    print(f"      Total: {wop_train_total}")

    print("   Validation Set:")
    wop_val_herb = statistics['without_pairs']['val']['herbarium']
    wop_val_field = statistics['without_pairs']['val']['field']
    wop_val_total = wop_val_herb + wop_val_field
    print(f"      Herbarium: {wop_val_herb}")
    print(f"      Field: {wop_val_field}")
    print(f"      Total: {wop_val_total}")
    print(f"   ✅ Strategy: Random split (herb→herb baseline)")

    # Overall comparison
    print("\n📈 OVERALL COMPARISON:")
    print(f"   Old Validation (Random 80/20):")
    print(f"      ~79% herbarium, ~21% field (same as training)")
    print(f"      ❌ Does NOT test cross-domain generalization")

    print(f"\n   New Validation (Cross-Domain):")
    total_val_herb = wp_val_herb + wop_val_herb
    total_val_field = wp_val_field + wop_val_field
    total_val = total_val_herb + total_val_field
    print(f"      {total_val_herb/total_val*100:.1f}% herbarium, {total_val_field/total_val*100:.1f}% field")
    print(f"      ✅ Tests cross-domain generalization (herb→field)")

    print("\n🎯 EXPECTED VALIDATION PERFORMANCE:")
    print("   Old validation: 99.7-99.9% (misleading - same distribution)")
    print("   New validation: Lower, more realistic (cross-domain challenge)")
    print("      - With-Pairs: 85-95% (herb→field is hard!)")
    print("      - Without-Pairs: ~99% (herb→herb, same domain)")
    print("=" * 70)


def main():
    args = parse_args()

    # Set UTF-8 encoding for Windows console
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 70)
    print("🌱 Cross-Domain Plant Dataset Balancing")
    print("=" * 70)
    print(f"Source: {args.source_dir}")
    print(f"Strategy: Field images → Validation (With-Pairs classes)")
    print(f"Target: {args.val_samples} val + {args.samples_per_class - args.val_samples} train per class")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Load class categories
    with_pairs, without_pairs = load_class_categories(args.with_pairs_txt, args.without_pairs_txt)

    # Load training data
    class_images = load_train_data(args.train_txt)
    print(f"\n📊 Loaded {len(class_images)} classes")
    print(f"   Total images: {sum(len(imgs) for imgs in class_images.values())}")

    # Cross-domain split
    train_data, val_data, statistics = split_crossdomain(
        class_images, with_pairs, without_pairs,
        val_samples=args.val_samples,
        total_samples=args.samples_per_class,
        seed=args.seed
    )

    # Copy training images
    train_stats = copy_images(train_data, args.source_dir, args.train_dir, 'training')

    # Copy validation images
    val_stats = copy_images(val_data, args.source_dir, args.val_dir, 'validation')

    # Print detailed comparison
    print_comparison_stats(statistics, train_stats, val_stats)

    # Final summary
    print("\n" + "=" * 70)
    print("✨ CROSS-DOMAIN BALANCING COMPLETE!")
    print("=" * 70)
    print(f"📁 Training set: {args.train_dir}")
    print(f"   {train_stats['total_images']} images across {train_stats['classes']} classes")
    print(f"\n📁 Validation set: {args.val_dir}")
    print(f"   {val_stats['total_images']} images across {val_stats['classes']} classes")
    print(f"   ⭐ Optimized for cross-domain evaluation!")
    print("\n🚀 Next Steps:")
    print("   1. Re-extract features with new validation split")
    print("   2. Retrain classifiers")
    print("   3. Compare old vs new validation performance")
    print("=" * 70)


if __name__ == '__main__':
    main()
