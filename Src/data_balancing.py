"""
Data Balancing Script for Cross-Domain Plant Identification

This script balances the training dataset to have exactly 200 samples per class,
then splits them into 80% training (160 samples) and 20% validation (40 samples).

The balanced dataset preserves domain information (herbarium vs field) and creates
a structured dataset for training DINOv2 models.

Usage:
    python Src/data_balancing.py --train_txt Dataset/list/train.txt \
                                  --source_dir Dataset/train \
                                  --train_dir Dataset/balanced_train \
                                  --val_dir Dataset/validation \
                                  --samples_per_class 200 \
                                  --train_split 0.8
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
    parser = argparse.ArgumentParser(description='Balance plant dataset')
    parser.add_argument('--train_txt', type=str, default='Dataset/list/train.txt',
                        help='Path to train.txt file')
    parser.add_argument('--source_dir', type=str, default='Dataset/train',
                        help='Source training directory')
    parser.add_argument('--train_dir', type=str, default='Dataset/balanced_train',
                        help='Output training directory')
    parser.add_argument('--val_dir', type=str, default='Dataset/validation',
                        help='Output validation directory')
    parser.add_argument('--samples_per_class', type=int, default=200,
                        help='Number of samples per class')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Training split ratio (0.8 = 80% train, 20% val)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def load_train_data(train_txt):
    """
    Load training data from train.txt file.

    Returns:
        dict: {class_id: [(image_path, domain), ...]}
    """
    class_images = defaultdict(list)

    print(f"📖 Reading {train_txt}...")
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


def balance_class_samples(class_images, samples_per_class, seed=42):
    """
    Balance dataset to have exactly samples_per_class per class.

    Strategy:
    - If class has > samples_per_class: random sample
    - If class has < samples_per_class: duplicate randomly to reach target

    Args:
        class_images: dict of {class_id: [(img_path, domain), ...]}
        samples_per_class: target number of samples per class
        seed: random seed

    Returns:
        dict: balanced {class_id: [(img_path, domain), ...]}
    """
    random.seed(seed)
    balanced = {}

    print(f"\n⚖️  Balancing classes to {samples_per_class} samples each...")

    for class_id, images in tqdm(class_images.items(), desc="Balancing"):
        num_images = len(images)

        if num_images >= samples_per_class:
            # Random sample without replacement
            balanced[class_id] = random.sample(images, samples_per_class)
        else:
            # Duplicate randomly to reach target
            balanced[class_id] = images.copy()
            while len(balanced[class_id]) < samples_per_class:
                # Randomly select and duplicate
                to_add = random.sample(images,
                                     min(len(images), samples_per_class - len(balanced[class_id])))
                balanced[class_id].extend(to_add)

    return balanced


def split_train_val(balanced_images, train_split, seed=42):
    """
    Split balanced dataset into training and validation sets.

    Args:
        balanced_images: dict of {class_id: [(img_path, domain), ...]}
        train_split: fraction for training (e.g., 0.8 for 80%)
        seed: random seed

    Returns:
        tuple: (train_dict, val_dict)
    """
    random.seed(seed)
    train_data = {}
    val_data = {}

    print(f"\n✂️  Splitting into {train_split*100:.0f}% train / {(1-train_split)*100:.0f}% val...")

    for class_id, images in balanced_images.items():
        # Shuffle images
        shuffled = images.copy()
        random.shuffle(shuffled)

        # Calculate split point
        split_idx = int(len(images) * train_split)

        train_data[class_id] = shuffled[:split_idx]
        val_data[class_id] = shuffled[split_idx:]

    return train_data, val_data


def copy_images(image_dict, source_dir, dest_dir, split_name):
    """
    Copy images from source to destination directory.

    Args:
        image_dict: {class_id: [(img_path, domain), ...]}
        source_dir: root source directory
        dest_dir: root destination directory
        split_name: 'train' or 'val' for logging
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
    print(f"   🏛️  Herbarium: {stats.get('herbarium', 0)}")
    print(f"   🌿 Field: {stats.get('field', 0)}")

    return stats


def main():
    args = parse_args()

    print("=" * 60)
    print("🌱 Plant Dataset Balancing & Splitting")
    print("=" * 60)
    print(f"Source: {args.source_dir}")
    print(f"Target samples/class: {args.samples_per_class}")
    print(f"Train split: {args.train_split*100:.0f}% / Val split: {(1-args.train_split)*100:.0f}%")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Load training data
    class_images = load_train_data(args.train_txt)
    print(f"\n📊 Loaded {len(class_images)} classes")
    print(f"   Total images: {sum(len(imgs) for imgs in class_images.values())}")

    # Show class distribution before balancing
    print("\n📈 Class distribution (before balancing):")
    sizes = [len(imgs) for imgs in class_images.values()]
    print(f"   Min: {min(sizes)} images")
    print(f"   Max: {max(sizes)} images")
    print(f"   Avg: {sum(sizes)/len(sizes):.1f} images")

    # Balance classes
    balanced = balance_class_samples(class_images, args.samples_per_class, args.seed)

    # Split into train and validation
    train_data, val_data = split_train_val(balanced, args.train_split, args.seed)

    # Copy training images
    train_stats = copy_images(train_data, args.source_dir, args.train_dir, 'train')

    # Copy validation images
    val_stats = copy_images(val_data, args.source_dir, args.val_dir, 'validation')

    # Final summary
    print("\n" + "=" * 60)
    print("✨ BALANCING COMPLETE!")
    print("=" * 60)
    print(f"📁 Training set: {args.train_dir}")
    print(f"   {train_stats['total_images']} images across {train_stats['classes']} classes")
    print(f"   ({args.samples_per_class * args.train_split:.0f} images/class)")
    print(f"\n📁 Validation set: {args.val_dir}")
    print(f"   {val_stats['total_images']} images across {val_stats['classes']} classes")
    print(f"   ({args.samples_per_class * (1-args.train_split):.0f} images/class)")
    print("=" * 60)


if __name__ == '__main__':
    main()
