"""
Data Exploration and Visualization for Cross-Domain Plant Dataset

This script analyzes the plant identification dataset and creates visualizations
to understand:
- Class distribution and imbalance
- Domain distribution (herbarium vs field images)
- Sample images from different classes and domains
- Data statistics and insights

Usage:
    python Src/data_exploration.py --train_txt Dataset/list/train.txt \
                                    --train_dir Dataset/train \
                                    --species_file classes.txt \
                                    --output_dir EDA_Results
"""

import os
import argparse
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
from tqdm import tqdm
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Explore plant dataset')
    parser.add_argument('--train_txt', type=str, default='Dataset/list/train.txt')
    parser.add_argument('--train_dir', type=str, default='Dataset/train')
    parser.add_argument('--species_file', type=str, default='classes.txt')
    parser.add_argument('--output_dir', type=str, default='EDA_Results')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_species_names(species_file):
    """Load species names from classes.txt"""
    with open(species_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


def analyze_dataset(train_txt):
    """
    Analyze dataset structure and statistics.

    Returns:
        dict: Analysis results
    """
    class_images = defaultdict(list)
    domain_counts = Counter()
    class_domains = defaultdict(lambda: {'herbarium': 0, 'field': 0})

    print("📖 Analyzing dataset...")
    with open(train_txt, 'r') as f:
        for line in tqdm(f, desc="Reading"):
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            img_path, class_id = parts[0], parts[1]

            # Determine domain
            if '/herbarium/' in img_path:
                domain = 'herbarium'
            elif '/photo/' in img_path:
                domain = 'field'
            else:
                domain = 'unknown'

            class_images[class_id].append((img_path, domain))
            domain_counts[domain] += 1
            class_domains[class_id][domain] += 1

    return {
        'class_images': class_images,
        'domain_counts': domain_counts,
        'class_domains': class_domains
    }


def plot_class_distribution(class_images, species_names, output_dir):
    """Plot class distribution histogram"""
    plt.figure(figsize=(14, 6))

    class_sizes = [len(imgs) for imgs in class_images.values()]

    plt.subplot(1, 2, 1)
    plt.hist(class_sizes, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Number of Images per Class', fontsize=12)
    plt.ylabel('Number of Classes', fontsize=12)
    plt.title('Class Distribution (Imbalance Analysis)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = f'Min: {min(class_sizes)}\nMax: {max(class_sizes)}\n' \
                 f'Mean: {np.mean(class_sizes):.1f}\nMedian: {np.median(class_sizes):.1f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Sorted bar chart (top 20 and bottom 20)
    plt.subplot(1, 2, 2)
    sorted_classes = sorted(class_images.items(), key=lambda x: len(x[1]), reverse=True)

    # Get top 10 and bottom 10
    top_10 = sorted_classes[:10]
    bottom_10 = sorted_classes[-10:]

    combined = top_10 + [('...', [])] + bottom_10
    labels = [f"Class {cls}" for cls, _ in combined]
    values = [len(imgs) for _, imgs in combined]

    colors = ['darkgreen']*10 + ['gray'] + ['darkred']*10
    plt.barh(range(len(labels)), values, color=colors, alpha=0.7)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel('Number of Images', fontsize=12)
    plt.title('Top 10 & Bottom 10 Classes by Size', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: class_distribution.png")


def plot_domain_distribution(domain_counts, class_domains, output_dir):
    """Plot domain distribution analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Overall domain distribution
    domains = list(domain_counts.keys())
    counts = list(domain_counts.values())
    colors_pie = ['#8B4513', '#228B22', '#808080'][:len(domains)]

    axes[0].pie(counts, labels=domains, autopct='%1.1f%%', startangle=90,
                colors=colors_pie, textprops={'fontsize': 12})
    axes[0].set_title('Overall Domain Distribution', fontsize=14, fontweight='bold')

    # Classes with/without field images
    classes_with_field = sum(1 for d in class_domains.values() if d['field'] > 0)
    classes_without_field = len(class_domains) - classes_with_field

    axes[1].bar(['With Field Images', 'Without Field Images'],
                [classes_with_field, classes_without_field],
                color=['#228B22', '#DC143C'], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Number of Classes', fontsize=12)
    axes[1].set_title('Classes with/without Field Images', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Add counts on bars
    for i, v in enumerate([classes_with_field, classes_without_field]):
        axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold', fontsize=12)

    # Domain distribution per class (stacked bar)
    class_ids = sorted(class_domains.keys())
    herbarium_counts = [class_domains[cid]['herbarium'] for cid in class_ids]
    field_counts = [class_domains[cid]['field'] for cid in class_ids]

    # Plot every 5th class for readability
    indices = range(0, len(class_ids), 5)
    x_labels = [f"C{class_ids[i]}" for i in indices]
    herb_subset = [herbarium_counts[i] for i in indices]
    field_subset = [field_counts[i] for i in indices]

    x_pos = np.arange(len(indices))
    axes[2].bar(x_pos, herb_subset, label='Herbarium', color='#8B4513', alpha=0.7)
    axes[2].bar(x_pos, field_subset, bottom=herb_subset, label='Field', color='#228B22', alpha=0.7)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
    axes[2].set_ylabel('Number of Images', fontsize=12)
    axes[2].set_title('Domain Distribution per Class (Every 5th)', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: domain_distribution.png")


def create_sample_grid(class_images, train_dir, output_dir, species_names, seed=42):
    """Create a 5x5 grid of sample images from different classes"""
    random.seed(seed)

    # Select 25 random classes
    class_ids = random.sample(list(class_images.keys()), min(25, len(class_images)))

    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.flatten()

    print("🖼️  Creating sample image grid...")
    for idx, class_id in enumerate(tqdm(class_ids, desc="Loading images")):
        images_list = class_images[class_id]

        # Try to get one herbarium and one field image if available
        herbarium_imgs = [img for img, dom in images_list if dom == 'herbarium']
        field_imgs = [img for img, dom in images_list if dom == 'field']

        # Select image (prefer field if available, otherwise herbarium)
        if field_imgs:
            img_path = random.choice(field_imgs)
            domain_label = 'Field'
        elif herbarium_imgs:
            img_path = random.choice(herbarium_imgs)
            domain_label = 'Herbarium'
        else:
            img_path = images_list[0][0]
            domain_label = 'Unknown'

        # Load and display image
        full_path = os.path.join(train_dir, '/'.join(img_path.split('/')[1:]))

        try:
            img = Image.open(full_path).convert('RGB')
            axes[idx].imshow(img)
            axes[idx].axis('off')

            # Add title with class info
            species_idx = list(class_images.keys()).index(class_id)
            species_name = species_names[species_idx] if species_idx < len(species_names) else f"Class {class_id}"
            title = f"{species_name[:30]}\n({domain_label})"
            axes[idx].set_title(title, fontsize=8, fontweight='bold')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error loading\nClass {class_id}",
                          ha='center', va='center', fontsize=10)
            axes[idx].axis('off')

    plt.suptitle('Sample Images from Different Plant Classes', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: sample_images_grid.png")


def create_domain_comparison(class_images, train_dir, output_dir, seed=42):
    """Create side-by-side comparison of herbarium vs field images"""
    random.seed(seed)

    # Find classes with both herbarium and field images
    classes_with_both = []
    for class_id, images_list in class_images.items():
        herbarium = [img for img, dom in images_list if dom == 'herbarium']
        field = [img for img, dom in images_list if dom == 'field']
        if herbarium and field:
            classes_with_both.append(class_id)

    # Select 5 random classes
    selected_classes = random.sample(classes_with_both, min(5, len(classes_with_both)))

    fig, axes = plt.subplots(5, 2, figsize=(12, 20))

    print("🔍 Creating domain comparison grid...")
    for row, class_id in enumerate(tqdm(selected_classes, desc="Comparing domains")):
        images_list = class_images[class_id]

        herbarium_imgs = [img for img, dom in images_list if dom == 'herbarium']
        field_imgs = [img for img, dom in images_list if dom == 'field']

        # Load herbarium image
        herb_path = os.path.join(train_dir, '/'.join(random.choice(herbarium_imgs).split('/')[1:]))
        field_path = os.path.join(train_dir, '/'.join(random.choice(field_imgs).split('/')[1:]))

        try:
            herb_img = Image.open(herb_path).convert('RGB')
            axes[row, 0].imshow(herb_img)
            axes[row, 0].axis('off')
            axes[row, 0].set_title(f'Herbarium (Class {class_id})', fontsize=10, fontweight='bold')
        except Exception as e:
            axes[row, 0].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[row, 0].axis('off')

        try:
            field_img = Image.open(field_path).convert('RGB')
            axes[row, 1].imshow(field_img)
            axes[row, 1].axis('off')
            axes[row, 1].set_title(f'Field (Class {class_id})', fontsize=10, fontweight='bold')
        except Exception as e:
            axes[row, 1].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[row, 1].axis('off')

    plt.suptitle('Domain Shift: Herbarium vs Field Images', fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: domain_comparison.png")


def save_statistics(analysis, output_dir):
    """Save dataset statistics as JSON"""
    class_images = analysis['class_images']
    domain_counts = analysis['domain_counts']
    class_domains = analysis['class_domains']

    stats = {
        'total_classes': len(class_images),
        'total_images': sum(len(imgs) for imgs in class_images.values()),
        'domain_distribution': dict(domain_counts),
        'classes_with_field_images': sum(1 for d in class_domains.values() if d['field'] > 0),
        'classes_without_field_images': sum(1 for d in class_domains.values() if d['field'] == 0),
        'class_size_statistics': {
            'min': min(len(imgs) for imgs in class_images.values()),
            'max': max(len(imgs) for imgs in class_images.values()),
            'mean': np.mean([len(imgs) for imgs in class_images.values()]),
            'median': float(np.median([len(imgs) for imgs in class_images.values()])),
            'std': float(np.std([len(imgs) for imgs in class_images.values()]))
        }
    }

    stats_file = os.path.join(output_dir, 'dataset_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"✅ Saved: dataset_statistics.json")

    # Print summary
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS SUMMARY")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("="*60)


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("🌱 PLANT DATASET EXPLORATION")
    print("="*60)

    # Load species names
    species_names = load_species_names(args.species_file)
    print(f"📖 Loaded {len(species_names)} species names")

    # Analyze dataset
    analysis = analyze_dataset(args.train_txt)

    # Create visualizations
    print("\n📊 Creating visualizations...")
    plot_class_distribution(analysis['class_images'], species_names, args.output_dir)
    plot_domain_distribution(analysis['domain_counts'], analysis['class_domains'], args.output_dir)
    create_sample_grid(analysis['class_images'], args.train_dir, args.output_dir, species_names, args.seed)
    create_domain_comparison(analysis['class_images'], args.train_dir, args.output_dir, args.seed)

    # Save statistics
    print("\n💾 Saving statistics...")
    save_statistics(analysis, args.output_dir)

    print(f"\n✨ Exploration complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
