
import os
import shutil
from collections import defaultdict
import argparse

def balance_data(train_path, balanced_train_path, num_samples_per_class):
    """
    Balances the training data by sampling a specified number of images from each class.
    """
    if os.path.exists(balanced_train_path):
        shutil.rmtree(balanced_train_path)
    os.makedirs(balanced_train_path)

    class_counts = defaultdict(int)
    for domain in os.listdir(train_path):
        domain_path = os.path.join(train_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for class_name in os.listdir(domain_path):
            class_path = os.path.join(domain_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            dest_class_path = os.path.join(balanced_train_path, class_name)
            os.makedirs(dest_class_path, exist_ok=True)
            
            images = os.listdir(class_path)
            for i, image_name in enumerate(images):
                if i >= num_samples_per_class:
                    break
                src_image_path = os.path.join(class_path, image_name)
                dest_image_path = os.path.join(dest_class_path, image_name)
                shutil.copy(src_image_path, dest_image_path)
                class_counts[class_name] += 1

    print(f"Data balancing complete. Each class now has up to {num_samples_per_class} samples.")
    print("Class distribution in balanced set:")
    for class_name, count in sorted(class_counts.items()):
        print(f"- {class_name}: {count} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance the training dataset.")
    parser.add_argument("--train_path", type=str, default="../Dataset/train", help="Path to the original training data.")
    parser.add_argument("--balanced_train_path", type=str, default="../Dataset/balanced_train", help="Path to save the balanced training data.")
    parser.add_argument("--samples_per_class", type=int, default=100, help="Number of samples per class in the balanced dataset.")
    args = parser.parse_args()

    balance_data(args.train_path, args.balanced_train_path, args.samples_per_class)
