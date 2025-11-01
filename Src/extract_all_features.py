
import os
import numpy as np
from feature_extractor import DinoV2FeatureExtractor
from tqdm import tqdm
import argparse

def extract_features_from_directory(image_dir, feature_dir, model_name):
    """
    Extracts features from all images in a directory and saves them to a new directory.
    """
    os.makedirs(feature_dir, exist_ok=True)
    extractor = DinoV2FeatureExtractor(model_name=model_name)

    for class_name in tqdm(os.listdir(image_dir), desc=f"Processing classes in {os.path.basename(image_dir)}"):
        class_path = os.path.join(image_dir, class_name)
        feature_class_path = os.path.join(feature_dir, class_name)
        os.makedirs(feature_class_path, exist_ok=True)

        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            feature_path = os.path.join(feature_class_path, f"{os.path.splitext(image_name)[0]}.npy")

            if os.path.exists(feature_path):
                continue

            features = extractor.extract_features(image_path)
            if features is not None:
                np.save(feature_path, features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from a dataset.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of images to process.")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory to save the features.")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14_reg", help="Name of the DINOv2 model to use.")
    args = parser.parse_args()

    extract_features_from_directory(args.image_dir, args.feature_dir, args.model_name)
