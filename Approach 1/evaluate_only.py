import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import json
import plot_utils

# Local Imports 
import config
from dataset import PlantDataset, get_transforms
from model import create_resnet50_model, create_convnextv2_model, create_xception_model
from train import get_test_set_performance

def main():
    """ 
    Loads a pre-trained model and runs the evaluation metrics 
    (Overall, With Pairs, Without Pairs) without re-training.
    """
    # 1. Setup 
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    print(f"Evaluating Model: {config.MODEL_NAME}")
    print(f"Data Mode: {config.DATA_MODE}")

    # 2. Load Class Mappings (Required for Detailed Metrics)
    with_pairs_path = os.path.join(config.DATA_DIR, 'list/class_with_pairs.txt')
    without_pairs_path = os.path.join(config.DATA_DIR, 'list/class_without_pairs.txt')

    with_pairs_df = pd.read_csv(with_pairs_path, header=None, names=['class_id'])
    without_pairs_df = pd.read_csv(without_pairs_path, header=None, names=['class_id'])

    species_df = pd.concat([with_pairs_df, without_pairs_df]).dropna()
    species_df['class_id'] = species_df['class_id'].astype(int)
    species_to_consider = species_df['class_id'].tolist()
    
    class_to_idx = {class_id: i for i, class_id in enumerate(species_to_consider)}

    # 3. Prepare Test Loader
    test_df = pd.read_csv(os.path.join(config.DATA_DIR, config.TEST_LIST), sep=' ', header=None, names=['image_path', 'class_id'])
    
    # Filter and map test data
    test_df = test_df[test_df['class_id'].isin(species_to_consider)].copy()
    test_df['class_id'] = test_df['class_id'].map(class_to_idx)
    test_df.dropna(subset=['class_id'], inplace=True)
    test_df['class_id'] = test_df['class_id'].astype(int)

    data_transforms = get_transforms(model_name=config.MODEL_NAME)
    test_dataset = PlantDataset(root_dir=config.DATA_DIR, dataframe=test_df, transform=data_transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    print(f"Test set size: {len(test_dataset)}")

    # 4. Load Model Structure
    model = None
    if config.MODEL_NAME == 'resnet50': model = create_resnet50_model(config.NUM_CLASSES)
    elif config.MODEL_NAME == 'convnextv2': model = create_convnextv2_model(config.NUM_CLASSES)
    elif config.MODEL_NAME == 'xception': model = create_xception_model(config.NUM_CLASSES)
    else: raise ValueError(f"Model {config.MODEL_NAME} is not supported.")
    model.to(device)

    # 5. Load Saved Weights
    base_save_path = ''
    if config.MODEL_NAME == 'resnet50': base_save_path = config.MODEL_SAVE_PATH_RESNET50
    elif config.MODEL_NAME == 'convnextv2': base_save_path = config.MODEL_SAVE_PATH_CONVNEXTV2
    elif config.MODEL_NAME == 'xception': base_save_path = config.MODEL_SAVE_PATH_XCEPTION
    
    filename, ext = os.path.splitext(base_save_path)
    save_path = f"{filename}_{config.DATA_MODE}{ext}"

    if not os.path.exists(save_path):
        print(f"\nERROR: Model file not found at {save_path}")
        print("Please make sure you have trained this configuration first.")
        return

    print(f"Loading weights from: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))

    # 6. Run Evaluation
    print("Running evaluation...")
    
    # Prepare indices for detailed split metrics
    with_pairs_ids = with_pairs_df['class_id'].astype(int).tolist()
    without_pairs_ids = without_pairs_df['class_id'].astype(int).tolist()

    mapped_with_pairs = set([class_to_idx[cid] for cid in with_pairs_ids if cid in class_to_idx])
    mapped_without_pairs = set([class_to_idx[cid] for cid in without_pairs_ids if cid in class_to_idx])

    performance = get_test_set_performance(
        model, 
        test_loader, 
        device, 
        with_pairs_indices=mapped_with_pairs, 
        without_pairs_indices=mapped_without_pairs
    )

    # 7. Print Summary
    print("\n=======================================================")
    print("FINAL SUMMARY (Evaluation Only)")
    print("=======================================================")
    print(f"Top-1 Accuracy:             {performance['Top-1 Accuracy']:.4f}")
    print(f"Average Accuracy Per Class: {performance['Average Accuracy Per Class']:.4f}")
    print(f"Mean Precision:             {performance['Mean Precision']:.4f}")
    print(f"Mean Recall:                {performance['Mean Recall']:.4f}")
    print(f"Mean F1-Score:              {performance['Mean F1-Score']:.4f}")
    print("=======================================================")

    # 8. Save Detailed Metrics to JSON
    json_save_path = f"results_{config.MODEL_NAME}_{config.DATA_MODE}.json"
    with open(json_save_path, 'w') as f:
        json.dump(performance['JSON_Stats'], f, indent=4)
    print(f"Detailed JSON metrics saved to {json_save_path}")

    # 9. Generate/Regenerate Plots (Optional but helpful)
    cm_save_path = f"confusion_matrix_{config.MODEL_NAME}_{config.DATA_MODE}.png"
    plot_utils.plot_confusion_matrix(
        performance['All Labels'], 
        performance['All Predictions'], 
        save_path=cm_save_path
    )
    print(f"Confusion matrix saved to {cm_save_path}")

if __name__ == '__main__':
    main()

