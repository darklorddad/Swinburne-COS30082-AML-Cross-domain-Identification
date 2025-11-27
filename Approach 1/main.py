import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import json
import plot_utils

# Local Imports 
import config
from dataset import PlantDataset, get_transforms
from model import create_resnet50_model, create_convnextv2_model, create_xception_model
from train import train_model, validate_model, get_test_set_performance

def main():
    """ Main function to orchestrate the model training and evaluation process. """
    # 1. Setup 
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    print(f"Selected model: {config.MODEL_NAME}")

    data_transforms = get_transforms()

    #  2. Datasets and Dataloaders (WITH VALIDATION SPLIT)
    # The class IDs in the original files are large numbers (e.g., 125412) and not zero-indexed.
    # We must map them to a 0 to (NUM_CLASSES - 1) range before feeding them to the model.

    # Load the class IDs from class_with_pairs.txt and class_without_pairs.txt.
    # These two files together define the 100 classes for the experiment.
    with_pairs_path = os.path.join(config.DATA_DIR, 'list/class_with_pairs.txt')
    without_pairs_path = os.path.join(config.DATA_DIR, 'list/class_without_pairs.txt')

    with_pairs_df = pd.read_csv(with_pairs_path, header=None, names=['class_id'])
    without_pairs_df = pd.read_csv(without_pairs_path, header=None, names=['class_id'])

    # Convert to integers
    with_pairs_df['class_id'] = with_pairs_df['class_id'].astype(int)
    without_pairs_df['class_id'] = without_pairs_df['class_id'].astype(int)
    
    # Filter classes based on CLASS_MODE
    if config.CLASS_MODE == 'with_pairs':
        # Only use classes that have pairs (both herbarium and field photos)
        species_to_consider = with_pairs_df['class_id'].tolist()
        print(f"Using {len(species_to_consider)} classes WITH pairs")
    elif config.CLASS_MODE == 'without_pairs':
        # Only use classes that don't have pairs (herbarium only)
        species_to_consider = without_pairs_df['class_id'].tolist()
        print(f"Using {len(species_to_consider)} classes WITHOUT pairs")
    else:  # 'mixed' or 'all'
        # Use all classes (both with and without pairs)
        species_to_consider = pd.concat([with_pairs_df, without_pairs_df])['class_id'].dropna().astype(int).tolist()
        print(f"Using {len(species_to_consider)} classes (MIXED: {len(with_pairs_df)} with pairs, {len(without_pairs_df)} without pairs)")
    
    # Create a mapping from the original class ID (e.g., 125412) to a new, zero-based index (e.g., 5).
    class_to_idx = {class_id: i for i, class_id in enumerate(species_to_consider)}
    
    # Also create reverse mapping and sets for later use
    idx_to_class = {i: class_id for class_id, i in class_to_idx.items()}
    with_pairs_set = set(with_pairs_df['class_id'].tolist())
    without_pairs_set = set(without_pairs_df['class_id'].tolist())

    # Load the full training data list
    full_train_df = pd.read_csv(os.path.join(config.DATA_DIR, config.TRAIN_LIST), sep=' ', header=None, names=['image_path', 'class_id'])
    
    # 1. Filter the DataFrame to only include the 100 classes we are interested in.
    # 2. Apply the mapping to convert original class IDs to 0-indexed labels.
    full_train_df = full_train_df[full_train_df['class_id'].isin(species_to_consider)].copy()
    full_train_df['class_id'] = full_train_df['class_id'].map(class_to_idx)

    # Drop any rows that failed to map (if any classes in train.txt were not in the master list)
    full_train_df.dropna(subset=['class_id'], inplace=True)
    full_train_df['class_id'] = full_train_df['class_id'].astype(int)
    
    # Domain-aware train/val split based on class type
    # Strategy:
    # - WITH PAIRS classes: val = field photos, train = herbarium + leftover field photos
    # - WITHOUT PAIRS classes: val = herbarium, train = herbarium (rest)
    # - MIXED: Apply above logic per class type
    
    herbarium_df = full_train_df[full_train_df['image_path'].str.contains('herbarium')].copy()
    photo_df = full_train_df[full_train_df['image_path'].str.contains('photo')].copy()
    
    train_df_list = []
    val_df_list = []
    
    # For each class, determine train/val split based on class type
    for class_id in full_train_df['class_id'].unique():
        # Get original class ID to check if it's with_pairs or without_pairs
        original_class_id = idx_to_class[class_id]
        is_with_pairs = original_class_id in with_pairs_set
        
        class_herbarium = herbarium_df[herbarium_df['class_id'] == class_id]
        class_photo = photo_df[photo_df['class_id'] == class_id]
        
        if is_with_pairs and len(class_photo) > 0:
            # WITH PAIRS: Validation = field photos, Training = herbarium + leftover field photos
            # Split field photos: use ~20% for validation, rest for training
            if len(class_photo) >= 5:
                photo_train, photo_val = train_test_split(
                    class_photo, test_size=0.2, random_state=42, stratify=None
                )
            else:
                # Very few photos: use all for validation
                photo_train = class_photo.iloc[:0].copy()  # Empty
                photo_val = class_photo.copy()
            
            # Training: All herbarium + leftover field photos
            train_df_list.append(class_herbarium)
            train_df_list.append(photo_train)
            # Validation: Field photos only
            val_df_list.append(photo_val)
            
        else:
            # WITHOUT PAIRS (or WITH PAIRS but no field photos available): 
            # Validation = herbarium, Training = herbarium (rest)
            if len(class_herbarium) >= 2:
                herb_train, herb_val = train_test_split(
                    class_herbarium, test_size=0.2, random_state=42, stratify=None
                )
                train_df_list.append(herb_train)
                val_df_list.append(herb_val)
            else:
                # Only 1 image: put in training
                train_df_list.append(class_herbarium)
    
    # Combine all splits
    train_df = pd.concat(train_df_list, ignore_index=True) if train_df_list else pd.DataFrame()
    val_df = pd.concat(val_df_list, ignore_index=True) if val_df_list else pd.DataFrame()
    
    print(f"\nDomain-aware split summary (CLASS_MODE: {config.CLASS_MODE}):")
    print(f"  Training: {len(train_df)} images ({len(train_df[train_df['image_path'].str.contains('herbarium')])} herbarium, {len(train_df[train_df['image_path'].str.contains('photo')])} photo)")
    print(f"  Validation: {len(val_df)} images ({len(val_df[val_df['image_path'].str.contains('herbarium')])} herbarium, {len(val_df[val_df['image_path'].str.contains('photo')])} photo)")

    # Load the test data list
    test_df = pd.read_csv(os.path.join(config.DATA_DIR, config.TEST_LIST), sep=' ', header=None, names=['image_path', 'class_id'])

    # Filter and map the test data as well to ensure consistency.
    test_df = test_df[test_df['class_id'].isin(species_to_consider)].copy()
    test_df['class_id'] = test_df['class_id'].map(class_to_idx)

    # Drop any rows that failed to map and ensure integer type
    test_df.dropna(subset=['class_id'], inplace=True)
    test_df['class_id'] = test_df['class_id'].astype(int)

    # Create dataset instances from the dataframes
    # Pass model_name to get_transforms to select appropriate augmentations
    data_transforms = get_transforms(model_name=config.MODEL_NAME)
    
    train_dataset = PlantDataset(root_dir=config.DATA_DIR, dataframe=train_df, transform=data_transforms['train'])
    val_dataset = PlantDataset(root_dir=config.DATA_DIR, dataframe=val_df, transform=data_transforms['val'])
    test_dataset = PlantDataset(root_dir=config.DATA_DIR, dataframe=test_df, transform=data_transforms['val']) # Use 'val' transforms for test set

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    print(f"Original training images: {len(full_train_df)}")
    print(f"New training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # 3. Model, Loss, and Optimizer 
    model = None
    if config.MODEL_NAME == 'resnet50': model = create_resnet50_model(config.NUM_CLASSES)
    elif config.MODEL_NAME == 'convnextv2': model = create_convnextv2_model(config.NUM_CLASSES)
    elif config.MODEL_NAME == 'xception': model = create_xception_model(config.NUM_CLASSES)
    else: raise ValueError(f"Model {config.MODEL_NAME} is not supported.")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # Pass all trainable parameters to the optimizer for fine-tuning
    # Implement Differential Learning Rates: Lower LR for backbone, Higher LR for head
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Separate head and backbone parameters
    head_params = []
    backbone_params = []
    
    # Identify head parameters based on model type
    head_names = []
    if config.MODEL_NAME in ['resnet50', 'xception']:
        head_names = ['fc']
    elif config.MODEL_NAME == 'convnextv2':
        head_names = ['head']
        
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name for h in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.LEARNING_RATE * 0.1}, # Lower LR for backbone
        {'params': head_params, 'lr': config.LEARNING_RATE}           # Base LR for head
    ], weight_decay=config.WEIGHT_DECAY)
    
    # Dynamic Scheduler Selection
    if config.MODEL_NAME == 'convnextv2':
        # Cosine Annealing with Warm Restarts for ConvNeXt
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_MULT, eta_min=config.MIN_LR)
    else:
        # ReduceLROnPlateau for ResNet50 and Xception
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=config.SCHEDULER_FACTOR, 
                                                         patience=config.SCHEDULER_PATIENCE, min_lr=config.MIN_LR)

    # 4. Training and Validation Loop 
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # Initialize CSV Logger
    log_file = f"history_{config.MODEL_NAME}_{config.CLASS_MODE}.csv"
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    for epoch in range(config.NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS}')
        print('-' * 10)
        
        # Train for one epoch
        train_loss, train_top1, train_top5 = train_model(model, train_loader, criterion, optimizer, device, model_name=config.MODEL_NAME)
        print(f'Train Loss: {train_loss:.4f} | Top-1 Acc: {train_top1:.2f}% | Top-5 Acc: {train_top5:.2f}%')

        # Validate at the end of the epoch
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc*100:.2f}%')
        
        # Log history to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_top1, val_loss, val_acc*100])
        
        # Step the scheduler dynamically
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        # Check if this is the best model so far and save it
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0 # Reset counter
            
            # Determine base save path
            base_save_path = ''
            if config.MODEL_NAME == 'resnet50': base_save_path = config.MODEL_SAVE_PATH_RESNET50
            elif config.MODEL_NAME == 'convnextv2': base_save_path = config.MODEL_SAVE_PATH_CONVNEXTV2
            elif config.MODEL_NAME == 'xception': base_save_path = config.MODEL_SAVE_PATH_XCEPTION
            
            # Construct dynamic save path with CLASS_MODE
            filename, ext = os.path.splitext(base_save_path)
            save_path = f"{filename}_{config.CLASS_MODE}{ext}"
            
            torch.save(model.state_dict(), save_path)
            print(f"✨ New best model saved to {save_path} with accuracy: {best_val_acc*100:.2f}%")
        else:
            epochs_no_improve += 1
        
        # Early stopping check
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n--- Early stopping triggered after {config.EARLY_STOPPING_PATIENCE} epochs with no improvement. ---")
            break

    # 5. Final Evaluation on Test Set 
    print("\n--- Loading best model for final evaluation on test set ---")
    base_save_path = ''
    if config.MODEL_NAME == 'resnet50': base_save_path = config.MODEL_SAVE_PATH_RESNET50
    elif config.MODEL_NAME == 'convnextv2': base_save_path = config.MODEL_SAVE_PATH_CONVNEXTV2
    elif config.MODEL_NAME == 'xception': base_save_path = config.MODEL_SAVE_PATH_XCEPTION
    
    # Construct dynamic save path with DATA_MODE
    filename, ext = os.path.splitext(base_save_path)
    save_path = f"{filename}_{config.DATA_MODE}{ext}"
    
    # Load the best weights into the model
    model.load_state_dict(torch.load(save_path))
    
    # Get the final performance summary
    
    # Prepare sets of mapped indices for detailed evaluation (With/Without Pairs)
    with_pairs_ids = with_pairs_df['class_id'].astype(int).tolist()
    without_pairs_ids = without_pairs_df['class_id'].astype(int).tolist()

    # Map original IDs to current 0-99 IDs
    mapped_with_pairs = set([class_to_idx[cid] for cid in with_pairs_ids if cid in class_to_idx])
    mapped_without_pairs = set([class_to_idx[cid] for cid in without_pairs_ids if cid in class_to_idx])

    performance = get_test_set_performance(
        model, 
        test_loader, 
        device, 
        with_pairs_indices=mapped_with_pairs, 
        without_pairs_indices=mapped_without_pairs
    )

    print("\n=======================================================")
    print("FINAL SUMMARY")
    print("=======================================================")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("-------------------------------------------------------")
    print("Final Test Set Performance (on best model):")
    print(f"Top-1 Accuracy:             {performance['Top-1 Accuracy']:.4f} ({performance['Top-1 Accuracy']*100:.2f}%)")
    print(f"Top-5 Accuracy:             {performance['Top-5 Accuracy']:.4f} ({performance['Top-5 Accuracy']*100:.2f}%)")
    print(f"Average Accuracy Per Class: {performance['Average Accuracy Per Class']:.4f}")
    print(f"Mean Precision:             {performance['Mean Precision']:.4f}")
    print(f"Mean Recall:                {performance['Mean Recall']:.4f}")
    print(f"Mean F1-Score:              {performance['Mean F1-Score']:.4f}")
    print("=======================================================")
    print(f"\nBest model weights are saved in {save_path}")

    # --- Save Detailed Metrics to JSON ---
    json_save_path = f"results_{config.MODEL_NAME}_{config.CLASS_MODE}.json"
    with open(json_save_path, 'w') as f:
        json.dump(performance['JSON_Stats'], f, indent=4)
    print(f"Detailed JSON metrics saved to {json_save_path}")

    # --- Plotting ---
    print("\nGenerating plots...")
    plot_utils.plot_training_history(log_file)
    
    cm_save_path = f"confusion_matrix_{config.MODEL_NAME}_{config.CLASS_MODE}.png"
    plot_utils.plot_confusion_matrix(
        performance['All Labels'], 
        performance['All Predictions'], 
        save_path=cm_save_path
    )

if __name__ == '__main__':
    main()