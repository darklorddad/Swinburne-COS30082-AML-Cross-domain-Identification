import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

    # Combine them, drop any missing values, and ensure the IDs are integers.
    species_df = pd.concat([with_pairs_df, without_pairs_df]).dropna()
    species_df['class_id'] = species_df['class_id'].astype(int)
    
    species_to_consider = species_df['class_id'].tolist()
    
    # Create a mapping from the original class ID (e.g., 125412) to a new, zero-based index (e.g., 5).
    class_to_idx = {class_id: i for i, class_id in enumerate(species_to_consider)}
    
    # Verify that the number of classes in the list matches your configuration.
    if len(class_to_idx) != config.NUM_CLASSES:
        print(f"Warning: The number of classes from the lists ({len(class_to_idx)}) does not match NUM_CLASSES in config.py ({config.NUM_CLASSES}).")

    # Load the full training data list
    full_train_df = pd.read_csv(os.path.join(config.DATA_DIR, config.TRAIN_LIST), sep=' ', header=None, names=['image_path', 'class_id'])
    
    # 1. Filter the DataFrame to only include the 100 classes we are interested in.
    # 2. Apply the mapping to convert original class IDs to 0-indexed labels.
    full_train_df = full_train_df[full_train_df['class_id'].isin(species_to_consider)].copy()
    full_train_df['class_id'] = full_train_df['class_id'].map(class_to_idx)

    # Drop any rows that failed to map (if any classes in train.txt were not in the master list)
    full_train_df.dropna(subset=['class_id'], inplace=True)
    full_train_df['class_id'] = full_train_df['class_id'].astype(int)
    
    # Split training data into training and validation sets
    # Stratify ensures that the class distribution is similar in both train and val sets
    train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42, stratify=full_train_df['class_id'])

    # Load the test data list
    test_df = pd.read_csv(os.path.join(config.DATA_DIR, config.TEST_LIST), sep=' ', header=None, names=['image_path', 'class_id'])

    # Filter and map the test data as well to ensure consistency.
    test_df = test_df[test_df['class_id'].isin(species_to_consider)].copy()
    test_df['class_id'] = test_df['class_id'].map(class_to_idx)

    # Drop any rows that failed to map and ensure integer type
    test_df.dropna(subset=['class_id'], inplace=True)
    test_df['class_id'] = test_df['class_id'].astype(int)

    # Create dataset instances from the dataframes
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

    for epoch in range(config.NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS}')
        print('-' * 10)
        
        # Train for one epoch
        train_loss, train_top1, train_top5 = train_model(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f} | Top-1 Acc: {train_top1:.2f}% | Top-5 Acc: {train_top5:.2f}%')

        # Validate at the end of the epoch
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc*100:.2f}%')
        
        # Step the scheduler dynamically
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        # Check if this is the best model so far and save it
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0 # Reset counter
            save_path = ''
            if config.MODEL_NAME == 'resnet50': save_path = config.MODEL_SAVE_PATH_RESNET50
            elif config.MODEL_NAME == 'convnextv2': save_path = config.MODEL_SAVE_PATH_CONVNEXTV2
            elif config.MODEL_NAME == 'xception': save_path = config.MODEL_SAVE_PATH_XCEPTION
            
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
    save_path = ''
    if config.MODEL_NAME == 'resnet50': save_path = config.MODEL_SAVE_PATH_RESNET50
    elif config.MODEL_NAME == 'convnextv2': save_path = config.MODEL_SAVE_PATH_CONVNEXTV2
    elif config.MODEL_NAME == 'xception': save_path = config.MODEL_SAVE_PATH_XCEPTION
    
    # Load the best weights into the model
    model.load_state_dict(torch.load(save_path))
    
    # Get the final performance summary
    performance = get_test_set_performance(model, test_loader, device)

    print("\n=======================================================")
    print("FINAL SUMMARY")
    print("=======================================================")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("-------------------------------------------------------")
    print("Final Test Set Performance (on best model):")
    print(f"Top-1 Accuracy:             {performance['Top-1 Accuracy']:.4f}")
    print(f"Average Accuracy Per Class: {performance['Average Accuracy Per Class']:.4f}")
    print(f"Mean Precision:             {performance['Mean Precision']:.4f}")
    print(f"Mean Recall:                {performance['Mean Recall']:.4f}")
    print(f"Mean F1-Score:              {performance['Mean F1-Score']:.4f}")
    print("=======================================================")
    print(f"\nBest model weights are saved in {save_path}")

if __name__ == '__main__':
    main()