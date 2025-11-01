
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.models import create_model
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import argparse

# --- Main Training Function ---
def train_dinov2(data_dir, results_dir, model_name, num_epochs, batch_size, lr):
    os.makedirs(results_dir, exist_ok=True)

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model and Transformations ---
    model = create_model(model_name, pretrained=True, num_classes=0).to(device)
    data_config = resolve_data_config(model.default_cfg)
    transform = create_transform(**data_config)

    # --- Datasets and Dataloaders ---
    train_dir = os.path.join(data_dir, 'balanced_train')
    val_dir = os.path.join(data_dir, 'validation')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    # --- Freeze Backbone and Add Classifier ---
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Linear(model.embed_dim, num_classes).to(device)

    # --- Optimizer, Scheduler, and Loss ---
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training
        model.eval() # Keep backbone frozen
        classifier.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                features = model(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += inputs.size(0)

        epoch_loss = running_loss / total_predictions
        epoch_acc = correct_predictions.double() / total_predictions
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        # Validation
        classifier.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)

                features = model(inputs)
                outputs = classifier(features)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct_predictions += torch.sum(preds == labels.data)
                val_total_predictions += inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / val_total_predictions
        val_epoch_acc = val_correct_predictions.double() / val_total_predictions
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

        scheduler.step()

    # --- Save Results ---
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    # Overfitting threshold
    overfitting_threshold = np.array(history['train_acc']) - np.array(history['val_acc'])
    plt.plot(overfitting_threshold, label='Overfitting Threshold (Train-Val Acc)', linestyle='--')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(results_dir, "training_plots.png"))

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Hyperparameters and Final Metrics
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Num Epochs: {num_epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}\n")

    # Save Model
    torch.save(classifier.state_dict(), os.path.join(results_dir, "dinov2_linear_probe.pth"))

    print(f"Training complete. Results saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a linear probe on top of a frozen DINOv2 model.")
    parser.add_argument("--data_dir", type=str, default="./Dataset", help="Directory containing the dataset.")
    parser.add_argument("--results_dir", type=str, default="./results/DINOv2_Linear_Probe", help="Directory to save the results.")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14_reg", help="Name of the DINOv2 model to use.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()

    train_dinov2(args.data_dir, args.results_dir, args.model_name, args.num_epochs, args.batch_size, args.lr)
