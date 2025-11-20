import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from tqdm import tqdm 

def accuracy(output, target, topk=(1, 5)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # Ensure output and target are on the same device and detached
        output = output.detach()
        target = target.detach()
        
        # Check for invalid values
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN or Inf detected in output tensor")
            return [torch.tensor(0.0, device=output.device) for _ in topk]
        
        # Ensure maxk doesn't exceed the number of classes
        num_classes = output.size(1)
        maxk = min(maxk, num_classes)
        
        if maxk <= 0:
            return [torch.tensor(0.0, device=output.device) for _ in topk]

        # Ensure tensor is contiguous for CUDA operations
        output = output.contiguous()
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k = min(k, maxk)  # Ensure k doesn't exceed maxk
            if k > 0:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                res.append(torch.tensor(0.0, device=output.device))
        return res

def train_model(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch on the training set.
    """
    model.train()
    running_loss = 0.0
    total_top1_acc = 0.0
    total_top5_acc = 0.0

    # Wrap the train_loader with tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc="Training")

    for batch_idx, (inputs, labels) in enumerate(progress_bar): # <--- USE THE PROGRESS BAR
        try:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Move accuracy computation outside of gradient context to avoid CUDA issues
                acc1, acc5 = accuracy(outputs.detach(), labels, topk=(1, 5))
                
                loss.backward()
                optimizer.step()
            
            # Synchronize CUDA operations
            if device.type == 'cuda':
                torch.cuda.synchronize()

            running_loss += loss.item() * inputs.size(0)
            total_top1_acc += acc1.item()
            total_top5_acc += acc5.item()
            
        except RuntimeError as e:
            if 'CUDA' in str(e):
                print(f"\nCUDA error at batch {batch_idx}: {e}")
                print("Attempting to clear CUDA cache and continue...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                # Skip this batch and continue
                continue
            else:
                raise

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_top1_acc = total_top1_acc / len(train_loader)
    epoch_top5_acc = total_top5_acc / len(train_loader)
    
    return epoch_loss, epoch_top1_acc, epoch_top5_acc

def validate_model(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Wrap the val_loader with tqdm for a progress bar
    progress_bar = tqdm(val_loader, desc="Validation")

    with torch.no_grad():
        for inputs, labels in progress_bar: # <--- USE THE PROGRESS BAR
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

def get_test_set_performance(model, test_loader, device):
    """
    Performs a final evaluation on the test set with a detailed report.
    """
    model.eval()
    all_preds = []
    all_labels = []

    # Wrap the test_loader with tqdm for a progress bar
    progress_bar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for inputs, labels in progress_bar: # <--- USE THE PROGRESS BAR
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    top1_accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    avg_per_class_accuracy = np.mean(per_class_accuracy)

    performance_summary = {
        "Top-1 Accuracy": top1_accuracy,
        "Average Accuracy Per Class": avg_per_class_accuracy,
        "Mean Precision": precision,
        "Mean Recall": recall,
        "Mean F1-Score": f1
    }
    return performance_summary