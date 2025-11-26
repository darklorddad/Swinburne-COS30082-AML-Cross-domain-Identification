import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from tqdm import tqdm
import json

def freeze_batchnorm_layers(model):
    """
    Freezes all BatchNorm layers in the model (sets them to eval mode).
    This is useful for fine-tuning when the batch size is small.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            # Optional: also turn off gradient computation for BN params
            for param in module.parameters():
                param.requires_grad = False

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

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

def train_model(model, train_loader, criterion, optimizer, device, model_name='convnextv2'):
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
            
            # Mixup / CutMix probabilities
            # Disable Mixup/CutMix for ResNet/Xception as they struggle to learn from it with limited epochs
            use_mixup = False
            if model_name == 'convnextv2':
                if np.random.rand() < 0.5: # 50% chance to apply Mixup/CutMix
                    use_mixup = True
            
            with torch.set_grad_enabled(True):
                if use_mixup:
                    if np.random.rand() < 0.5:
                        # Apply Mixup
                        alpha = 1.0
                        lam = np.random.beta(alpha, alpha)
                        rand_index = torch.randperm(inputs.size(0)).to(device)
                        target_a = labels
                        target_b = labels[rand_index]
                        inputs = lam * inputs + (1 - lam) * inputs[rand_index]
                        
                        outputs = model(inputs)
                        loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
                    else:
                        # Apply CutMix
                        alpha = 1.0
                        lam = np.random.beta(alpha, alpha)
                        rand_index = torch.randperm(inputs.size(0)).to(device)
                        target_a = labels
                        target_b = labels[rand_index]
                        
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        
                        # Adjust lambda to match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                        
                        outputs = model(inputs)
                        loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
                else:
                    # Standard Training
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Move accuracy computation outside of gradient context to avoid CUDA issues
                acc1, acc5 = accuracy(outputs.detach(), labels, topk=(1, 5))
                
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

def get_test_set_performance(model, test_loader, device, with_pairs_indices=None, without_pairs_indices=None):
    """
    Performs a final evaluation on the test set with a detailed report.
    """
    model.eval()
    all_preds = []
    all_labels_np = []
    
    # We store raw outputs (logits) and labels as tensors for Top-K calculations
    all_outputs_tensor = []
    all_labels_tensor = []

    # Wrap the test_loader with tqdm for a progress bar
    progress_bar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for inputs, labels in progress_bar: # <--- USE THE PROGRESS BAR
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # TTA: Original + Horizontal Flip
            outputs = model(inputs)
            
            inputs_flipped = torch.flip(inputs, [3]) # Flip width
            outputs_flipped = model(inputs_flipped)
            
            # Average predictions
            outputs = (outputs + outputs_flipped) / 2
            
            # Collect tensors for split metrics (Overall, With Pairs, Without Pairs)
            all_outputs_tensor.append(outputs)
            all_labels_tensor.append(labels)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_np.extend(labels.cpu().numpy())
            
    # Concatenate all batches
    all_outputs_tensor = torch.cat(all_outputs_tensor)
    all_labels_tensor = torch.cat(all_labels_tensor)

    # --- Calculate Overall, With Pairs, Without Pairs Stats ---
    def calc_metrics_for_subset(indices_set=None):
        """Calculates N, Top-1, Top-5 for a subset of class indices."""
        if indices_set is not None:
            # Create a mask for samples whose label is in the indices_set
            # Move indices_set to device tensor for comparison
            indices_tensor = torch.tensor(list(indices_set), device=device)
            mask = torch.isin(all_labels_tensor, indices_tensor)
            
            subset_outputs = all_outputs_tensor[mask]
            subset_labels = all_labels_tensor[mask]
        else:
            subset_outputs = all_outputs_tensor
            subset_labels = all_labels_tensor
            
        N = len(subset_labels)
        if N == 0:
            return N, 0.0, 0.0
        
        # Use the accuracy function defined above (returns percentages)
        acc1, acc5 = accuracy(subset_outputs, subset_labels, topk=(1, 5))
        return N, acc1.item(), acc5.item()

    # Calculate stats
    n_all, t1_all, t5_all = calc_metrics_for_subset(None) # Overall
    
    stats_output = {
        "overall": {
            "N": n_all,
            "top1": t1_all,
            "top5": t5_all
        },
        "with_pairs": {
            "N": 0, "top1": 0.0, "top5": 0.0
        },
        "without_pairs": {
            "N": 0, "top1": 0.0, "top5": 0.0
        }
    }

    if with_pairs_indices is not None:
        n_wp, t1_wp, t5_wp = calc_metrics_for_subset(with_pairs_indices)
        stats_output["with_pairs"] = {"N": n_wp, "top1": t1_wp, "top5": t5_wp}
        
    if without_pairs_indices is not None:
        n_wop, t1_wop, t5_wop = calc_metrics_for_subset(without_pairs_indices)
        stats_output["without_pairs"] = {"N": n_wop, "top1": t1_wop, "top5": t5_wop}
    
    print("\n" + "="*30)
    print("DETAILED PERFORMANCE METRICS (JSON)")
    print("="*30)
    print(json.dumps(stats_output, indent=4))
    print("="*30 + "\n")
    
    # Save stats to JSON file
    # We don't have access to config here, so we'll return it in the summary
    # or save it with a generic name if needed, but returning it is cleaner.
    
    # Standard Metrics Calculation (existing code)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels_np, all_preds, average='macro', zero_division=0)
    top1_accuracy = accuracy_score(all_labels_np, all_preds)
    cm = confusion_matrix(all_labels_np, all_preds)
    # Handle potential division by zero in per-class accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        per_class_accuracy = np.nan_to_num(per_class_accuracy) # Replace NaNs with 0
    avg_per_class_accuracy = np.mean(per_class_accuracy)

    performance_summary = {
        "Top-1 Accuracy": top1_accuracy,
        "Average Accuracy Per Class": avg_per_class_accuracy,
        "Mean Precision": precision,
        "Mean Recall": recall,
        "Mean F1-Score": f1,
        "All Predictions": all_preds,
        "All Labels": all_labels_np,
        "JSON_Stats": stats_output # <--- Added this
    }
    return performance_summary